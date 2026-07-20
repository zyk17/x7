//! Bounded cooperative streaming pipeline.
//!
//! Reference: LC3 overview, "Workers" and "No batch concept":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! This first S2 implementation keeps the stages on one controller thread but
//! uses the final owned queue boundaries. It establishes queue backpressure,
//! generation gating, batch evaluation, and stop/drain before stages are moved
//! to persistent threads.

use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError, TrySendError};
use xiangqi_core::{GameResult, PositionHistory};

use crate::neural::backend::{Backend, EvalPosition};
use crate::EnginError;

use super::{
    select_edge, terminal_value_for_side_to_move, BackpropEvent, ExpansionState, NodeEvent, NodeKey, NodeRepository,
    SearchGeneration, StreamStats,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamPipelineConfig {
    pub queue_capacity: usize,
    pub eval_batch_size: usize,
    pub cpuct_milli: u32,
}

impl Default for StreamPipelineConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 256,
            eval_batch_size: 32,
            cpuct_milli: 1_000,
        }
    }
}

impl StreamPipelineConfig {
    pub(crate) fn cpuct(self) -> f32 {
        self.cpuct_milli as f32 / 1_000.0
    }

    pub(crate) fn validate(self) {
        assert!(self.queue_capacity > 0, "stream queue capacity must be non-zero");
        assert!(self.eval_batch_size > 0, "stream eval batch size must be non-zero");
        assert!(
            self.eval_batch_size <= self.queue_capacity,
            "stream eval batch size must fit the backprop queue"
        );
    }
}

/// Queue-based stream search controller. Queues only transport owned events;
/// repository nodes and edge reservations retain all mutable search state.
pub struct StreamPipeline {
    backend: Arc<dyn Backend>,
    repository: Arc<NodeRepository>,
    generation: SearchGeneration,
    root_history: Arc<PositionHistory>,
    root_key: NodeKey,
    config: StreamPipelineConfig,
    gather_tx: Sender<NodeEvent>,
    gather_rx: Receiver<NodeEvent>,
    eval_tx: Sender<NodeEvent>,
    eval_rx: Receiver<NodeEvent>,
    backprop_tx: Sender<BackpropEvent>,
    backprop_rx: Receiver<BackpropEvent>,
    stopped: bool,
    stats: StreamStats,
}

impl StreamPipeline {
    pub fn new(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        root_history: Arc<PositionHistory>,
        config: StreamPipelineConfig,
    ) -> Self {
        config.validate();
        let (gather_tx, gather_rx) = bounded(config.queue_capacity);
        let (eval_tx, eval_rx) = bounded(config.queue_capacity);
        let (backprop_tx, backprop_rx) = bounded(config.queue_capacity);
        let root_key = NodeKey::root(root_history.last().hash());
        Self {
            backend,
            repository: Arc::new(NodeRepository::default()),
            generation,
            root_history,
            root_key,
            config,
            gather_tx,
            gather_rx,
            eval_tx,
            eval_rx,
            backprop_tx,
            backprop_rx,
            stopped: false,
            stats: StreamStats::default(),
        }
    }

    pub fn repository(&self) -> &Arc<NodeRepository> {
        &self.repository
    }

    pub fn root_key(&self) -> NodeKey {
        self.root_key
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    pub fn submit_playout(&mut self) -> Result<(), EnginError> {
        let event = NodeEvent::root(self.generation, Arc::clone(&self.root_history));
        self.submit_event(event)
    }

    /// Rejects old events before they can reach Gather after a UCI replacement.
    pub fn submit_event(&mut self, event: NodeEvent) -> Result<(), EnginError> {
        if self.stopped {
            event.cancel();
            return Err(EnginError::PortIncomplete("stream pipeline is stopped"));
        }
        if event.generation != self.generation {
            event.cancel();
            return Err(EnginError::PortIncomplete("stale stream search generation"));
        }
        match self.gather_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream gather queue full"))
            }
            Err(TrySendError::Disconnected(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream gather queue disconnected"))
            }
        }
    }

    pub fn run_playouts(&mut self, count: u64) -> Result<StreamStats, EnginError> {
        let target = self.stats.completed_playouts + count;
        while self.stats.completed_playouts < target {
            let root_is_expanded = self
                .repository
                .get(self.root_key)
                .is_some_and(|root| root.expansion_state() == ExpansionState::Expanded);
            let submit_count = if root_is_expanded {
                self.config
                    .queue_capacity
                    .min((target - self.stats.completed_playouts) as usize)
            } else {
                1
            };
            for _ in 0..submit_count {
                self.submit_playout()?;
            }
            self.pump_until_stalled()?;
        }
        Ok(self.stats)
    }

    /// Executes all currently ready stages. Returns only after all submitted
    /// events either complete, collide, or remain blocked by an external stop.
    pub fn pump_until_stalled(&mut self) -> Result<(), EnginError> {
        loop {
            let mut progressed = self.process_backprop();
            while self.eval_rx.len() < self.config.eval_batch_size {
                if !self.process_gather()? {
                    break;
                }
                progressed = true;
            }
            progressed |= self.process_eval_batch()?;
            progressed |= self.process_backprop();
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Stop is a generation boundary: queued events must cancel reservations
    /// instead of being silently dropped. A later UCI search creates a new
    /// pipeline with a new generation.
    pub fn stop_and_drain(&mut self) {
        self.stopped = true;
        while let Ok(event) = self.gather_rx.try_recv() {
            event.cancel();
        }
        while let Ok(event) = self.eval_rx.try_recv() {
            event.cancel();
        }
        while let Ok(event) = self.backprop_rx.try_recv() {
            event.cancel();
        }
    }

    fn process_gather(&mut self) -> Result<bool, EnginError> {
        let event = match self.gather_rx.try_recv() {
            Ok(event) => event,
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => return Ok(false),
        };
        let node = self.repository.get_or_insert(event.node_key);
        match node.expansion_state() {
            ExpansionState::Unexpanded => {
                if node.try_begin_evaluation() {
                    self.forward_eval(event)?;
                } else {
                    self.forward_gather(event)?;
                }
            }
            ExpansionState::Evaluating => {
                event.cancel();
                self.stats.collisions += 1;
            }
            ExpansionState::Terminal => {
                let (value, draw) = node.terminal_value().expect("terminal stream value");
                self.forward_backprop(BackpropEvent {
                    node: event,
                    value,
                    draw,
                })?;
            }
            ExpansionState::Expanded => {
                let edges = node.edges();
                let edge_index = select_edge(&edges, node.completed_visits(), self.config.cpuct())
                    .expect("expanded stream node must have an edge");
                let reservation = node.reserve_edge(edge_index).expect("selected stream edge");
                let child_key = event.node_key.child(reservation.mv());
                self.forward_gather(event.descend(child_key, reservation))?;
            }
        }
        Ok(true)
    }

    fn process_eval_batch(&mut self) -> Result<bool, EnginError> {
        let mut events = Vec::with_capacity(self.config.eval_batch_size);
        while events.len() < self.config.eval_batch_size {
            match self.eval_rx.try_recv() {
                Ok(event) => events.push(event),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        if events.is_empty() {
            return Ok(false);
        }

        let computation = self.backend.create_computation()?;
        let mut pending = Vec::new();
        for event in events {
            let node = self.repository.get_or_insert(event.node_key);
            let history = event.variation.replay_history();
            match history.compute_game_result() {
                GameResult::Undecided => {
                    let legal_moves = history.last().board().generate_legal_moves();
                    let input = EvalPosition {
                        positions: history.positions().to_vec(),
                        legal_moves,
                    };
                    match computation.add_input(input) {
                        Ok((_, ticket)) => pending.push((event, node, ticket)),
                        Err(error) => {
                            event.cancel();
                            node.abort_evaluation();
                            self.stop_and_drain();
                            return Err(error);
                        }
                    }
                }
                result => {
                    let (value, draw) = terminal_value_for_side_to_move(result, history.last().is_black_to_move());
                    node.mark_terminal(value, draw);
                    self.forward_backprop(BackpropEvent {
                        node: event,
                        value,
                        draw,
                    })?;
                }
            }
        }

        let used_batch_size = computation.used_batch_size();
        if used_batch_size > 0 {
            if let Err(error) = computation.compute_blocking() {
                for (event, node, _) in pending {
                    event.cancel();
                    node.abort_evaluation();
                }
                self.stop_and_drain();
                return Err(error);
            }
            self.stats.network_batches += 1;
            self.stats.network_evaluations += used_batch_size as u64;
        }
        for (event, node, ticket) in pending {
            let eval = match computation.take_result(ticket) {
                Ok(eval) => eval,
                Err(error) => {
                    event.cancel();
                    node.abort_evaluation();
                    self.stop_and_drain();
                    return Err(error);
                }
            };
            let legal_count = event
                .variation
                .replay_history()
                .last()
                .board()
                .generate_legal_moves()
                .len();
            if eval.policies.len() != legal_count {
                event.cancel();
                node.abort_evaluation();
                self.stop_and_drain();
                return Err(EnginError::PortIncomplete("stream backend policy length"));
            }
            let legal_moves = event.variation.replay_history().last().board().generate_legal_moves();
            node.publish_edges(legal_moves.into_iter().zip(eval.policies.iter().copied()).collect());
            self.forward_backprop(BackpropEvent {
                node: event,
                value: eval.wl,
                draw: eval.d,
            })?;
        }
        Ok(true)
    }

    fn process_backprop(&mut self) -> bool {
        let mut progressed = false;
        while let Ok(event) = self.backprop_rx.try_recv() {
            event.complete(&self.repository);
            self.stats.completed_playouts += 1;
            progressed = true;
        }
        progressed
    }

    fn forward_gather(&self, event: NodeEvent) -> Result<(), EnginError> {
        match self.gather_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream gather queue backpressure"))
            }
            Err(TrySendError::Disconnected(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream gather queue disconnected"))
            }
        }
    }

    fn forward_eval(&self, event: NodeEvent) -> Result<(), EnginError> {
        match self.eval_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream eval queue backpressure"))
            }
            Err(TrySendError::Disconnected(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream eval queue disconnected"))
            }
        }
    }

    fn forward_backprop(&self, event: BackpropEvent) -> Result<(), EnginError> {
        match self.backprop_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream backprop queue backpressure"))
            }
            Err(TrySendError::Disconnected(event)) => {
                event.cancel();
                Err(EnginError::PortIncomplete("stream backprop queue disconnected"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use super::{StreamPipeline, StreamPipelineConfig};
    use crate::neural::backend::UniformBackend;
    use crate::search::stream::SearchGeneration;

    fn startpos_history() -> Arc<PositionHistory> {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        Arc::new(PositionHistory::from_positions(state.positions()))
    }

    #[test]
    fn bounded_pipeline_batches_and_drains_all_reservations() {
        let mut pipeline = StreamPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(11),
            startpos_history(),
            StreamPipelineConfig {
                queue_capacity: 8,
                eval_batch_size: 4,
                ..StreamPipelineConfig::default()
            },
        );
        let stats = pipeline.run_playouts(32).expect("stream pipeline");
        assert_eq!(stats.completed_playouts, 32);
        assert!(stats.network_batches > 0);
        assert!(stats.network_batches < stats.completed_playouts);
        assert!(stats.network_evaluations >= stats.network_batches);
        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        assert_eq!(root.completed_visits(), 32);
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
        }
    }

    #[test]
    fn stale_generation_is_rejected_before_gather() {
        let root_history = startpos_history();
        let mut pipeline = StreamPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(12),
            Arc::clone(&root_history),
            StreamPipelineConfig::default(),
        );
        let stale = super::NodeEvent::root(SearchGeneration(11), root_history);
        assert!(pipeline.submit_event(stale).is_err());
        pipeline.pump_until_stalled().expect("no stale work remains");
        assert_eq!(pipeline.stats().completed_playouts, 0);
    }

    #[test]
    fn stop_drain_cancels_queued_descendant_reservation() {
        let mut pipeline = StreamPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(13),
            startpos_history(),
            StreamPipelineConfig::default(),
        );
        pipeline.run_playouts(1).expect("expand root");
        pipeline.submit_playout().expect("root event");
        assert!(pipeline.process_gather().expect("gather root"));

        pipeline.stop_and_drain();
        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
        }
        assert!(pipeline.submit_playout().is_err());
    }
}
