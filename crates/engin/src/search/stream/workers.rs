//! Persistent worker version of the S2 streaming pipeline.
//!
//! Reference: LC3 overview, "Workers":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! Workers exchange owned events only. They never borrow a mutable DFS tree or
//! backend workspace from another worker.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError, TrySendError};
use parking_lot::{Condvar, Mutex};
use xiangqi_core::{GameResult, PositionHistory};

use crate::neural::backend::{Backend, EvalPosition};
use crate::EnginError;

use super::{
    select_edge, terminal_value_for_side_to_move, BackpropEvent, ExpansionState, NodeEvent, NodeKey, NodeRepository,
    SearchGeneration, StreamNode, StreamPipelineConfig, StreamStats,
};

const RECEIVE_POLL: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamWorkerConfig {
    pub pipeline: StreamPipelineConfig,
    pub gather_workers: usize,
    pub backprop_workers: usize,
}

impl Default for StreamWorkerConfig {
    fn default() -> Self {
        Self {
            pipeline: StreamPipelineConfig::default(),
            gather_workers: 2,
            backprop_workers: 1,
        }
    }
}

impl StreamWorkerConfig {
    fn validate(self) {
        self.pipeline.validate();
        assert!(self.gather_workers > 0, "stream requires at least one gather worker");
        assert!(
            self.backprop_workers > 0,
            "stream requires at least one backprop worker"
        );
    }
}

struct SharedPipeline {
    backend: Arc<dyn Backend>,
    repository: Arc<NodeRepository>,
    generation: SearchGeneration,
    cpuct: f32,
    stopping: AtomicBool,
    outstanding: AtomicUsize,
    completed: AtomicU64,
    collisions: AtomicU64,
    network_batches: AtomicU64,
    network_evaluations: AtomicU64,
    error: Mutex<Option<EnginError>>,
    idle_lock: Mutex<()>,
    idle: Condvar,
    gather_tx: Sender<NodeEvent>,
    eval_tx: Sender<NodeEvent>,
    backprop_tx: Sender<BackpropEvent>,
}

impl SharedPipeline {
    fn finish(&self, completed: bool) {
        if completed {
            self.completed.fetch_add(1, Ordering::AcqRel);
        }
        let previous = self.outstanding.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "stream outstanding task underflow");
        if previous == 1 {
            let _guard = self.idle_lock.lock();
            self.idle.notify_all();
        }
    }

    fn cancel_and_finish(&self, event: NodeEvent, collision: bool) {
        event.cancel();
        if collision {
            self.collisions.fetch_add(1, Ordering::AcqRel);
        }
        self.finish(false);
    }

    fn fail(&self, error: EnginError) {
        let mut current = self.error.lock();
        if current.is_none() {
            *current = Some(error);
        }
        self.stopping.store(true, Ordering::Release);
        self.idle.notify_all();
    }

    fn send_eval(&self, event: NodeEvent) {
        if self.stopping.load(Ordering::Acquire) {
            self.cancel_and_finish(event, false);
            return;
        }
        if let Err(error) = self.eval_tx.send(event) {
            self.cancel_and_finish(error.0, false);
        }
    }

    fn send_backprop(&self, event: BackpropEvent) {
        if self.stopping.load(Ordering::Acquire) {
            event.cancel();
            self.finish(false);
            return;
        }
        if let Err(error) = self.backprop_tx.send(event) {
            error.0.cancel();
            self.finish(false);
        }
    }

    fn stats(&self) -> StreamStats {
        StreamStats {
            completed_playouts: self.completed.load(Ordering::Acquire),
            collisions: self.collisions.load(Ordering::Acquire),
            network_batches: self.network_batches.load(Ordering::Acquire),
            network_evaluations: self.network_evaluations.load(Ordering::Acquire),
        }
    }
}

/// Persistent LC3-style stages. Eval is intentionally one worker so one
/// `BackendComputation` owns each batch; Gather and Backprop can scale without
/// sharing its workspace.
pub struct StreamWorkerPipeline {
    shared: Arc<SharedPipeline>,
    root_history: Arc<PositionHistory>,
    root_key: NodeKey,
    queue_capacity: usize,
    workers: Vec<JoinHandle<()>>,
}

impl StreamWorkerPipeline {
    pub fn new(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        root_history: Arc<PositionHistory>,
        config: StreamWorkerConfig,
    ) -> Self {
        config.validate();
        let (gather_tx, gather_rx) = bounded(config.pipeline.queue_capacity);
        let (eval_tx, eval_rx) = bounded(config.pipeline.queue_capacity);
        let (backprop_tx, backprop_rx) = bounded(config.pipeline.queue_capacity);
        let root_key = NodeKey::root(root_history.last().hash());
        let shared = Arc::new(SharedPipeline {
            backend,
            repository: Arc::new(NodeRepository::default()),
            generation,
            cpuct: config.pipeline.cpuct(),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            network_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            gather_tx,
            eval_tx,
            backprop_tx,
        });
        let mut workers = Vec::with_capacity(config.gather_workers + config.backprop_workers + 1);
        for _ in 0..config.gather_workers {
            let shared = Arc::clone(&shared);
            let receiver = gather_rx.clone();
            workers.push(thread::spawn(move || gather_worker(shared, receiver)));
        }
        {
            let shared = Arc::clone(&shared);
            workers.push(thread::spawn(move || {
                eval_worker(shared, eval_rx, config.pipeline.eval_batch_size)
            }));
        }
        for _ in 0..config.backprop_workers {
            let shared = Arc::clone(&shared);
            let receiver = backprop_rx.clone();
            workers.push(thread::spawn(move || backprop_worker(shared, receiver)));
        }
        Self {
            shared,
            root_history,
            root_key,
            queue_capacity: config.pipeline.queue_capacity,
            workers,
        }
    }

    pub fn repository(&self) -> &Arc<NodeRepository> {
        &self.shared.repository
    }

    pub fn root_key(&self) -> NodeKey {
        self.root_key
    }

    pub fn stats(&self) -> StreamStats {
        self.shared.stats()
    }

    pub fn submit_playout(&self) -> Result<(), EnginError> {
        self.submit_event(NodeEvent::root(self.shared.generation, Arc::clone(&self.root_history)))
    }

    pub fn submit_event(&self, event: NodeEvent) -> Result<(), EnginError> {
        if self.shared.stopping.load(Ordering::Acquire) {
            event.cancel();
            return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
        }
        if event.generation != self.shared.generation {
            event.cancel();
            return Err(EnginError::PortIncomplete("stale stream search generation"));
        }
        self.shared.outstanding.fetch_add(1, Ordering::AcqRel);
        match self.shared.gather_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                self.shared.cancel_and_finish(event, false);
                Err(EnginError::PortIncomplete("stream gather queue full"))
            }
            Err(TrySendError::Disconnected(event)) => {
                self.shared.cancel_and_finish(event, false);
                Err(EnginError::PortIncomplete("stream gather queue disconnected"))
            }
        }
    }

    pub fn run_playouts(&self, count: u64) -> Result<StreamStats, EnginError> {
        let target = self.stats().completed_playouts + count;
        while self.stats().completed_playouts < target {
            let root_is_expanded = self
                .shared
                .repository
                .get(self.root_key)
                .is_some_and(|root| root.expansion_state() == ExpansionState::Expanded);
            let submit_count = if root_is_expanded {
                self.queue_capacity
                    .min((target - self.stats().completed_playouts) as usize)
            } else {
                1
            };
            for _ in 0..submit_count {
                self.submit_playout()?;
            }
            self.wait_for_idle()?;
        }
        Ok(self.stats())
    }

    pub fn wait_for_idle(&self) -> Result<(), EnginError> {
        let mut guard = self.shared.idle_lock.lock();
        while self.shared.outstanding.load(Ordering::Acquire) != 0 && self.shared.error.lock().is_none() {
            self.shared.idle.wait(&mut guard);
        }
        if let Some(error) = self.shared.error.lock().clone() {
            return Err(error);
        }
        Ok(())
    }

    pub fn stop_and_join(&mut self) {
        self.shared.stopping.store(true, Ordering::Release);
        let _ = self.wait_for_idle();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for StreamWorkerPipeline {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn gather_worker(shared: Arc<SharedPipeline>, receiver: Receiver<NodeEvent>) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(event) => {
                if shared.stopping.load(Ordering::Acquire) {
                    shared.cancel_and_finish(event, false);
                    continue;
                }
                process_gather_event(&shared, event);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn process_gather_event(shared: &SharedPipeline, mut event: NodeEvent) {
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            shared.cancel_and_finish(event, false);
            return;
        }
        let node = shared.repository.get_or_insert(event.node_key);
        match node.expansion_state() {
            ExpansionState::Unexpanded => {
                if node.try_begin_evaluation() {
                    shared.send_eval(event);
                    return;
                }
            }
            ExpansionState::Evaluating => {
                shared.cancel_and_finish(event, true);
                return;
            }
            ExpansionState::Terminal => {
                let (value, draw) = node.terminal_value().expect("terminal stream value");
                shared.send_backprop(BackpropEvent {
                    node: event,
                    value,
                    draw,
                });
                return;
            }
            ExpansionState::Expanded => {
                let edges = node.edges();
                let edge_index = select_edge(&edges, node.completed_visits(), shared.cpuct)
                    .expect("expanded stream node must have an edge");
                let reservation = node.reserve_edge(edge_index).expect("selected stream edge");
                let child_key = event.node_key.child(reservation.mv());
                event = event.descend(child_key, reservation);
            }
        }
    }
}

struct PendingEval {
    event: NodeEvent,
    node: Arc<StreamNode>,
    ticket: crate::neural::backend::EvalTicket,
    legal_moves: Vec<xiangqi_core::Move>,
}

fn eval_worker(shared: Arc<SharedPipeline>, receiver: Receiver<NodeEvent>, batch_size: usize) {
    loop {
        let first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(event) => event,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        let mut events = vec![first];
        while events.len() < batch_size {
            match receiver.try_recv() {
                Ok(event) => events.push(event),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        if shared.stopping.load(Ordering::Acquire) {
            for event in events {
                shared.cancel_and_finish(event, false);
            }
            continue;
        }
        if let Err(error) = process_eval_events(&shared, events) {
            shared.fail(error);
        }
    }
}

fn process_eval_events(shared: &SharedPipeline, events: Vec<NodeEvent>) -> Result<(), EnginError> {
    let computation = shared.backend.create_computation()?;
    let mut pending = Vec::new();
    for event in events {
        let node = shared.repository.get_or_insert(event.node_key);
        let history = event.variation.replay_history();
        match history.compute_game_result() {
            GameResult::Undecided => {
                let legal_moves = history.last().board().generate_legal_moves();
                let input = EvalPosition {
                    positions: history.positions().to_vec(),
                    legal_moves: legal_moves.clone(),
                };
                match computation.add_input(input) {
                    Ok((_, ticket)) => pending.push(PendingEval {
                        event,
                        node,
                        ticket,
                        legal_moves,
                    }),
                    Err(error) => {
                        event.cancel();
                        node.abort_evaluation();
                        shared.finish(false);
                        cancel_pending(shared, pending);
                        return Err(error);
                    }
                }
            }
            result => {
                let (value, draw) = terminal_value_for_side_to_move(result, history.last().is_black_to_move());
                node.mark_terminal(value, draw);
                shared.send_backprop(BackpropEvent {
                    node: event,
                    value,
                    draw,
                });
            }
        }
    }
    let used_batch_size = computation.used_batch_size();
    if used_batch_size > 0 {
        if let Err(error) = computation.compute_blocking() {
            cancel_pending(shared, pending);
            return Err(error);
        }
        shared.network_batches.fetch_add(1, Ordering::AcqRel);
        shared
            .network_evaluations
            .fetch_add(used_batch_size as u64, Ordering::AcqRel);
    }
    while let Some(item) = pending.pop() {
        let eval = match computation.take_result(item.ticket) {
            Ok(eval) => eval,
            Err(error) => {
                item.event.cancel();
                item.node.abort_evaluation();
                shared.finish(false);
                cancel_pending(shared, pending);
                return Err(error);
            }
        };
        if eval.policies.len() != item.legal_moves.len() {
            item.event.cancel();
            item.node.abort_evaluation();
            shared.finish(false);
            cancel_pending(shared, pending);
            return Err(EnginError::PortIncomplete("stream backend policy length"));
        }
        item.node.publish_edges(
            item.legal_moves
                .into_iter()
                .zip(eval.policies.iter().copied())
                .collect(),
        );
        shared.send_backprop(BackpropEvent {
            node: item.event,
            value: eval.wl,
            draw: eval.d,
        });
    }
    Ok(())
}

fn cancel_pending(shared: &SharedPipeline, pending: Vec<PendingEval>) {
    for item in pending {
        item.event.cancel();
        item.node.abort_evaluation();
        shared.finish(false);
    }
}

fn backprop_worker(shared: Arc<SharedPipeline>, receiver: Receiver<BackpropEvent>) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(event) => {
                if shared.stopping.load(Ordering::Acquire) {
                    event.cancel();
                    shared.finish(false);
                } else {
                    event.complete(&shared.repository);
                    shared.finish(true);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use super::{StreamWorkerConfig, StreamWorkerPipeline};
    use crate::neural::backend::UniformBackend;
    use crate::search::stream::SearchGeneration;

    fn startpos_history() -> Arc<PositionHistory> {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        Arc::new(PositionHistory::from_positions(state.positions()))
    }

    #[test]
    fn persistent_workers_complete_batched_playouts_and_join() {
        let mut pipeline = StreamWorkerPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(21),
            startpos_history(),
            StreamWorkerConfig {
                pipeline: super::StreamPipelineConfig {
                    queue_capacity: 8,
                    eval_batch_size: 4,
                    ..super::StreamPipelineConfig::default()
                },
                gather_workers: 2,
                backprop_workers: 1,
            },
        );
        let stats = pipeline.run_playouts(32).expect("playouts");
        assert_eq!(stats.completed_playouts, 32);
        assert!(stats.network_batches > 0);
        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
        }
        pipeline.stop_and_join();
    }

    #[test]
    fn stop_join_drains_in_flight_reservations() {
        let mut pipeline = StreamWorkerPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(22),
            startpos_history(),
            StreamWorkerConfig {
                pipeline: super::StreamPipelineConfig {
                    queue_capacity: 16,
                    eval_batch_size: 8,
                    ..super::StreamPipelineConfig::default()
                },
                gather_workers: 4,
                backprop_workers: 2,
            },
        );
        pipeline.run_playouts(1).expect("expand root");
        for _ in 0..16 {
            pipeline.submit_playout().expect("queued root event");
        }
        pipeline.stop_and_join();

        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
        }
    }
}
