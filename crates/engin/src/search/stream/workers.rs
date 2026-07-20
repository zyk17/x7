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
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, SendTimeoutError, Sender, TryRecvError};
use parking_lot::{Condvar, Mutex};
use xiangqi_core::{GameResult, PositionHistory};

use crate::neural::backend::{Backend, EvalPosition};
use crate::EnginError;

use super::{
    select_edge, terminal_value_for_side_to_move, BackpropEvent, ExpansionState, NodeEvent, NodeKey, NodeRepository,
    SearchGeneration, StreamNode, StreamPipelineConfig, StreamStats,
};

const RECEIVE_POLL: Duration = Duration::from_millis(10);

/// Search budget consumed relative to the pipeline's current completed
/// playout count. A later UCI watchdog maps `go nodes` / `go movetime` onto
/// this boundary; workers themselves only observe the immutable limit and
/// the explicit `request_stop()` signal.
///
/// Reference: LC3 overview, "Stats Collection" and "WatchdogWorker".
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamSearchLimits {
    pub max_playouts: Option<u64>,
    pub deadline: Option<Instant>,
}

impl StreamSearchLimits {
    fn is_exhausted(self, completed: u64, target: u64) -> bool {
        completed >= target || self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
    }
}

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

    /// Requests a normal stream-search stop without tearing down worker
    /// threads. Gather/Eval/Backprop cancel every unfinished event and its
    /// edge reservation before becoming idle. This is the boundary a later
    /// UCI controller will use for `stop`; `stop_and_join()` remains the
    /// terminal owner cleanup path.
    ///
    /// Reference: LC3 overview, "Watchdog" and worker stop coordination.
    pub fn request_stop(&self) {
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.idle.notify_all();
    }

    pub fn is_stopping(&self) -> bool {
        self.shared.stopping.load(Ordering::Acquire)
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
        let mut event = event;
        loop {
            if self.shared.stopping.load(Ordering::Acquire) {
                self.shared.cancel_and_finish(event, false);
                return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
            }
            // LC3 workers communicate through bounded event queues. Queue
            // capacity is normal backpressure, not a failed search request;
            // use a short timeout so a concurrent Stop can still interrupt a
            // producer that is waiting for Gather capacity.
            match self.shared.gather_tx.send_timeout(event, RECEIVE_POLL) {
                Ok(()) => return Ok(()),
                Err(SendTimeoutError::Timeout(returned)) => event = returned,
                Err(SendTimeoutError::Disconnected(returned)) => {
                    self.shared.cancel_and_finish(returned, false);
                    return Err(EnginError::PortIncomplete("stream gather queue disconnected"));
                }
            }
        }
    }

    pub fn run_playouts(&self, count: u64) -> Result<StreamStats, EnginError> {
        self.run_with_limits(StreamSearchLimits {
            max_playouts: Some(count),
            deadline: None,
        })
    }

    /// Runs owned stream events until a relative playout budget, deadline, or
    /// explicit stop is reached. It always waits for submitted events to
    /// complete or cancel before returning, so root snapshots never observe
    /// a leaked edge reservation.
    pub fn run_with_limits(&self, limits: StreamSearchLimits) -> Result<StreamStats, EnginError> {
        let initial_completed = self.stats().completed_playouts;
        let target = initial_completed.saturating_add(limits.max_playouts.unwrap_or(u64::MAX));
        while !self.is_stopping() && !limits.is_exhausted(self.stats().completed_playouts, target) {
            let root_is_expanded = self
                .shared
                .repository
                .get(self.root_key)
                .is_some_and(|root| root.expansion_state() == ExpansionState::Expanded);
            let submit_count = if root_is_expanded {
                usize::try_from(target.saturating_sub(self.stats().completed_playouts))
                    .unwrap_or(usize::MAX)
                    .min(self.queue_capacity)
            } else {
                1
            };
            for _ in 0..submit_count {
                if self.is_stopping() || limits.is_exhausted(self.stats().completed_playouts, target) {
                    break;
                }
                if let Err(error) = self.submit_playout() {
                    if self.is_stopping() {
                        break;
                    }
                    return Err(error);
                }
            }
            self.wait_for_idle()?;
        }
        // A requested stop is a normal search outcome. `wait_for_idle()` has
        // already ensured every queued event either completed or cancelled its
        // reservation, so callers can safely snapshot the partial tree.
        self.wait_for_idle()?;
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
        self.request_stop();
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
    let computation = match shared.backend.create_computation() {
        Ok(computation) => computation,
        Err(error) => {
            cancel_unstarted_eval_events(shared, events.into_iter());
            return Err(error);
        }
    };
    let mut pending = Vec::new();
    let mut events = events.into_iter();
    while let Some(event) = events.next() {
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
                        cancel_unstarted_eval_events(shared, events);
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

/// Cancels events that Gather already claimed for Eval but whose computation
/// was never created or accepted. Every event owns its reservations, so this
/// is the only safe failure path that can release in-flight edges without a
/// repository-wide lock.
fn cancel_unstarted_eval_events(shared: &SharedPipeline, events: impl Iterator<Item = NodeEvent>) {
    for event in events {
        let node = shared.repository.get_or_insert(event.node_key);
        if node.expansion_state() == ExpansionState::Evaluating {
            node.abort_evaluation();
        }
        event.cancel();
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
    use std::thread;
    use std::time::{Duration, Instant};

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

    use super::{StreamSearchLimits, StreamWorkerConfig, StreamWorkerPipeline};
    use crate::neural::backend::{Backend, BackendAttributes, BackendComputation, EvalResult, UniformBackend};
    use crate::search::stream::{root_stats, SearchGeneration, StreamSearch};
    use crate::EnginError;

    struct FailingComputationBackend;

    impl Backend for FailingComputationBackend {
        fn evaluate(&self, _history: &PositionHistory, _legal_moves: &[Move]) -> Arc<EvalResult> {
            unreachable!("stream worker must use BackendComputation")
        }

        fn attributes(&self) -> BackendAttributes {
            BackendAttributes::default()
        }

        fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
            Err(EnginError::Onnx("test computation failure".to_owned()))
        }
    }

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

    #[test]
    fn bounded_gather_queue_backpressures_without_failing_submission() {
        let mut pipeline = StreamWorkerPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(23),
            startpos_history(),
            StreamWorkerConfig {
                pipeline: super::StreamPipelineConfig {
                    queue_capacity: 1,
                    eval_batch_size: 1,
                    ..super::StreamPipelineConfig::default()
                },
                gather_workers: 1,
                backprop_workers: 1,
            },
        );
        pipeline.run_playouts(1).expect("expand root");
        for _ in 0..32 {
            pipeline.submit_playout().expect("bounded queue waits for gather");
        }
        pipeline.wait_for_idle().expect("all submitted work drains");
        pipeline.stop_and_join();
    }

    #[test]
    fn requested_stop_returns_partial_stats_and_releases_reservations() {
        let mut pipeline = StreamWorkerPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(24),
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

        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(1));
                pipeline.request_stop();
            });
            let stats = pipeline.run_playouts(1_000_000).expect("normal stop");
            assert!(pipeline.is_stopping());
            assert!(stats.completed_playouts < 1_000_000);
        });

        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
        }
        pipeline.stop_and_join();
    }

    #[test]
    fn expired_deadline_submits_no_new_playout() {
        let mut pipeline = StreamWorkerPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(25),
            startpos_history(),
            StreamWorkerConfig::default(),
        );
        let stats = pipeline
            .run_with_limits(StreamSearchLimits {
                max_playouts: Some(64),
                deadline: Some(Instant::now()),
            })
            .expect("expired deadline is a normal result");
        assert_eq!(stats.completed_playouts, 0);
        assert!(pipeline.repository().get(pipeline.root_key()).is_none());
        pipeline.stop_and_join();
    }

    #[test]
    fn serial_and_workers_share_fixed_playout_root_contract() {
        let count = 64;
        let mut serial = StreamSearch::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(26),
            startpos_history(),
            1.0,
        );
        serial.run_playouts(count).expect("serial playouts");
        let serial_root = root_stats(serial.repository(), serial.root_key()).expect("serial root");

        let mut workers = StreamWorkerPipeline::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(27),
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
        let worker_stats = workers.run_playouts(count).expect("worker playouts");
        let worker_root = root_stats(workers.repository(), workers.root_key()).expect("worker root");

        assert_eq!(serial_root.completed_visits, count as u32);
        assert_eq!(worker_stats.completed_playouts, count);
        assert_eq!(worker_root.completed_visits, count as u32);
        assert_eq!(serial_root.edges.len(), worker_root.edges.len());
        assert!(serial_root
            .edges
            .iter()
            .all(|edge| edge.started_visits == edge.completed_visits));
        assert!(worker_root
            .edges
            .iter()
            .all(|edge| edge.started_visits == edge.completed_visits));

        let mut serial_priors = serial_root
            .edges
            .iter()
            .map(|edge| (edge.mv, edge.prior))
            .collect::<Vec<_>>();
        let mut worker_priors = worker_root
            .edges
            .iter()
            .map(|edge| (edge.mv, edge.prior))
            .collect::<Vec<_>>();
        serial_priors.sort_unstable_by_key(|(mv, _)| mv.raw());
        worker_priors.sort_unstable_by_key(|(mv, _)| mv.raw());
        assert_eq!(serial_priors, worker_priors);

        workers.stop_and_join();
    }

    #[test]
    fn computation_creation_failure_drains_claimed_events() {
        let mut pipeline = StreamWorkerPipeline::new(
            Arc::new(FailingComputationBackend),
            SearchGeneration(28),
            startpos_history(),
            StreamWorkerConfig::default(),
        );
        let error = pipeline.run_playouts(1).expect_err("computation creation must fail");
        assert_eq!(error, EnginError::Onnx("test computation failure".to_owned()));
        pipeline.wait_for_idle().expect_err("pipeline retains the worker error");
        assert_eq!(pipeline.stats().completed_playouts, 0);
        pipeline.stop_and_join();
    }
}
