//! Stream search mainline: Gather / Eval / NN / Backprop workers.
//!
//! Reference: LC3 overview, "Workers":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! Eval handles terminal/cache/legal moves and encodes planes; one NN thread only
//! runs ONNX on queued tensors. Workers exchange owned events only.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, SendTimeoutError, Sender, TryRecvError, TrySendError};
use parking_lot::{Condvar, Mutex};
use xiangqi_core::{Move, PositionHistory};

use crate::neural::backend::{Backend, EvalPosition, EvalResult};
use crate::neural::onnx::softmax_legal_policy;
use crate::neural::{encode_position_for_nn, FillEmptyHistory, BOARD_COLS, BOARD_ROWS, INPUT_PLANES, POLICY_SIZE};
use crate::EnginError;

use super::extension::{classify_extension, ExtensionKind};
use super::{
    network_wl_to_node, select_edge_from_node, BackpropEvent, ExpansionState, Node, NodeEvent, NodeKey, NodeRepository,
    SearchGeneration, SearchParams, Tree,
};

const RECEIVE_POLL: Duration = Duration::from_millis(10);

/// Search budget consumed relative to the pipeline's current completed
/// playout count. A later UCI watchdog maps `go nodes` / `go movetime` onto
/// this boundary; workers themselves only observe the immutable limit and
/// the explicit `request_stop()` signal.
///
/// Reference: LC3 overview, "Stats Collection" and "WatchdogWorker".
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchLimits {
    pub max_playouts: Option<u64>,
    pub deadline: Option<Instant>,
}

impl SearchLimits {
    fn is_exhausted(self, completed: u64, target: u64) -> bool {
        completed >= target || self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
    }
}

/// One queue's optional benchmark timing snapshot.
///
/// Reference: LC3 overview, "Stats Collection".
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueueStats {
    pub samples: u64,
    pub total_wait_ns: u64,
    pub max_wait_ns: u64,
}

/// Normal search counters plus benchmark-only pipeline telemetry.
///
/// Reference: LC3 overview, "Stats Collection".
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    pub submitted_playouts: u64,
    pub completed_playouts: u64,
    /// Mean completed leaf depth, with the root at depth one. This follows
    /// classic `ThinkingInfo::depth` rather than current PV length.
    pub average_depth: u64,
    /// Deepest completed leaf depth, matching classic `seldepth`.
    pub max_depth: u64,
    pub collisions: u64,
    pub network_batches: u64,
    pub network_evaluations: u64,
    /// Largest `BackendComputation` batch observed this search.
    pub network_batch_size_max: u64,
    pub peak_in_flight: u64,
    /// Collisions indexed by variation depth from the root.
    pub collisions_by_depth: Vec<u64>,
    pub gather_queue: QueueStats,
    pub eval_queue: QueueStats,
    pub nn_queue: QueueStats,
    pub backprop_queue: QueueStats,
}

#[derive(Default)]
struct QueueMetrics {
    samples: AtomicU64,
    total_wait_ns: AtomicU64,
    max_wait_ns: AtomicU64,
}

impl QueueMetrics {
    /// Records one event handoff for `feed_bench` only.
    ///
    /// Reference: LC3 overview, "Stats Collection".
    fn record(&self, wait: Duration) {
        let nanos = wait.as_nanos().min(u64::MAX as u128) as u64;
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.total_wait_ns.fetch_add(nanos, Ordering::Relaxed);
        self.max_wait_ns.fetch_max(nanos, Ordering::Relaxed);
    }

    /// Reads a consistent-enough diagnostic snapshot; it is not search state.
    ///
    /// Reference: LC3 overview, "Stats Collection".
    fn snapshot(&self) -> QueueStats {
        QueueStats {
            samples: self.samples.load(Ordering::Relaxed),
            total_wait_ns: self.total_wait_ns.load(Ordering::Relaxed),
            max_wait_ns: self.max_wait_ns.load(Ordering::Relaxed),
        }
    }
}

/// Cloneable stop boundary for a running stream search.
///
/// The controller may hold this while the owning thread retains `Search` and
/// its worker join handles. Reference: LC3 overview, "WatchdogWorker".
#[derive(Clone)]
pub struct SearchControl {
    shared: Arc<Shared>,
}

impl SearchControl {
    pub fn request_stop(&self) {
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.idle.notify_all();
    }

    pub fn stats(&self) -> Stats {
        self.shared.stats()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Gather/Eval/Backprop/NN queue depth. `0` → `max(4096, 64 * resolved_batch)`.
    pub queue_capacity: usize,
    /// NN GPU merge size when several encoded positions are already queued. `0` →
    /// backend `recommended_batch_size`. Eval does not wait to fill this.
    pub eval_batch_size: usize,
    pub params: SearchParams,
    pub gather_workers: usize,
    /// LC3 EvalWorker count. Prep/cache/legal moves; NN inference is a separate thread.
    pub eval_workers: usize,
    pub backprop_workers: usize,
    /// px0 `root_move_filter_` from UCI `go searchmoves` (empty = unrestricted).
    pub root_move_filter: Vec<Move>,
    /// Enables timing/distribution counters used only by `feed_bench`.
    /// Reference: LC3 overview, "Stats Collection".
    pub benchmark_telemetry: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 0,
            eval_batch_size: 0,
            params: SearchParams::default(),
            gather_workers: 4,
            eval_workers: 2,
            backprop_workers: 1,
            root_move_filter: Vec::new(),
            benchmark_telemetry: false,
        }
    }
}

impl SearchConfig {
    fn validate(&self) {
        self.params.validate();
        assert!(self.gather_workers > 0, "stream requires at least one gather worker");
        assert!(self.eval_workers > 0, "stream requires at least one eval worker");
        assert!(
            self.backprop_workers > 0,
            "stream requires at least one backprop worker"
        );
    }

    /// Fills `0` sentinels from the backend; returns concrete queue/batch sizes.
    fn resolve(&self, backend: &dyn Backend) -> ResolvedSearchConfig {
        let recommended = backend.attributes().recommended_batch_size.max(1);
        let eval_batch_size = if self.eval_batch_size == 0 {
            recommended
        } else {
            self.eval_batch_size
        };
        let queue_capacity = if self.queue_capacity == 0 {
            (eval_batch_size.saturating_mul(64)).max(4096)
        } else {
            self.queue_capacity
        };
        assert!(queue_capacity > 0, "stream queue capacity must be non-zero");
        assert!(eval_batch_size > 0, "stream eval batch size must be non-zero");
        assert!(
            eval_batch_size <= queue_capacity,
            "stream eval batch size must fit the queue capacity"
        );
        ResolvedSearchConfig {
            queue_capacity,
            eval_batch_size,
            params: self.params,
            benchmark_telemetry: self.benchmark_telemetry,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResolvedSearchConfig {
    queue_capacity: usize,
    eval_batch_size: usize,
    params: SearchParams,
    benchmark_telemetry: bool,
}

struct Shared {
    backend: Arc<dyn Backend>,
    repository: Arc<NodeRepository>,
    root_key: NodeKey,
    generation: SearchGeneration,
    params: SearchParams,
    benchmark_telemetry: bool,
    root_move_filter: Vec<Move>,
    stopping: AtomicBool,
    outstanding: AtomicUsize,
    submitted: AtomicU64,
    peak_in_flight: AtomicU64,
    completed: AtomicU64,
    completed_depth: AtomicU64,
    max_depth: AtomicU64,
    collisions: AtomicU64,
    network_batches: AtomicU64,
    network_evaluations: AtomicU64,
    network_batch_size_max: AtomicU64,
    collisions_by_depth: Mutex<Vec<u64>>,
    gather_queue: QueueMetrics,
    eval_queue: QueueMetrics,
    nn_queue: QueueMetrics,
    backprop_queue: QueueMetrics,
    error: Mutex<Option<EnginError>>,
    idle_lock: Mutex<()>,
    idle: Condvar,
    gather_tx: Sender<NodeEvent>,
    eval_tx: Sender<NodeEvent>,
    backprop_tx: Sender<BackpropEvent>,
}

impl Shared {
    fn finish(&self, completed: bool) {
        if completed {
            self.completed.fetch_add(1, Ordering::AcqRel);
        }
        let previous = self.outstanding.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "stream outstanding task underflow");
        // Wake submitters waiting on in-flight caps, not only full idle.
        let _guard = self.idle_lock.lock();
        self.idle.notify_all();
        let _ = previous;
    }

    fn cancel_and_finish(&self, event: NodeEvent, collision: bool) {
        event.cancel();
        if collision {
            self.collisions.fetch_add(1, Ordering::AcqRel);
        }
        self.finish(false);
    }

    /// Cancels an event after Gather has claimed its leaf for Eval.
    ///
    /// Reference: LC3 overview's EvalWorker ownership model. Releasing only
    /// the edge reservations would leave the claimed node permanently
    /// `Evaluating`, so this also restores it to `Unexpanded`.
    fn cancel_claimed_evaluation(&self, event: NodeEvent) {
        let node = self
            .repository
            .get(event.node_key)
            .expect("claimed stream node remains in the repository");
        cancel_evaluation(self, event, node);
    }

    fn fail(&self, error: EnginError) {
        let mut current = self.error.lock();
        if current.is_none() {
            *current = Some(error);
        }
        self.stopping.store(true, Ordering::Release);
        self.idle.notify_all();
    }

    /// Non-blocking enqueue: `try_send` + yield instead of parking on a full
    /// queue so Gather can keep polling `stopping` and stay schedulable.
    fn send_eval(&self, mut event: NodeEvent) {
        if self.benchmark_telemetry {
            event.mark_queued();
        }
        loop {
            if self.stopping.load(Ordering::Acquire) {
                self.cancel_claimed_evaluation(event);
                return;
            }
            match self.eval_tx.try_send(event) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    event = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    self.cancel_claimed_evaluation(returned);
                    return;
                }
            }
        }
    }

    fn send_backprop(&self, mut event: BackpropEvent) {
        if self.benchmark_telemetry {
            event.mark_queued();
        }
        loop {
            if self.stopping.load(Ordering::Acquire) {
                event.cancel();
                self.finish(false);
                return;
            }
            match self.backprop_tx.try_send(event) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    event = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    returned.cancel();
                    self.finish(false);
                    return;
                }
            }
        }
    }

    fn stats(&self) -> Stats {
        let completed_playouts = self.completed.load(Ordering::Acquire);
        let (
            submitted_playouts,
            peak_in_flight,
            collisions_by_depth,
            gather_queue,
            eval_queue,
            nn_queue,
            backprop_queue,
        ) = if self.benchmark_telemetry {
            (
                self.submitted.load(Ordering::Acquire),
                self.peak_in_flight.load(Ordering::Acquire),
                self.collisions_by_depth.lock().clone(),
                self.gather_queue.snapshot(),
                self.eval_queue.snapshot(),
                self.nn_queue.snapshot(),
                self.backprop_queue.snapshot(),
            )
        } else {
            (
                0,
                0,
                Vec::new(),
                QueueStats::default(),
                QueueStats::default(),
                QueueStats::default(),
                QueueStats::default(),
            )
        };
        Stats {
            submitted_playouts,
            completed_playouts,
            average_depth: self.completed_depth.load(Ordering::Acquire) / completed_playouts.max(1),
            max_depth: self.max_depth.load(Ordering::Acquire),
            collisions: self.collisions.load(Ordering::Acquire),
            network_batches: self.network_batches.load(Ordering::Acquire),
            network_evaluations: self.network_evaluations.load(Ordering::Acquire),
            network_batch_size_max: self.network_batch_size_max.load(Ordering::Acquire),
            peak_in_flight,
            collisions_by_depth,
            gather_queue,
            eval_queue,
            nn_queue,
            backprop_queue,
        }
    }

    /// Adds per-depth collision counts without changing selection behavior.
    ///
    /// Reference: LC3 overview, "Stats Collection".
    fn record_collision_depths(&self, depths: &[usize]) {
        if !self.benchmark_telemetry {
            return;
        }
        let mut counts = self.collisions_by_depth.lock();
        for &depth in depths {
            if counts.len() <= depth {
                counts.resize(depth + 1, 0);
            }
            counts[depth] += 1;
        }
    }
}

/// LC3-style streaming search: Gather / Eval / NN / Backprop.
/// Eval: terminal | cache → Backprop; else encode planes → NN queue; on NN
/// completion → softmax/edges → Backprop. NN: dequeue tensors → GPU → reply.
pub struct Search {
    shared: Arc<Shared>,
    root_history: Arc<PositionHistory>,
    root_key: NodeKey,
    /// Cap concurrent owned playouts after root expansion. Much smaller than
    /// queue capacity: saturating thousands of in-flight walks explodes collisions.
    max_in_flight: usize,
    workers: Vec<JoinHandle<()>>,
}

impl Search {
    pub fn new(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        root_history: Arc<PositionHistory>,
        config: SearchConfig,
    ) -> Self {
        let tree = Tree::new(root_history);
        Self::new_with_tree(backend, generation, &tree, config)
    }

    /// Starts workers from a previously retained tree root. The caller must
    /// stop and join this search before advancing or rewinding `tree`.
    pub fn new_with_tree(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        tree: &Tree,
        config: SearchConfig,
    ) -> Self {
        config.validate();
        let resolved = config.resolve(backend.as_ref());
        let (gather_tx, gather_rx) = bounded(resolved.queue_capacity);
        let (eval_tx, eval_rx) = bounded(resolved.queue_capacity);
        let (backprop_tx, backprop_rx) = bounded(resolved.queue_capacity);
        let root_history = Arc::clone(tree.root_history());
        let root_key = tree.root_key();
        let shared = Arc::new(Shared {
            backend,
            repository: Arc::clone(tree.repository()),
            root_key,
            generation,
            params: resolved.params,
            benchmark_telemetry: resolved.benchmark_telemetry,
            root_move_filter: config.root_move_filter.clone(),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(0),
            submitted: AtomicU64::new(0),
            peak_in_flight: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            network_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            network_batch_size_max: AtomicU64::new(0),
            collisions_by_depth: Mutex::new(Vec::new()),
            gather_queue: QueueMetrics::default(),
            eval_queue: QueueMetrics::default(),
            nn_queue: QueueMetrics::default(),
            backprop_queue: QueueMetrics::default(),
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            gather_tx,
            eval_tx,
            backprop_tx,
        });
        let (nn_tx, nn_rx) = bounded::<NnRequest>(resolved.queue_capacity);
        let mut workers = Vec::with_capacity(config.gather_workers + config.eval_workers + config.backprop_workers + 1);
        for _ in 0..config.gather_workers {
            let shared = Arc::clone(&shared);
            let receiver = gather_rx.clone();
            workers.push(thread::spawn(move || gather_worker(shared, receiver)));
        }
        {
            let shared = Arc::clone(&shared);
            let batch_size = resolved.eval_batch_size;
            workers.push(thread::spawn(move || nn_worker(shared, nn_rx, batch_size)));
        }
        for _ in 0..config.eval_workers {
            let shared = Arc::clone(&shared);
            let receiver = eval_rx.clone();
            let nn_tx = nn_tx.clone();
            workers.push(thread::spawn(move || eval_worker(shared, receiver, nn_tx)));
        }
        drop(nn_tx);
        for _ in 0..config.backprop_workers {
            let shared = Arc::clone(&shared);
            let receiver = backprop_rx.clone();
            workers.push(thread::spawn(move || backprop_worker(shared, receiver)));
        }
        Self {
            shared,
            root_history,
            root_key,
            // Keep a few Eval batches worth of leaves in flight, not the full queue depth.
            max_in_flight: resolved
                .eval_batch_size
                .saturating_mul(4)
                .min(resolved.queue_capacity)
                .max(1),
            workers,
        }
    }

    pub fn repository(&self) -> &Arc<NodeRepository> {
        &self.shared.repository
    }

    pub fn root_key(&self) -> NodeKey {
        self.root_key
    }

    pub fn stats(&self) -> Stats {
        self.shared.stats()
    }

    pub fn control(&self) -> SearchControl {
        SearchControl {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Requests a normal stream-search stop without tearing down worker
    /// threads. Gather/Eval/Backprop cancel every unfinished event and its
    /// edge reservation before becoming idle. This is the boundary a later
    /// UCI controller will use for `stop`; `stop_and_join()` remains the
    /// terminal owner cleanup path.
    ///
    /// Reference: LC3 overview, "Watchdog" and worker stop coordination.
    pub fn request_stop(&self) {
        self.control().request_stop();
    }

    pub fn is_stopping(&self) -> bool {
        self.shared.stopping.load(Ordering::Acquire)
    }

    pub fn submit_playout(&self) -> Result<(), EnginError> {
        self.submit_event(NodeEvent::at_root(
            self.shared.generation,
            self.root_key,
            Arc::clone(&self.root_history),
        ))
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
        let outstanding = self.shared.outstanding.fetch_add(1, Ordering::AcqRel) + 1;
        if self.shared.benchmark_telemetry {
            self.shared.submitted.fetch_add(1, Ordering::Relaxed);
            self.shared
                .peak_in_flight
                .fetch_max(outstanding as u64, Ordering::Relaxed);
        }
        let mut event = event;
        if self.shared.benchmark_telemetry {
            event.mark_queued();
        }
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

    pub fn run_playouts(&self, count: u64) -> Result<Stats, EnginError> {
        self.run_with_limits(SearchLimits {
            max_playouts: Some(count),
            deadline: None,
        })
    }

    /// Runs owned stream events until a relative playout budget, deadline, or
    /// explicit stop is reached. After root expansion it keeps the pipeline
    /// filled up to `max_in_flight` (~4× eval batch) instead of draining to idle
    /// between waves. It always waits for submitted events to complete or
    /// cancel before returning, so root snapshots never observe a leaked edge
    /// reservation.
    pub fn run_with_limits(&self, limits: SearchLimits) -> Result<Stats, EnginError> {
        let initial_completed = self.stats().completed_playouts;
        let target = initial_completed.saturating_add(limits.max_playouts.unwrap_or(u64::MAX));
        let max_in_flight = self.max_in_flight;
        while !self.is_stopping() && !limits.is_exhausted(self.stats().completed_playouts, target) {
            let root_state = self
                .shared
                .repository
                .get(self.root_key)
                .map(|root| root.expansion_state());
            if root_state == Some(ExpansionState::Terminal) {
                break;
            }
            if root_state != Some(ExpansionState::Expanded) {
                // Bootstrap: one playout at a time until the root is expanded.
                if let Err(error) = self.submit_playout() {
                    if self.is_stopping() {
                        break;
                    }
                    return Err(error);
                }
                self.wait_for_idle()?;
                continue;
            }
            let outstanding = self.shared.outstanding.load(Ordering::Acquire);
            let completed = self.stats().completed_playouts;
            if completed.saturating_add(outstanding as u64) >= target {
                if outstanding == 0 {
                    break;
                }
                // Let in-flight work finish toward the budget; do not overshoot with more submits.
                self.wait_until_outstanding_below(outstanding)?;
                continue;
            }
            if outstanding >= max_in_flight {
                self.wait_until_outstanding_below(max_in_flight)?;
                continue;
            }
            if let Err(error) = self.submit_playout() {
                if self.is_stopping() {
                    break;
                }
                return Err(error);
            }
        }
        // A requested stop is a normal search outcome. `wait_for_idle()` has
        // already ensured every queued event either completed or cancelled its
        // reservation, so callers can safely snapshot the partial tree.
        self.wait_for_idle()?;
        Ok(self.stats())
    }

    pub fn wait_for_idle(&self) -> Result<(), EnginError> {
        self.wait_until_outstanding_below(1)
    }

    /// Blocks until `outstanding < limit` (or an error/stop drains work).
    fn wait_until_outstanding_below(&self, limit: usize) -> Result<(), EnginError> {
        let mut guard = self.shared.idle_lock.lock();
        while self.shared.outstanding.load(Ordering::Acquire) >= limit && self.shared.error.lock().is_none() {
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

impl Drop for Search {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn gather_worker(shared: Arc<Shared>, receiver: Receiver<NodeEvent>) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(mut event) => {
                if shared.benchmark_telemetry {
                    if let Some(wait) = event.take_queue_wait() {
                        shared.gather_queue.record(wait);
                    }
                }
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

fn process_gather_event(shared: &Shared, mut event: NodeEvent) {
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
                shared.send_backprop(BackpropEvent::collision(event));
                return;
            }
            ExpansionState::Terminal => {
                let (wl, draw) = node.terminal_wl().expect("terminal stream wl");
                shared.send_backprop(BackpropEvent::evaluation(
                    event,
                    wl,
                    draw,
                    node.terminal_plies_left().expect("terminal stream plies"),
                ));
                return;
            }
            ExpansionState::Expanded => {
                let depth = event.variation.moves().len();
                let edge_index = select_edge_from_node(node.as_ref(), depth, &shared.params, &shared.root_move_filter)
                    .expect("expanded stream node must have an edge");
                let reservation = node.reserve_edge(edge_index).expect("selected stream edge");
                let child_key = event.node_key.child(reservation.mv());
                event = event.descend(child_key, reservation);
            }
        }
    }
}

/// Raw logits[POLICY], WDL[3] and moves-left[1] returned by the NN worker.
type NnReply = Result<(Vec<f32>, Vec<f32>, f32), EnginError>;

/// One encoded position for the NN thread. Reply owns all three model heads.
struct NnRequest {
    planes: Vec<f32>,
    reply: Sender<NnReply>,
    queued_at: Option<Instant>,
}

impl NnRequest {
    /// Records the NN queue handoff when benchmark telemetry is enabled.
    ///
    /// Reference: LC3 overview, "Stats Collection".
    fn mark_queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    /// Returns the NN queue delay for benchmark reporting.
    ///
    /// Reference: LC3 overview, "Stats Collection".
    fn take_queue_wait(&mut self) -> Option<Duration> {
        self.queued_at.take().map(|queued_at| queued_at.elapsed())
    }
}

/// Eval is waiting on NN for this node (LC3: after NN completes → Backprop).
struct WaitingNn {
    event: NodeEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
    input: EvalPosition,
    reply: Receiver<NnReply>,
}

/// LC3 EvalWorker:
/// terminal | cache → Backprop; else legal moves + encode → NN queue;
/// when NN replies → softmax/edges → Backprop.
/// Does not block the whole worker on one GPU call: poll completions interleaved
/// with new NodeEvents.
fn eval_worker(shared: Arc<Shared>, receiver: Receiver<NodeEvent>, nn_tx: Sender<NnRequest>) {
    let mut waiting: Vec<WaitingNn> = Vec::new();
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            for item in waiting.drain(..) {
                cancel_waiting_item(&shared, item);
            }
            while let Ok(event) = receiver.try_recv() {
                shared.cancel_claimed_evaluation(event);
            }
            break;
        }

        poll_nn_completions(&shared, &mut waiting);

        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(mut event) => {
                if shared.benchmark_telemetry {
                    if let Some(wait) = event.take_queue_wait() {
                        shared.eval_queue.record(wait);
                    }
                }
                if let Err(error) = handle_eval_event(&shared, &nn_tx, &mut waiting, event) {
                    shared.fail(error);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if waiting.is_empty() {
                    continue;
                }
                // No new leaves: wait briefly for at least one NN reply.
                wait_one_nn_completion(&shared, &mut waiting);
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn poll_nn_completions(shared: &Shared, waiting: &mut Vec<WaitingNn>) {
    let mut i = 0;
    while i < waiting.len() {
        match waiting[i].reply.try_recv() {
            Ok(Ok((logits, wdl, moves_left))) => {
                let item = waiting.swap_remove(i);
                if let Err(error) = complete_nn_item(shared, item, logits, wdl, moves_left) {
                    shared.fail(error);
                    return;
                }
            }
            Ok(Err(error)) => {
                let item = waiting.swap_remove(i);
                cancel_waiting_item(shared, item);
                if !shared.stopping.load(Ordering::Acquire) {
                    shared.fail(error);
                }
                return;
            }
            Err(TryRecvError::Empty) => i += 1,
            Err(TryRecvError::Disconnected) => {
                let item = waiting.swap_remove(i);
                cancel_waiting_item(shared, item);
                if !shared.stopping.load(Ordering::Acquire) {
                    shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
                }
                return;
            }
        }
    }
}

fn wait_one_nn_completion(shared: &Shared, waiting: &mut Vec<WaitingNn>) {
    if waiting.is_empty() {
        return;
    }
    match waiting[0].reply.recv_timeout(RECEIVE_POLL) {
        Ok(Ok((logits, wdl, moves_left))) => {
            let item = waiting.remove(0);
            if let Err(error) = complete_nn_item(shared, item, logits, wdl, moves_left) {
                shared.fail(error);
            }
        }
        Ok(Err(error)) => {
            let item = waiting.remove(0);
            cancel_waiting_item(shared, item);
            if !shared.stopping.load(Ordering::Acquire) {
                shared.fail(error);
            }
        }
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
            let item = waiting.remove(0);
            cancel_waiting_item(shared, item);
            if !shared.stopping.load(Ordering::Acquire) {
                shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
            }
        }
    }
}

fn handle_eval_event(
    shared: &Shared,
    nn_tx: &Sender<NnRequest>,
    waiting: &mut Vec<WaitingNn>,
    event: NodeEvent,
) -> Result<(), EnginError> {
    if shared.stopping.load(Ordering::Acquire) {
        shared.cancel_claimed_evaluation(event);
        return Ok(());
    }
    let node = shared.repository.get_or_insert(event.node_key);
    let history = event.variation.replay_history();
    let depth = event.variation.moves().len();
    match classify_extension(&history, depth) {
        ExtensionKind::Terminal { wl, draw, plies_left } => {
            node.mark_terminal(wl, draw, plies_left);
            shared
                .repository
                .propagate_proven_bounds(event.node_path(), shared.root_key);
            shared.send_backprop(BackpropEvent::evaluation(event, wl, draw, plies_left));
            Ok(())
        }
        ExtensionKind::Evaluate => {
            let legal_moves = history.last().board().generate_legal_moves();
            let input = EvalPosition {
                positions: history.positions().to_vec(),
                legal_moves: legal_moves.clone(),
            };
            if let Some(eval) = shared.backend.cached_evaluation(&input) {
                return publish_eval(shared, event, node, legal_moves, eval);
            }
            let planes = encode_position_for_nn(&history, FillEmptyHistory::FenOnly);
            let (reply_tx, reply_rx) = bounded(1);
            if let Err(error) = send_nn_request(
                shared,
                nn_tx,
                NnRequest {
                    planes,
                    reply: reply_tx,
                    queued_at: None,
                },
            ) {
                cancel_evaluation(shared, event, node);
                return if shared.stopping.load(Ordering::Acquire) {
                    Ok(())
                } else {
                    Err(error)
                };
            }
            waiting.push(WaitingNn {
                event,
                node,
                legal_moves,
                input,
                reply: reply_rx,
            });
            Ok(())
        }
    }
}

fn send_nn_request(shared: &Shared, nn_tx: &Sender<NnRequest>, mut request: NnRequest) -> Result<(), EnginError> {
    if shared.benchmark_telemetry {
        request.mark_queued();
    }
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            return Err(EnginError::PortIncomplete("stream nn stopping"));
        }
        match nn_tx.try_send(request) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned)) => {
                request = returned;
                thread::yield_now();
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(EnginError::PortIncomplete("stream nn queue disconnected"));
            }
        }
    }
}

fn complete_nn_item(
    shared: &Shared,
    item: WaitingNn,
    logits: Vec<f32>,
    wdl: Vec<f32>,
    moves_left: f32,
) -> Result<(), EnginError> {
    let policies = match softmax_legal_policy(&logits, &item.legal_moves) {
        Ok(policies) => policies,
        Err(error) => {
            cancel_waiting_item(shared, item);
            return Err(error);
        }
    };
    if wdl.len() < 3 {
        cancel_waiting_item(shared, item);
        return Err(EnginError::PortIncomplete("stream nn wdl length"));
    }
    if !wdl.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        || (wdl.iter().sum::<f32>() - 1.0).abs() > 1e-3
        || !moves_left.is_finite()
        || moves_left < 0.0
    {
        cancel_waiting_item(shared, item);
        return Err(EnginError::Onnx("stream nn values are invalid".into()));
    }
    let eval = Arc::new(EvalResult {
        wl: wdl[0] - wdl[2],
        d: wdl[1],
        plies_left: moves_left,
        policies,
    });
    shared.backend.store_evaluation(&item.input, Arc::clone(&eval));
    publish_eval(shared, item.event, item.node, item.legal_moves, eval)
}

/// Validates a completed backend result before it reaches the tree.
///
/// Reference: the fixed `WDL + moves-left` backend contract in ARCHITECTURE;
/// worker failures must return through LC3's cancellation path, never panic
/// after retaining an owned event.
fn publish_eval(
    shared: &Shared,
    event: NodeEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
    eval: Arc<EvalResult>,
) -> Result<(), EnginError> {
    let value_is_valid = eval.wl.is_finite()
        && eval.d.is_finite()
        && (0.0..=1.0).contains(&eval.d)
        && eval.wl.abs() <= 1.0 - eval.d + f32::EPSILON
        && eval.plies_left.is_finite()
        && eval.plies_left >= 0.0;
    let policy_sum: f32 = eval.policies.iter().sum();
    let policy_is_valid = eval.policies.len() == legal_moves.len()
        && eval.policies.iter().all(|policy| policy.is_finite() && *policy >= 0.0)
        && policy_sum.is_finite()
        && (policy_sum - 1.0).abs() <= 1e-3;
    if !value_is_valid || !policy_is_valid {
        cancel_evaluation(shared, event, node);
        return Err(EnginError::Onnx("stream backend evaluation is invalid".into()));
    }
    node.publish_edges(legal_moves.into_iter().zip(eval.policies.iter().copied()).collect());
    shared.send_backprop(BackpropEvent::evaluation(
        event,
        network_wl_to_node(eval.wl),
        eval.d,
        eval.plies_left,
    ));
    Ok(())
}

fn cancel_waiting_item(shared: &Shared, item: WaitingNn) {
    cancel_evaluation(shared, item.event, item.node);
}

/// Releases a claimed evaluation event that will not publish a result.
///
/// Reference: LC3 overview's EvalWorker ownership model: each owned event
/// must finish through backpropagation or explicit cancellation.
fn cancel_evaluation(shared: &Shared, event: NodeEvent, node: Arc<Node>) {
    event.cancel();
    node.abort_evaluation();
    shared.finish(false);
}

/// NN worker: pull encoded planes from the queue, run ONNX, reply. Merges whatever
/// is already queued (up to `batch_size`) so the GPU stays busy; Eval never waits
/// to fill that size.
fn nn_worker(shared: Arc<Shared>, receiver: Receiver<NnRequest>, batch_size: usize) {
    let plane_len = INPUT_PLANES * BOARD_ROWS * BOARD_COLS;
    loop {
        let mut first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        if shared.benchmark_telemetry {
            if let Some(wait) = first.take_queue_wait() {
                shared.nn_queue.record(wait);
            }
        }
        let mut requests = vec![first];
        while requests.len() < batch_size {
            match receiver.try_recv() {
                Ok(mut request) => {
                    if shared.benchmark_telemetry {
                        if let Some(wait) = request.take_queue_wait() {
                            shared.nn_queue.record(wait);
                        }
                    }
                    requests.push(request);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        if shared.stopping.load(Ordering::Acquire) {
            for request in requests {
                let _ = request
                    .reply
                    .send(Err(EnginError::PortIncomplete("stream nn stopping")));
            }
            continue;
        }
        let batch = requests.len();
        let mut packed = Vec::with_capacity(batch * plane_len);
        for request in &requests {
            debug_assert_eq!(request.planes.len(), plane_len);
            packed.extend_from_slice(&request.planes);
        }
        match shared.backend.infer_encoded(&packed, batch) {
            Ok((logits, wdl, moves_left)) => {
                if logits.len() != batch * POLICY_SIZE || wdl.len() != batch * 3 || moves_left.len() != batch {
                    let error = EnginError::PortIncomplete("stream nn output shape");
                    for request in requests {
                        let _ = request.reply.send(Err(error.clone()));
                    }
                    continue;
                }
                shared.network_batches.fetch_add(1, Ordering::AcqRel);
                shared.network_evaluations.fetch_add(batch as u64, Ordering::AcqRel);
                shared.network_batch_size_max.fetch_max(batch as u64, Ordering::AcqRel);
                for (index, request) in requests.into_iter().enumerate() {
                    let part_logits = logits[index * POLICY_SIZE..(index + 1) * POLICY_SIZE].to_vec();
                    let part_wdl = wdl[index * 3..(index + 1) * 3].to_vec();
                    let _ = request.reply.send(Ok((part_logits, part_wdl, moves_left[index])));
                }
            }
            Err(error) => {
                for request in requests {
                    let _ = request.reply.send(Err(error.clone()));
                }
            }
        }
    }
}

fn backprop_worker(shared: Arc<Shared>, receiver: Receiver<BackpropEvent>) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(mut first) => {
                if shared.benchmark_telemetry {
                    if let Some(wait) = first.take_queue_wait() {
                        shared.backprop_queue.record(wait);
                    }
                }
                let mut events = vec![first];
                for mut event in receiver.try_iter() {
                    if shared.benchmark_telemetry {
                        if let Some(wait) = event.take_queue_wait() {
                            shared.backprop_queue.record(wait);
                        }
                    }
                    events.push(event);
                }
                if shared.stopping.load(Ordering::Acquire) {
                    for event in events {
                        event.cancel();
                        shared.finish(false);
                    }
                } else {
                    let result = BackpropEvent::complete_batch(events, &shared.repository);
                    shared.record_collision_depths(&result.collision_depths);
                    shared
                        .completed_depth
                        .fetch_add(result.completed_depth, Ordering::AcqRel);
                    shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
                    for _ in 0..result.completed_playouts {
                        shared.finish(true);
                    }
                    for _ in 0..result.collisions {
                        shared.collisions.fetch_add(1, Ordering::AcqRel);
                        shared.finish(false);
                    }
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
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use crossbeam_channel::bounded;
    use parking_lot::{Condvar, Mutex};
    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

    use super::{NodeEvent, Search, SearchConfig, SearchLimits, Shared};
    use crate::neural::backend::{
        Backend, BackendAttributes, BackendComputation, EncodedInference, EvalResult, UniformBackend,
    };
    use crate::search::{best_move, root_stats, ExpansionState, NodeRepository, SearchGeneration, SearchParams};
    use crate::EnginError;

    struct FailingComputationBackend;

    struct InvalidValueBackend;

    impl Backend for FailingComputationBackend {
        fn evaluate(&self, _history: &PositionHistory, _legal_moves: &[Move]) -> Arc<EvalResult> {
            unreachable!("stream worker must use encoded inference")
        }

        fn attributes(&self) -> BackendAttributes {
            BackendAttributes::default()
        }

        fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
            Err(EnginError::Onnx("test computation failure".to_owned()))
        }

        fn infer_encoded(&self, _planes: &[f32], _batch: usize) -> Result<EncodedInference, EnginError> {
            Err(EnginError::Onnx("test computation failure".to_owned()))
        }
    }

    impl Backend for InvalidValueBackend {
        fn evaluate(&self, _history: &PositionHistory, _legal_moves: &[Move]) -> Arc<EvalResult> {
            unreachable!("stream worker must use encoded inference")
        }

        fn attributes(&self) -> BackendAttributes {
            BackendAttributes::default()
        }

        fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
            Err(EnginError::Onnx("test computation failure".to_owned()))
        }

        fn infer_encoded(&self, _planes: &[f32], batch: usize) -> Result<EncodedInference, EnginError> {
            Ok((
                vec![0.0; batch * crate::neural::POLICY_SIZE],
                (0..batch).flat_map(|_| [f32::NAN, 0.0, 0.0]).collect(),
                vec![0.0; batch],
            ))
        }
    }

    fn startpos_history() -> Arc<PositionHistory> {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        Arc::new(PositionHistory::from_positions(state.positions()))
    }

    #[test]
    fn persistent_workers_complete_batched_playouts_and_join() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(21),
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 8,
                eval_batch_size: 4,
                gather_workers: 2,
                backprop_workers: 1,
                ..SearchConfig::default()
            },
        );
        let stats = pipeline.run_playouts(32).expect("playouts");
        assert_eq!(stats.completed_playouts, 32);
        assert!(stats.network_batches > 0);
        // A very fast stop may win the race before Gather expands the root.
        // If it did expand, every reservation must still be released.
        if let Some(root) = pipeline.repository().get(pipeline.root_key()) {
            for edge in root.edges().iter() {
                assert_eq!(edge.visits(), edge.completed_visits());
            }
        }
        pipeline.stop_and_join();
    }

    #[test]
    fn benchmark_telemetry_reports_pipeline_handoffs() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(22),
            startpos_history(),
            SearchConfig {
                benchmark_telemetry: true,
                ..SearchConfig::default()
            },
        );

        let stats = pipeline.run_playouts(16).expect("benchmark playouts");

        assert!(stats.submitted_playouts >= stats.completed_playouts);
        assert!(stats.peak_in_flight > 0);
        assert!(stats.gather_queue.samples >= stats.completed_playouts);
        assert!(stats.eval_queue.samples > 0);
        assert!(stats.backprop_queue.samples > 0);
        pipeline.stop_and_join();
    }

    #[test]
    fn stop_join_drains_in_flight_reservations() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(22),
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 16,
                eval_batch_size: 8,
                gather_workers: 4,
                backprop_workers: 2,
                ..SearchConfig::default()
            },
        );
        pipeline.run_playouts(1).expect("expand root");
        for _ in 0..16 {
            pipeline.submit_playout().expect("queued root event");
        }
        pipeline.stop_and_join();

        // Stop may arrive before Gather expands the root. If it did expand,
        // every reservation still has to be released.
        if let Some(root) = pipeline.repository().get(pipeline.root_key()) {
            for edge in root.edges().iter() {
                assert_eq!(edge.visits(), edge.completed_visits());
            }
        }
    }

    #[test]
    fn bounded_gather_queue_backpressures_without_failing_submission() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(23),
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 1,
                eval_batch_size: 1,
                gather_workers: 1,
                backprop_workers: 1,
                ..SearchConfig::default()
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
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(24),
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 8,
                eval_batch_size: 4,
                gather_workers: 2,
                backprop_workers: 1,
                ..SearchConfig::default()
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

        // Stop may win the race before Gather creates the root. If it did
        // create it, every reservation still has to be released.
        if let Some(root) = pipeline.repository().get(pipeline.root_key()) {
            for edge in root.edges().iter() {
                assert_eq!(edge.visits(), edge.completed_visits());
            }
        }
        pipeline.stop_and_join();
    }

    #[test]
    fn expired_deadline_submits_no_new_playout() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(25),
            startpos_history(),
            SearchConfig::default(),
        );
        let stats = pipeline
            .run_with_limits(SearchLimits {
                max_playouts: Some(64),
                deadline: Some(Instant::now()),
            })
            .expect("expired deadline is a normal result");
        assert_eq!(stats.completed_playouts, 0);
        assert!(pipeline.repository().get(pipeline.root_key()).is_none());
        pipeline.stop_and_join();
    }

    #[test]
    fn terminal_root_finishes_after_its_initial_evaluation() {
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmated root");
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(26),
            Arc::new(PositionHistory::from_positions(state.positions())),
            SearchConfig::default(),
        );

        let stats = pipeline.run_playouts(8).expect("terminal search");

        assert_eq!(stats.completed_playouts, 1);
        assert_eq!(
            pipeline
                .repository()
                .get(pipeline.root_key())
                .expect("terminal root")
                .expansion_state(),
            super::ExpansionState::Terminal
        );
        pipeline.stop_and_join();
    }

    #[test]
    fn fixed_playout_root_contract() {
        let count = 64;
        let mut workers = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(27),
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 8,
                eval_batch_size: 4,
                gather_workers: 2,
                backprop_workers: 1,
                ..SearchConfig::default()
            },
        );
        let worker_stats = workers.run_playouts(count).expect("worker playouts");
        let worker_root = root_stats(workers.repository(), workers.root_key()).expect("worker root");

        assert_eq!(worker_stats.completed_playouts, count);
        assert!(worker_stats.average_depth >= 1);
        assert!(worker_stats.max_depth >= worker_stats.average_depth);
        assert_eq!(worker_root.completed_visits, count as u32);
        assert!(worker_root
            .edges
            .iter()
            .all(|edge| edge.started_visits == edge.completed_visits));
        assert!(!worker_root.edges.is_empty());

        workers.stop_and_join();
    }

    #[test]
    fn nn_inference_failure_drains_claimed_events() {
        let mut pipeline = Search::new(
            Arc::new(FailingComputationBackend),
            SearchGeneration(28),
            startpos_history(),
            SearchConfig::default(),
        );
        let error = pipeline.run_playouts(1).expect_err("nn inference must fail");
        assert_eq!(error, EnginError::Onnx("test computation failure".to_owned()));
        pipeline.wait_for_idle().expect_err("pipeline retains the worker error");
        assert_eq!(pipeline.stats().completed_playouts, 0);
        pipeline.stop_and_join();
    }

    #[test]
    fn invalid_nn_values_fail_without_leaking_the_claimed_event() {
        let mut pipeline = Search::new(
            Arc::new(InvalidValueBackend),
            SearchGeneration(29),
            startpos_history(),
            SearchConfig::default(),
        );

        let error = pipeline.run_playouts(1).expect_err("invalid network values must fail");

        assert_eq!(error, EnginError::Onnx("stream nn values are invalid".into()));
        assert_eq!(pipeline.stats().completed_playouts, 0);
        pipeline.stop_and_join();
    }

    #[test]
    fn failed_eval_enqueue_releases_the_claimed_node() {
        let history = startpos_history();
        let root_key = crate::search::NodeKey::root(history.last().hash());
        let repository = Arc::new(NodeRepository::default());
        let root = repository.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        let (gather_tx, _gather_rx) = bounded(1);
        let (eval_tx, eval_rx) = bounded(1);
        let (backprop_tx, _backprop_rx) = bounded(1);
        drop(eval_rx);
        let shared = Shared {
            backend: Arc::new(UniformBackend::default()),
            repository,
            root_key,
            generation: SearchGeneration(30),
            params: SearchParams::default(),
            benchmark_telemetry: false,
            root_move_filter: Vec::new(),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(1),
            submitted: AtomicU64::new(0),
            peak_in_flight: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            network_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            network_batch_size_max: AtomicU64::new(0),
            collisions_by_depth: Mutex::new(Vec::new()),
            gather_queue: super::QueueMetrics::default(),
            eval_queue: super::QueueMetrics::default(),
            nn_queue: super::QueueMetrics::default(),
            backprop_queue: super::QueueMetrics::default(),
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            gather_tx,
            eval_tx,
            backprop_tx,
        };

        shared.send_eval(NodeEvent::root(SearchGeneration(30), history));

        assert_eq!(root.expansion_state(), ExpansionState::Unexpanded);
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 0);
    }

    #[test]
    fn completed_search_can_reuse_a_tree_at_its_played_child() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = super::Tree::new(history);
        let backend = Arc::new(UniformBackend::default()) as Arc<dyn Backend>;

        let mut first = Search::new_with_tree(
            Arc::clone(&backend),
            SearchGeneration(29),
            &tree,
            SearchConfig::default(),
        );
        first.run_playouts(16).expect("first search");
        let old_root = tree.root_key();
        let played = best_move(first.repository(), first.root_key(), false).expect("best move");
        first.stop_and_join();

        tree.advance(played).expect("advance retained tree");
        assert!(tree.repository().get(old_root).is_some());

        let mut second = Search::new_with_tree(backend, SearchGeneration(30), &tree, SearchConfig::default());
        second.run_playouts(8).expect("reused search");
        let root = second.repository().get(second.root_key()).expect("reused root");
        assert!(root.edges().iter().all(|edge| edge.visits() == edge.completed_visits()));
        second.stop_and_join();
    }
}
