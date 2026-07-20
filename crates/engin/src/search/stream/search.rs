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
use crate::neural::{encode_position_for_nn, FillEmptyHistory, INPUT_PLANES, BOARD_ROWS, BOARD_COLS, POLICY_SIZE};
use crate::search::classic::params::SearchParams;
use crate::EnginError;

use super::extension::{classify_extension, ExtensionKind};
use super::{
    network_wl_to_node, select_edge_from_node, BackpropEvent, ExpansionState, NodeEvent, NodeKey,
    NodeRepository, SearchGeneration, Node,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    pub completed_playouts: u64,
    pub collisions: u64,
    pub network_batches: u64,
    pub network_evaluations: u64,
    /// Largest `BackendComputation` batch observed this search.
    pub network_batch_size_max: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Gather/Eval/Backprop/NN queue depth. `0` → `max(4096, 64 * resolved_batch)`.
    pub queue_capacity: usize,
    /// NN GPU merge size when several encoded positions are already queued. `0` →
    /// backend `recommended_batch_size`. Eval does not wait to fill this.
    pub eval_batch_size: usize,
    /// px0 `CPuct` default is `1.0` (`params.cc:547`).
    pub cpuct: f32,
    pub gather_workers: usize,
    /// LC3 EvalWorker count. Prep/cache/legal moves; NN inference is a separate thread.
    pub eval_workers: usize,
    pub backprop_workers: usize,
    /// px0 `root_move_filter_` from UCI `go searchmoves` (empty = unrestricted).
    pub root_move_filter: Vec<Move>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 0,
            eval_batch_size: 0,
            cpuct: 1.0,
            gather_workers: 4,
            eval_workers: 2,
            backprop_workers: 1,
            root_move_filter: Vec::new(),
        }
    }
}

impl SearchConfig {
    fn validate(&self) {
        assert!(
            self.cpuct.is_finite() && self.cpuct >= 0.0,
            "stream cpuct must be finite and non-negative"
        );
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
            cpuct: self.cpuct,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResolvedSearchConfig {
    queue_capacity: usize,
    eval_batch_size: usize,
    cpuct: f32,
}

impl ResolvedSearchConfig {
    fn search_params(self) -> SearchParams {
        SearchParams {
            cpuct: self.cpuct,
            ..SearchParams::default()
        }
    }
}

struct Shared {
    backend: Arc<dyn Backend>,
    repository: Arc<NodeRepository>,
    generation: SearchGeneration,
    params: SearchParams,
    root_move_filter: Vec<Move>,
    stopping: AtomicBool,
    outstanding: AtomicUsize,
    completed: AtomicU64,
    collisions: AtomicU64,
    network_batches: AtomicU64,
    network_evaluations: AtomicU64,
    network_batch_size_max: AtomicU64,
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
        loop {
            if self.stopping.load(Ordering::Acquire) {
                self.cancel_and_finish(event, false);
                return;
            }
            match self.eval_tx.try_send(event) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    event = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    self.cancel_and_finish(returned, false);
                    return;
                }
            }
        }
    }

    fn send_backprop(&self, mut event: BackpropEvent) {
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
        Stats {
            completed_playouts: self.completed.load(Ordering::Acquire),
            collisions: self.collisions.load(Ordering::Acquire),
            network_batches: self.network_batches.load(Ordering::Acquire),
            network_evaluations: self.network_evaluations.load(Ordering::Acquire),
            network_batch_size_max: self.network_batch_size_max.load(Ordering::Acquire),
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
        config.validate();
        let resolved = config.resolve(backend.as_ref());
        let (gather_tx, gather_rx) = bounded(resolved.queue_capacity);
        let (eval_tx, eval_rx) = bounded(resolved.queue_capacity);
        let (backprop_tx, backprop_rx) = bounded(resolved.queue_capacity);
        let root_key = NodeKey::root(root_history.last().hash());
        let shared = Arc::new(Shared {
            backend,
            repository: Arc::new(NodeRepository::default()),
            generation,
            params: resolved.search_params(),
            root_move_filter: config.root_move_filter.clone(),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            network_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            network_batch_size_max: AtomicU64::new(0),
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            gather_tx,
            eval_tx,
            backprop_tx,
        });
        let (nn_tx, nn_rx) = bounded::<NnRequest>(resolved.queue_capacity);
        let mut workers = Vec::with_capacity(
            config.gather_workers + config.eval_workers + config.backprop_workers + 1,
        );
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
            let root_is_expanded = self
                .shared
                .repository
                .get(self.root_key)
                .is_some_and(|root| root.expansion_state() == ExpansionState::Expanded);
            if !root_is_expanded {
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
        while self.shared.outstanding.load(Ordering::Acquire) >= limit
            && self.shared.error.lock().is_none()
        {
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
                shared.send_backprop(BackpropEvent::evaluation(event, wl, draw));
                return;
            }
            ExpansionState::Expanded => {
                let depth = event.variation.moves().len();
                let edge_index =
                    select_edge_from_node(node.as_ref(), depth, &shared.params, &shared.root_move_filter)
                        .expect("expanded stream node must have an edge");
                let reservation = node.reserve_edge(edge_index).expect("selected stream edge");
                let child_key = event.node_key.child(reservation.mv());
                event = event.descend(child_key, reservation);
            }
        }
    }
}

/// One encoded position for the NN thread. Reply is raw logits[POLICY] + WDL[3].
struct NnRequest {
    planes: Vec<f32>,
    reply: Sender<Result<(Vec<f32>, Vec<f32>), EnginError>>,
}

/// Eval is waiting on NN for this node (LC3: after NN completes → Backprop).
struct WaitingNn {
    event: NodeEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
    input: EvalPosition,
    reply: Receiver<Result<(Vec<f32>, Vec<f32>), EnginError>>,
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
                shared.cancel_and_finish(event, false);
            }
            break;
        }

        poll_nn_completions(&shared, &mut waiting);

        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(event) => {
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
            Ok(Ok((logits, wdl))) => {
                let item = waiting.swap_remove(i);
                if let Err(error) = complete_nn_item(shared, item, logits, wdl) {
                    shared.fail(error);
                    return;
                }
            }
            Ok(Err(error)) => {
                let item = waiting.swap_remove(i);
                cancel_waiting_item(shared, item);
                shared.fail(error);
                return;
            }
            Err(TryRecvError::Empty) => i += 1,
            Err(TryRecvError::Disconnected) => {
                let item = waiting.swap_remove(i);
                cancel_waiting_item(shared, item);
                shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
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
        Ok(Ok((logits, wdl))) => {
            let item = waiting.remove(0);
            if let Err(error) = complete_nn_item(shared, item, logits, wdl) {
                shared.fail(error);
            }
        }
        Ok(Err(error)) => {
            let item = waiting.remove(0);
            cancel_waiting_item(shared, item);
            shared.fail(error);
        }
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
            let item = waiting.remove(0);
            cancel_waiting_item(shared, item);
            shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
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
        shared.cancel_and_finish(event, false);
        return Ok(());
    }
    let node = shared.repository.get_or_insert(event.node_key);
    let history = event.variation.replay_history();
    let depth = event.variation.moves().len();
    match classify_extension(&history, depth, shared.params.two_fold_draws) {
        ExtensionKind::Terminal {
            wl,
            draw,
            plies_left,
        } => {
            node.mark_terminal(wl, draw, plies_left);
            shared.send_backprop(BackpropEvent::evaluation(event, wl, draw));
            Ok(())
        }
        ExtensionKind::Evaluate => {
            let legal_moves = history.last().board().generate_legal_moves();
            let input = EvalPosition {
                positions: history.positions().to_vec(),
                legal_moves: legal_moves.clone(),
            };
            if let Some(eval) = shared.backend.cached_evaluation(&input) {
                publish_eval(shared, event, node, legal_moves, eval);
                return Ok(());
            }
            let planes = encode_position_for_nn(&history, FillEmptyHistory::FenOnly);
            let (reply_tx, reply_rx) = bounded(1);
            send_nn_request(nn_tx, NnRequest {
                planes,
                reply: reply_tx,
            })?;
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

fn send_nn_request(nn_tx: &Sender<NnRequest>, mut request: NnRequest) -> Result<(), EnginError> {
    loop {
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
    let eval = Arc::new(EvalResult {
        wl: wdl[0] - wdl[2],
        d: wdl[1],
        m: 0.0,
        policies,
    });
    shared.backend.store_evaluation(&item.input, Arc::clone(&eval));
    publish_eval(shared, item.event, item.node, item.legal_moves, eval);
    Ok(())
}

fn publish_eval(
    shared: &Shared,
    event: NodeEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
    eval: Arc<EvalResult>,
) {
    if eval.policies.len() != legal_moves.len() {
        event.cancel();
        node.abort_evaluation();
        shared.finish(false);
        shared.fail(EnginError::PortIncomplete("stream backend policy length"));
        return;
    }
    node.publish_edges(
        legal_moves
            .into_iter()
            .zip(eval.policies.iter().copied())
            .collect(),
    );
    shared.send_backprop(BackpropEvent::evaluation(
        event,
        network_wl_to_node(eval.wl),
        eval.d,
    ));
}

fn cancel_waiting_item(shared: &Shared, item: WaitingNn) {
    item.event.cancel();
    item.node.abort_evaluation();
    shared.finish(false);
}

/// NN worker: pull encoded planes from the queue, run ONNX, reply. Merges whatever
/// is already queued (up to `batch_size`) so the GPU stays busy; Eval never waits
/// to fill that size.
fn nn_worker(shared: Arc<Shared>, receiver: Receiver<NnRequest>, batch_size: usize) {
    let plane_len = INPUT_PLANES * BOARD_ROWS * BOARD_COLS;
    loop {
        let first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(crossbeam_channel::RecvTimeoutError::Timeout)
                if shared.stopping.load(Ordering::Acquire) =>
            {
                break
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        let mut requests = vec![first];
        while requests.len() < batch_size {
            match receiver.try_recv() {
                Ok(request) => requests.push(request),
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
            Ok((logits, wdl)) => {
                shared.network_batches.fetch_add(1, Ordering::AcqRel);
                shared
                    .network_evaluations
                    .fetch_add(batch as u64, Ordering::AcqRel);
                shared
                    .network_batch_size_max
                    .fetch_max(batch as u64, Ordering::AcqRel);
                for (index, request) in requests.into_iter().enumerate() {
                    let part_logits = logits[index * POLICY_SIZE..(index + 1) * POLICY_SIZE].to_vec();
                    let part_wdl = wdl[index * 3..(index + 1) * 3].to_vec();
                    let _ = request.reply.send(Ok((part_logits, part_wdl)));
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
            Ok(first) => {
                let mut events = vec![first];
                events.extend(receiver.try_iter());
                if shared.stopping.load(Ordering::Acquire) {
                    for event in events {
                        event.cancel();
                        shared.finish(false);
                    }
                } else {
                    let result = BackpropEvent::complete_batch(events, &shared.repository);
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
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

    use super::{Search, SearchConfig, SearchLimits};
    use crate::neural::backend::{Backend, BackendAttributes, BackendComputation, EvalResult, UniformBackend};
    use crate::search::stream::{root_stats, SearchGeneration};
    use crate::EnginError;

    struct FailingComputationBackend;

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

        fn infer_encoded(&self, _planes: &[f32], _batch: usize) -> Result<(Vec<f32>, Vec<f32>), EnginError> {
            Err(EnginError::Onnx("test computation failure".to_owned()))
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
        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
        }
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

        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
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

        let root = pipeline.repository().get(pipeline.root_key()).expect("root");
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
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
}
