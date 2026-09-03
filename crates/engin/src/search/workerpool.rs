//! 事件定义 + WorkerPool + 当前固定拓扑的 Gather/Eval/NN/Backprop 线程循环壳。
//!
//! 循环只调度；算法在 `pipeline`（Gather 树走）/ `eval` / `backprop`。

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, unbounded};
use parking_lot::Mutex;
use xiangqi_core::{LegalMoveList, Move, PositionHistory};

use crate::EnginError;
use crate::neural::backend::{Backend, EvalCacheKey};
use crate::neural::{EncodedBatch, InputPlanes};
use crate::search::EdgeReservation;

use super::backprop::complete_batch;
use super::eval::{
    cancel_evaluation, drain_waiting, handle_eval_event, infer_nn_batch, poll_nn_completions, wait_one_nn_completion,
};
use super::observer::{NoQueueStamp, NoopObserver, QueueKind, QueueStamp, SearchObserver, observe_queue_wait};
use super::param::{ResolvedSearchConfig, SearchConfig};
use super::pipeline::{RECEIVE_POLL, Shared, process_gather_event};
use super::{NodeId, ValueDelta};

// --- Event -------------------------------------------------------------------

/// root history 加上从 root 到 repository node 的走法。
#[derive(Clone, Debug)]
pub struct Variation {
    base_history: Arc<PositionHistory>,
    moves: smallvec::SmallVec<[Move; 32]>,
}

impl Variation {
    pub fn root(root_history: Arc<PositionHistory>) -> Self {
        Self {
            base_history: root_history,
            moves: smallvec::SmallVec::new(),
        }
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    pub(crate) fn history(&self) -> PositionHistory {
        let mut history = self.base_history.as_ref().clone();
        for &mv in &self.moves {
            history.append(mv);
        }
        history
    }

    pub fn push(&mut self, mv: Move) {
        self.moves.push(mv);
    }
}

/// Eval 后仍需的路径。它不携带 Gather 专用的规则上下文。
#[derive(Debug)]
pub struct Event {
    pub(crate) node_id: NodeId,
    pub(crate) node_path: Vec<NodeId>,
    pub(crate) reservations: Vec<EdgeReservation>,
}

impl Event {
    pub fn cancel(self) {
        for reservation in self.reservations.into_iter().rev() {
            reservation.cancel();
        }
    }

    pub fn node_path(&self) -> &[NodeId] {
        &self.node_path
    }
}

/// Gather 到 Eval 前的完整 playout：路径、reservation 与规则上下文。
#[derive(Debug)]
pub struct GatherEvent<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: Event,
    pub variation: Variation,
    pub(crate) queued_at: S,
}

impl<S: QueueStamp> GatherEvent<S> {
    pub fn at_root(root_id: NodeId, root_history: Arc<PositionHistory>) -> Self {
        Self {
            event: Event {
                node_id: root_id,
                node_path: vec![root_id],
                reservations: Vec::new(),
            },
            variation: Variation::root(root_history),
            queued_at: S::default(),
        }
    }

    pub fn descend(mut self, child_id: NodeId, reservation: EdgeReservation) -> Self {
        self.variation.push(reservation.mv());
        self.event.node_id = child_id;
        self.event.node_path.push(child_id);
        self.event.reservations.push(reservation);
        self
    }

    pub fn cancel(self) {
        self.event.cancel();
    }

    pub fn node_path(&self) -> &[NodeId] {
        self.event.node_path()
    }

    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }

    pub(crate) fn into_event(self) -> Event {
        self.event
    }
}

pub(crate) type NnReply = Result<(Arc<EncodedBatch>, usize), EnginError>;

/// 同一 NN 事务的发送半边：NN worker 输入与其排队时间。
pub(crate) struct NnRequest<S: QueueStamp = NoQueueStamp> {
    pub(crate) planes: InputPlanes,
    pub(crate) reply: Sender<NnReply>,
    pub(crate) queued_at: S,
}

impl<S: QueueStamp> NnRequest<S> {
    pub(crate) fn new(planes: InputPlanes, reply: Sender<NnReply>) -> Self {
        Self {
            planes,
            reply,
            queued_at: S::default(),
        }
    }

    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

/// 同一 NN 事务的等待半边：Eval 收到 reply 后据此发布 edge 或回传。
pub(crate) struct NnPending {
    pub(crate) event: Event,
    pub(crate) legal_moves: LegalMoveList,
    pub(crate) cache_key: EvalCacheKey,
    pub(crate) reply: Receiver<NnReply>,
}

/// 由 Gather/Eval 路由给 Backprop 的结果（算法在 `backprop::complete_batch`）。
#[derive(Debug)]
pub struct BackpropEvent<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: Event,
    pub(crate) value: ValueDelta,
    /// 走过 `send_eval` 的叶子；backprop 完成后释放对应 claim slot。
    pub(crate) held_eval_claim: bool,
    pub(crate) queued_at: S,
}

impl<S: QueueStamp> BackpropEvent<S> {
    /// Eval 路径：走过 `send_eval`，backprop 后释放 claim。
    pub(crate) fn from_eval(event: Event, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self {
            event,
            value: ValueDelta::with_plies_left(wl, draw, plies_left),
            held_eval_claim: true,
            queued_at: S::default(),
        }
    }

    /// Gather 直发：未占 eval claim。
    pub(crate) fn from_gather(event: Event, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self {
            event,
            value: ValueDelta::with_plies_left(wl, draw, plies_left),
            held_eval_claim: false,
            queued_at: S::default(),
        }
    }

    pub fn cancel(self) {
        self.event.cancel();
    }

    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

// --- Pool --------------------------------------------------------------------

enum GatherCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<GatherEvent<O::Stamp>>),
    Shutdown,
}

enum EvalCommand<O: SearchObserver> {
    Run(
        Arc<Shared<O>>,
        Receiver<GatherEvent<O::Stamp>>,
        Sender<NnRequest<O::Stamp>>,
    ),
    Shutdown,
}

enum NnCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<NnRequest<O::Stamp>>),
    Shutdown,
}

enum BackpropCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<BackpropEvent<O::Stamp>>),
    Shutdown,
}

/// Engine 持有的当前固定 worker 拓扑。
///
/// 每个 job 独占树视图与队列；线程池只跨 job 保留线程。NN 与 Backprop
/// 各固定一个 worker；Gather/Eval 的数量也在建池时固定。这只是当前实验配置：两者会彼此
/// 受队列、claim 与回传速度制约，未来调度器可按实时压力在 Gather、Eval、proof 等工作间
/// 分配 CPU worker，而不把静态线程数量当作 job 契约。
pub(crate) struct WorkerPool<O: SearchObserver = NoopObserver> {
    gather_commands: Vec<Sender<GatherCommand<O>>>,
    eval_commands: Vec<Sender<EvalCommand<O>>>,
    nn_commands: Sender<NnCommand<O>>,
    backprop_commands: Vec<Sender<BackpropCommand<O>>>,
    job_done: Receiver<()>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    eval_batch_size: usize,
    eval_claim_limit: usize,
}

impl<O: SearchObserver> WorkerPool<O> {
    pub(crate) fn new(backend: &dyn Backend, config: &SearchConfig) -> Self {
        config.validate();
        Self::from_resolved(&config.resolve(backend))
    }

    /// 当前固定拓扑实现中，batch 或 Gather/Eval worker 数改变时必须换新 pool。
    /// 动态调度器落地后可将静态比例改为总 CPU 容量与调度策略，而非 pool 兼容性条件。
    pub(crate) fn matches_config(&self, backend: &dyn Backend, config: &SearchConfig) -> bool {
        let config = config.resolve(backend);
        self.eval_batch_size == config.eval_batch_size
            && self.eval_claim_limit == config.eval_claim_limit
            && self.gather_commands.len() == config.gather_workers
            && self.eval_commands.len() == config.eval_workers
    }

    fn from_resolved(config: &ResolvedSearchConfig) -> Self {
        let (job_done_tx, job_done) = unbounded();
        let eval_batch_size = config.eval_batch_size;
        let eval_claim_limit = config.eval_claim_limit;
        let mut gather_commands = Vec::with_capacity(config.gather_workers);
        let mut eval_commands = Vec::with_capacity(config.eval_workers);
        let (nn_commands, nn_rx) = unbounded();
        let (backprop_commands, backprop_rx) = unbounded();
        let mut threads = Vec::with_capacity(config.gather_workers + config.eval_workers + 2);
        for _ in 0..config.gather_workers {
            let (tx, rx) = unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || {
                persistent_gather_worker::<O>(rx, job_done, eval_claim_limit)
            }));
            gather_commands.push(tx);
        }
        for _ in 0..config.eval_workers {
            let (tx, rx) = unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || persistent_eval_worker::<O>(rx, job_done)));
            eval_commands.push(tx);
        }
        threads.push(thread::spawn({
            let job_done = job_done_tx.clone();
            move || persistent_nn_worker::<O>(nn_rx, job_done, eval_batch_size)
        }));
        threads.push(thread::spawn({
            let job_done = job_done_tx.clone();
            move || persistent_backprop_worker::<O>(backprop_rx, job_done)
        }));
        Self {
            gather_commands,
            eval_commands,
            nn_commands,
            backprop_commands: vec![backprop_commands],
            job_done,
            threads: Mutex::new(threads),
            eval_batch_size: config.eval_batch_size,
            eval_claim_limit: config.eval_claim_limit,
        }
    }

    pub(crate) fn start_job(
        &self,
        shared: &Arc<Shared<O>>,
        gather_rx: &Receiver<GatherEvent<O::Stamp>>,
        eval_rx: &Receiver<GatherEvent<O::Stamp>>,
        nn_tx: &Sender<NnRequest<O::Stamp>>,
        nn_rx: &Receiver<NnRequest<O::Stamp>>,
        backprop_rx: &Receiver<BackpropEvent<O::Stamp>>,
    ) {
        for sender in &self.gather_commands {
            sender
                .send(GatherCommand::Run(Arc::clone(shared), gather_rx.clone()))
                .expect("persistent search worker is alive");
        }
        self.nn_commands
            .send(NnCommand::Run(Arc::clone(shared), nn_rx.clone()))
            .expect("persistent nn worker is alive");
        for sender in &self.eval_commands {
            sender
                .send(EvalCommand::Run(Arc::clone(shared), eval_rx.clone(), nn_tx.clone()))
                .expect("persistent eval worker is alive");
        }
        for sender in &self.backprop_commands {
            sender
                .send(BackpropCommand::Run(Arc::clone(shared), backprop_rx.clone()))
                .expect("persistent backprop worker is alive");
        }
    }

    pub(crate) fn finish_job(&self) {
        for _ in 0..self.gather_commands.len() + self.eval_commands.len() + self.backprop_commands.len() + 1 {
            self.job_done.recv().expect("persistent worker completion");
        }
    }

    pub(crate) fn eval_claim_limit(&self) -> usize {
        self.eval_claim_limit
    }

    pub(crate) fn eval_batch_size(&self) -> usize {
        self.eval_batch_size
    }

    pub(crate) fn assert_compatible(&self, config: &ResolvedSearchConfig) {
        debug_assert_eq!(
            self.gather_commands.len(),
            config.gather_workers,
            "worker pool gather topology changed"
        );
        debug_assert_eq!(
            self.eval_commands.len(),
            config.eval_workers,
            "worker pool eval topology changed"
        );
        debug_assert_eq!(
            self.eval_batch_size, config.eval_batch_size,
            "worker pool batch size changed"
        );
        debug_assert_eq!(
            self.eval_claim_limit, config.eval_claim_limit,
            "worker pool nn window changed"
        );
    }
}

impl<O: SearchObserver> Drop for WorkerPool<O> {
    fn drop(&mut self) {
        for sender in &self.gather_commands {
            let _ = sender.send(GatherCommand::Shutdown);
        }
        let _ = self.nn_commands.send(NnCommand::Shutdown);
        for sender in &self.eval_commands {
            let _ = sender.send(EvalCommand::Shutdown);
        }
        for sender in &self.backprop_commands {
            let _ = sender.send(BackpropCommand::Shutdown);
        }
        for worker in self.threads.get_mut().drain(..) {
            let _ = worker.join();
        }
    }
}

// --- Gather / Eval / NN / Backprop loops

fn gather_worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    receiver: Receiver<GatherEvent<O::Stamp>>,
    eval_claim_limit: usize,
) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(mut event) => {
                if O::ENABLED {
                    observe_queue_wait(&mut event.queued_at, &shared.observer, QueueKind::Gather);
                }
                if shared.stopping.load(Ordering::Acquire) {
                    event.cancel();
                    shared.finish(1, false);
                } else {
                    process_gather_event(&shared, event, eval_claim_limit);
                }
            }
            Err(RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn persistent_gather_worker<O: SearchObserver>(
    commands: Receiver<GatherCommand<O>>,
    job_done: Sender<()>,
    eval_claim_limit: usize,
) {
    while let Ok(command) = commands.recv() {
        match command {
            GatherCommand::Run(shared, receiver) => {
                gather_worker(shared, receiver, eval_claim_limit);
                let _ = job_done.send(());
            }
            GatherCommand::Shutdown => break,
        }
    }
}

fn eval_worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    receiver: Receiver<GatherEvent<O::Stamp>>,
    nn_tx: Sender<NnRequest<O::Stamp>>,
) {
    let mut waiting = Vec::<NnPending>::new();
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            drain_waiting(&shared, &mut waiting);
            while let Ok(event) = receiver.try_recv() {
                cancel_evaluation(&shared, event.into_event());
            }
            shared.cancel_all_collisions();
            break;
        }

        poll_nn_completions(&shared, &mut waiting);

        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(mut event) => {
                if O::ENABLED {
                    observe_queue_wait(&mut event.queued_at, &shared.observer, QueueKind::Eval);
                }
                if let Err(error) = handle_eval_event(&shared, &nn_tx, &mut waiting, event) {
                    shared.fail(error);
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if waiting.is_empty() || shared.stopping.load(Ordering::Acquire) {
                    continue;
                }
                wait_one_nn_completion(&shared, &mut waiting);
            }
            Err(RecvTimeoutError::Disconnected) => {
                drain_waiting(&shared, &mut waiting);
                while let Ok(event) = receiver.try_recv() {
                    cancel_evaluation(&shared, event.into_event());
                }
                shared.cancel_all_collisions();
                break;
            }
        }
    }
}

fn persistent_eval_worker<O: SearchObserver>(commands: Receiver<EvalCommand<O>>, job_done: Sender<()>) {
    while let Ok(command) = commands.recv() {
        match command {
            EvalCommand::Run(shared, receiver, nn_tx) => {
                eval_worker(shared, receiver, nn_tx);
                let _ = job_done.send(());
            }
            EvalCommand::Shutdown => break,
        }
    }
}

fn nn_worker<O: SearchObserver>(shared: Arc<Shared<O>>, receiver: Receiver<NnRequest<O::Stamp>>, batch_size: usize) {
    loop {
        let mut first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        if O::ENABLED {
            observe_queue_wait(&mut first.queued_at, &shared.observer, QueueKind::Nn);
        }
        let mut requests = vec![first];
        while requests.len() < batch_size {
            match receiver.try_recv() {
                Ok(mut request) => {
                    if O::ENABLED {
                        observe_queue_wait(&mut request.queued_at, &shared.observer, QueueKind::Nn);
                    }
                    requests.push(request);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        infer_nn_batch(&shared, requests);
    }
}

fn persistent_nn_worker<O: SearchObserver>(commands: Receiver<NnCommand<O>>, job_done: Sender<()>, batch_size: usize) {
    while let Ok(command) = commands.recv() {
        match command {
            NnCommand::Run(shared, receiver) => {
                nn_worker(shared, receiver, batch_size);
                let _ = job_done.send(());
            }
            NnCommand::Shutdown => break,
        }
    }
}

fn backprop_worker<O: SearchObserver>(shared: Arc<Shared<O>>, receiver: Receiver<BackpropEvent<O::Stamp>>) {
    loop {
        let first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let mut events = Vec::with_capacity(1 + receiver.len());
        events.push(first);
        events.extend(receiver.try_iter());
        if events.is_empty() {
            continue;
        }
        if shared.stopping.load(Ordering::Acquire) {
            let n = events.len();
            let held = events.iter().filter(|event| event.held_eval_claim).count();
            for event in events {
                event.cancel();
            }
            shared.release_eval_claims(held);
            shared.finish(n, false);
            continue;
        }
        if O::ENABLED {
            for event in &mut events {
                observe_queue_wait(&mut event.queued_at, &shared.observer, QueueKind::Backprop);
            }
        }
        let claims: Vec<(bool, NodeId)> = events
            .iter()
            .map(|event| (event.held_eval_claim, event.event.node_id))
            .collect();
        let result = complete_batch(events, &shared.arena);
        for (_, id) in &claims {
            shared.cancel_collisions(*id);
        }
        let held = claims.iter().filter(|(held, _)| *held).count();
        shared.release_eval_claims(held);
        shared
            .completed_depth
            .fetch_add(result.completed_depth, Ordering::AcqRel);
        shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
        shared.finish(result.completed_playouts as usize, true);
    }
}

fn persistent_backprop_worker<O: SearchObserver>(commands: Receiver<BackpropCommand<O>>, job_done: Sender<()>) {
    while let Ok(command) = commands.recv() {
        match command {
            BackpropCommand::Run(shared, receiver) => {
                backprop_worker(shared, receiver);
                let _ = job_done.send(());
            }
            BackpropCommand::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::thread;

    use crossbeam_channel::bounded;
    use parking_lot::{Condvar, Mutex};
    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use super::{BackpropEvent, GatherEvent, backprop_worker};
    use crate::neural::backend::{Backend, UniformBackend};
    use crate::search::observer::NoopObserver;
    use crate::search::param::SearchParams;
    use crate::search::pipeline::Shared;
    use crate::search::{NoQueueStamp, NodeArena};

    #[test]
    fn backprop_completion_releases_the_eval_claim() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let arena = Arc::new(NodeArena::default());
        let root = arena.allocate();
        let (gather_tx, _) = bounded(1);
        let (eval_tx, _) = bounded(1);
        let (backprop_tx, backprop_rx) = bounded(1);
        let shared = Arc::new(Shared {
            backend: Arc::new(UniformBackend::default()) as Arc<dyn Backend>,
            arena: Arc::clone(&arena),
            params: SearchParams::default(),
            root_move_filter: Mutex::new(Vec::new()),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(1),
            nn_inflight: AtomicUsize::new(1),
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            observer: NoopObserver,
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            gather_tx,
            eval_tx,
            backprop_tx: backprop_tx.clone(),
            collision_waiters: Mutex::new(Vec::new()),
        });
        backprop_tx
            .send(BackpropEvent::from_eval(
                GatherEvent::<NoQueueStamp>::at_root(root, history).into_event(),
                0.4,
                0.2,
                2.0,
            ))
            .expect("backprop event");
        let worker_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || backprop_worker(worker_shared, backprop_rx));
        while shared.completed.load(Ordering::Acquire) == 0 {
            thread::yield_now();
        }
        shared.stopping.store(true, Ordering::Release);
        worker.join().expect("backprop worker");

        assert_eq!(arena.get(root).expect("root").completed_visits(), 1);
        assert_eq!(shared.nn_inflight.load(Ordering::Acquire), 0);
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 0);
    }
}
