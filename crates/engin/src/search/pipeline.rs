//! Stream 搜索主线：Search / Eval / NN worker。
//!
//! worker 角色划分可参考 LC3 Overview 的 "Workers"：
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! Eval 负责终局、缓存、合法着和编码；NN 线程只做「取批 → 推理 → 交回」，
//! 不处理象棋/搜索逻辑。Search worker 优先处理回传，再 gather 下一个 NN batch。
//! 每个 batch 的 logical visit 在每层重新按 PUCT 分配；碰撞只取消 reservation。

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, SendTimeoutError, Sender, TryRecvError, TrySendError, bounded};
use parking_lot::{Condvar, Mutex};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;
use crate::neural::backend::{Backend, EvalCacheKey, EvalResult};
use crate::neural::{
    EncodedBatch, FillEmptyHistory, InputPlanes, MOVE_HISTORY, encode_position_input_planes,
    eval_result_from_encoded_row,
};

use super::extension::{ExtensionKind, classify_extension, path_terminal_value};
use super::graph::ChildLink;
use super::{
    BackpropEvent, ExpansionState, Node, NodeEvent, NodeKey, NodeRepository, SearchGraph, SearchParams, ValueDelta,
    network_wl_to_node, select_edge,
};

const RECEIVE_POLL: Duration = Duration::from_millis(10);

/// 当前 root 的累计 visit 搜索预算。
///
/// `go nodes N` 包含 graph reuse 前已有的 root N；时钟仍只约束本次 job。
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

/// 一条队列的 benchmark 等待时间快照。
#[cfg(feature = "benchmark")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueueStats {
    pub samples: u64,
    pub total_wait_ns: u64,
    pub max_wait_ns: u64,
}

/// 正式搜索计数。
/// 参考：LC3 Overview 的 "Stats Collection"。
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    #[cfg(feature = "benchmark")]
    pub submitted_playouts: u64,
    pub completed_playouts: u64,
    /// 已完成 leaf 的平均深度，root 深度为一；语义对应 px0 `ThinkingInfo::depth`，
    /// 不是当前 PV 长度。
    pub average_depth: u64,
    /// 最深已完成 leaf，语义对应 px0 `seldepth`。
    pub max_depth: u64,
    #[cfg(feature = "benchmark")]
    pub collisions: u64,
    pub network_batches: u64,
    pub network_evaluations: u64,
    /// 正常 Eval 命中的 NN cache 次数。
    pub cache_hits: u64,
    /// 本次搜索观察到的最大 NN batch。
    pub network_batch_size_max: u64,
    #[cfg(feature = "benchmark")]
    pub peak_in_flight: u64,
    /// 按距 root 的 variation 深度统计 collision。
    #[cfg(feature = "benchmark")]
    pub collisions_by_depth: Vec<u64>,
    #[cfg(feature = "benchmark")]
    pub gather_queue: QueueStats,
    #[cfg(feature = "benchmark")]
    pub eval_queue: QueueStats,
    #[cfg(feature = "benchmark")]
    pub nn_queue: QueueStats,
    #[cfg(feature = "benchmark")]
    pub backprop_queue: QueueStats,
}

#[cfg(feature = "benchmark")]
#[derive(Default)]
struct QueueMetrics {
    samples: AtomicU64,
    total_wait_ns: AtomicU64,
    max_wait_ns: AtomicU64,
}

#[cfg(feature = "benchmark")]
impl QueueMetrics {
    /// 记录一次 event 交接；仅 benchmark 启用。
    /// 参考：LC3 Overview 的 "Stats Collection"。
    fn record(&self, wait: Duration) {
        let nanos = wait.as_nanos().min(u64::MAX as u128) as u64;
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.total_wait_ns.fetch_add(nanos, Ordering::Relaxed);
        self.max_wait_ns.fetch_max(nanos, Ordering::Relaxed);
    }

    /// 读取足够一致的诊断快照；它不是搜索状态。
    /// 参考：LC3 Overview 的 "Stats Collection"。
    fn snapshot(&self) -> QueueStats {
        QueueStats {
            samples: self.samples.load(Ordering::Relaxed),
            total_wait_ns: self.total_wait_ns.load(Ordering::Relaxed),
            max_wait_ns: self.max_wait_ns.load(Ordering::Relaxed),
        }
    }
}

/// 运行中搜索可克隆的 stop 句柄。
///
/// controller 可持有它；`Search` owner 负责 drain 当前 job 并归还 worker。
/// 参考：LC3 Overview 的 "WatchdogWorker"。
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
    /// Search/Eval/NN 队列深度。`0` 表示 `max(4096, 64 * resolved_batch)`。
    pub queue_capacity: usize,
    /// 已有多个编码局面时的 NN GPU 合批大小。`0` 表示 backend 的
    /// `recommended_batch_size`。
    pub eval_batch_size: usize,
    pub params: SearchParams,
    pub gather_workers: usize,
    /// Eval worker 数。它负责准备、缓存、合法着；NN inference 是独立线程。
    pub eval_workers: usize,
    /// UCI `go searchmoves` 空表示不限制。
    pub root_move_filter: Vec<Move>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 0,
            eval_batch_size: 0,
            params: SearchParams::default(),
            gather_workers: 4,
            eval_workers: 4,
            root_move_filter: Vec::new(),
        }
    }
}

impl SearchConfig {
    fn validate(&self) {
        self.params.validate();
        assert!(self.gather_workers > 0, "stream requires at least one gather worker");
        assert!(self.eval_workers > 0, "stream requires at least one eval worker");
    }

    /// Fills `0` sentinels from the backend; returns concrete queue/batch sizes.
    fn resolve(&self, backend: &dyn Backend) -> ResolvedSearchConfig {
        let recommended = backend.attributes().recommended_batch_size.max(1);
        let maximum = backend.attributes().maximum_batch_size.max(1);
        let eval_batch_size = if self.eval_batch_size == 0 {
            recommended
        } else {
            self.eval_batch_size.min(maximum)
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
            gather_workers: self.gather_workers,
            eval_workers: self.eval_workers,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResolvedSearchConfig {
    queue_capacity: usize,
    eval_batch_size: usize,
    params: SearchParams,
    gather_workers: usize,
    eval_workers: usize,
}

/// 一个逻辑 NN batch 的生命周期。
///
/// Search owner 最多同时保留两个：一个由 NN/Eval/回传完成，另一个可并行 gather。
/// 它不是一轮全局屏障；batch 收集完叶子即可关闭，回传仍由 Search worker 优先执行。
struct SearchBatch {
    state: Mutex<SearchBatchState>,
    ready: Condvar,
}

struct SearchBatchState {
    gather_left: usize,
    eval_left: usize,
    nn_requests: usize,
    playouts_left: usize,
}

impl SearchBatch {
    fn new(gather_tasks: usize) -> Self {
        Self {
            state: Mutex::new(SearchBatchState {
                gather_left: gather_tasks,
                eval_left: 0,
                nn_requests: 0,
                playouts_left: 0,
            }),
            ready: Condvar::new(),
        }
    }

    fn begin_eval(&self) {
        self.state.lock().eval_left += 1;
    }

    fn begin_playout(&self) {
        self.state.lock().playouts_left += 1;
    }

    fn finish_playout(&self) {
        let mut state = self.state.lock();
        assert!(state.playouts_left > 0, "search batch playout underflow");
        state.playouts_left -= 1;
        if state.gather_left == 0 && state.eval_left == 0 && state.playouts_left == 0 {
            self.ready.notify_all();
        }
    }

    fn finish_eval(&self, needs_nn: bool) {
        let mut state = self.state.lock();
        assert!(state.eval_left > 0, "search batch eval underflow");
        state.eval_left -= 1;
        state.nn_requests += usize::from(needs_nn);
        if state.gather_left == 0 && state.eval_left == 0 {
            self.ready.notify_all();
        }
    }

    fn finish_gather(&self) {
        let mut state = self.state.lock();
        assert!(state.gather_left > 0, "search batch worker underflow");
        state.gather_left -= 1;
        if state.gather_left == 0 && state.eval_left == 0 {
            self.ready.notify_all();
        }
    }

    fn wait_for_nn_requests(&self, stopping: &AtomicBool) -> usize {
        let mut state = self.state.lock();
        while (state.gather_left != 0 || state.eval_left != 0) && !stopping.load(Ordering::Acquire) {
            self.ready.wait_for(&mut state, RECEIVE_POLL);
        }
        state.nn_requests
    }

    fn wait_until_closed(&self, stopping: &AtomicBool) {
        let mut state = self.state.lock();
        while (state.gather_left != 0 || state.eval_left != 0) && !stopping.load(Ordering::Acquire) {
            self.ready.wait_for(&mut state, RECEIVE_POLL);
        }
    }

    fn wait_until_finished(&self, stopping: &AtomicBool) {
        let mut state = self.state.lock();
        while (state.gather_left != 0 || state.eval_left != 0 || state.playouts_left != 0)
            && !stopping.load(Ordering::Acquire)
        {
            self.ready.wait_for(&mut state, RECEIVE_POLL);
        }
    }
}

/// 一个 Search worker 在 batch 内分配的逻辑 visit 预算。展开节点会按临时 PUCT
/// 把它递归拆分为多个叶子；collision 不进入 completed N。
struct SearchTask {
    batch: Arc<SearchBatch>,
    root_key: NodeKey,
    root_history: Arc<PositionHistory>,
    visits: u32,
    /// 该串行 gather 允许累计的 collision logical visit；恰好等于它的 logical
    /// visit 配额。碰撞只消耗本批已分配的预算，不会因跨回合 root N 增长而扩张。
    collision_budget: u32,
}

/// Eval 的输入与其所属 batch 一起移动；同一 batch 的 Eval 都完成编码/缓存判断后，
/// NN 才执行该 batch 的真实推理。
struct EvalTask {
    event: NodeEvent,
    batch: Arc<SearchBatch>,
}

struct BackpropTask {
    event: BackpropEvent,
    batch: Arc<SearchBatch>,
}

enum SearchCommand {
    Run(Arc<Shared>, Receiver<SearchTask>, Receiver<BackpropTask>),
    Shutdown,
}

enum EvalCommand {
    Run(Arc<Shared>, Receiver<EvalTask>, Sender<NnRequest>),
    Shutdown,
}

enum NnCommand {
    Run(Arc<Shared>, Receiver<NnRequest>),
    Shutdown,
}

/// Engine 持有的固定 worker 拓扑。
///
/// 每个 job 独占树视图、队列与 generation；线程池只跨 job 保留线程。
/// 参考：LC3 Overview 的 "Workers" / "Search"。
pub(crate) struct WorkerPool {
    search_commands: Vec<Sender<SearchCommand>>,
    eval_commands: Vec<Sender<EvalCommand>>,
    nn_commands: Sender<NnCommand>,
    job_done: Receiver<()>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    eval_batch_size: usize,
}

impl WorkerPool {
    pub(crate) fn new(backend: &dyn Backend, config: &SearchConfig) -> Self {
        config.validate();
        Self::from_resolved(&config.resolve(backend))
    }

    /// 对齐 LC3 Overview 的固定 worker job：batch 或任一 worker 数改变时，必须
    /// 使用相应拓扑的新 pool。
    pub(crate) fn matches_config(&self, backend: &dyn Backend, config: &SearchConfig) -> bool {
        let config = config.resolve(backend);
        self.eval_batch_size == config.eval_batch_size
            && self.search_commands.len() == config.gather_workers
            && self.eval_commands.len() == config.eval_workers
    }

    fn from_resolved(config: &ResolvedSearchConfig) -> Self {
        let (job_done_tx, job_done) = crossbeam_channel::unbounded();
        let eval_batch_size = config.eval_batch_size;
        let mut search_commands = Vec::with_capacity(config.gather_workers);
        let mut eval_commands = Vec::with_capacity(config.eval_workers);
        let mut threads = Vec::with_capacity(config.gather_workers + config.eval_workers + 1);
        for _ in 0..config.gather_workers {
            let (tx, rx) = crossbeam_channel::unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || persistent_search_worker(rx, job_done)));
            search_commands.push(tx);
        }
        let (nn_commands, nn_rx) = crossbeam_channel::unbounded();
        threads.push(thread::spawn({
            let job_done = job_done_tx.clone();
            move || persistent_nn_worker(nn_rx, job_done, eval_batch_size)
        }));
        for _ in 0..config.eval_workers {
            let (tx, rx) = crossbeam_channel::unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || persistent_eval_worker(rx, job_done)));
            eval_commands.push(tx);
        }
        Self {
            search_commands,
            eval_commands,
            nn_commands,
            job_done,
            threads: Mutex::new(threads),
            eval_batch_size: config.eval_batch_size,
        }
    }

    fn start_job(
        &self,
        shared: &Arc<Shared>,
        search_rx: &Receiver<SearchTask>,
        eval_rx: &Receiver<EvalTask>,
        nn_tx: &Sender<NnRequest>,
        nn_rx: &Receiver<NnRequest>,
        backprop_rx: &Receiver<BackpropTask>,
    ) {
        for sender in &self.search_commands {
            sender
                .send(SearchCommand::Run(
                    Arc::clone(shared),
                    search_rx.clone(),
                    backprop_rx.clone(),
                ))
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
    }

    fn finish_job(&self) {
        for _ in 0..self.search_commands.len() + self.eval_commands.len() + 1 {
            self.job_done.recv().expect("persistent worker completion");
        }
    }

    fn assert_compatible(&self, config: &ResolvedSearchConfig) {
        assert_eq!(
            self.search_commands.len(),
            config.gather_workers,
            "worker pool gather topology changed"
        );
        assert_eq!(
            self.eval_commands.len(),
            config.eval_workers,
            "worker pool eval topology changed"
        );
        assert_eq!(
            self.eval_batch_size, config.eval_batch_size,
            "worker pool batch size changed"
        );
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        for sender in &self.search_commands {
            let _ = sender.send(SearchCommand::Shutdown);
        }
        let _ = self.nn_commands.send(NnCommand::Shutdown);
        for sender in &self.eval_commands {
            let _ = sender.send(EvalCommand::Shutdown);
        }
        for worker in self.threads.get_mut().drain(..) {
            let _ = worker.join();
        }
    }
}

struct Shared {
    backend: Arc<dyn Backend>,
    repository: Arc<NodeRepository>,
    generation: u64,
    params: SearchParams,
    root_move_filter: Vec<Move>,
    /// 当前 root 的 history 已裁决结束，但不能污染同 board 的共享 node。
    root_path_terminal: AtomicBool,
    stopping: AtomicBool,
    outstanding: AtomicUsize,
    #[cfg(feature = "benchmark")]
    submitted: AtomicU64,
    #[cfg(feature = "benchmark")]
    peak_in_flight: AtomicU64,
    completed: AtomicU64,
    completed_depth: AtomicU64,
    max_depth: AtomicU64,
    #[cfg(feature = "benchmark")]
    collisions: AtomicU64,
    network_batches: AtomicU64,
    network_evaluations: AtomicU64,
    cache_hits: AtomicU64,
    network_batch_size_max: AtomicU64,
    #[cfg(feature = "benchmark")]
    collisions_by_depth: Mutex<Vec<u64>>,
    #[cfg(feature = "benchmark")]
    gather_queue: QueueMetrics,
    #[cfg(feature = "benchmark")]
    eval_queue: QueueMetrics,
    #[cfg(feature = "benchmark")]
    nn_queue: QueueMetrics,
    #[cfg(feature = "benchmark")]
    backprop_queue: QueueMetrics,
    error: Mutex<Option<EnginError>>,
    idle_lock: Mutex<()>,
    idle: Condvar,
    search_tx: Sender<SearchTask>,
    eval_tx: Sender<EvalTask>,
    backprop_tx: Sender<BackpropTask>,
}

impl Shared {
    /// 一条真实 leaf 开始进入 Eval/Backprop。collision 只保留临时 reservation，
    /// 不计为 playout。
    fn start_playout(&self, batch: &SearchBatch) {
        batch.begin_playout();
        let outstanding = self.outstanding.fetch_add(1, Ordering::AcqRel) + 1;
        #[cfg(feature = "benchmark")]
        {
            self.submitted.fetch_add(1, Ordering::Relaxed);
            self.peak_in_flight.fetch_max(outstanding as u64, Ordering::Relaxed);
        }
        #[cfg(not(feature = "benchmark"))]
        let _ = outstanding;
    }

    fn complete_root_terminal(&self) {
        self.completed.fetch_add(1, Ordering::AcqRel);
    }

    fn finish(&self) {
        let previous = self.outstanding.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "stream outstanding task underflow");
        // 唤醒等待本轮真实 leaf 完成的 owner。
        let _guard = self.idle_lock.lock();
        self.idle.notify_all();
        let _ = previous;
    }

    fn add_completed_visits(&self, visits: u32) {
        self.completed.fetch_add(visits as u64, Ordering::AcqRel);
    }

    /// 一个 batch gather 结束后归还 collision 的 virtual visit；它从未进入 Eval/NN/Backprop。
    fn finish_collision(&self, event: NodeEvent) {
        #[cfg(feature = "benchmark")]
        {
            let visits = event
                .reservations
                .iter()
                .map(|reservation| reservation.visits())
                .max()
                .unwrap_or(1);
            self.collisions.fetch_add(visits as u64, Ordering::AcqRel);
            self.record_collision_depths(&[event.variation.moves().len()]);
        }
        event.cancel();
    }

    /// Cancels an event after Gather has claimed its leaf for Eval.
    ///
    /// Reference: LC3 overview's EvalWorker ownership model. Releasing only
    /// the edge reservations would leave the claimed node permanently
    /// `Evaluating`, so this also restores it to `Unexpanded`.
    fn cancel_claimed_evaluation(&self, event: NodeEvent, batch: Arc<SearchBatch>) {
        let node = self
            .repository
            .get(event.node_key)
            .expect("claimed stream node remains in the repository");
        cancel_evaluation(self, event, node, batch);
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
    fn send_eval(&self, mut task: EvalTask) {
        #[cfg(feature = "benchmark")]
        task.event.mark_queued();
        loop {
            if self.stopping.load(Ordering::Acquire) {
                task.batch.finish_eval(false);
                self.cancel_claimed_evaluation(task.event, task.batch);
                return;
            }
            match self.eval_tx.try_send(task) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    task = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    returned.batch.finish_eval(false);
                    self.cancel_claimed_evaluation(returned.event, returned.batch);
                    return;
                }
            }
        }
    }

    fn send_backprop(&self, mut task: BackpropTask) {
        #[cfg(feature = "benchmark")]
        task.event.mark_queued();
        loop {
            if self.stopping.load(Ordering::Acquire) {
                task.event.cancel();
                task.batch.finish_playout();
                self.finish();
                return;
            }
            match self.backprop_tx.try_send(task) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    task = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    returned.event.cancel();
                    returned.batch.finish_playout();
                    self.finish();
                    return;
                }
            }
        }
    }

    fn stats(&self) -> Stats {
        let completed_playouts = self.completed.load(Ordering::Acquire);
        Stats {
            #[cfg(feature = "benchmark")]
            submitted_playouts: self.submitted.load(Ordering::Acquire),
            completed_playouts,
            average_depth: self.completed_depth.load(Ordering::Acquire) / completed_playouts.max(1),
            max_depth: self.max_depth.load(Ordering::Acquire),
            #[cfg(feature = "benchmark")]
            collisions: self.collisions.load(Ordering::Acquire),
            network_batches: self.network_batches.load(Ordering::Acquire),
            network_evaluations: self.network_evaluations.load(Ordering::Acquire),
            cache_hits: self.cache_hits.load(Ordering::Acquire),
            network_batch_size_max: self.network_batch_size_max.load(Ordering::Acquire),
            #[cfg(feature = "benchmark")]
            peak_in_flight: self.peak_in_flight.load(Ordering::Acquire),
            #[cfg(feature = "benchmark")]
            collisions_by_depth: self.collisions_by_depth.lock().clone(),
            #[cfg(feature = "benchmark")]
            gather_queue: self.gather_queue.snapshot(),
            #[cfg(feature = "benchmark")]
            eval_queue: self.eval_queue.snapshot(),
            #[cfg(feature = "benchmark")]
            nn_queue: self.nn_queue.snapshot(),
            #[cfg(feature = "benchmark")]
            backprop_queue: self.backprop_queue.snapshot(),
        }
    }

    /// Adds per-depth collision counts without changing selection behavior.
    ///
    /// Reference: LC3 overview, "Stats Collection".
    #[cfg(feature = "benchmark")]
    fn record_collision_depths(&self, depths: &[usize]) {
        let mut counts = self.collisions_by_depth.lock();
        for &depth in depths {
            if counts.len() <= depth {
                counts.resize(depth + 1, 0);
            }
            counts[depth] += 1;
        }
    }
}

/// LC3 风格流式搜索：Search / Eval / NN；Backprop 由 Search worker 优先处理。
/// Eval：terminal | cache → Backprop；否则稀疏编码 → NN queue；NN expand + 合批推理后
/// 整批交回 → Eval 切行/softmax/edges → Backprop。
pub struct Search {
    shared: Arc<Shared>,
    root_history: Arc<PositionHistory>,
    root_key: NodeKey,
    /// 启动本次 job 前 root 已有的 completed N。它计入 UCI `go nodes`，但不计入本次 NPS。
    initial_visits: u64,
    worker_pool: Arc<WorkerPool>,
    workers_idle: bool,
}

impl Search {
    pub fn new(
        backend: Arc<dyn Backend>,
        generation: u64,
        root_history: Arc<PositionHistory>,
        config: SearchConfig,
    ) -> Self {
        let graph = SearchGraph::new(root_history);
        Self::new_with_graph(backend, generation, &graph, config)
    }

    /// 从保留图创建独立搜索；该搜索自己创建、销毁 worker pool。
    pub fn new_with_graph(
        backend: Arc<dyn Backend>,
        generation: u64,
        graph: &SearchGraph,
        config: SearchConfig,
    ) -> Self {
        let worker_pool = Arc::new(WorkerPool::new(backend.as_ref(), &config));
        Self::new_with_graph_in_pool(backend, generation, graph, config, worker_pool)
    }

    pub(crate) fn new_with_graph_in_pool(
        backend: Arc<dyn Backend>,
        generation: u64,
        graph: &SearchGraph,
        config: SearchConfig,
        worker_pool: Arc<WorkerPool>,
    ) -> Self {
        config.validate();
        let resolved = config.resolve(backend.as_ref());
        worker_pool.assert_compatible(&resolved);
        let (search_tx, search_rx) = bounded(resolved.queue_capacity);
        let (eval_tx, eval_rx) = bounded(resolved.queue_capacity);
        let (backprop_tx, backprop_rx) = bounded(resolved.queue_capacity);
        // UCI/graph 持有完整 history 用于跨回合定位；每个 event 只需要重复规则自
        // 最近零化着以来的后缀，以及 NN 的最近 8 层。这里一次裁剪后由整次 job 共享。
        let root_history = Arc::new(graph.root_history().search_window(MOVE_HISTORY));
        let root_key = graph.root_key();
        let initial_visits = graph
            .repository()
            .get(root_key)
            .map_or(0, |root| root.completed_visits() as u64);
        let shared = Arc::new(Shared {
            backend,
            repository: Arc::clone(graph.repository()),
            generation,
            params: resolved.params,
            root_move_filter: config.root_move_filter.clone(),
            root_path_terminal: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(0),
            #[cfg(feature = "benchmark")]
            submitted: AtomicU64::new(0),
            #[cfg(feature = "benchmark")]
            peak_in_flight: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            #[cfg(feature = "benchmark")]
            collisions: AtomicU64::new(0),
            network_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            network_batch_size_max: AtomicU64::new(0),
            #[cfg(feature = "benchmark")]
            collisions_by_depth: Mutex::new(Vec::new()),
            #[cfg(feature = "benchmark")]
            gather_queue: QueueMetrics::default(),
            #[cfg(feature = "benchmark")]
            eval_queue: QueueMetrics::default(),
            #[cfg(feature = "benchmark")]
            nn_queue: QueueMetrics::default(),
            #[cfg(feature = "benchmark")]
            backprop_queue: QueueMetrics::default(),
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            search_tx,
            eval_tx,
            backprop_tx,
        });
        let (nn_tx, nn_rx) = bounded::<NnRequest>(resolved.queue_capacity);
        worker_pool.start_job(&shared, &search_rx, &eval_rx, &nn_tx, &nn_rx, &backprop_rx);
        drop(nn_tx);
        Self {
            shared,
            root_history,
            root_key,
            initial_visits,
            worker_pool,
            workers_idle: false,
        }
    }

    pub fn repository(&self) -> &Arc<NodeRepository> {
        &self.shared.repository
    }

    pub fn root_key(&self) -> NodeKey {
        self.root_key
    }

    pub fn initial_visits(&self) -> u64 {
        self.initial_visits
    }

    /// 当前完整 history 已在路径规则下裁决结束。它不能写入 board-key shared node，
    /// 但 UCI 仍必须把当前 root 当作终局处理。
    pub(crate) fn root_is_path_terminal(&self) -> bool {
        self.shared.root_path_terminal.load(Ordering::Acquire)
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
    /// UCI controller uses this for `stop`; owner cleanup drains this job and
    /// returns its workers to the pool.
    ///
    /// Reference: LC3 overview, "Watchdog" and worker stop coordination.
    pub fn request_stop(&self) {
        self.control().request_stop();
    }

    pub fn is_stopping(&self) -> bool {
        self.shared.stopping.load(Ordering::Acquire)
    }

    /// 提交一个逻辑 NN batch。真实 leaf 总数最多为一个 NN batch，并按 Search worker 均分；
    /// 同轮 collision 仅作为 temporary virtual visit，绝不进入 completed N。
    fn submit_batch(&self, visits: usize) -> Result<Arc<SearchBatch>, EnginError> {
        if self.shared.stopping.load(Ordering::Acquire) {
            return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
        }
        let workers = self.worker_pool.search_commands.len().min(visits).max(1);
        let batch = Arc::new(SearchBatch::new(workers));
        for worker in 0..workers {
            let visit_quota = visits / workers + usize::from(worker < visits % workers);
            let mut task = SearchTask {
                batch: Arc::clone(&batch),
                root_key: self.root_key,
                root_history: Arc::clone(&self.root_history),
                visits: visit_quota as u32,
                collision_budget: visit_quota as u32,
            };
            loop {
                if self.shared.stopping.load(Ordering::Acquire) {
                    task.batch.finish_gather();
                    return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
                }
                match self.shared.search_tx.send_timeout(task, RECEIVE_POLL) {
                    Ok(()) => break,
                    Err(SendTimeoutError::Timeout(returned)) => task = returned,
                    Err(SendTimeoutError::Disconnected(_)) => {
                        return Err(EnginError::PortIncomplete("stream search queue disconnected"));
                    }
                }
            }
        }
        Ok(batch)
    }

    /// 仅供队列/drain 回归逐轮提交；正式 owner 一律按 backend batch 提交整轮。
    #[cfg(test)]
    fn submit_playout(&self) -> Result<(), EnginError> {
        self.submit_batch(1).map(|_| ())
    }

    pub fn run_playouts(&self, count: u64) -> Result<Stats, EnginError> {
        self.run_with_limits(SearchLimits {
            max_playouts: Some(count),
            deadline: None,
        })
    }

    /// Runs logical batches until a cumulative root-visit budget, deadline, or explicit stop.
    pub fn run_with_limits(&self, limits: SearchLimits) -> Result<Stats, EnginError> {
        self.run_with_limits_reporting(limits, None, |_| {})
    }

    /// 与 `run_with_limits` 相同，但在不 drain 在途流水线的前提下定期归还一次 owner
    /// 控制权。UCI owner 用它判断是否需要输出 `info`；搜索层不解释输出语义。
    pub(crate) fn run_with_limits_reporting(
        &self,
        limits: SearchLimits,
        report_interval: Option<Duration>,
        mut report: impl FnMut(Stats),
    ) -> Result<Stats, EnginError> {
        let target = limits.max_playouts.unwrap_or(u64::MAX);
        let mut next_report = report_interval.and_then(|interval| Instant::now().checked_add(interval));
        let mut window: VecDeque<(Arc<SearchBatch>, u64)> = VecDeque::new();
        let mut reserved_visits = 0_u64;
        while !self.is_stopping()
            && !limits.is_exhausted(
                self.initial_visits.saturating_add(self.stats().completed_playouts),
                target,
            )
        {
            if next_report.is_some_and(|deadline| Instant::now() >= deadline) {
                report(self.stats());
                next_report = report_interval.and_then(|interval| Instant::now().checked_add(interval));
            }
            let root_state = self
                .shared
                .repository
                .get(self.root_key)
                .map(|root| root.expansion_state());
            if root_state == Some(ExpansionState::Terminal) || self.shared.root_path_terminal.load(Ordering::Acquire) {
                break;
            }
            if root_state == Some(ExpansionState::Evaluating) {
                self.wait_for_idle()?;
                continue;
            }
            if root_state != Some(ExpansionState::Expanded) {
                // root 展开前只需要一个真实 leaf。
                let batch = match self.submit_batch(1) {
                    Ok(batch) => batch,
                    Err(_error) if self.is_stopping() => break,
                    Err(error) => return Err(error),
                };
                batch.wait_until_finished(&self.shared.stopping);
                continue;
            }
            while window.len() < 2 {
                let completed = self.stats().completed_playouts;
                let remaining = target.saturating_sub(
                    self.initial_visits
                        .saturating_add(completed)
                        .saturating_add(reserved_visits),
                );
                if remaining == 0 {
                    break;
                }
                let visits = self.worker_pool.eval_batch_size.min(remaining as usize).max(1);
                let batch = match self.submit_batch(visits) {
                    Ok(batch) => batch,
                    Err(_error) if self.is_stopping() => break,
                    Err(error) => return Err(error),
                };
                // 先让这一 batch 的 Eval 完成缓存/编码判断，保证 NN 队列按 batch 顺序
                // 接收；随后可并行准备唯一的后继 batch。
                batch.wait_until_closed(&self.shared.stopping);
                reserved_visits += visits as u64;
                window.push_back((batch, visits as u64));
            }
            let Some((batch, visits)) = window.pop_front() else {
                break;
            };
            batch.wait_until_finished(&self.shared.stopping);
            reserved_visits = reserved_visits.saturating_sub(visits);
            if self.is_stopping() {
                break;
            }
        }
        // 请求停止是正常搜索结果。`wait_for_idle()` 已保证每个入队 event 都已完成或取消
        // reservation，因此调用方可安全快照部分 graph。
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

    /// 结束当前 job；常驻 worker 完成 drain 后回到等待，不在这里退出线程。
    pub fn stop_and_finish(&mut self) {
        if self.workers_idle {
            return;
        }
        self.request_stop();
        let _ = self.wait_for_idle();
        self.worker_pool.finish_job();
        self.workers_idle = true;
    }
}

impl Drop for Search {
    fn drop(&mut self) {
        self.stop_and_finish();
    }
}

fn search_worker(shared: Arc<Shared>, gather: Receiver<SearchTask>, backprop: Receiver<BackpropTask>) {
    loop {
        crossbeam_channel::select_biased! {
            recv(backprop) -> task => match task {
                Ok(first) => complete_backprop(&shared, std::iter::once(first).chain(backprop.try_iter())),
                Err(_) => break,
            },
            recv(gather) -> task => match task {
                Ok(task) if shared.stopping.load(Ordering::Acquire) => task.batch.finish_gather(),
                Ok(task) => process_search_task(&shared, task),
                Err(_) => break,
            },
            default(RECEIVE_POLL) => {
                if !shared.stopping.load(Ordering::Acquire) {
                    continue;
                }
                while let Ok(task) = backprop.try_recv() {
                    complete_backprop(&shared, std::iter::once(task));
                }
                while let Ok(task) = gather.try_recv() {
                    task.batch.finish_gather();
                }
                break;
            }
        }
    }
}

fn persistent_search_worker(commands: Receiver<SearchCommand>, job_done: Sender<()>) {
    while let Ok(command) = commands.recv() {
        match command {
            SearchCommand::Run(shared, gather, backprop) => {
                search_worker(shared, gather, backprop);
                let _ = job_done.send(());
            }
            SearchCommand::Shutdown => break,
        }
    }
}

fn process_search_task(shared: &Shared, task: SearchTask) {
    let mut collisions = Vec::new();
    let mut collision_visits = 0_u32;
    let mut pending = VecDeque::from([NodeEvent::at_root_with_visits(
        shared.generation,
        task.root_key,
        Arc::clone(&task.root_history),
        task.visits,
    )]);
    while let Some(event) = pending.pop_front() {
        if shared.stopping.load(Ordering::Acquire) {
            event.cancel();
            break;
        }
        match process_gather_event(shared, &task.batch, event) {
            GatherResult::Eval(event) => {
                task.batch.begin_eval();
                shared.start_playout(&task.batch);
                shared.send_eval(EvalTask {
                    event,
                    batch: Arc::clone(&task.batch),
                });
            }
            GatherResult::Completed => {}
            GatherResult::Collision(event) => {
                if collision_visits.saturating_add(event.logical_visits) <= task.collision_budget {
                    collision_visits += event.logical_visits;
                    collisions.push(event);
                } else {
                    // 预算耗尽不改变“已经撞到 Evaluating node”这一事实；统计仍应
                    // 反映真实 collision，只是立即归还 reservation。
                    shared.finish_collision(event);
                    break;
                }
            }
            GatherResult::Branch(events) => pending.extend(events),
            GatherResult::Stopped => break,
        }
    }
    // collision 预算用尽或 stop 后，尚未取出的分支没有真实 leaf；必须归还它们
    // 已在每层占用的 virtual visit，不能把它们留到下一 batch。
    for event in pending {
        event.cancel();
    }
    for event in collisions {
        shared.finish_collision(event);
    }
    task.batch.finish_gather();
}

enum GatherResult {
    Eval(NodeEvent),
    Completed,
    Collision(NodeEvent),
    Branch(Vec<NodeEvent>),
    Stopped,
}

#[derive(Clone, Copy)]
enum ChildTarget {
    Graph(NodeKey),
    Continuation(NodeKey),
}

fn selected_child(shared: &Shared, event: &mut NodeEvent, node: &Node, edge_index: usize) -> Option<ChildTarget> {
    let edge = &node.edges()[edge_index];
    if !event.node_key.is_continuation()
        && let Some(child) = event.repeated_child_key(edge.mv())
    {
        shared.repository.get_or_insert(child);
        return Some(ChildTarget::Continuation(child));
    }
    let child = if event.node_key.is_continuation() {
        event.variation.child_key_for_history(edge.mv())
    } else {
        event.variation.child_board_key(edge.mv())
    };
    match shared.repository.bind_child_or_cut_cycle(event.node_key, edge, child) {
        ChildLink::Bound => Some(ChildTarget::Graph(child)),
        ChildLink::TopologyPruned => None,
    }
}

fn branch_at_expanded_node(
    shared: &Shared,
    batch: &Arc<SearchBatch>,
    mut event: NodeEvent,
    node: &Node,
    depth: usize,
) -> GatherResult {
    let mut groups: Vec<(usize, ChildTarget, Vec<super::EdgeReservation>)> = Vec::new();
    let mut assigned = 0;
    while assigned < event.logical_visits {
        let Some(edge_index) = select_edge(
            &shared.repository,
            &node.edges(),
            node.completed_visits(),
            node.q(),
            depth,
            &shared.params,
            &shared.root_move_filter,
        ) else {
            if groups.is_empty() {
                if event.reservations.is_empty() {
                    shared.root_path_terminal.store(true, Ordering::Release);
                    shared.complete_root_terminal();
                    return GatherResult::Stopped;
                }
                shared.start_playout(batch);
                shared.send_backprop(BackpropTask {
                    event: BackpropEvent::evaluation(event),
                    batch: Arc::clone(batch),
                });
                return GatherResult::Completed;
            }
            break;
        };
        let Some(target) = selected_child(shared, &mut event, node, edge_index) else {
            continue;
        };
        let reservation = node.reserve_edge(edge_index).expect("selected stream edge");
        assigned += 1;
        if let Some((_, _, reservations)) = groups.iter_mut().find(|(candidate_edge, candidate, _)| {
            *candidate_edge == edge_index
                && match (*candidate, target) {
                    (ChildTarget::Graph(left), ChildTarget::Graph(right))
                    | (ChildTarget::Continuation(left), ChildTarget::Continuation(right)) => left == right,
                    _ => false,
                }
        }) {
            reservations.push(reservation);
        } else {
            groups.push((edge_index, target, vec![reservation]));
        }
    }
    if groups.is_empty() {
        event.cancel();
        return GatherResult::Stopped;
    }
    let weights: Vec<_> = groups
        .iter()
        .map(|(_, _, reservations)| reservations.len() as u32)
        .collect();
    let mut children = event.split(&weights).into_iter();
    let branches = groups
        .into_iter()
        .map(|(_, target, reservations)| {
            let event = children.next().expect("one event per PUCT branch");
            let reservation = super::EdgeReservation::merge(reservations);
            match target {
                ChildTarget::Graph(child) => event.descend(child, reservation),
                ChildTarget::Continuation(child) => event.descend_continuation(child, reservation),
            }
        })
        .collect();
    GatherResult::Branch(branches)
}

fn process_gather_event(shared: &Shared, batch: &Arc<SearchBatch>, mut event: NodeEvent) -> GatherResult {
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            event.cancel();
            return GatherResult::Stopped;
        }
        let node = shared.repository.get_or_insert(event.node_key);
        match node.expansion_state() {
            ExpansionState::Unexpanded => {
                if node.try_begin_evaluation() {
                    return GatherResult::Eval(event);
                }
            }
            ExpansionState::Evaluating => {
                return GatherResult::Collision(event);
            }
            ExpansionState::Terminal => {
                shared.start_playout(batch);
                shared.send_backprop(BackpropTask {
                    event: BackpropEvent::evaluation(event),
                    batch: Arc::clone(batch),
                });
                return GatherResult::Completed;
            }
            ExpansionState::Expanded => {
                // board-key node 可由历史中已展开的局面复用；重复、长将/长捉和 rule60
                // 仍只属于当前 variation，不能因 node 已展开而绕过裁决。
                let depth = event.variation.moves().len();
                // root history 在一个 go 内不可变；已展开 root 的每个 playout 直接借用
                // 它，避免仅为重复/rule60 检查而复制整条历史。离开 root 后才物化本
                // event 的私有 history。
                let terminal = if depth == 0 {
                    path_terminal_value(event.variation.root_history().as_ref(), 0)
                } else {
                    path_terminal_value(event.variation.history(), depth)
                };
                if let Some((wl, draw, plies_left)) = terminal {
                    let value = ValueDelta::with_plies_left(wl, draw, plies_left);
                    if event.reservations.is_empty() {
                        shared.root_path_terminal.store(true, Ordering::Release);
                        shared.complete_root_terminal();
                        return GatherResult::Stopped;
                    } else {
                        shared.start_playout(batch);
                        shared.send_backprop(BackpropTask {
                            event: BackpropEvent::local_leaf(event.discard_leaf_node(), value),
                            batch: Arc::clone(batch),
                        });
                    }
                    return GatherResult::Completed;
                }
                // MCGS 的共享 child 可能刚刚由另一条 variation 更新。回传只重算实际
                // 路径上的 node；因此每次再次访问本 node 前都要按最新 child Q 重算。
                // 参考 KataGo `docs/GraphSearch.md` “Stale Q Values”。
                shared.repository.recompute_graph_node(event.node_key);
                return branch_at_expanded_node(shared, batch, event, node.as_ref(), depth);
            }
        }
    }
}

/// NN 只交回整批 [`EncodedBatch`] + 行号；切行与 softmax 在 Eval。
type NnReply = Result<(Arc<EncodedBatch>, usize), EnginError>;

/// 交给 NN 线程的一个已编码局面（稀疏 InputPlanes；ORT 前再 expand）。
struct NnRequest {
    planes: InputPlanes,
    reply: Sender<NnReply>,
    batch: Arc<SearchBatch>,
    #[cfg(feature = "benchmark")]
    queued_at: Option<Instant>,
}

impl NnRequest {
    #[cfg(feature = "benchmark")]
    fn mark_queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    #[cfg(feature = "benchmark")]
    fn take_queue_wait(&mut self) -> Option<Duration> {
        self.queued_at.take().map(|queued_at| queued_at.elapsed())
    }
}

/// Eval 正在等待此 node 的 NN（LC3：NN 完成后 → Backprop）。
struct WaitingNn {
    event: NodeEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
    cache_key: EvalCacheKey,
    reply: Receiver<NnReply>,
    batch: Arc<SearchBatch>,
}

/// LC3 EvalWorker：terminal | cache → Backprop；否则合法着 + 编码 → NN queue；
/// NN 回复后 → softmax/edges → Backprop。它不会因一次 GPU 调用阻塞整个 worker，
/// 而是在新 NodeEvent 之间轮询已完成结果。
fn eval_worker(shared: Arc<Shared>, receiver: Receiver<EvalTask>, nn_tx: Sender<NnRequest>) {
    let mut waiting: Vec<WaitingNn> = Vec::new();
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            for item in waiting.drain(..) {
                cancel_waiting_item(&shared, item);
            }
            while let Ok(task) = receiver.try_recv() {
                task.batch.finish_eval(false);
                shared.cancel_claimed_evaluation(task.event, task.batch);
            }
            break;
        }

        poll_nn_completions(&shared, &mut waiting);

        match receiver.recv_timeout(RECEIVE_POLL) {
            #[cfg(feature = "benchmark")]
            Ok(mut task) => {
                #[cfg(feature = "benchmark")]
                if let Some(wait) = task.event.take_queue_wait() {
                    shared.eval_queue.record(wait);
                }
                if let Err(error) = handle_eval_event(&shared, &nn_tx, &mut waiting, task) {
                    shared.fail(error);
                }
            }
            #[cfg(not(feature = "benchmark"))]
            Ok(task) => {
                if let Err(error) = handle_eval_event(&shared, &nn_tx, &mut waiting, task) {
                    shared.fail(error);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if waiting.is_empty() {
                    continue;
                }
                // 没有新叶子：短暂等待至少一个 NN 回复。
                wait_one_nn_completion(&shared, &mut waiting);
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn persistent_eval_worker(commands: Receiver<EvalCommand>, job_done: Sender<()>) {
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

fn poll_nn_completions(shared: &Shared, waiting: &mut Vec<WaitingNn>) {
    let mut i = 0;
    while i < waiting.len() {
        match waiting[i].reply.try_recv() {
            Ok(Ok((batch, row))) => {
                let item = waiting.swap_remove(i);
                if let Err(error) = complete_nn_item(shared, item, batch, row) {
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
        Ok(Ok((batch, row))) => {
            let item = waiting.remove(0);
            if let Err(error) = complete_nn_item(shared, item, batch, row) {
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
    task: EvalTask,
) -> Result<(), EnginError> {
    let EvalTask { mut event, batch } = task;
    if shared.stopping.load(Ordering::Acquire) {
        batch.finish_eval(false);
        shared.cancel_claimed_evaluation(event, batch);
        return Ok(());
    }
    let node = shared.repository.get_or_insert(event.node_key);
    let depth = event.variation.moves().len();
    let history = event.variation.history();
    match classify_extension(history, depth) {
        ExtensionKind::SharedTerminal { wl, draw, plies_left } => {
            node.set_terminal_value_weighted(wl, draw, plies_left, event.logical_visits);
            batch.finish_eval(false);
            shared.send_backprop(BackpropTask {
                event: BackpropEvent::evaluation(event),
                batch,
            });
            Ok(())
        }
        ExtensionKind::PathTerminal { wl, draw, plies_left } => {
            let value = ValueDelta::with_plies_left(wl, draw, plies_left);
            node.abort_evaluation();
            if event.reservations.is_empty() {
                shared.root_path_terminal.store(true, Ordering::Release);
                batch.finish_eval(false);
                batch.finish_playout();
                shared.add_completed_visits(1);
                shared.finish();
            } else {
                batch.finish_eval(false);
                shared.send_backprop(BackpropTask {
                    event: BackpropEvent::local_leaf(event.discard_leaf_node(), value),
                    batch,
                });
            }
            Ok(())
        }
        ExtensionKind::Evaluate => {
            let legal_moves = history.last().board().generate_legal_moves();
            let cache_key = EvalCacheKey::new(history.last(), legal_moves.len());
            if let Some(eval) = shared.backend.cached_evaluation(cache_key) {
                shared.cache_hits.fetch_add(1, Ordering::AcqRel);
                batch.finish_eval(false);
                return publish_eval(shared, event, node, legal_moves, eval, batch);
            }
            let planes = encode_position_input_planes(history, FillEmptyHistory::FenOnly);
            let (reply_tx, reply_rx) = bounded(1);
            if let Err(error) = send_nn_request(
                shared,
                nn_tx,
                NnRequest {
                    planes,
                    reply: reply_tx,
                    batch: Arc::clone(&batch),
                    #[cfg(feature = "benchmark")]
                    queued_at: None,
                },
            ) {
                batch.finish_eval(false);
                cancel_evaluation(shared, event, node, batch);
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
                cache_key,
                reply: reply_rx,
                batch: Arc::clone(&batch),
            });
            batch.finish_eval(true);
            Ok(())
        }
    }
}

fn send_nn_request(shared: &Shared, nn_tx: &Sender<NnRequest>, mut request: NnRequest) -> Result<(), EnginError> {
    #[cfg(feature = "benchmark")]
    request.mark_queued();
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

fn complete_nn_item(shared: &Shared, item: WaitingNn, batch: Arc<EncodedBatch>, row: usize) -> Result<(), EnginError> {
    let eval = match eval_result_from_encoded_row(&batch, row, &item.legal_moves) {
        Ok(eval) => eval,
        Err(error) => {
            cancel_waiting_item(shared, item);
            return Err(error);
        }
    };
    shared.backend.store_evaluation(item.cache_key, Arc::clone(&eval));
    publish_eval(shared, item.event, item.node, item.legal_moves, eval, item.batch)
}

/// 在 backend 完成结果写入 graph 前校验它。
///
/// 参考：ARCHITECTURE 固定的 `WDL + moves-left` backend 契约。worker 失败必须通过
/// LC3 取消路径返回；持有 owned event 后不得 panic。
fn publish_eval(
    shared: &Shared,
    event: NodeEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
    eval: Arc<EvalResult>,
    batch: Arc<SearchBatch>,
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
        cancel_evaluation(shared, event, node, batch);
        return Err(EnginError::Onnx("stream backend evaluation is invalid".into()));
    }
    node.set_graph_value(
        ValueDelta::with_plies_left(network_wl_to_node(eval.wl), eval.d, eval.plies_left)
            .repeated(event.logical_visits),
    );
    // 先发布共享基值，再把 node 变为 Expanded；否则并发 Gather 可能在两者之间
    // 完成一次回传，而那次幂等重算会看不到基值。
    node.publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()).collect());
    shared.send_backprop(BackpropTask {
        event: BackpropEvent::evaluation(event),
        batch,
    });
    Ok(())
}

fn cancel_waiting_item(shared: &Shared, item: WaitingNn) {
    cancel_evaluation(shared, item.event, item.node, item.batch);
}

/// 释放已 claim 但不会发布结果的 evaluation event。
///
/// 参考：LC3 Overview 的 EvalWorker 所有权模型：每个 owned event 必须经 backpropagation
/// 完成，或显式取消。
fn cancel_evaluation(shared: &Shared, event: NodeEvent, node: Arc<Node>, batch: Arc<SearchBatch>) {
    event.cancel();
    node.abort_evaluation();
    batch.finish_playout();
    shared.finish();
}

/// NN：取队列 → 稀疏合批推理 → 整批交回 → 继续取。
/// expand/pad 在 ONNX 输入 scratch 内完成；不负责局面编码、合法着过滤或 softmax。
fn nn_worker(shared: Arc<Shared>, receiver: Receiver<NnRequest>, batch_size: usize) {
    let mut samples = Vec::new();
    let mut logits = Vec::new();
    let mut wdl = Vec::new();
    let mut moves_left = Vec::new();
    loop {
        #[cfg(feature = "benchmark")]
        let mut first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        #[cfg(not(feature = "benchmark"))]
        let first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        #[cfg(feature = "benchmark")]
        record_nn_queue_wait(&shared, &mut first);
        let search_batch = Arc::clone(&first.batch);
        let expected = search_batch.wait_for_nn_requests(&shared.stopping);
        let mut requests = vec![first];
        while requests.len() < expected {
            match receiver.recv_timeout(RECEIVE_POLL) {
                #[cfg(feature = "benchmark")]
                Ok(mut request) => {
                    #[cfg(feature = "benchmark")]
                    record_nn_queue_wait(&shared, &mut request);
                    requests.push(request);
                }
                #[cfg(not(feature = "benchmark"))]
                Ok(request) => requests.push(request),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }
        debug_assert!(
            requests.len() <= batch_size,
            "one search batch must not exceed one NN batch"
        );
        if shared.stopping.load(Ordering::Acquire) {
            reject_nn_requests(requests, EnginError::PortIncomplete("stream nn stopping"));
            continue;
        }
        let batch = requests.len();
        samples.clear();
        samples.reserve(batch);
        for request in &requests {
            samples.push(request.planes);
        }
        let infer_result = shared
            .backend
            .infer_input_planes_into(&samples, &mut logits, &mut wdl, &mut moves_left);
        match infer_result {
            Ok(()) => {
                let output = EncodedBatch::take_from(&mut logits, &mut wdl, &mut moves_left);
                if let Err(error) = output.ensure_batch_len(batch) {
                    reject_nn_requests(requests, error);
                    continue;
                }
                shared.network_batches.fetch_add(1, Ordering::AcqRel);
                shared.network_evaluations.fetch_add(batch as u64, Ordering::AcqRel);
                shared.network_batch_size_max.fetch_max(batch as u64, Ordering::AcqRel);
                EncodedBatch::reserve_scratch(&mut logits, &mut wdl, &mut moves_left, batch);
                let output = Arc::new(output);
                for (row, request) in requests.into_iter().enumerate() {
                    let _ = request.reply.send(Ok((Arc::clone(&output), row)));
                }
            }
            Err(error) => {
                reject_nn_requests(requests, error);
            }
        }
    }
}

#[cfg(feature = "benchmark")]
fn record_nn_queue_wait(shared: &Shared, request: &mut NnRequest) {
    if let Some(wait) = request.take_queue_wait() {
        shared.nn_queue.record(wait);
    }
}

fn reject_nn_requests(requests: Vec<NnRequest>, error: EnginError) {
    for request in requests {
        let _ = request.reply.send(Err(error.clone()));
    }
}

fn persistent_nn_worker(commands: Receiver<NnCommand>, job_done: Sender<()>, batch_size: usize) {
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

fn complete_backprop(shared: &Shared, tasks: impl IntoIterator<Item = BackpropTask>) {
    let tasks: Vec<_> = tasks.into_iter().collect();
    if tasks.is_empty() {
        return;
    }
    if shared.stopping.load(Ordering::Acquire) {
        for task in tasks {
            task.event.cancel();
            task.batch.finish_playout();
            shared.finish();
        }
        return;
    }
    let mut events = Vec::with_capacity(tasks.len());
    let mut batches = Vec::with_capacity(tasks.len());
    for task in tasks {
        #[cfg(feature = "benchmark")]
        let mut task = task;
        #[cfg(feature = "benchmark")]
        if let Some(wait) = task.event.take_queue_wait() {
            shared.backprop_queue.record(wait);
        }
        batches.push(task.batch);
        events.push(task.event);
    }
    let result = BackpropEvent::complete_batch(events, &shared.repository);
    shared
        .completed_depth
        .fetch_add(result.completed_depth, Ordering::AcqRel);
    shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
    shared.add_completed_visits(result.completed_playouts);
    for batch in batches {
        batch.finish_playout();
        shared.finish();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    use crossbeam_channel::bounded;
    use parking_lot::{Condvar, Mutex};
    use xiangqi_core::{ChessBoard, GameState, Move, Position, PositionHistory, STARTPOS_FEN, Square};

    use super::{EvalTask, NodeEvent, Search, SearchBatch, SearchConfig, SearchLimits, Shared};
    use crate::EnginError;
    use crate::neural::backend::{Backend, BackendAttributes, UniformBackend};
    use crate::search::{ExpansionState, NodeRepository, SearchGraph, SearchParams, ValueDelta, best_move, root_stats};

    struct FailingInferenceBackend;

    struct InvalidValueBackend;

    impl Backend for FailingInferenceBackend {
        fn attributes(&self) -> BackendAttributes {
            BackendAttributes::default()
        }

        fn infer_input_planes_into(
            &self,
            _samples: &[crate::neural::InputPlanes],
            _logits: &mut Vec<f32>,
            _wdl: &mut Vec<f32>,
            _moves_left: &mut Vec<f32>,
        ) -> Result<(), EnginError> {
            Err(EnginError::Onnx("test computation failure".to_owned()))
        }
    }

    impl Backend for InvalidValueBackend {
        fn attributes(&self) -> BackendAttributes {
            BackendAttributes::default()
        }

        fn infer_input_planes_into(
            &self,
            samples: &[crate::neural::InputPlanes],
            logits: &mut Vec<f32>,
            wdl: &mut Vec<f32>,
            moves_left: &mut Vec<f32>,
        ) -> Result<(), EnginError> {
            logits.clear();
            logits.resize(samples.len() * crate::neural::POLICY_SIZE, 0.0);
            wdl.clear();
            wdl.extend((0..samples.len()).flat_map(|_| [f32::NAN, 0.0, 0.0]));
            moves_left.clear();
            moves_left.resize(samples.len(), 0.0);
            Ok(())
        }
    }

    fn startpos_history() -> Arc<PositionHistory> {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        Arc::new(PositionHistory::from_positions(state.positions()))
    }

    #[test]
    fn search_completes_batched_playouts_and_returns_workers() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            21,
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 8,
                eval_batch_size: 4,
                gather_workers: 2,
                ..SearchConfig::default()
            },
        );
        let stats = pipeline.run_playouts(32).expect("playouts");
        assert_eq!(stats.completed_playouts, 32);
        assert!(stats.network_batches > 0);
        // 很快的 stop 可能在 Gather 展开 root 前竞争获胜；如果已展开，仍必须释放每个
        // reservation。
        if let Some(root) = pipeline.repository().get(pipeline.root_key()) {
            for edge in root.edges().iter() {
                assert_eq!(edge.visits(), edge.completed_visits());
            }
        }
        pipeline.stop_and_finish();
    }

    #[cfg(feature = "benchmark")]
    #[test]
    fn benchmark_telemetry_reports_pipeline_handoffs() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            22,
            startpos_history(),
            SearchConfig::default(),
        );

        let stats = pipeline.run_playouts(16).expect("benchmark playouts");

        assert!(stats.submitted_playouts >= stats.completed_playouts);
        assert!(stats.peak_in_flight > 0);
        assert!(stats.gather_queue.samples >= stats.completed_playouts);
        assert!(stats.eval_queue.samples > 0);
        assert!(stats.backprop_queue.samples > 0);
        pipeline.stop_and_finish();
    }

    #[test]
    fn stop_and_finish_drains_in_flight_reservations() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            22,
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 16,
                eval_batch_size: 8,
                gather_workers: 4,
                ..SearchConfig::default()
            },
        );
        pipeline.run_playouts(1).expect("expand root");
        for _ in 0..16 {
            pipeline.submit_playout().expect("queued root event");
        }
        pipeline.stop_and_finish();

        // Stop 可能在 Gather 展开 root 前到达；如果已展开，仍必须释放每个 reservation。
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
            23,
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 1,
                eval_batch_size: 1,
                gather_workers: 1,
                ..SearchConfig::default()
            },
        );
        pipeline.run_playouts(1).expect("expand root");
        for _ in 0..32 {
            pipeline.submit_playout().expect("bounded queue waits for gather");
        }
        pipeline.wait_for_idle().expect("all submitted work drains");
        pipeline.stop_and_finish();
    }

    #[test]
    fn requested_stop_returns_partial_stats_and_releases_reservations() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            24,
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 8,
                eval_batch_size: 4,
                gather_workers: 2,
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

        // Stop 可能在 Gather 创建 root 前竞争获胜；如果已创建，仍必须释放每个 reservation。
        if let Some(root) = pipeline.repository().get(pipeline.root_key()) {
            for edge in root.edges().iter() {
                assert_eq!(edge.visits(), edge.completed_visits());
            }
        }
        pipeline.stop_and_finish();
    }

    #[test]
    fn expired_deadline_submits_no_new_playout() {
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            25,
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
        pipeline.stop_and_finish();
    }

    #[test]
    fn terminal_root_finishes_after_its_initial_evaluation() {
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmated root");
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            26,
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
        pipeline.stop_and_finish();
    }

    #[test]
    fn reused_expanded_root_still_applies_perpetual_check_rule() {
        // 同一 board 曾作为正常局面展开；随后以带完整历史的相同 board 作为 root
        // 重新进入时，不能因 node 已 Expanded 而跳过 path-local 长将裁决。
        let (board, _) = ChessBoard::from_fen("3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = PositionHistory::default();
        history.reset(board, 2, 30);
        for mv in ["d9e9", "d2e2", "e9d9", "e2d2", "d9e9", "d2e2", "e9d9", "e2d2"] {
            history.append(history.last().board().parse_move(mv).expect(mv));
        }
        assert!(history.last().repetitions() >= 2);

        let history = Arc::new(history);
        let tree = super::SearchGraph::new(Arc::clone(&history));
        let root = tree.repository().get_or_insert(tree.root_key());
        assert!(root.try_begin_evaluation());
        let legal = history.last().board().generate_legal_moves();
        root.publish_edges(legal.iter().map(|&mv| (mv, 1.0 / legal.len() as f32)).collect());
        root.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));

        let mut search =
            Search::new_with_graph(Arc::new(UniformBackend::default()), 27, &tree, SearchConfig::default());
        let stats = search.run_playouts(1).expect("path terminal root");

        assert_eq!(stats.completed_playouts, 1);
        assert!(root.edges().iter().all(|edge| edge.completed_visits() == 0));
        assert!(search.root_is_path_terminal());
        search.stop_and_finish();
    }

    #[test]
    fn gather_prunes_a_global_graph_cycle_and_tries_the_next_edge() {
        // 真实 Gather 路径：root -> child 若接上 child -> root 的既有图边会闭环。
        // 该 edge 不记 N/Q，并在本 node 继续选择下一条可用 edge。
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let tree = SearchGraph::new(Arc::clone(&history));
        let root_key = tree.root_key();
        let mv = history.last().board().parse_move("b2b3").expect("legal root move");
        let fallback = history.last().board().parse_move("g3g4").expect("legal fallback move");
        let child_key = NodeEvent::root(41, Arc::clone(&history)).variation.child_board_key(mv);

        let root = tree.repository().get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(mv, 0.9), (fallback, 0.1)]);

        let child = tree.repository().get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.set_graph_value(ValueDelta::one(0.4, 0.0));
        child.publish_edges(vec![(
            Move::new(Square::parse("b9").unwrap(), Square::parse("b8").unwrap()),
            1.0,
        )]);
        child.edges()[0].bind_child_key(root_key);

        let mut search =
            Search::new_with_graph(Arc::new(UniformBackend::default()), 41, &tree, SearchConfig::default());
        let stats = search.run_playouts(1).expect("cycle-pruned playout");
        let root_edge = &root.edges()[0];
        assert_eq!(stats.completed_playouts, 1);
        assert_eq!(root_edge.completed_visits(), 0);
        assert_eq!(root_edge.child_key(), None);
        assert!(root_edge.topology_pruned());
        assert_eq!(root.edges()[1].completed_visits(), 1);
        search.stop_and_finish();
    }

    #[test]
    fn all_topology_pruned_children_finish_at_the_shared_node_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let tree = SearchGraph::new(Arc::clone(&history));
        let root_key = tree.root_key();
        let first = history.last().board().parse_move("b2b3").expect("legal first move");
        let child_key = NodeEvent::root(43, Arc::clone(&history))
            .variation
            .child_board_key(first);
        let child_position = Position::after(history.last(), first);
        let reply = child_position.board().parse_move("b9b8").expect("legal reply");
        let grandchild = Position::after(&child_position, reply);
        let grandchild_key = crate::search::NodeKey::board(grandchild.board().hash());

        let root = tree.repository().get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(first, 1.0)]);
        root.edges()[0].bind_child_key(child_key);

        let child = tree.repository().get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.set_graph_value(ValueDelta::one(0.4, 0.0));
        child.publish_edges(vec![(reply, 1.0)]);

        let grandchild_node = tree.repository().get_or_insert(grandchild_key);
        assert!(grandchild_node.try_begin_evaluation());
        grandchild_node.set_graph_value(ValueDelta::one(0.2, 0.0));
        grandchild_node.publish_edges(vec![(
            Move::new(Square::parse("c0").unwrap(), Square::parse("c1").unwrap()),
            1.0,
        )]);
        grandchild_node.edges()[0].bind_child_key(child_key);

        let mut search =
            Search::new_with_graph(Arc::new(UniformBackend::default()), 43, &tree, SearchConfig::default());
        let stats = search.run_playouts(1).expect("topology boundary playout");

        assert_eq!(stats.completed_playouts, 1);
        assert!(child.edges()[0].topology_pruned());
        assert_eq!(child.edges()[0].completed_visits(), 0);
        assert_eq!(root.edges()[0].completed_visits(), 1);
        assert!((root.q() + 0.2).abs() < f32::EPSILON);
        search.stop_and_finish();
    }

    #[test]
    fn gather_enters_continuation_tree_on_the_first_real_repetition() {
        let (board, _) = ChessBoard::from_fen("3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = PositionHistory::default();
        history.reset(board, 2, 30);
        // 前三步尚未重复；第四步才回到初始局面。这样 root 仍是 shared
        // board node，才能验证首次重复不把它的入边绑定到 shared graph。
        for text in ["d9e9", "d2e2", "e9d9"] {
            let mv = history.last().board().parse_move(text).expect(text);
            history.append(mv);
        }
        let history = Arc::new(history);
        let tree = SearchGraph::new(Arc::clone(&history));
        let root = tree.repository().get_or_insert(tree.root_key());
        let mv = history.last().board().parse_move("e2d2").expect("repeat move");
        let mut event = NodeEvent::root(42, Arc::clone(&history));
        let continuation = event.repeated_child_key(mv).expect("first repetition");

        assert!(root.try_begin_evaluation());
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(mv, 1.0)]);
        let mut search =
            Search::new_with_graph(Arc::new(UniformBackend::default()), 42, &tree, SearchConfig::default());
        let stats = search.run_playouts(1).expect("continuation playout");

        assert_eq!(stats.completed_playouts, 1);
        assert_eq!(root.edges()[0].completed_visits(), 1);
        assert!(root.edges()[0].child_key().is_none());
        assert_eq!(
            tree.repository()
                .get(continuation)
                .expect("continuation root")
                .expansion_state(),
            ExpansionState::Expanded
        );
        search.stop_and_finish();
    }

    #[test]
    fn zeroing_edge_from_a_tree_node_rejoins_the_shared_graph() {
        let (board, _) = ChessBoard::from_fen("3k5/9/9/r3R4/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = PositionHistory::default();
        history.reset(board, 2, 30);
        for text in ["d9e9", "d2e2", "e9d9", "e2d2", "d9e9", "d2e2"] {
            let mv = history.last().board().parse_move(text).expect(text);
            history.append(mv);
        }
        assert!(history.did_repeat_since_last_zeroing_move());
        let history = Arc::new(history);
        let mut graph = SearchGraph::new(Arc::clone(&history));
        assert!(graph.root_key().is_continuation());
        let root = graph.repository().get_or_insert(graph.root_key());
        let capture = history
            .last()
            .board()
            .generate_legal_moves()
            .into_iter()
            .find(|mv| Position::after(history.last(), *mv).rule60_ply() == 0)
            .expect("legal capture");
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(capture, 1.0)]);

        let mut search =
            Search::new_with_graph(Arc::new(UniformBackend::default()), 43, &graph, SearchConfig::default());
        assert_eq!(search.run_playouts(1).expect("one playout").completed_playouts, 1);

        let child = root.edges()[0].child_key().expect("zeroing edge binds graph child");
        assert!(matches!(child, crate::search::NodeKey::GraphNode { .. }));
        assert!(graph.repository().get(child).is_some());
        search.stop_and_finish();

        let mut target = history.as_ref().clone();
        target.append(capture);
        assert!(
            graph
                .reset_to_history(Arc::new(target))
                .expect("played zeroing move reuses graph child")
                .is_none()
        );
        assert_eq!(graph.root_key(), child);
        assert!(graph.repository().get(child).expect("reused root").completed_visits() > 0);
    }

    #[test]
    fn path_terminal_root_does_not_mark_the_shared_node() {
        let state =
            GameState::from_fen_moves("4k4/9/9/9/9/9/9/9/R8/4K4 w - - 120 1", &[] as &[&str]).expect("rule60 root");
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            26,
            Arc::new(PositionHistory::from_positions(state.positions())),
            SearchConfig::default(),
        );

        let stats = pipeline.run_playouts(8).expect("path terminal search");

        assert_eq!(stats.completed_playouts, 1);
        assert_eq!(
            pipeline
                .repository()
                .get(pipeline.root_key())
                .expect("root exists")
                .expansion_state(),
            ExpansionState::Unexpanded
        );
        assert!(pipeline.root_is_path_terminal());
        pipeline.stop_and_finish();
    }

    #[test]
    fn fixed_playout_root_contract() {
        let count = 64;
        let mut workers = Search::new(
            Arc::new(UniformBackend::default()),
            27,
            startpos_history(),
            SearchConfig {
                root_move_filter: Vec::new(),
                queue_capacity: 8,
                eval_batch_size: 4,
                gather_workers: 2,
                ..SearchConfig::default()
            },
        );
        let worker_stats = workers.run_playouts(count).expect("worker playouts");
        let worker_root = root_stats(workers.repository(), workers.root_key()).expect("worker root");

        assert_eq!(worker_stats.completed_playouts, count);
        assert!(worker_stats.average_depth >= 1);
        assert!(worker_stats.max_depth >= worker_stats.average_depth);
        assert_eq!(worker_root.completed_visits, count as u32);
        assert!(
            worker_root
                .edges
                .iter()
                .all(|edge| edge.started_visits == edge.completed_visits)
        );
        assert!(!worker_root.edges.is_empty());

        workers.stop_and_finish();
    }

    #[test]
    fn nn_inference_failure_drains_claimed_events() {
        let mut pipeline = Search::new(
            Arc::new(FailingInferenceBackend),
            28,
            startpos_history(),
            SearchConfig::default(),
        );
        let error = pipeline.run_playouts(1).expect_err("nn inference must fail");
        assert_eq!(error, EnginError::Onnx("test computation failure".to_owned()));
        pipeline.wait_for_idle().expect_err("pipeline retains the worker error");
        assert_eq!(pipeline.stats().completed_playouts, 0);
        pipeline.stop_and_finish();
    }

    #[test]
    fn invalid_nn_values_fail_without_leaking_the_claimed_event() {
        let mut pipeline = Search::new(
            Arc::new(InvalidValueBackend),
            29,
            startpos_history(),
            SearchConfig::default(),
        );

        let error = pipeline.run_playouts(1).expect_err("invalid network values must fail");
        match error {
            EnginError::Onnx(message) => assert!(
                message.starts_with("stream nn values are invalid"),
                "unexpected onnx error: {message}"
            ),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(pipeline.stats().completed_playouts, 0);
        pipeline.stop_and_finish();
    }

    #[test]
    fn failed_eval_enqueue_releases_the_claimed_node() {
        let history = startpos_history();
        let root_key = crate::search::NodeKey::board(history.last().board().hash());
        let repository = Arc::new(NodeRepository::default());
        let root = repository.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        let (search_tx, _search_rx) = bounded(1);
        let (eval_tx, eval_rx) = bounded(1);
        let (backprop_tx, _backprop_rx) = bounded(1);
        drop(eval_rx);
        let shared = Shared {
            backend: Arc::new(UniformBackend::default()),
            repository,
            generation: 30,
            params: SearchParams::default(),
            root_move_filter: Vec::new(),
            root_path_terminal: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(1),
            #[cfg(feature = "benchmark")]
            submitted: AtomicU64::new(0),
            #[cfg(feature = "benchmark")]
            peak_in_flight: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            #[cfg(feature = "benchmark")]
            collisions: AtomicU64::new(0),
            network_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            network_batch_size_max: AtomicU64::new(0),
            #[cfg(feature = "benchmark")]
            collisions_by_depth: Mutex::new(Vec::new()),
            #[cfg(feature = "benchmark")]
            gather_queue: super::QueueMetrics::default(),
            #[cfg(feature = "benchmark")]
            eval_queue: super::QueueMetrics::default(),
            #[cfg(feature = "benchmark")]
            nn_queue: super::QueueMetrics::default(),
            #[cfg(feature = "benchmark")]
            backprop_queue: super::QueueMetrics::default(),
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            search_tx,
            eval_tx,
            backprop_tx,
        };

        let batch = Arc::new(SearchBatch::new(0));
        batch.begin_eval();
        batch.begin_playout();
        shared.send_eval(EvalTask {
            event: NodeEvent::root(30, history),
            batch,
        });

        assert_eq!(root.expansion_state(), ExpansionState::Unexpanded);
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 0);
    }

    #[test]
    fn completed_search_can_reuse_a_graph_at_its_played_child() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = super::SearchGraph::new(history);
        let backend = Arc::new(UniformBackend::default()) as Arc<dyn Backend>;

        let mut first = Search::new_with_graph(Arc::clone(&backend), 29, &tree, SearchConfig::default());
        first.run_playouts(16).expect("first search");
        let old_root = tree.root_key();
        let played = best_move(first.repository(), first.root_key(), false).expect("best move");
        first.stop_and_finish();

        tree.advance(played).expect("advance retained tree");
        assert!(tree.repository().get(old_root).is_some());

        let mut second = Search::new_with_graph(backend, 30, &tree, SearchConfig::default());
        second.run_playouts(8).expect("reused search");
        let root = second.repository().get(second.root_key()).expect("reused root");
        assert!(root.edges().iter().all(|edge| edge.visits() == edge.completed_visits()));
        second.stop_and_finish();
    }
}
