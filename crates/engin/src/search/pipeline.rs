//! Stream 搜索主线：Gather / Eval / NN / Backprop worker。
//!
//! 参考：LC3 Overview 的 "Workers"：
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! Eval 负责终局、缓存、合法着和编码；NN 线程只对队列中的 tensor 执行 ONNX。
//! worker 之间只传递 owned event。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, SendTimeoutError, Sender, TryRecvError, TrySendError, bounded};
use parking_lot::{Condvar, Mutex};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;
use crate::neural::backend::{Backend, EvalPosition, EvalResult};
use crate::neural::onnx::softmax_legal_policy;
use crate::neural::{BOARD_COLS, BOARD_ROWS, FillEmptyHistory, INPUT_PLANES, POLICY_SIZE, encode_position_for_nn};

use super::extension::{ExtensionKind, classify_extension, path_terminal_value};
use super::graph::ChildLink;
use super::{
    BackpropEvent, ExpansionState, Node, NodeEvent, NodeKey, NodeRepository, SearchGeneration, SearchGraph,
    SearchParams, ValueDelta, Variation, network_wl_to_node, select_edge_from_node,
};

const RECEIVE_POLL: Duration = Duration::from_millis(10);

/// 当前 root 的累计 visit 搜索预算。
///
/// `go nodes N` 包含 tree reuse 前已有的 root N。参考 px0 `Search::Search` 的
/// `initial_visits_` 和 `VisitsStopper`（`classic/search.cc:149,919`，
/// `classic/stoppers/stoppers.cc:59-69`）；时钟仍只约束本次 job。
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
    /// 本次搜索观察到的最大 `BackendComputation` batch。
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
    /// Gather/Eval/Backprop/NN 队列深度。`0` 表示 `max(4096, 64 * resolved_batch)`。
    pub queue_capacity: usize,
    /// 已有多个编码局面时的 NN GPU 合批大小。`0` 表示 backend 的
    /// `recommended_batch_size`；Eval 不会等待凑满 batch。
    pub eval_batch_size: usize,
    pub params: SearchParams,
    pub gather_workers: usize,
    /// Eval worker 数。它负责准备、缓存、合法着；NN inference 是独立线程。
    pub eval_workers: usize,
    pub backprop_workers: usize,
    /// UCI `go searchmoves` 对应 px0 `root_move_filter_`（空表示不限制）。
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
            backprop_workers: 1,
            root_move_filter: Vec::new(),
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
            gather_workers: self.gather_workers,
            eval_workers: self.eval_workers,
            backprop_workers: self.backprop_workers,
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
    backprop_workers: usize,
}

enum GatherCommand {
    Run(Arc<Shared>, Receiver<NodeEvent>),
    Shutdown,
}

enum EvalCommand {
    Run(Arc<Shared>, Receiver<NodeEvent>, Sender<NnRequest>),
    Shutdown,
}

enum NnCommand {
    Run(Arc<Shared>, Receiver<NnRequest>),
    Shutdown,
}

enum BackpropCommand {
    Run(Arc<Shared>, Receiver<BackpropEvent>),
    Shutdown,
}

/// Engine 持有的固定 worker 拓扑。
///
/// 每个 job 独占树视图、队列与 generation；线程池只跨 job 保留线程。
/// 参考：LC3 Overview 的 "Workers" / "Search"。
pub(crate) struct WorkerPool {
    gather_commands: Vec<Sender<GatherCommand>>,
    eval_commands: Vec<Sender<EvalCommand>>,
    nn_commands: Sender<NnCommand>,
    backprop_commands: Vec<Sender<BackpropCommand>>,
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
            && self.gather_commands.len() == config.gather_workers
            && self.eval_commands.len() == config.eval_workers
            && self.backprop_commands.len() == config.backprop_workers
    }

    fn from_resolved(config: &ResolvedSearchConfig) -> Self {
        let (job_done_tx, job_done) = crossbeam_channel::unbounded();
        let eval_batch_size = config.eval_batch_size;
        let mut gather_commands = Vec::with_capacity(config.gather_workers);
        let mut eval_commands = Vec::with_capacity(config.eval_workers);
        let mut backprop_commands = Vec::with_capacity(config.backprop_workers);
        let mut threads = Vec::with_capacity(config.gather_workers + config.eval_workers + config.backprop_workers + 1);
        for _ in 0..config.gather_workers {
            let (tx, rx) = crossbeam_channel::unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || persistent_gather_worker(rx, job_done)));
            gather_commands.push(tx);
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
        for _ in 0..config.backprop_workers {
            let (tx, rx) = crossbeam_channel::unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || persistent_backprop_worker(rx, job_done)));
            backprop_commands.push(tx);
        }
        Self {
            gather_commands,
            eval_commands,
            nn_commands,
            backprop_commands,
            job_done,
            threads: Mutex::new(threads),
            eval_batch_size: config.eval_batch_size,
        }
    }

    fn start_job(
        &self,
        shared: &Arc<Shared>,
        gather_rx: &Receiver<NodeEvent>,
        eval_rx: &Receiver<NodeEvent>,
        nn_tx: &Sender<NnRequest>,
        nn_rx: &Receiver<NnRequest>,
        backprop_rx: &Receiver<BackpropEvent>,
    ) {
        for sender in &self.gather_commands {
            sender
                .send(GatherCommand::Run(Arc::clone(shared), gather_rx.clone()))
                .expect("persistent gather worker is alive");
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

    fn finish_job(&self) {
        for _ in 0..self.gather_commands.len() + self.eval_commands.len() + self.backprop_commands.len() + 1 {
            self.job_done.recv().expect("persistent worker completion");
        }
    }

    fn assert_compatible(&self, config: &ResolvedSearchConfig) {
        assert_eq!(
            self.gather_commands.len(),
            config.gather_workers,
            "worker pool gather topology changed"
        );
        assert_eq!(
            self.eval_commands.len(),
            config.eval_workers,
            "worker pool eval topology changed"
        );
        assert_eq!(
            self.backprop_commands.len(),
            config.backprop_workers,
            "worker pool backprop topology changed"
        );
        assert_eq!(
            self.eval_batch_size, config.eval_batch_size,
            "worker pool batch size changed"
        );
    }
}

impl Drop for WorkerPool {
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

struct Shared {
    backend: Arc<dyn Backend>,
    repository: Arc<NodeRepository>,
    generation: SearchGeneration,
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
        // 唤醒等待 in-flight 上限的提交者，而不只是在完全空闲时唤醒。
        let _guard = self.idle_lock.lock();
        self.idle.notify_all();
        let _ = previous;
    }

    fn cancel_and_finish(&self, event: NodeEvent, collision: bool) {
        event.cancel();
        #[cfg(feature = "benchmark")]
        if collision {
            self.collisions.fetch_add(1, Ordering::AcqRel);
        }
        #[cfg(not(feature = "benchmark"))]
        let _ = collision;
        self.finish(false);
    }

    /// 正在评估的叶子不重复计算；直接归还这条未完成路径的 reservation。
    /// 参考：LC3 Overview 的 Gather/Eval worker 所有权边界。
    fn cancel_collision(&self, event: NodeEvent) {
        #[cfg(feature = "benchmark")]
        self.record_collision_depths(&[event.variation.moves().len()]);
        self.cancel_and_finish(event, true);
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
        #[cfg(feature = "benchmark")]
        event.mark_queued();
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
        #[cfg(feature = "benchmark")]
        event.mark_queued();
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

/// LC3 风格流式搜索：Gather / Eval / NN / Backprop。
/// Eval：terminal | cache → Backprop；否则编码 planes → NN queue；NN 返回后
/// softmax/edges → Backprop。NN：取出 tensor → GPU → 回复。
pub struct Search {
    shared: Arc<Shared>,
    root_history: Arc<PositionHistory>,
    root_key: NodeKey,
    /// 启动本次 job 前 root 已有的 completed N。它计入 UCI `go nodes`，但不计入本次 NPS。
    initial_visits: u64,
    /// Cap concurrent owned playouts after root expansion. Much smaller than
    /// queue capacity: saturating thousands of in-flight walks explodes collisions.
    max_in_flight: usize,
    worker_pool: Arc<WorkerPool>,
    workers_idle: bool,
}

impl Search {
    pub fn new(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        root_history: Arc<PositionHistory>,
        config: SearchConfig,
    ) -> Self {
        let graph = SearchGraph::new(root_history);
        Self::new_with_graph(backend, generation, &graph, config)
    }

    /// 从保留图创建独立搜索；该搜索自己创建、销毁 worker pool。
    pub fn new_with_graph(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        graph: &SearchGraph,
        config: SearchConfig,
    ) -> Self {
        let worker_pool = Arc::new(WorkerPool::new(backend.as_ref(), &config));
        Self::new_with_graph_in_pool(backend, generation, graph, config, worker_pool)
    }

    pub(crate) fn new_with_graph_in_pool(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        graph: &SearchGraph,
        config: SearchConfig,
        worker_pool: Arc<WorkerPool>,
    ) -> Self {
        config.validate();
        let resolved = config.resolve(backend.as_ref());
        worker_pool.assert_compatible(&resolved);
        let (gather_tx, gather_rx) = bounded(resolved.queue_capacity);
        let (eval_tx, eval_rx) = bounded(resolved.queue_capacity);
        let (backprop_tx, backprop_rx) = bounded(resolved.queue_capacity);
        let root_history = Arc::clone(graph.root_history());
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
            gather_tx,
            eval_tx,
            backprop_tx,
        });
        let (nn_tx, nn_rx) = bounded::<NnRequest>(resolved.queue_capacity);
        worker_pool.start_job(&shared, &gather_rx, &eval_rx, &nn_tx, &nn_rx, &backprop_rx);
        drop(nn_tx);
        Self {
            shared,
            root_history,
            root_key,
            initial_visits,
            // 保持数个 Eval batch 的叶子处于 in-flight，而非占满全部队列深度。
            max_in_flight: resolved
                .eval_batch_size
                .saturating_mul(4)
                .min(resolved.queue_capacity)
                .max(1),
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
        #[cfg(feature = "benchmark")]
        {
            self.shared.submitted.fetch_add(1, Ordering::Relaxed);
            self.shared
                .peak_in_flight
                .fetch_max(outstanding as u64, Ordering::Relaxed);
        }
        #[cfg(not(feature = "benchmark"))]
        let _ = outstanding;
        let mut event = event;
        #[cfg(feature = "benchmark")]
        event.mark_queued();
        loop {
            if self.shared.stopping.load(Ordering::Acquire) {
                self.shared.cancel_and_finish(event, false);
                return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
            }
            // LC3 worker 通过有界 event queue 通信。队列容量耗尽是正常背压，不是搜索请求
            // 失败；使用短超时，以便并发 Stop 仍能中断等待 Gather 容量的生产者。
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

    /// Runs owned stream events until a cumulative root-visit budget, deadline, or
    /// explicit stop is reached. After root expansion it keeps the pipeline
    /// filled up to `max_in_flight` (~4× eval batch) instead of draining to idle
    /// between waves. It always waits for submitted events to complete or
    /// cancel before returning, so root snapshots never observe a leaked edge
    /// reservation.
    pub fn run_with_limits(&self, limits: SearchLimits) -> Result<Stats, EnginError> {
        let target = limits.max_playouts.unwrap_or(u64::MAX);
        let max_in_flight = self.max_in_flight;
        while !self.is_stopping()
            && !limits.is_exhausted(
                self.initial_visits.saturating_add(self.stats().completed_playouts),
                target,
            )
        {
            let root_state = self
                .shared
                .repository
                .get(self.root_key)
                .map(|root| root.expansion_state());
            if root_state == Some(ExpansionState::Terminal) || self.shared.root_path_terminal.load(Ordering::Acquire) {
                break;
            }
            if root_state != Some(ExpansionState::Expanded) {
                // 启动阶段：root 展开前一次只提交一个 playout。
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
            if self
                .initial_visits
                .saturating_add(completed)
                .saturating_add(outstanding as u64)
                >= target
            {
                if outstanding == 0 {
                    break;
                }
                // 让已有 in-flight 工作完成至预算；不要继续提交而超出预算。
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

fn gather_worker(shared: Arc<Shared>, receiver: Receiver<NodeEvent>) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            #[cfg(feature = "benchmark")]
            Ok(mut event) => {
                #[cfg(feature = "benchmark")]
                if let Some(wait) = event.take_queue_wait() {
                    shared.gather_queue.record(wait);
                }
                if shared.stopping.load(Ordering::Acquire) {
                    shared.cancel_and_finish(event, false);
                    continue;
                }
                process_gather_event(&shared, event);
            }
            #[cfg(not(feature = "benchmark"))]
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

fn persistent_gather_worker(commands: Receiver<GatherCommand>, job_done: Sender<()>) {
    while let Ok(command) = commands.recv() {
        match command {
            GatherCommand::Run(shared, receiver) => {
                gather_worker(shared, receiver);
                let _ = job_done.send(());
            }
            GatherCommand::Shutdown => break,
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
                shared.cancel_collision(event);
                return;
            }
            ExpansionState::Terminal => {
                shared.send_backprop(BackpropEvent::evaluation(event));
                return;
            }
            ExpansionState::Expanded => {
                // board-key node 可由历史中已展开的局面复用；重复、长将/长捉和 rule60
                // 仍只属于当前 variation，不能因 node 已展开而绕过裁决。
                let depth = event.variation.moves().len();
                if let Some((wl, draw, plies_left)) = path_terminal_value(event.variation.history(), depth) {
                    let value = ValueDelta::with_plies_left(wl, draw, plies_left);
                    if event.reservations.is_empty() {
                        shared.root_path_terminal.store(true, Ordering::Release);
                        shared.finish(true);
                    } else {
                        shared.send_backprop(BackpropEvent::local_leaf(event.discard_leaf_node(), value));
                    }
                    return;
                }
                // MCGS 的共享 child 可能刚刚由另一条 variation 更新。回传只重算实际
                // 路径上的 node；因此每次再次访问本 node 前都要按最新 child Q 重算。
                // 参考 KataGo `docs/GraphSearch.md` “Stale Q Values”。
                shared.repository.recompute_graph_node(event.node_key);
                let edge_index = select_edge_from_node(
                    &shared.repository,
                    node.as_ref(),
                    depth,
                    &shared.params,
                    &shared.root_move_filter,
                )
                .expect("expanded stream node must have an edge");
                let reservation = node.reserve_edge(edge_index).expect("selected stream edge");
                let graph_child = event.variation.child_board_key(reservation.mv());
                if event.node_path().contains(&graph_child) || event.variation.returns_to_root_history(graph_child) {
                    let value = cycle_value(&mut event.variation, reservation.mv());
                    shared.send_backprop(BackpropEvent::local_leaf(event.descend_cycle(reservation), value));
                    return;
                }
                match shared
                    .repository
                    .bind_child_or_cut_cycle(event.node_key, &node.edges()[edge_index], graph_child)
                {
                    ChildLink::Bound => event = event.descend(graph_child, reservation),
                    ChildLink::Cycle(value) => {
                        shared.send_backprop(BackpropEvent::local_leaf(event.descend_cycle(reservation), value));
                        return;
                    }
                }
            }
        }
    }
}

/// MCGS 的 path-local 环只在当前 variation 内裁决，不能把历史相关结果写入共享 node。优先复用
/// px0 风格的 extension 判定；尚未达到 two-fold 门槛的闭环按首版约定视为本地和棋。
fn cycle_value(variation: &mut Variation, mv: Move) -> ValueDelta {
    let mut history = variation.history().clone();
    history.append(mv);
    match classify_extension(&history, variation.moves().len() + 1) {
        ExtensionKind::SharedTerminal { wl, draw, plies_left }
        | ExtensionKind::PathTerminal { wl, draw, plies_left } => ValueDelta::with_plies_left(wl, draw, plies_left),
        ExtensionKind::Evaluate => ValueDelta::with_plies_left(0.0, 1.0, 0.0),
    }
}

/// NN worker 返回的原始 logits[POLICY]、WDL[3] 与 moves-left[1]。
type NnReply = Result<(Vec<f32>, Vec<f32>, f32), EnginError>;

/// 交给 NN 线程的一个已编码局面。
struct NnRequest {
    planes: Vec<f32>,
    reply: Sender<NnReply>,
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
    input: EvalPosition,
    reply: Receiver<NnReply>,
}

/// LC3 EvalWorker：terminal | cache → Backprop；否则合法着 + 编码 → NN queue；
/// NN 回复后 → softmax/edges → Backprop。它不会因一次 GPU 调用阻塞整个 worker，
/// 而是在新 NodeEvent 之间轮询已完成结果。
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
            #[cfg(feature = "benchmark")]
            Ok(mut event) => {
                #[cfg(feature = "benchmark")]
                if let Some(wait) = event.take_queue_wait() {
                    shared.eval_queue.record(wait);
                }
                if let Err(error) = handle_eval_event(&shared, &nn_tx, &mut waiting, event) {
                    shared.fail(error);
                }
            }
            #[cfg(not(feature = "benchmark"))]
            Ok(event) => {
                if let Err(error) = handle_eval_event(&shared, &nn_tx, &mut waiting, event) {
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
    mut event: NodeEvent,
) -> Result<(), EnginError> {
    if shared.stopping.load(Ordering::Acquire) {
        shared.cancel_claimed_evaluation(event);
        return Ok(());
    }
    let node = shared.repository.get_or_insert(event.node_key);
    let depth = event.variation.moves().len();
    let history = event.variation.history();
    match classify_extension(history, depth) {
        ExtensionKind::SharedTerminal { wl, draw, plies_left } => {
            node.mark_terminal(wl, draw, plies_left);
            shared.send_backprop(BackpropEvent::evaluation(event));
            Ok(())
        }
        ExtensionKind::PathTerminal { wl, draw, plies_left } => {
            let value = ValueDelta::with_plies_left(wl, draw, plies_left);
            node.abort_evaluation();
            if event.reservations.is_empty() {
                shared.root_path_terminal.store(true, Ordering::Release);
                shared.finish(true);
            } else {
                shared.send_backprop(BackpropEvent::local_leaf(event.discard_leaf_node(), value));
            }
            Ok(())
        }
        ExtensionKind::Evaluate => {
            let legal_moves = history.last().board().generate_legal_moves();
            let input = EvalPosition {
                positions: history.positions().to_vec(),
                legal_moves: legal_moves.clone(),
            };
            if let Some(eval) = shared.backend.cached_evaluation(&input) {
                shared.cache_hits.fetch_add(1, Ordering::AcqRel);
                return publish_eval(shared, event, node, legal_moves, eval);
            }
            let planes = encode_position_for_nn(history, FillEmptyHistory::FenOnly);
            let (reply_tx, reply_rx) = bounded(1);
            if let Err(error) = send_nn_request(
                shared,
                nn_tx,
                NnRequest {
                    planes,
                    reply: reply_tx,
                    #[cfg(feature = "benchmark")]
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

fn complete_nn_item(
    shared: &Shared,
    item: WaitingNn,
    logits: Vec<f32>,
    wdl: Vec<f32>,
    moves_left: f32,
) -> Result<(), EnginError> {
    let eval = match decode_nn_eval(logits, wdl, moves_left, &item.legal_moves) {
        Ok(eval) => eval,
        Err(error) => {
            cancel_waiting_item(shared, item);
            return Err(error);
        }
    };
    shared.backend.store_evaluation(&item.input, Arc::clone(&eval));
    publish_eval(shared, item.event, item.node, item.legal_moves, eval)
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
    node.set_graph_value(ValueDelta::with_plies_left(
        network_wl_to_node(eval.wl),
        eval.d,
        eval.plies_left,
    ));
    // 先发布共享基值，再把 node 变为 Expanded；否则并发 Gather 可能在两者之间
    // 完成一次回传，而那次幂等重算会看不到基值。
    node.publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()).collect());
    shared.send_backprop(BackpropEvent::evaluation(event));
    Ok(())
}

/// 将 raw ONNX 输出转为可缓存的正式 `EvalResult`。
///
/// 参考：px0 `BackendComputation` 的统一结果路径（`src/neural/backend.h:75-87`）。
fn decode_nn_eval(
    logits: Vec<f32>,
    wdl: Vec<f32>,
    moves_left: f32,
    legal_moves: &[Move],
) -> Result<Arc<EvalResult>, EnginError> {
    let policies = softmax_legal_policy(&logits, legal_moves)?;
    if wdl.len() < 3 {
        return Err(EnginError::PortIncomplete("stream nn wdl length"));
    }
    if !wdl.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        || (wdl.iter().sum::<f32>() - 1.0).abs() > 1e-3
        || !moves_left.is_finite()
        || moves_left < 0.0
    {
        return Err(EnginError::Onnx("stream nn values are invalid".into()));
    }
    Ok(Arc::new(EvalResult {
        wl: wdl[0] - wdl[2],
        d: wdl[1],
        plies_left: moves_left,
        policies,
    }))
}

fn cancel_waiting_item(shared: &Shared, item: WaitingNn) {
    cancel_evaluation(shared, item.event, item.node);
}

/// 释放已 claim 但不会发布结果的 evaluation event。
///
/// 参考：LC3 Overview 的 EvalWorker 所有权模型：每个 owned event 必须经 backpropagation
/// 完成，或显式取消。
fn cancel_evaluation(shared: &Shared, event: NodeEvent, node: Arc<Node>) {
    event.cancel();
    node.abort_evaluation();
    shared.finish(false);
}

/// NN worker 按当前队列立即合并请求，Eval 不等待凑满 batch。
fn nn_worker(shared: Arc<Shared>, receiver: Receiver<NnRequest>, batch_size: usize) {
    let plane_len = INPUT_PLANES * BOARD_ROWS * BOARD_COLS;
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
        let mut requests = vec![first];
        while requests.len() < batch_size {
            match receiver.try_recv() {
                #[cfg(feature = "benchmark")]
                Ok(mut request) => {
                    #[cfg(feature = "benchmark")]
                    record_nn_queue_wait(&shared, &mut request);
                    requests.push(request);
                }
                #[cfg(not(feature = "benchmark"))]
                Ok(request) => requests.push(request),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        if shared.stopping.load(Ordering::Acquire) {
            reject_nn_requests(requests, EnginError::PortIncomplete("stream nn stopping"));
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
                    reject_nn_requests(requests, error);
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

fn backprop_worker(shared: Arc<Shared>, receiver: Receiver<BackpropEvent>) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            #[cfg(feature = "benchmark")]
            Ok(mut first) => {
                #[cfg(feature = "benchmark")]
                if let Some(wait) = first.take_queue_wait() {
                    shared.backprop_queue.record(wait);
                }
                let mut events = vec![first];
                for mut event in receiver.try_iter() {
                    #[cfg(feature = "benchmark")]
                    if let Some(wait) = event.take_queue_wait() {
                        shared.backprop_queue.record(wait);
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
                    shared
                        .completed_depth
                        .fetch_add(result.completed_depth, Ordering::AcqRel);
                    shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
                    for _ in 0..result.completed_playouts {
                        shared.finish(true);
                    }
                }
            }
            #[cfg(not(feature = "benchmark"))]
            Ok(first) => {
                let events: Vec<_> = std::iter::once(first).chain(receiver.try_iter()).collect();
                if shared.stopping.load(Ordering::Acquire) {
                    for event in events {
                        event.cancel();
                        shared.finish(false);
                    }
                } else {
                    let result = BackpropEvent::complete_batch(events, &shared.repository);
                    shared
                        .completed_depth
                        .fetch_add(result.completed_depth, Ordering::AcqRel);
                    shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
                    for _ in 0..result.completed_playouts {
                        shared.finish(true);
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn persistent_backprop_worker(commands: Receiver<BackpropCommand>, job_done: Sender<()>) {
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
    use std::time::{Duration, Instant};

    use crossbeam_channel::bounded;
    use parking_lot::{Condvar, Mutex};
    use xiangqi_core::{ChessBoard, GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{NodeEvent, Search, SearchConfig, SearchLimits, Shared};
    use crate::EnginError;
    use crate::neural::backend::{
        Backend, BackendAttributes, BackendComputation, EncodedInference, EvalResult, UniformBackend,
    };
    use crate::search::{
        ExpansionState, NodeRepository, SearchGeneration, SearchGraph, SearchParams, ValueDelta, best_move, root_stats,
    };

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
    fn search_completes_batched_playouts_and_returns_workers() {
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
            SearchGeneration(22),
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
        pipeline.stop_and_finish();
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
        pipeline.stop_and_finish();
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

        let mut search = Search::new_with_graph(
            Arc::new(UniformBackend::default()),
            SearchGeneration(27),
            &tree,
            SearchConfig::default(),
        );
        let stats = search.run_playouts(1).expect("path terminal root");

        assert_eq!(stats.completed_playouts, 1);
        assert!(root.edges().iter().all(|edge| edge.completed_visits() == 0));
        assert!(search.root_is_path_terminal());
        search.stop_and_finish();
    }

    #[test]
    fn gather_cuts_a_global_graph_cycle_to_an_edge_local_nn_leaf() {
        // 以真实 Gather/Backprop 路径覆盖：root -> child 若接上 child -> root 的既有
        // 图边会闭环。它必须完成 root edge 的 reservation，但不能绑定 child。
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let tree = SearchGraph::new(Arc::clone(&history));
        let root_key = tree.root_key();
        let mv = history.last().board().parse_move("b2b3").expect("legal root move");
        let child_key = NodeEvent::root(SearchGeneration(41), Arc::clone(&history))
            .variation
            .child_board_key(mv);

        let root = tree.repository().get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(mv, 1.0)]);

        let child = tree.repository().get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.set_graph_value(ValueDelta::one(0.4, 0.0));
        child.publish_edges(vec![(
            Move::new(Square::parse("b9").unwrap(), Square::parse("b8").unwrap()),
            1.0,
        )]);
        child.edges()[0].bind_child_key(root_key);

        let mut search = Search::new_with_graph(
            Arc::new(UniformBackend::default()),
            SearchGeneration(41),
            &tree,
            SearchConfig::default(),
        );
        let stats = search.run_playouts(1).expect("cycle-cut playout");
        let root_edge = &root.edges()[0];
        assert_eq!(stats.completed_playouts, 1);
        assert_eq!(root_edge.completed_visits(), 1);
        assert_eq!(root_edge.child_key(), None);
        assert_eq!(root_edge.cycle_leaf(), Some(ValueDelta::one(0.4, 0.0)));
        search.stop_and_finish();
    }

    #[test]
    fn path_terminal_root_does_not_mark_the_shared_node() {
        let state =
            GameState::from_fen_moves("4k4/9/9/9/9/9/9/9/R8/4K4 w - - 120 1", &[] as &[&str]).expect("rule60 root");
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(26),
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
            Arc::new(FailingComputationBackend),
            SearchGeneration(28),
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
            SearchGeneration(29),
            startpos_history(),
            SearchConfig::default(),
        );

        let error = pipeline.run_playouts(1).expect_err("invalid network values must fail");

        assert_eq!(error, EnginError::Onnx("stream nn values are invalid".into()));
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
        let (gather_tx, _gather_rx) = bounded(1);
        let (eval_tx, eval_rx) = bounded(1);
        let (backprop_tx, _backprop_rx) = bounded(1);
        drop(eval_rx);
        let shared = Shared {
            backend: Arc::new(UniformBackend::default()),
            repository,
            generation: SearchGeneration(30),
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
            gather_tx,
            eval_tx,
            backprop_tx,
        };

        shared.send_eval(NodeEvent::root(SearchGeneration(30), history));

        assert_eq!(root.expansion_state(), ExpansionState::Unexpanded);
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 0);
    }

    #[test]
    fn completed_search_can_reuse_a_graph_at_its_played_child() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = super::SearchGraph::new(history);
        let backend = Arc::new(UniformBackend::default()) as Arc<dyn Backend>;

        let mut first = Search::new_with_graph(
            Arc::clone(&backend),
            SearchGeneration(29),
            &tree,
            SearchConfig::default(),
        );
        first.run_playouts(16).expect("first search");
        let old_root = tree.root_key();
        let played = best_move(first.repository(), first.root_key(), false).expect("best move");
        first.stop_and_finish();

        tree.advance(played).expect("advance retained tree");
        assert!(tree.repository().get(old_root).is_some());

        let mut second = Search::new_with_graph(backend, SearchGeneration(30), &tree, SearchConfig::default());
        second.run_playouts(8).expect("reused search");
        let root = second.repository().get(second.root_key()).expect("reused root");
        assert!(root.edges().iter().all(|edge| edge.visits() == edge.completed_visits()));
        second.stop_and_finish();
    }
}
