//! Stream 搜索主线：Gather / Eval / NN / Backprop worker。
//!
//! worker 角色划分可参考 LC3 Overview 的 "Workers"：
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! Eval 负责终局、缓存、合法着和编码；NN 线程只做「取批 → 推理 → 交回」，
//! 不处理象棋/搜索逻辑。每个 owned event 独立流过队列；NN 只消费已经编码的 tensor，
//! 因而当前推理期间 Eval 可以持续准备下一批。

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
    BackpropEvent, ExpansionState, Node, NodeKey, NodeRepository, PlayoutEvent, SearchGraph, SearchParams, ValueDelta,
    select_edge,
};

const RECEIVE_POLL: Duration = Duration::from_millis(10);
const SHORT_CLOCK: Duration = Duration::from_millis(500);
const SHORT_CLOCK_IN_FLIGHT_CAP: usize = 8;

/// 当前 root 的累计 visit 搜索预算。
///
/// `go nodes N` 包含 graph reuse 前已有的 root N；时钟仍只约束本次 job。
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchLimits {
    pub max_playouts: Option<u64>,
    pub deadline: Option<Instant>,
}

impl SearchLimits {
    fn is_exhausted(self, completed: u64, target: u64, now: Instant) -> bool {
        completed >= target || self.deadline.is_some_and(|deadline| now >= deadline)
    }
}

/// 剩余思考时间不足 `SHORT_CLOCK` 时，不要按推荐 MiniBatch 填满窗口。
/// 否则第一刀 GPU 推理就会单独超过 `go movetime`。
fn deadline_in_flight_limit(deadline: Option<Instant>, window: usize, batch: usize, now: Instant) -> usize {
    let Some(deadline) = deadline else {
        return window;
    };
    if deadline.saturating_duration_since(now) >= SHORT_CLOCK {
        return window;
    }
    batch.clamp(1, SHORT_CLOCK_IN_FLIGHT_CAP).min(window)
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
    /// 本次搜索观察到的最大在途 Eval 叶子数（claim 后、NN 回复或提前释放前）。
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
    /// 已交给 Eval 的叶子上限倍率：`limit = ceil(MiniBatchSize × nn_window)`，启动时算一次。
    pub nn_window: f32,
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
            nn_window: 2.3,
            params: SearchParams::default(),
            gather_workers: 3,
            eval_workers: 5,
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
            self.nn_window.is_finite() && self.nn_window > 0.0,
            "stream nn window factor must be finite and positive"
        );
    }

    /// UCI `Threads` 尽量按 Gather:Eval = 1:2；除不尽时多给 Gather。
    /// 3→1/2，4→2/2，5→2/3，6→2/4，7→3/4，8→3/5。
    pub(crate) fn gather_eval_from_threads(threads: usize) -> (usize, usize) {
        let eval = ((threads * 2) / 3).max(1);
        let gather = threads.saturating_sub(eval).max(1);
        (gather, eval)
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
        let eval_claim_limit = ((eval_batch_size as f32) * self.nn_window).ceil().max(1.0) as usize;
        ResolvedSearchConfig {
            queue_capacity,
            eval_batch_size,
            eval_claim_limit,
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
    eval_claim_limit: usize,
    params: SearchParams,
    gather_workers: usize,
    eval_workers: usize,
}

enum GatherCommand {
    Run(Arc<Shared>, Receiver<PlayoutEvent>),
    Shutdown,
}

enum EvalCommand {
    Run(Arc<Shared>, Receiver<PlayoutEvent>, Sender<NnRequest>),
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
    eval_claim_limit: usize,
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
            && self.eval_claim_limit == config.eval_claim_limit
            && self.gather_commands.len() == config.gather_workers
            && self.eval_commands.len() == config.eval_workers
    }

    fn from_resolved(config: &ResolvedSearchConfig) -> Self {
        let (job_done_tx, job_done) = crossbeam_channel::unbounded();
        let eval_batch_size = config.eval_batch_size;
        let eval_claim_limit = config.eval_claim_limit;
        let mut gather_commands = Vec::with_capacity(config.gather_workers);
        let mut eval_commands = Vec::with_capacity(config.eval_workers);
        let (backprop_commands, backprop_rx) = crossbeam_channel::unbounded();
        let mut threads = Vec::with_capacity(config.gather_workers + config.eval_workers + 2);
        for _ in 0..config.gather_workers {
            let (tx, rx) = crossbeam_channel::unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || {
                persistent_gather_worker(rx, job_done, eval_claim_limit)
            }));
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
        threads.push(thread::spawn({
            let job_done = job_done_tx.clone();
            move || persistent_backprop_worker(backprop_rx, job_done)
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

    fn start_job(
        &self,
        shared: &Arc<Shared>,
        gather_rx: &Receiver<PlayoutEvent>,
        eval_rx: &Receiver<PlayoutEvent>,
        nn_tx: &Sender<NnRequest>,
        nn_rx: &Receiver<NnRequest>,
        backprop_rx: &Receiver<BackpropEvent>,
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
        self.backprop_commands[0]
            .send(BackpropCommand::Run(Arc::clone(shared), backprop_rx.clone()))
            .expect("persistent backprop worker is alive");
    }

    fn finish_job(&self) {
        for _ in 0..self.gather_commands.len() + self.eval_commands.len() + 2 {
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
            self.eval_batch_size, config.eval_batch_size,
            "worker pool batch size changed"
        );
        assert_eq!(
            self.eval_claim_limit, config.eval_claim_limit,
            "worker pool nn window changed"
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
    generation: u64,
    params: SearchParams,
    root_move_filter: Vec<Move>,
    /// 当前 root 的 history 已裁决结束，但不能污染同 board 的共享 node。
    root_path_terminal: AtomicBool,
    stopping: AtomicBool,
    /// 未 `finish` 的 owned event：drain / 节点预算。
    outstanding: AtomicUsize,
    /// 已交给 Eval 的叶子（编码或 NN）。满窗口就不再 claim 新叶子。
    nn_inflight: AtomicUsize,
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
    gather_tx: Sender<PlayoutEvent>,
    eval_tx: Sender<PlayoutEvent>,
    backprop_tx: Sender<BackpropEvent>,
    /// 撞上 `Evaluating` 叶子的 playout。先留着 reservation / μ；该叶子自己的
    /// backprop `complete` 之后再按 `node_key` 摘出来 cancel。
    collision_waiters: Mutex<Vec<PlayoutEvent>>,
}

impl Shared {
    fn start_playout(&self) {
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        #[cfg(feature = "benchmark")]
        self.submitted.fetch_add(1, Ordering::Relaxed);
    }

    fn complete_root_terminal(&self) {
        self.completed.fetch_add(1, Ordering::AcqRel);
    }

    fn finish(&self, completed: bool) {
        if completed {
            self.completed.fetch_add(1, Ordering::AcqRel);
        }
        let previous = self.outstanding.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "stream outstanding task underflow");
        // 唤醒等待本轮真实 leaf 完成的 owner。
        let _guard = self.idle_lock.lock();
        self.idle.notify_all();
        let _ = previous;
    }

    fn release_eval_claim(&self) {
        let previous = self.nn_inflight.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "stream eval claim underflow");
        let _guard = self.idle_lock.lock();
        self.idle.notify_all();
    }

    fn wait_while(&self, deadline: Option<Instant>, stop_on_stopping: bool, mut busy: impl FnMut(&Self) -> bool) {
        let mut guard = self.idle_lock.lock();
        while busy(self) && !(stop_on_stopping && self.stopping.load(Ordering::Acquire)) && self.error.lock().is_none()
        {
            let Some(deadline) = deadline else {
                self.idle.wait(&mut guard);
                continue;
            };
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let wait = deadline.saturating_duration_since(now).min(RECEIVE_POLL);
            if wait.is_zero() {
                break;
            }
            self.idle.wait_for(&mut guard, wait);
        }
    }

    /// 撞上正在评估的叶子：挂起整条 reservation，让 μ 继续分流。
    fn park_collision(&self, event: PlayoutEvent) {
        #[cfg(feature = "benchmark")]
        {
            self.collisions.fetch_add(1, Ordering::AcqRel);
            self.record_collision_depths(&[event.variation.moves().len()]);
        }
        {
            let mut waiters = self.collision_waiters.lock();
            if !self.stopping.load(Ordering::Acquire)
                && self
                    .repository
                    .get(event.node_key)
                    .is_some_and(|node| node.expansion_state() == ExpansionState::Evaluating)
            {
                waiters.push(event);
                return;
            }
        }
        event.cancel();
        self.finish(false);
    }

    fn cancel_collisions(&self, key: NodeKey) {
        let mut waiters = self.collision_waiters.lock();
        let mut i = 0;
        let mut parked = Vec::new();
        while i < waiters.len() {
            if waiters[i].node_key == key {
                parked.push(waiters.swap_remove(i));
            } else {
                i += 1;
            }
        }
        drop(waiters);
        for event in parked {
            event.cancel();
            self.finish(false);
        }
    }

    fn cancel_all_collisions(&self) {
        let parked = std::mem::take(&mut *self.collision_waiters.lock());
        for event in parked {
            event.cancel();
            self.finish(false);
        }
    }

    /// Cancels an event after Gather has claimed its leaf for Eval.
    ///
    /// Reference: LC3 overview's EvalWorker ownership model. Releasing only
    /// the edge reservations would leave the claimed node permanently
    /// `Evaluating`, so this also restores it to `Unexpanded`.
    fn cancel_claimed_evaluation(&self, event: PlayoutEvent) {
        let Some(node) = self.repository.get(event.node_key) else {
            event.cancel();
            self.finish(false);
            return;
        };
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

    /// 先增加 `nn_inflight` 再入队：图复用后 cache hit 极快，若 `try_send` 成功后才 +1，
    /// Eval 可能已经 `release_eval_claim`，第二手 `go` 就会 underflow 闪退。
    fn send_eval(&self, mut event: PlayoutEvent) {
        #[cfg(feature = "benchmark")]
        event.mark_queued();
        self.nn_inflight.fetch_add(1, Ordering::AcqRel);
        #[cfg(feature = "benchmark")]
        self.peak_in_flight
            .fetch_max(self.nn_inflight.load(Ordering::Relaxed) as u64, Ordering::Relaxed);
        loop {
            if self.stopping.load(Ordering::Acquire) {
                self.release_eval_claim();
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
                    self.release_eval_claim();
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
                let key = event.local_leaf.is_none().then_some(event.playout.node_key);
                event.cancel();
                self.finish(false);
                if let Some(key) = key {
                    self.cancel_collisions(key);
                }
                return;
            }
            match self.backprop_tx.try_send(event) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    event = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    let key = returned.local_leaf.is_none().then_some(returned.playout.node_key);
                    returned.cancel();
                    self.finish(false);
                    if let Some(key) = key {
                        self.cancel_collisions(key);
                    }
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

/// 连续流式搜索：Gather / Eval / NN / Backprop。
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
        let (gather_tx, gather_rx) = bounded(resolved.queue_capacity);
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
            nn_inflight: AtomicUsize::new(0),
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
            collision_waiters: Mutex::new(Vec::new()),
        });
        let (nn_tx, nn_rx) = bounded::<NnRequest>(resolved.queue_capacity);
        worker_pool.start_job(&shared, &gather_rx, &eval_rx, &nn_tx, &nn_rx, &backprop_rx);
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

    fn submit_playout(&self) -> Result<(), EnginError> {
        if self.shared.stopping.load(Ordering::Acquire) {
            return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
        }
        self.shared.start_playout();
        let mut event = PlayoutEvent::at_root(self.shared.generation, self.root_key, Arc::clone(&self.root_history));
        #[cfg(feature = "benchmark")]
        event.mark_queued();
        loop {
            if self.shared.stopping.load(Ordering::Acquire) {
                event.cancel();
                self.shared.finish(false);
                return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
            }
            match self.shared.gather_tx.send_timeout(event, RECEIVE_POLL) {
                Ok(()) => return Ok(()),
                Err(SendTimeoutError::Timeout(returned)) => event = returned,
                Err(SendTimeoutError::Disconnected(returned)) => {
                    returned.cancel();
                    self.shared.finish(false);
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
        loop {
            let now = Instant::now();
            if self.is_stopping()
                || limits.is_exhausted(
                    self.initial_visits.saturating_add(self.stats().completed_playouts),
                    target,
                    now,
                )
            {
                break;
            }
            if next_report.is_some_and(|deadline| now >= deadline) {
                report(self.stats());
                next_report = report_interval.and_then(|interval| now.checked_add(interval));
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
                self.wait_until_outstanding_below(1, limits.deadline)?;
                continue;
            }
            if root_state != Some(ExpansionState::Expanded) {
                // root 展开前只需要一个真实 leaf。
                if let Err(error) = self.submit_playout() {
                    if self.is_stopping() {
                        break;
                    }
                    return Err(error);
                }
                self.wait_until_outstanding_below(1, limits.deadline)?;
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
                self.wait_until_outstanding_below(outstanding, limits.deadline)?;
                continue;
            }
            let in_flight_limit = deadline_in_flight_limit(
                limits.deadline,
                self.worker_pool.eval_claim_limit,
                self.worker_pool.eval_batch_size,
                now,
            );
            if self.shared.nn_inflight.load(Ordering::Acquire) >= in_flight_limit {
                self.shared.wait_while(limits.deadline, true, |shared| {
                    shared.nn_inflight.load(Ordering::Acquire) >= in_flight_limit
                });
                if let Some(error) = self.shared.error.lock().clone() {
                    return Err(error);
                }
                continue;
            }
            if let Err(error) = self.submit_playout() {
                if self.is_stopping() {
                    break;
                }
                return Err(error);
            }
        }
        // 时钟到期后必须 request_stop：否则 Eval 会等当前 GPU 整批跑完，200ms
        // 的 go 就会变成一次推荐 batch 的推理时间。节点预算仍等在途完成。
        if limits.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            self.request_stop();
        }
        // 请求停止是正常搜索结果。`wait_for_idle()` 已保证每个入队 event 都已完成或取消
        // reservation，因此调用方可安全快照部分 graph。
        self.wait_for_idle()?;
        Ok(self.stats())
    }

    pub fn wait_for_idle(&self) -> Result<(), EnginError> {
        self.wait_until_outstanding_below(1, None)
    }

    fn wait_until_outstanding_below(&self, limit: usize, deadline: Option<Instant>) -> Result<(), EnginError> {
        self.shared.wait_while(deadline, false, |shared| {
            shared.outstanding.load(Ordering::Acquire) >= limit
        });
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

fn gather_worker(shared: Arc<Shared>, receiver: Receiver<PlayoutEvent>, eval_claim_limit: usize) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            #[cfg(feature = "benchmark")]
            Ok(mut event) => {
                if let Some(wait) = event.take_queue_wait() {
                    shared.gather_queue.record(wait);
                }
                if shared.stopping.load(Ordering::Acquire) {
                    event.cancel();
                    shared.finish(false);
                } else {
                    process_gather_event(&shared, event, eval_claim_limit);
                }
            }
            #[cfg(not(feature = "benchmark"))]
            Ok(event) if shared.stopping.load(Ordering::Acquire) => {
                event.cancel();
                shared.finish(false);
            }
            #[cfg(not(feature = "benchmark"))]
            Ok(event) => process_gather_event(&shared, event, eval_claim_limit),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn persistent_gather_worker(commands: Receiver<GatherCommand>, job_done: Sender<()>, eval_claim_limit: usize) {
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

#[derive(Clone, Copy)]
enum ChildTarget {
    Graph(NodeKey),
    Continuation(NodeKey),
}

fn selected_child(shared: &Shared, event: &mut PlayoutEvent, node: &Node, edge_index: usize) -> Option<ChildTarget> {
    let edge = &node.edges()[edge_index];
    if !event.node_key.is_continuation()
        && let Some(child) = event.repeated_child_key(edge.mv())
    {
        let child_node = shared.repository.get_or_insert(child);
        // Graph→Tree 不走 bind，深度不会在那里赋值。入口深度必须是父节点首次到达+1，
        // 否则 Tree 稍后当父节点绑边时会被写成 0，浅层 Graph 回边剪不掉。
        node.try_set_first_depth(0);
        child_node.try_set_first_depth(node.depth().unwrap_or(0) + 1);
        return Some(ChildTarget::Continuation(child));
    }
    let child = if event.node_key.is_continuation() {
        event.variation.child_key_for_history(edge.mv())
    } else {
        event.variation.child_key(edge.mv())
    };
    match shared.repository.bind_child_or_cut_cycle(node, edge, child) {
        ChildLink::Bound => Some(ChildTarget::Graph(child)),
        ChildLink::TopologyPruned => None,
    }
}

fn branch_at_expanded_node(
    shared: &Shared,
    event: &mut PlayoutEvent,
    node: &Node,
    depth: usize,
) -> Option<(ChildTarget, super::EdgeReservation)> {
    loop {
        let (edge_index, fpu) = select_edge(
            &shared.repository,
            &node.edges(),
            node.completed_visits(),
            node.q(),
            depth,
            &shared.params,
            &shared.root_move_filter,
        )?;
        // 复用本次 PUCT selection 的 FPU。`selected_child` 可能绑定一个已有 graph
        // child；那会改变下一次 selection 的 FPU 参与集合，但不能倒写本次 pending
        // sample 的均值。
        let virtual_mean = if shared.params.virtual_mean_fpu_scale > 0.0 {
            Some(shared.params.virtual_mean_fpu_scale * fpu)
        } else {
            None
        };
        let Some(target) = selected_child(shared, event, node, edge_index) else {
            continue;
        };
        let reservation = node
            .reserve_edge_with_virtual_mean(edge_index, virtual_mean)
            .expect("selected stream edge");
        return Some((target, reservation));
    }
}

fn process_gather_event(shared: &Shared, mut event: PlayoutEvent, eval_claim_limit: usize) {
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            event.cancel();
            shared.finish(false);
            return;
        }
        let node = shared.repository.get_or_insert(event.node_key);
        match node.expansion_state() {
            ExpansionState::Unexpanded => {
                if shared.nn_inflight.load(Ordering::Acquire) >= eval_claim_limit {
                    shared.wait_while(None, true, |shared| {
                        shared.nn_inflight.load(Ordering::Acquire) >= eval_claim_limit
                    });
                    continue;
                }
                if node.try_begin_evaluation() {
                    shared.send_eval(event);
                    return;
                }
            }
            ExpansionState::Evaluating => {
                shared.park_collision(event);
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
                        shared.finish(false);
                        return;
                    } else {
                        shared.send_backprop(BackpropEvent::local_leaf(event.discard_leaf_node(), value));
                    }
                    return;
                }
                // MCGS 的共享 child 可能刚刚由另一条 variation 更新。回传只重算实际
                // 路径上的 node；因此每次再次访问本 node 前都要按最新 child Q 重算。
                // 参考 KataGo `docs/GraphSearch.md` “Stale Q Values”。
                shared.repository.recompute_node(event.node_key);
                let Some((target, reservation)) = branch_at_expanded_node(shared, &mut event, node.as_ref(), depth)
                else {
                    if event.reservations.is_empty() {
                        shared.root_path_terminal.store(true, Ordering::Release);
                        shared.complete_root_terminal();
                        shared.finish(false);
                    } else {
                        shared.send_backprop(BackpropEvent::evaluation(event));
                    }
                    return;
                };
                event = match target {
                    ChildTarget::Graph(child) => event.descend(child, reservation),
                    ChildTarget::Continuation(child) => event.descend_continuation(child, reservation),
                };
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
    event: PlayoutEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
    cache_key: EvalCacheKey,
    reply: Receiver<NnReply>,
}

/// LC3 EvalWorker：terminal | cache → Backprop；否则合法着 + 编码 → NN queue；
/// NN 回复后 → softmax/edges → Backprop。它不会因一次 GPU 调用阻塞整个 worker，
/// 而是在新 PlayoutEvent 之间轮询已完成结果。
fn eval_worker(shared: Arc<Shared>, receiver: Receiver<PlayoutEvent>, nn_tx: Sender<NnRequest>) {
    let mut waiting: Vec<WaitingNn> = Vec::new();
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            for item in waiting.drain(..) {
                cancel_waiting_item(&shared, item);
            }
            while let Ok(event) = receiver.try_recv() {
                shared.release_eval_claim();
                shared.cancel_claimed_evaluation(event);
            }
            shared.cancel_all_collisions();
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
                if waiting.is_empty() || shared.stopping.load(Ordering::Acquire) {
                    continue;
                }
                // 没有新叶子：短暂等待至少一个 NN 回复。
                wait_one_nn_completion(&shared, &mut waiting);
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                for item in waiting.drain(..) {
                    cancel_waiting_item(&shared, item);
                }
                while let Ok(event) = receiver.try_recv() {
                    shared.release_eval_claim();
                    shared.cancel_claimed_evaluation(event);
                }
                shared.cancel_all_collisions();
                break;
            }
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
    if waiting.is_empty() || shared.stopping.load(Ordering::Acquire) {
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
    mut event: PlayoutEvent,
) -> Result<(), EnginError> {
    if shared.stopping.load(Ordering::Acquire) {
        shared.release_eval_claim();
        shared.cancel_claimed_evaluation(event);
        return Ok(());
    }
    let node = shared.repository.get_or_insert(event.node_key);
    let depth = event.variation.moves().len();
    let history = event.variation.history();
    match classify_extension(history, depth) {
        ExtensionKind::SharedTerminal { wl, draw, plies_left } => {
            node.mark_terminal(wl, draw, plies_left);
            shared.release_eval_claim();
            shared.send_backprop(BackpropEvent::evaluation(event));
            Ok(())
        }
        ExtensionKind::PathTerminal { wl, draw, plies_left } => {
            let value = ValueDelta::with_plies_left(wl, draw, plies_left);
            // TreeNode 的 key 含完整规则 history，第三次重复等对这个局部 state
            // 是确定的，可 `mark_terminal`。GraphNode 落到这里只可能是 rule60：
            // 同一棋盘从另一条 history 进来可以不到 120，只能写本次 edge 的 local leaf。
            if event.node_key.is_continuation() {
                node.mark_terminal(wl, draw, plies_left);
                shared.release_eval_claim();
                shared.send_backprop(BackpropEvent::evaluation(event));
                return Ok(());
            }
            shared.release_eval_claim();
            node.abort_evaluation();
            shared.cancel_collisions(event.node_key);
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
            let cache_key = EvalCacheKey::new(history.last(), legal_moves.len());
            if let Some(eval) = shared.backend.cached_evaluation(cache_key) {
                shared.cache_hits.fetch_add(1, Ordering::AcqRel);
                shared.release_eval_claim();
                return publish_eval(shared, event, node, legal_moves, eval);
            }
            let planes = encode_position_input_planes(history, FillEmptyHistory::FenOnly);
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
                shared.release_eval_claim();
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

fn complete_nn_item(shared: &Shared, item: WaitingNn, batch: Arc<EncodedBatch>, row: usize) -> Result<(), EnginError> {
    shared.release_eval_claim();
    let eval = match eval_result_from_encoded_row(&batch, row, &item.legal_moves) {
        Ok(eval) => eval,
        Err(error) => {
            cancel_evaluation(shared, item.event, item.node);
            return Err(error);
        }
    };
    shared.backend.store_evaluation(item.cache_key, Arc::clone(&eval));
    publish_eval(shared, item.event, item.node, item.legal_moves, eval)
}

/// 在 backend 完成结果写入 graph 前校验它。
///
/// 参考：ARCHITECTURE 固定的 `WDL + moves-left` backend 契约。worker 失败必须通过
/// LC3 取消路径返回；持有 owned event 后不得 panic。
fn publish_eval(
    shared: &Shared,
    event: PlayoutEvent,
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
    node.set_base_value(ValueDelta::with_plies_left(
        -eval.wl, // 将 NN 的 side-to-move WDL 转为 node / incoming-edge 视角：取反。
        eval.d,
        eval.plies_left,
    ));
    // 先发布共享基值，再把 node 变为 Expanded；否则并发 Gather 可能在两者之间
    // 完成一次回传，而那次幂等重算会看不到基值。
    node.publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()).collect());
    shared.send_backprop(BackpropEvent::evaluation(event));
    Ok(())
}

fn cancel_waiting_item(shared: &Shared, item: WaitingNn) {
    shared.release_eval_claim();
    cancel_evaluation(shared, item.event, item.node);
}

/// 释放已 claim 但不会发布结果的 evaluation event。
///
/// 参考：LC3 Overview 的 EvalWorker 所有权模型：每个 owned event 必须经 backpropagation
/// 完成，或显式取消。
fn cancel_evaluation(shared: &Shared, event: PlayoutEvent, node: Arc<Node>) {
    let key = event.node_key;
    event.cancel();
    node.abort_evaluation();
    shared.cancel_collisions(key);
    shared.finish(false);
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

fn backprop_worker(shared: Arc<Shared>, receiver: Receiver<BackpropEvent>) {
    loop {
        let first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(event) => event,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        let mut events = Vec::with_capacity(1 + receiver.len());
        events.push(first);
        events.extend(receiver.try_iter());
        if events.is_empty() {
            continue;
        }
        if shared.stopping.load(Ordering::Acquire) {
            for event in events {
                event.cancel();
                shared.finish(false);
            }
            continue;
        }
        #[cfg(feature = "benchmark")]
        for event in &mut events {
            if let Some(wait) = event.take_queue_wait() {
                shared.backprop_queue.record(wait);
            }
        }
        let leaf_keys: Vec<NodeKey> = events
            .iter()
            .filter(|event| event.local_leaf.is_none())
            .map(|event| event.playout.node_key)
            .collect();
        let result = BackpropEvent::complete_batch(events, &shared.repository);
        for key in leaf_keys {
            shared.cancel_collisions(key);
        }
        shared
            .completed_depth
            .fetch_add(result.completed_depth, Ordering::AcqRel);
        shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
        for _ in 0..result.completed_playouts {
            shared.finish(true);
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
    use xiangqi_core::{ChessBoard, GameState, Move, Position, PositionHistory, STARTPOS_FEN, Square};

    use super::{BackpropEvent, PlayoutEvent, Search, SearchConfig, SearchLimits, Shared};
    use crate::EnginError;
    use crate::neural::backend::{Backend, BackendAttributes, UniformBackend};
    use crate::search::{
        ExpansionState, NodeKey, NodeRepository, SearchGraph, SearchParams, ValueDelta, best_move, root_stats,
    };

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

    fn detached_shared(repository: Arc<NodeRepository>, outstanding: usize) -> Shared {
        let (gather_tx, _gather_rx) = bounded(1);
        let (eval_tx, _eval_rx) = bounded(1);
        let (backprop_tx, _backprop_rx) = bounded(1);
        Shared {
            backend: Arc::new(UniformBackend::default()),
            repository,
            generation: 1,
            params: SearchParams::default(),
            root_move_filter: Vec::new(),
            root_path_terminal: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(outstanding),
            nn_inflight: AtomicUsize::new(0),
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
            collision_waiters: Mutex::new(Vec::new()),
        }
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
        // Uniform 默认 batch=1 → 窗口 2；Gather 竞态最多再多几个。
        assert!(stats.peak_in_flight <= 8);
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

    struct SlowInferenceBackend;

    impl Backend for SlowInferenceBackend {
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
            thread::sleep(Duration::from_millis(250));
            UniformBackend::default().infer_input_planes_into(samples, logits, wdl, moves_left)
        }
    }

    #[test]
    fn expired_deadline_does_not_wait_for_in_flight_inference() {
        let mut pipeline = Search::new(
            Arc::new(SlowInferenceBackend),
            31,
            startpos_history(),
            SearchConfig {
                eval_batch_size: 4,
                gather_workers: 2,
                eval_workers: 2,
                ..SearchConfig::default()
            },
        );
        let started = Instant::now();
        pipeline
            .run_with_limits(SearchLimits {
                max_playouts: Some(64),
                deadline: Some(started + Duration::from_millis(40)),
            })
            .expect("deadline is a normal result");
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "deadline waited for GPU: {:?}",
            started.elapsed()
        );
        pipeline.stop_and_finish();
    }

    #[test]
    fn deadline_in_flight_limit_shrinks_near_the_clock() {
        let now = Instant::now();
        assert_eq!(super::deadline_in_flight_limit(None, 160, 64, now), 160);
        assert_eq!(
            super::deadline_in_flight_limit(Some(now + Duration::from_millis(200)), 160, 64, now),
            8
        );
        assert_eq!(
            super::deadline_in_flight_limit(Some(now + Duration::from_secs(2)), 160, 64, now),
            160
        );
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
        root.set_base_value(crate::search::ValueDelta::one(0.0, 0.0));

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
        let child_key = PlayoutEvent::root(41, Arc::clone(&history)).variation.child_key(mv);

        let root = tree.repository().get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.set_base_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(mv, 0.9), (fallback, 0.1)]);

        let child = tree.repository().get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.set_base_value(ValueDelta::one(0.4, 0.0));
        child.publish_edges(vec![(
            Move::new(Square::parse("b9").unwrap(), Square::parse("b8").unwrap()),
            1.0,
        )]);
        assert!(matches!(
            tree.repository()
                .bind_child_or_cut_cycle(&child, &child.edges()[0], root_key),
            crate::search::graph::ChildLink::Bound
        ));

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
        let child_key = PlayoutEvent::root(43, Arc::clone(&history)).variation.child_key(first);
        let child_position = Position::after(history.last(), first);
        let reply = child_position.board().parse_move("b9b8").expect("legal reply");
        let grandchild = Position::after(&child_position, reply);
        let grandchild_key = crate::search::NodeKey::graph_node(grandchild.board().hash());

        let root = tree.repository().get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.set_base_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(first, 1.0)]);
        root.edges()[0].bind_child_key(child_key);

        let child = tree.repository().get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.set_base_value(ValueDelta::one(0.4, 0.0));
        child.publish_edges(vec![(reply, 1.0)]);

        let grandchild_node = tree.repository().get_or_insert(grandchild_key);
        assert!(grandchild_node.try_begin_evaluation());
        grandchild_node.set_base_value(ValueDelta::one(0.2, 0.0));
        grandchild_node.publish_edges(vec![(
            Move::new(Square::parse("c0").unwrap(), Square::parse("c1").unwrap()),
            1.0,
        )]);
        assert!(matches!(
            tree.repository()
                .bind_child_or_cut_cycle(&grandchild_node, &grandchild_node.edges()[0], child_key),
            crate::search::graph::ChildLink::Bound
        ));

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
        let mut event = PlayoutEvent::root(42, Arc::clone(&history));
        let continuation = event.repeated_child_key(mv).expect("first repetition");

        assert!(root.try_begin_evaluation());
        root.set_base_value(ValueDelta::one(0.0, 0.0));
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
        root.set_base_value(ValueDelta::one(0.0, 0.0));
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
    fn collision_keeps_virtual_mean_until_that_leaf_completes() {
        let history = startpos_history();
        let root_key = NodeKey::graph_node(history.last().board().hash());
        let first = history.last().board().parse_move("b2b3").expect("first");
        let second = history.last().board().parse_move("g3g4").expect("second");
        let child_a = PlayoutEvent::root(1, Arc::clone(&history)).variation.child_key(first);
        let child_b = PlayoutEvent::root(1, Arc::clone(&history)).variation.child_key(second);

        let repository = Arc::new(NodeRepository::default());
        let root = repository.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(first, 0.6), (second, 0.4)]);
        root.set_base_value(ValueDelta::one(0.0, 0.0));
        root.edges()[0].bind_child_key(child_a);
        root.edges()[1].bind_child_key(child_b);
        assert!(repository.get_or_insert(child_a).try_begin_evaluation());
        assert!(repository.get_or_insert(child_b).try_begin_evaluation());

        let claimer = PlayoutEvent::root(1, Arc::clone(&history)).descend(
            child_a,
            root.reserve_edge_with_virtual_mean(0, Some(0.25)).expect("claimer"),
        );
        let collider_a = PlayoutEvent::root(1, Arc::clone(&history)).descend(
            child_a,
            root.reserve_edge_with_virtual_mean(0, Some(0.5)).expect("collider a"),
        );
        let collider_b = PlayoutEvent::root(1, Arc::clone(&history)).descend(
            child_b,
            root.reserve_edge_with_virtual_mean(1, Some(0.75)).expect("collider b"),
        );

        let shared = detached_shared(repository, 3);
        shared.park_collision(collider_a);
        shared.park_collision(collider_b);

        let edge_a = root.edges()[0].clone();
        let edge_b = root.edges()[1].clone();
        assert_eq!(edge_a.visits(), 2);
        assert_eq!(edge_a.completed_visits(), 0);
        assert!((edge_a.stats().virtual_wl_sum - 0.75).abs() < 1e-5);
        assert_eq!(edge_b.visits(), 1);
        assert!((edge_b.stats().virtual_wl_sum - 0.75).abs() < 1e-5);

        BackpropEvent::complete_batch([BackpropEvent::evaluation(claimer)], &shared.repository);
        assert_eq!(edge_a.completed_visits(), 1);
        assert_eq!(edge_a.visits(), 2);
        assert!((edge_a.stats().virtual_wl_sum - 0.5).abs() < 1e-5);
        assert_eq!(edge_b.visits(), 1);

        shared.cancel_collisions(child_a);
        assert_eq!(edge_a.visits(), 1);
        assert_eq!(edge_a.completed_visits(), 1);
        assert_eq!(edge_a.stats().virtual_wl_sum, 0.0);
        assert_eq!(edge_b.visits(), 1);
        assert!((edge_b.stats().virtual_wl_sum - 0.75).abs() < 1e-5);

        shared.cancel_collisions(child_b);
        assert_eq!(edge_b.visits(), 0);
        assert_eq!(edge_b.stats().virtual_wl_sum, 0.0);
    }

    #[test]
    fn failed_eval_enqueue_releases_the_claimed_node() {
        let history = startpos_history();
        let root_key = NodeKey::graph_node(history.last().board().hash());
        let repository = Arc::new(NodeRepository::default());
        let root = repository.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        let shared = detached_shared(repository, 1);
        shared.send_eval(PlayoutEvent::root(30, history));
        assert_eq!(root.expansion_state(), ExpansionState::Unexpanded);
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 0);
    }

    #[test]
    fn reused_graph_survives_incremental_uci_chase() {
        // GUI 跨回合 `position fen … moves …` 复用图；一次性直接设到终局 fen+moves 不复现。
        let fen = "2bakc3/4a3n/4b4/2C1p4/P8/4P2cN/P8/4B1C2/4A4/4KAB2 b - - 0 1";
        let steps: [&[&str]; 4] = [&[], &["f9f4"], &["f9f4", "g2g4"], &["f9f4", "g2g4", "f4f3"]];
        let backend = Arc::new(UniformBackend::default()) as Arc<dyn Backend>;
        let start = GameState::from_fen_moves(fen, &[] as &[&str]).expect("fen");
        let mut graph = super::SearchGraph::new(Arc::new(PositionHistory::from_positions(start.positions())));
        for (generation, moves) in steps.into_iter().enumerate() {
            let state = GameState::from_fen_moves(fen, moves).expect("replay");
            let history = Arc::new(PositionHistory::from_positions(state.positions()));
            graph
                .reset_to_history_after_drain(Arc::clone(&history))
                .expect("reuse graph");
            if let Some(root) = graph.take_pending_gc_root() {
                graph.repository().retain_from_root(root);
            }
            let mut search = Search::new_with_graph(
                Arc::clone(&backend),
                40 + generation as u64,
                &graph,
                SearchConfig {
                    eval_batch_size: 8,
                    gather_workers: 2,
                    eval_workers: 2,
                    ..SearchConfig::default()
                },
            );
            search
                .run_playouts(2000)
                .unwrap_or_else(|error| panic!("reused search at ply {generation} failed: {error}"));
            let root_is_black = history.last().is_black_to_move();
            let mv = best_move(search.repository(), search.root_key(), root_is_black);
            assert!(
                mv.is_some_and(|mv| !mv.is_null()) || !history.last().board().generate_legal_moves().is_empty(),
                "reused root at ply {generation} produced no playable move"
            );
            search.stop_and_finish();
        }
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
