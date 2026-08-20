//! 搜索观察者：正式路径用 [`NoopObserver`]（空实现可内联消掉）；
//! bench 挂 [`BenchObserver`] 收集诊断指标。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// 队列等待所属阶段。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueKind {
    Gather,
    Eval,
    Nn,
    Backprop,
}

/// 事件入队戳存储。正式路径用 [`NoQueueStamp`]（ZST）；bench 用 [`InstantQueueStamp`]。
pub trait QueueStamp: Default + Send + 'static {
    fn mark(&mut self);
    fn take_wait(&mut self) -> Option<Duration>;
}

/// 正式路径：零大小，不占事件内存。
#[derive(Clone, Copy, Debug, Default)]
pub struct NoQueueStamp;

impl QueueStamp for NoQueueStamp {
    #[inline(always)]
    fn mark(&mut self) {}

    #[inline(always)]
    fn take_wait(&mut self) -> Option<Duration> {
        None
    }
}

/// bench：真正打 `Instant`。
#[derive(Clone, Debug, Default)]
pub struct InstantQueueStamp(Option<Instant>);

impl QueueStamp for InstantQueueStamp {
    #[inline(always)]
    fn mark(&mut self) {
        self.0 = Some(Instant::now());
    }

    #[inline(always)]
    fn take_wait(&mut self) -> Option<Duration> {
        self.0.take().map(|queued_at| queued_at.elapsed())
    }
}

/// 热路径诊断钩子。默认方法全部为空；[`NoopObserver`] 单态化后应被优化掉。
pub trait SearchObserver: Send + Sync + 'static {
    /// 入队戳类型；正式路径为 [`NoQueueStamp`]。
    type Stamp: QueueStamp;
    /// 为 true 时才走诊断记账；正式路径为 false，调用点整段可被 DCE。
    const ENABLED: bool = false;

    fn on_submitted(&self) {}
    fn on_collision(&self, _depth: usize) {}
    fn on_peak_inflight(&self, _n: usize) {}
    fn on_queue_wait(&self, _kind: QueueKind, _wait: Duration) {}
    /// 一次 NN 合批推理的实际 batch size。
    fn on_batch(&self, _size: usize) {}
}

/// 正式 UCI / 默认搜索：无诊断开销。
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopObserver;

impl SearchObserver for NoopObserver {
    type Stamp = NoQueueStamp;
}

/// 单段队列等待快照。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueueStats {
    pub samples: u64,
    pub total_wait_ns: u64,
    pub max_wait_ns: u64,
}

#[derive(Debug, Default)]
struct QueueMetrics {
    samples: AtomicU64,
    total_wait_ns: AtomicU64,
    max_wait_ns: AtomicU64,
}

impl QueueMetrics {
    fn record(&self, wait: Duration) {
        let nanos = wait.as_nanos().min(u64::MAX as u128) as u64;
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.total_wait_ns.fetch_add(nanos, Ordering::Relaxed);
        self.max_wait_ns.fetch_max(nanos, Ordering::Relaxed);
    }

    fn snapshot(&self) -> QueueStats {
        QueueStats {
            samples: self.samples.load(Ordering::Relaxed),
            total_wait_ns: self.total_wait_ns.load(Ordering::Relaxed),
            max_wait_ns: self.max_wait_ns.load(Ordering::Relaxed),
        }
    }
}

/// bench 用的诊断观察者。
#[derive(Debug, Default)]
pub struct BenchObserver {
    submitted: AtomicU64,
    collisions: AtomicU64,
    peak_in_flight: AtomicU64,
    collisions_by_depth: Mutex<Vec<u64>>,
    batches_by_size: Mutex<Vec<u64>>,
    gather_queue: QueueMetrics,
    eval_queue: QueueMetrics,
    nn_queue: QueueMetrics,
    backprop_queue: QueueMetrics,
}

impl BenchObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> BenchStats {
        BenchStats {
            submitted_playouts: self.submitted.load(Ordering::Acquire),
            collisions: self.collisions.load(Ordering::Acquire),
            peak_in_flight: self.peak_in_flight.load(Ordering::Acquire),
            collisions_by_depth: self.collisions_by_depth.lock().clone(),
            batches_by_size: self.batches_by_size.lock().clone(),
            gather_queue: self.gather_queue.snapshot(),
            eval_queue: self.eval_queue.snapshot(),
            nn_queue: self.nn_queue.snapshot(),
            backprop_queue: self.backprop_queue.snapshot(),
        }
    }
}

impl SearchObserver for BenchObserver {
    type Stamp = InstantQueueStamp;
    const ENABLED: bool = true;

    fn on_submitted(&self) {
        self.submitted.fetch_add(1, Ordering::Relaxed);
    }

    fn on_collision(&self, depth: usize) {
        self.collisions.fetch_add(1, Ordering::AcqRel);
        let mut counts = self.collisions_by_depth.lock();
        if counts.len() <= depth {
            counts.resize(depth + 1, 0);
        }
        counts[depth] += 1;
    }

    fn on_peak_inflight(&self, n: usize) {
        self.peak_in_flight.fetch_max(n as u64, Ordering::Relaxed);
    }

    fn on_queue_wait(&self, kind: QueueKind, wait: Duration) {
        match kind {
            QueueKind::Gather => self.gather_queue.record(wait),
            QueueKind::Eval => self.eval_queue.record(wait),
            QueueKind::Nn => self.nn_queue.record(wait),
            QueueKind::Backprop => self.backprop_queue.record(wait),
        }
    }

    fn on_batch(&self, size: usize) {
        if size == 0 {
            return;
        }
        let mut counts = self.batches_by_size.lock();
        if counts.len() <= size {
            counts.resize(size + 1, 0);
        }
        counts[size] += 1;
    }
}

/// [`BenchObserver::snapshot`] 的诊断汇总。
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BenchStats {
    pub submitted_playouts: u64,
    pub collisions: u64,
    pub peak_in_flight: u64,
    pub collisions_by_depth: Vec<u64>,
    /// 下标 = batch size，值 = 该 size 出现次数；`[0]` 恒为 0。
    pub batches_by_size: Vec<u64>,
    pub gather_queue: QueueStats,
    pub eval_queue: QueueStats,
    pub nn_queue: QueueStats,
    pub backprop_queue: QueueStats,
}
