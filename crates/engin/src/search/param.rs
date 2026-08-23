//! 搜索分层配置：
//!
//! - `SearchParams`：算法旋钮（PUCT / FPU / virtual mean / 根 LCB）。`Copy`，热路径只带这一包。
//! - `SearchConfig`：当前固定 worker 实现的线程、队列、batch、window 配置 + 嵌套的 `SearchParams`。
//! - `SearchLimits`（在 `pipeline`）：这一手 `go` 的停止条件与 `searchmoves`。
//! - `Options`（`options.rs`）：UCI / 引擎生命周期；`go` 时拍快照写入上面几层。

use crate::neural::backend::Backend;

/// MCTS 算法旋钮。不包含线程、队列或 `searchmoves`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParams {
    pub cpuct: f32,
    pub cpuct_base: f32,   // 增长何时开始。更小 → 更早、更快变宽；更大 → 更久保持利用 Q。
    pub cpuct_factor: f32, // 增长幅度。更大 → 后期更强地向 PUCT/P 分流；更小 → 后期更容易让已验证的高 Q 分支继续积累 N。
    pub fpu_reduction: f32,
    /// reservation 暂时以 `scale * FPU` 进入 action Q；0 退化为纯 virtual visit。
    pub virtual_mean_fpu_scale: f32,
    /// 根最终选边的 LCB 半径倍数；0 表示退回既有 N→Q→P 排名。
    pub lcb_stdevs: f32,
    /// LCB 候选至少须达到 N 第一候选的这一 completed-N 比例。
    pub lcb_min_visit_fraction: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            cpuct: 1.75,
            cpuct_base: 40_000.0,
            cpuct_factor: 4.0,
            // 小网络可能有系统性偏差；降低未知 edge 的首次进入门槛。
            fpu_reduction: 0.200,
            virtual_mean_fpu_scale: 1.0,
            // KataGo 搜索参数的经验起点；只用于根最终 Decision，不参与 PUCT。
            lcb_stdevs: 5.0,
            lcb_min_visit_fraction: 0.15,
        }
    }
}

impl SearchParams {
    pub(crate) fn validate(self) {
        assert!(
            self.cpuct.is_finite() && self.cpuct >= 0.0,
            "stream cpuct must be finite and non-negative"
        );
        assert!(
            self.cpuct_base.is_finite() && self.cpuct_base > 0.0,
            "stream cpuct base must be finite and positive"
        );
        assert!(
            self.cpuct_factor.is_finite() && self.cpuct_factor >= 0.0,
            "stream cpuct factor must be finite and non-negative"
        );
        assert!(
            self.fpu_reduction.is_finite() && self.fpu_reduction >= 0.0,
            "stream FPU reduction must be finite and non-negative"
        );
        assert!(
            self.virtual_mean_fpu_scale.is_finite() && self.virtual_mean_fpu_scale >= 0.0,
            "stream virtual mean FPU scale must be finite and non-negative"
        );
        assert!(
            self.lcb_stdevs.is_finite() && self.lcb_stdevs >= 0.0,
            "stream LCB stdevs must be finite and non-negative"
        );
        assert!(
            self.lcb_min_visit_fraction.is_finite() && (0.0..=1.0).contains(&self.lcb_min_visit_fraction),
            "stream LCB minimum visit fraction must be within [0, 1]"
        );
    }
}

/// 当前固定 worker pool 的 job 配置。算法旋钮在 `params`；`searchmoves` 在 `SearchLimits`。
/// Gather/Eval 的静态比例来自当前实验；后续动态调度可按队列压力在二者及 proof 间分配 CPU，
/// 而不改变搜索语义。
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Search/Eval/NN 队列深度。`0` 表示 `max(4096, 64 * resolved_batch)`。
    pub queue_capacity: usize,
    /// 已有多个编码局面时的 NN GPU 合批大小。`0` 表示 backend 的
    /// `recommended_batch_size`。
    pub eval_batch_size: usize,
    /// Eval claim 并发上限：`limit = ceil(MiniBatchSize × nn_window)`。
    /// Claim 在 backprop 写完 N/Q 后释放；调大可能提高eps、调小让 Gather 更贴最新统计。
    pub nn_window: f32,
    pub params: SearchParams,
    /// 当前固定 pool 的 Gather worker 数。
    pub gather_workers: usize,
    /// 当前固定 pool 的 Eval worker 数。它负责准备、缓存、合法着；NN inference 是单独的单 worker。
    pub eval_workers: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 0,
            eval_batch_size: 0,
            nn_window: 2.25,
            params: SearchParams::default(),
            gather_workers: 3,
            eval_workers: 5,
        }
    }
}

impl SearchConfig {
    pub(crate) fn validate(&self) {
        self.params.validate();
        assert!(self.gather_workers > 0, "stream requires at least one gather worker");
        assert!(self.eval_workers > 0, "stream requires at least one eval worker");
        assert!(
            self.nn_window.is_finite() && self.nn_window > 0.0,
            "stream nn window factor must be finite and positive"
        );
    }

    /// UCI `Threads` 尽量按 Gather:Eval = 1:2；除不尽时多给 Gather。
    pub(crate) fn gather_eval_from_threads(threads: usize) -> (usize, usize) {
        let eval = ((threads * 2) / 3).max(1);
        let gather = threads.saturating_sub(eval).max(1);
        (gather, eval)
    }

    /// 填充0配置, 推算具体队列/批量大小。
    pub(crate) fn resolve(&self, backend: &dyn Backend) -> ResolvedSearchConfig {
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
pub(crate) struct ResolvedSearchConfig {
    pub(crate) queue_capacity: usize,
    pub(crate) eval_batch_size: usize,
    pub(crate) eval_claim_limit: usize,
    pub(crate) params: SearchParams,
    pub(crate) gather_workers: usize,
    pub(crate) eval_workers: usize,
}
