//! 搜索分层配置：
//!
//! - `SearchParams`：算法旋钮（PUCT / FPU / virtual mean / Bvar / 根决策）。`Copy`，热路径只带这一包。
//! - `SearchConfig`：当前固定 worker 实现的线程、队列、batch、window 配置 + 嵌套的 `SearchParams`。
//! - `SearchLimits`（在 `pipeline`）：这一手 `go` 的停止条件与 `searchmoves`。
//! - `Options`（`options.rs`）：UCI / 引擎生命周期；`go` 时拍快照写入上面几层。

use super::decision::DecisionRule;
use crate::neural::backend::Backend;

/// MCTS 算法旋钮。不包含线程、队列或 `searchmoves`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParams {
    pub cpuct: f32,
    pub cpuct_base: f32,   // 增长何时开始。更小 → 更早、更快变宽；更大 → 更久保持利用 Q。
    pub cpuct_factor: f32, // 增长幅度。更大 → 后期更强地向 PUCT/P 分流；更小 → 后期更容易让已验证的高 Q 分支继续积累 N。
    pub fpu_reduction: f32,
    /// 不受 prior/PUCT U 抑制的 completed-evidence 标准误 bonus；0 为关闭。
    pub variance_bonus_scale: f32,
    /// reservation 暂时以 `scale * FPU` 进入 action Q；0 退化为纯 virtual visit。
    pub virtual_mean_fpu_scale: f32,
    /// `Lcb` 根决策的下置信半径倍数。
    pub decision_lcb_stdevs: f32,
    /// `Ucb` 根决策的上置信半径倍数。
    pub decision_ucb_stdevs: f32,
    /// 根最终选边规则；只读 completed evidence。
    pub decision_rule: DecisionRule,
    /// `MixNQ` 中归一化 N 的权重，单位与 Q 相同。
    pub decision_mix_n_weight: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            cpuct: 2.4,
            cpuct_base: 40_000.0,
            cpuct_factor: 0.0,
            // 小网络可能有系统性偏差；降低未知 edge 的首次进入门槛。
            fpu_reduction: 0.225,
            variance_bonus_scale: 1.5,
            virtual_mean_fpu_scale: 1.0,
            // 根最终 Decision 的温和一倍 SE 置信修正；不参与 PUCT。
            decision_lcb_stdevs: 1.0,
            decision_ucb_stdevs: 1.0,
            decision_rule: DecisionRule::Auto,
            // N 已归一化；最多提供 0.25 个 Q 单位的访问量偏好。
            decision_mix_n_weight: 0.25,
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
            self.variance_bonus_scale.is_finite() && self.variance_bonus_scale >= 0.0,
            "stream variance bonus scale must be finite and non-negative"
        );
        assert!(
            self.virtual_mean_fpu_scale.is_finite() && self.virtual_mean_fpu_scale >= 0.0,
            "stream virtual mean FPU scale must be finite and non-negative"
        );
        assert!(
            self.decision_lcb_stdevs.is_finite() && self.decision_lcb_stdevs >= 0.0,
            "stream decision LCB stdevs must be finite and non-negative"
        );
        assert!(
            self.decision_ucb_stdevs.is_finite() && self.decision_ucb_stdevs >= 0.0,
            "stream decision UCB stdevs must be finite and non-negative"
        );
        assert!(
            self.decision_mix_n_weight.is_finite() && self.decision_mix_n_weight >= 0.0,
            "stream decision MixNQ N weight must be finite and non-negative"
        );
    }
}

/// 当前 worker pool 的 job 配置。算法旋钮在 `params`；`searchmoves` 在 `SearchLimits`。
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConfig {
    /// Search/Eval/NN 队列深度。`0` 表示 `max(4096, 64 * resolved_batch)`。
    pub queue_capacity: usize,
    /// 已有多个编码局面时的 NN GPU 合批大小。`0` 表示 backend 的
    /// `recommended_batch_size`。
    pub eval_batch_size: usize,
    /// 真正已提交 NN 的请求并发上限：`limit = ceil(NnBatchSize × nn_window)`。
    /// cache/terminal 不占 slot；NN 结果完成 backprop 后才释放。调大可能提高 eps，调小让
    /// Gather 更贴最新统计。
    pub nn_window: f32,
    pub params: SearchParams,
    /// 通用 CPU worker 数；每个 worker 按就绪队列处理 Gather、Expand、Eval、NN 回包、Backprop，
    /// 后续也承接 Proof。
    pub cpu_workers: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 0,
            eval_batch_size: 0,
            nn_window: 2.25,
            params: SearchParams::default(),
            cpu_workers: 8,
        }
    }
}

impl SearchConfig {
    pub(crate) fn validate(&self) {
        self.params.validate();
        assert!(self.cpu_workers > 0, "stream requires at least one CPU worker");
        assert!(
            self.nn_window.is_finite() && self.nn_window > 0.0,
            "stream nn window factor must be finite and positive"
        );
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
            cpu_workers: self.cpu_workers,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedSearchConfig {
    pub(crate) queue_capacity: usize,
    pub(crate) eval_batch_size: usize,
    pub(crate) eval_claim_limit: usize,
    pub(crate) params: SearchParams,
    pub(crate) cpu_workers: usize,
}
