//! Stream 的选择参数。
//!
//! LC3 Policy 描述 worker/policy 架构，但未公开具体 PUCT 公式。因此在有
//! stream 原生公式前，默认值保持项目批准的 X7 策略
//! （px0 `src/search/classic/search.cc:408-433`）。

use crate::utils::fastmath::fast_log;

/// Stream 自己拥有的最小选择参数集。
///
/// 不包含 root 专用参数、absolute FPU、draw score、contempt 或旧
/// task-worker 参数；stream 固定使用中性和棋分数与 reduction FPU。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParams {
    pub cpuct: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub fpu_reduction: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            cpuct: 1.0,
            cpuct_base: 38_739.0,
            cpuct_factor: 3.894,
            fpu_reduction: 0.220,
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
        assert!(self.cpuct_factor.is_finite(), "stream cpuct factor must be finite");
        assert!(
            self.fpu_reduction.is_finite() && self.fpu_reduction >= 0.0,
            "stream FPU reduction must be finite and non-negative"
        );
    }
}

/// 项目批准的 PUCT 对齐；等待公开的 LC3 公式。
pub(crate) fn compute_cpuct(params: SearchParams, visits: u32) -> f32 {
    if params.cpuct_factor == 0.0 {
        params.cpuct
    } else {
        params.cpuct + params.cpuct_factor * fast_log((visits as f32 + params.cpuct_base) / params.cpuct_base)
    }
}

// 尚未定义并验证事件/生命周期语义的能力不预留字段：OOO evaluation、MultiPV、
// prefetch 与 DAG reuse。

#[cfg(test)]
mod tests {
    use super::{compute_cpuct, SearchParams};

    #[test]
    fn defaults_preserve_the_approved_x7_policy() {
        let params = SearchParams::default();
        assert_eq!(params.cpuct, 1.0);
        assert_eq!(params.cpuct_base, 38_739.0);
        assert_eq!(params.cpuct_factor, 3.894);
        assert_eq!(params.fpu_reduction, 0.220);
        assert_eq!(compute_cpuct(params, 0), params.cpuct);
    }

    #[test]
    fn zero_cpuct_factor_keeps_the_initial_value() {
        let params = SearchParams {
            cpuct: 1.25,
            cpuct_factor: 0.0,
            ..SearchParams::default()
        };
        assert_eq!(compute_cpuct(params, 10_000), 1.25);
    }
}
