use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::policy_onnx::BackendAttributes;

/// MCTS 搜索预算。
#[derive(Clone, Debug, Default)]
pub struct MctsBudget {
    pub max_playouts: Option<u32>,
    pub max_nodes: Option<u32>,
    pub max_depth: Option<u32>,
    pub deadline: Option<Instant>,
    pub stop: Option<Arc<AtomicBool>>,
}

impl MctsBudget {
    pub fn from_movetime_ms(ms: u64) -> Self {
        Self {
            max_playouts: None,
            max_nodes: None,
            max_depth: None,
            deadline: Some(Instant::now() + Duration::from_millis(ms)),
            stop: None,
        }
    }
}

/// lc0 classic 默认 NN 缓存条目数。
pub const DEFAULT_NN_CACHE_SIZE: usize = 200_000;

/// lc0 `FpuStrategy`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpuStrategy {
    Reduction,
    Absolute,
}

/// MCTS 主配置（lc0 classic 内化参数 + 少量 UCI 可配项）。
#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub cpuct: f32,
    pub cpuct_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub fpu_reduction: f32,
    pub fpu_reduction_root: f32,
    pub fpu_strategy: FpuStrategy,
    pub fpu_strategy_root: FpuStrategy,
    pub root_temperature: f32,
    pub root_dirichlet_epsilon: f32,
    pub root_dirichlet_alpha: f32,
    pub minibatch_size: i32,
    pub nn_cache_size: usize,
    pub draw_score: f32,
    pub max_concurrent_searchers: i32,
    pub max_collision_visits: i32,
    pub max_collision_visits_scaling_start: i32,
    pub max_collision_visits_scaling_end: i32,
    pub max_collision_visits_scaling_power: f32,
    pub max_collision_events: i32,
    pub thread_idling_threshold: i32,
    pub idling_minimum_work: i32,
    pub smart_pruning_factor: f32,
    pub root_inflight_fraction: f32,
    pub retry_yield_interval: u32,
    pub retry_sleep_interval: u32,
    pub out_of_order_eval: bool,
    pub max_out_of_order_evals_factor: f32,
    pub max_prefetch: i32,
    pub minimum_work_size_for_processing: i32,
    pub minimum_work_size_for_picking: i32,
    pub minimum_work_per_task_for_processing: i32,
    pub search_spin_backoff: bool,
    pub sticky_endgames: bool,
    pub two_fold_draws: bool,
    /// 0 = 关闭。
    pub nps_limit: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        // lc0 `BaseSearchParams::Populate` (params.cc:543-639)
        let cpuct = 1.745;
        let fpu_reduction = 0.330;
        Self {
            cpuct,
            cpuct_root: cpuct,
            cpuct_base: 38_739.0,
            cpuct_factor: 3.894,
            fpu_reduction,
            fpu_reduction_root: fpu_reduction,
            fpu_strategy: FpuStrategy::Reduction,
            fpu_strategy_root: FpuStrategy::Reduction,
            root_temperature: 0.0,
            root_dirichlet_epsilon: 0.0,
            root_dirichlet_alpha: 0.3,
            minibatch_size: 0,
            nn_cache_size: DEFAULT_NN_CACHE_SIZE,
            draw_score: 0.0,
            max_concurrent_searchers: 1,
            max_collision_visits: 80_000,
            max_collision_visits_scaling_start: 28,
            max_collision_visits_scaling_end: 145_000,
            max_collision_visits_scaling_power: 1.25,
            max_collision_events: 917,
            thread_idling_threshold: 1,
            idling_minimum_work: 0,
            smart_pruning_factor: 1.33,
            root_inflight_fraction: 0.5,
            retry_yield_interval: 64,
            retry_sleep_interval: 512,
            out_of_order_eval: true,
            max_out_of_order_evals_factor: 2.4,
            max_prefetch: 32,
            minimum_work_size_for_processing: 20,
            minimum_work_size_for_picking: 1,
            minimum_work_per_task_for_processing: 8,
            search_spin_backoff: false,
            sticky_endgames: true,
            two_fold_draws: true,
            nps_limit: 0,
        }
    }
}

impl MctsBudget {
    pub fn from_depth(depth: u32) -> Self {
        Self {
            max_playouts: None,
            max_nodes: None,
            max_depth: Some(depth),
            deadline: None,
            stop: None,
        }
    }
}

impl MctsConfig {
    #[inline]
    pub fn effective_minibatch_size(self, attrs: Option<&BackendAttributes>) -> usize {
        let target = if self.minibatch_size == 0 {
            attrs.map(|a| a.recommended_batch_size).unwrap_or(256)
        } else {
            self.minibatch_size as usize
        };
        let max = attrs.map(|a| a.maximum_batch_size).unwrap_or(1024);
        target.clamp(1, max)
    }

    #[inline]
    pub fn max_out_of_order(self, batch_limit: usize) -> usize {
        (self.max_out_of_order_evals_factor * batch_limit as f32).max(1.0) as usize
    }

    #[inline]
    pub fn cpuct_for(self, is_root: bool, parent_visits: u32) -> f32 {
        let init = if is_root { self.cpuct_root } else { self.cpuct };
        if self.cpuct_factor == 0.0 {
            return init;
        }
        let base = self.cpuct_base.max(1.0);
        init + self.cpuct_factor * (((parent_visits as f32) + base) / base).ln()
    }

    /// lc0 `GetFpu`（search.cc:408-424）。
    #[inline]
    pub fn get_fpu(self, is_root: bool, parent_q: f32, visited_policy: f32) -> f32 {
        let value = if is_root {
            self.fpu_reduction_root
        } else {
            self.fpu_reduction
        };
        let strategy = if is_root {
            self.fpu_strategy_root
        } else {
            self.fpu_strategy
        };
        match strategy {
            FpuStrategy::Absolute => value,
            FpuStrategy::Reduction => -parent_q - value * visited_policy.sqrt(),
        }
    }

    #[inline]
    pub fn fpu_for(self, is_root: bool) -> f32 {
        if is_root {
            self.fpu_reduction_root
        } else {
            self.fpu_reduction
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FpuStrategy, MctsConfig};

    #[test]
    fn lc0_default_search_params() {
        let c = MctsConfig::default();
        assert!((c.cpuct - 1.745).abs() < f32::EPSILON);
        assert!((c.fpu_reduction - 0.330).abs() < f32::EPSILON);
        assert!(c.out_of_order_eval);
        assert!((c.max_out_of_order_evals_factor - 2.4).abs() < f32::EPSILON);
        assert_eq!(c.max_prefetch, 32);
        assert_eq!(c.max_concurrent_searchers, 1);
    }

    #[test]
    fn get_fpu_reduction_matches_lc0() {
        let config = MctsConfig::default();
        let parent_q = 0.1f32;
        let visited_policy = 0.25f32;
        let fpu = config.get_fpu(false, parent_q, visited_policy);
        let expected = -parent_q - config.fpu_reduction * visited_policy.sqrt();
        assert!((fpu - expected).abs() < 1e-5);
    }

    #[test]
    fn get_fpu_absolute_strategy() {
        let config = MctsConfig {
            fpu_strategy: FpuStrategy::Absolute,
            fpu_reduction: 0.33,
            ..MctsConfig::default()
        };
        assert!((config.get_fpu(false, 0.5, 0.25) - 0.33).abs() < f32::EPSILON);
    }

    #[test]
    fn max_out_of_order_scales_with_batch() {
        let config = MctsConfig::default();
        assert_eq!(config.max_out_of_order(256), (2.4 * 256.0) as usize);
        assert_eq!(config.max_out_of_order(1), 2);
    }
}
