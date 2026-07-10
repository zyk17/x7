use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// MCTS 主配置。
#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    /// 非根节点 PUCT 初始探索系数。
    pub cpuct: f32,
    /// 根节点 PUCT 初始探索系数。
    pub cpuct_root: f32,
    /// PUCT base。
    pub cpuct_base: f32,
    /// PUCT factor。
    pub cpuct_factor: f32,
    /// 非根节点 lc0/px0 风格 first-play urgency reduction。
    pub fpu_reduction: f32,
    /// 根节点 first-play urgency reduction。
    pub fpu_reduction_root: f32,
    /// 根节点温度；UCI 对弈默认通常为 0，自对弈可放宽。
    pub root_temperature: f32,
    /// 根节点探索噪声占比；当前仅保留配置位。
    pub root_dirichlet_epsilon: f32,
    /// 根节点探索噪声 alpha；当前仅保留配置位。
    pub root_dirichlet_alpha: f32,
    /// 单线程搜索每轮 gather 的目标 batch 大小。
    pub search_batch_size: usize,
    /// px0 `DrawScore`；奇偶层 Q 会取反。
    pub draw_score: f32,
    /// px0 `MaxConcurrentSearchers`；0 表示不限制。
    pub max_concurrent_searchers: i32,
    /// px0 `MaxCollisionVisits`。
    pub max_collision_visits: i32,
    pub max_collision_visits_scaling_start: i32,
    pub max_collision_visits_scaling_end: i32,
    pub max_collision_visits_scaling_power: f32,
    /// px0 `ThreadIdlingThreshold` / `IdlingMinimumWork`。
    pub thread_idling_threshold: i32,
    pub idling_minimum_work: i32,
    /// px0 `SmartPruningFactor`；0 关闭根节点剪枝。
    pub smart_pruning_factor: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        let fpu_reduction = 0.22;
        Self {
            cpuct: 1.0,
            cpuct_root: 1.745,
            cpuct_base: 38_739.0,
            cpuct_factor: 3.894,
            fpu_reduction,
            // px0 默认 root FPU 策略为 "same"，实际根节点默认值与非根相同。
            fpu_reduction_root: fpu_reduction,
            root_temperature: 0.0,
            root_dirichlet_epsilon: 0.0,
            root_dirichlet_alpha: 0.3,
            search_batch_size: 2048,
            draw_score: 0.0,
            max_concurrent_searchers: 1,
            max_collision_visits: 80_000,
            max_collision_visits_scaling_start: 28,
            max_collision_visits_scaling_end: 100_000,
            max_collision_visits_scaling_power: 1.0,
            thread_idling_threshold: 1,
            idling_minimum_work: 0,
            smart_pruning_factor: 1.0,
        }
    }
}

impl MctsBudget {
    /// UCI `go depth`：平均 depth 达到该值后停止。
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
    pub fn cpuct_for(self, is_root: bool, parent_visits: u32) -> f32 {
        let init = if is_root { self.cpuct_root } else { self.cpuct };
        if self.cpuct_factor == 0.0 {
            return init;
        }
        let base = self.cpuct_base.max(1.0);
        init + self.cpuct_factor * (((parent_visits as f32) + base) / base).ln()
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
