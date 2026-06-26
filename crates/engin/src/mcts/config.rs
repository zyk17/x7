use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// MCTS 搜索预算。
#[derive(Clone, Debug, Default)]
pub struct MctsBudget {
    pub max_playouts: Option<u32>,
    pub max_nodes: Option<u32>,
    pub deadline: Option<Instant>,
    pub stop: Option<Arc<AtomicBool>>,
}

impl MctsBudget {
    pub fn from_movetime_ms(ms: u64) -> Self {
        Self {
            max_playouts: None,
            max_nodes: None,
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
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            cpuct: 1.0,
            cpuct_root: 1.745,
            cpuct_base: 38_739.0,
            cpuct_factor: 3.894,
            fpu_reduction: 0.22,
            fpu_reduction_root: 1.0,
            root_temperature: 0.0,
            root_dirichlet_epsilon: 0.0,
            root_dirichlet_alpha: 0.3,
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
