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
    /// PUCT 风格先验探索系数。
    pub cpuct: f32,
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
            cpuct: 1.25,
            root_temperature: 0.0,
            root_dirichlet_epsilon: 0.0,
            root_dirichlet_alpha: 0.3,
        }
    }
}
