//! px0 `src/neural/backend.h` 的 P3 确定性 stub（无 ONNX）。

use xiangqi_core::{Move, PositionHistory};

/// px0 `EvalResult` / backend 评估输出子集。
#[derive(Clone, Debug, PartialEq)]
pub struct EvalResult {
    pub wl: f32,
    pub d: f32,
    pub m: f32,
    pub policies: Vec<f32>,
}

/// px0 `Backend` 评估边界（P3 单线程）。
pub trait Backend: Send + Sync {
    fn evaluate(&self, history: &PositionHistory, legal_moves: &[Move]) -> EvalResult;
    fn recommended_batch_size(&self) -> usize {
        1
    }
}

/// 测试用：均匀 policy + 固定 WDL。
#[derive(Clone, Debug)]
pub struct UniformBackend {
    pub wl: f32,
    pub d: f32,
    pub m: f32,
}

impl Default for UniformBackend {
    fn default() -> Self {
        Self {
            wl: 0.0,
            d: 0.0,
            m: 0.0,
        }
    }
}

impl Backend for UniformBackend {
    fn evaluate(&self, _history: &PositionHistory, legal_moves: &[Move]) -> EvalResult {
        let count = legal_moves.len().max(1);
        let p = 1.0 / count as f32;
        EvalResult {
            wl: self.wl,
            d: self.d,
            m: self.m,
            policies: vec![p; legal_moves.len()],
        }
    }
}
