//! NN backend 边界：属性、cache 与批量评估结果。
//!
//! 正式推理走 ONNX，测试可用 `UniformBackend`。

use crate::EnginError;

use super::{InputPlanes, POLICY_SIZE};

/// Backend 的 batch 大小边界。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendAttributes {
    pub recommended_batch_size: usize,
    pub maximum_batch_size: usize,
}

impl Default for BackendAttributes {
    fn default() -> Self {
        Self { recommended_batch_size: 1, maximum_batch_size: 1 }
    }
}

/// 单次局面评估输出：policy、WDL、moves-left。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvalResult {
    /// 当前行棋方的胜率减负率。
    pub wl: f32,
    /// 当前行棋方的和棋概率。
    pub d: f32,
    /// 预测到结果的距离，单位为 ply（半回合）。
    ///
    /// ONNX 输出名为 `moves_left`，但训练记录与搜索回传均以 ply 而非完整回合度量此值。
    pub plies_left: f32,
    /// 与传入合法着列表对齐的概率。
    pub policies: Vec<f32>,
}

/// Backend 评估边界：属性与 stream NN worker 的稀疏 batch 推理。
pub trait Backend: Send + Sync {
    fn attributes(&self) -> BackendAttributes;

    /// stream NN worker：稀疏 `InputPlanes` 合批推理。
    fn infer_input_planes_into(
        &self,
        samples: &[InputPlanes],
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError>;
}

/// 测试用：均匀 policy + 固定 WDL。
#[derive(Clone, Debug)]
pub struct UniformBackend {
    pub wl: f32,
    pub d: f32,
    pub plies_left: f32,
}

impl Default for UniformBackend {
    fn default() -> Self {
        Self { wl: 0.0, d: 0.0, plies_left: 0.0 }
    }
}

impl UniformBackend {
    pub fn with_wdl(wl: f32, d: f32, plies_left: f32) -> Self {
        Self { wl, d, plies_left }
    }
}

impl Backend for UniformBackend {
    fn attributes(&self) -> BackendAttributes {
        BackendAttributes::default()
    }

    fn infer_input_planes_into(
        &self,
        samples: &[InputPlanes],
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError> {
        let batch = samples.len();
        // 相等 logits → Eval softmax 在合法着上均匀分布。
        logits.clear();
        logits.resize(batch * POLICY_SIZE, 0.0);
        wdl.clear();
        wdl.reserve(batch * 3);
        let win = (1.0 - self.d + self.wl) * 0.5;
        let loss = (1.0 - self.d - self.wl) * 0.5;
        for _ in 0..batch {
            wdl.extend_from_slice(&[win, self.d, loss]);
        }
        moves_left.clear();
        moves_left.resize(batch, self.plies_left);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_inference_keeps_moves_left_per_position() {
        let backend = UniformBackend::with_wdl(0.0, 0.0, 17.0);
        let samples = vec![[super::super::InputPlane::default(); super::super::INPUT_PLANES]; 2];
        let mut logits = Vec::new();
        let mut wdl = Vec::new();
        let mut moves_left = Vec::new();
        backend.infer_input_planes_into(&samples, &mut logits, &mut wdl, &mut moves_left).expect("infer");
        assert_eq!(moves_left, vec![17.0, 17.0]);
    }

    #[test]
    fn uniform_inference_uses_configured_wdl() {
        let backend = UniformBackend::with_wdl(0.4, 0.2, 0.0);
        let samples = vec![[super::super::InputPlane::default(); super::super::INPUT_PLANES]];
        let mut logits = Vec::new();
        let mut wdl = Vec::new();
        let mut moves_left = Vec::new();
        backend.infer_input_planes_into(&samples, &mut logits, &mut wdl, &mut moves_left).expect("infer");
        assert_eq!(wdl, vec![0.6, 0.2, 0.2]);
    }
}
