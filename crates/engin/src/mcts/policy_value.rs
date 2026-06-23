use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use xiangqi_core::types::Move;
use xiangqi_core::write_move_uci_bytes;
use xiangqi_core::Position;
use xiangqi_core::Color;

use crate::policy_onnx::PolicyOnnx;
use crate::px0_policy::px0_policy_index;

pub type SharedPolicy = Option<Arc<Mutex<PolicyOnnx>>>;

/// 评估输入。
pub struct PolicyValueInput<'a> {
    pub position: &'a Position,
    pub legal_moves: &'a [Move],
}

/// 单次网络评估输出。
#[derive(Clone, Debug, Default)]
pub struct PolicyValueOutput {
    /// 与 `legal_moves` 对齐的先验分布。
    pub priors: Vec<f32>,
    /// 当前行棋方视角 q = w - l，范围预期为 [-1, 1]。
    pub value: f32,
}

/// MCTS 对网络评估器的最小依赖接口。
pub trait PolicyValueEval {
    type Error;

    fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<PolicyValueOutput, Self::Error>;
}

/// 复用现有 `PolicyOnnx` 的最小 MCTS 评估桥。
pub struct OnnxPolicyValueEval<'a> {
    pub policy: &'a SharedPolicy,
    pub vocab: &'a HashMap<String, usize>,
}

impl<'a> PolicyValueEval for OnnxPolicyValueEval<'a> {
    type Error = String;

    fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<PolicyValueOutput, Self::Error> {
        let Some(policy) = self.policy.as_ref() else {
            return Ok(uniform_output(input.legal_moves.len()));
        };
        let out = {
            let mut net = policy.lock().map_err(|_| "policy 锁中毒".to_string())?;
            net.eval_position(input.position).map_err(|e| e.to_string())?
        };

        let mut priors = Vec::with_capacity(input.legal_moves.len());
        let mut scratch = [0u8; 8];
        let black_to_move = input.position.side_to_move == Color::Black;
        for mv in input.legal_moves {
            let len = write_move_uci_bytes(*mv, &mut scratch);
            let u = std::str::from_utf8(&scratch[..len]).map_err(|e| e.to_string())?;
            let prior = self
                .vocab
                .get(u)
                .copied()
                .or_else(|| px0_policy_index(*mv, black_to_move))
                .and_then(|idx| out.logits.get(idx))
                .copied()
                .unwrap_or(0.0);
            priors.push(prior);
        }

        normalize_priors(&mut priors);

        Ok(PolicyValueOutput {
            priors,
            value: out
                .wdl
                .map(|wdl| (wdl[0] - wdl[2]).clamp(-1.0, 1.0))
                .unwrap_or(0.0),
        })
    }
}

fn uniform_output(len: usize) -> PolicyValueOutput {
    let p = if len == 0 { 0.0 } else { 1.0 / len as f32 };
    PolicyValueOutput {
        priors: vec![p; len],
        value: 0.0,
    }
}

fn normalize_priors(priors: &mut [f32]) {
    if priors.is_empty() {
        return;
    }

    let max_logit = priors.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        let p = 1.0 / priors.len() as f32;
        priors.fill(p);
        return;
    }

    let mut sum = 0.0;
    for prior in priors.iter_mut() {
        *prior = (*prior - max_logit).exp();
        sum += *prior;
    }

    if sum <= 0.0 || !sum.is_finite() {
        let p = 1.0 / priors.len() as f32;
        priors.fill(p);
        return;
    }

    for prior in priors.iter_mut() {
        *prior /= sum;
    }
}
