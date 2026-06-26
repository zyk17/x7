use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use xiangqi_core::types::Move;
use xiangqi_core::Color;
use xiangqi_core::Position;

use crate::history::PositionHistory;
use crate::policy_onnx::PolicyOnnx;
use crate::px0_policy::px0_policy_index;

pub type SharedPolicy = Option<Arc<Mutex<PolicyOnnx>>>;

/// 评估输入。
pub struct PolicyValueInput<'a> {
    pub position: &'a Position,
    pub history: &'a PositionHistory,
    pub legal_moves: &'a [Move],
}

#[derive(Clone)]
pub struct PolicyValueTask {
    pub position: Position,
    pub history: PositionHistory,
    pub legal_moves: Vec<Move>,
}

impl PolicyValueTask {
    pub fn as_input(&self) -> PolicyValueInput<'_> {
        PolicyValueInput {
            position: &self.position,
            history: &self.history,
            legal_moves: &self.legal_moves,
        }
    }
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

    fn evaluate_many(&mut self, tasks: &[PolicyValueTask]) -> Result<Vec<PolicyValueOutput>, Self::Error> {
        let mut out = Vec::with_capacity(tasks.len());
        for task in tasks {
            out.push(self.evaluate(task.as_input())?);
        }
        Ok(out)
    }
}

/// 复用现有 `PolicyOnnx` 的最小 MCTS 评估桥。
pub struct OnnxPolicyValueEval {
    pub policy: SharedPolicy,
    cache: HashMap<u64, CachedEval>,
}

#[derive(Clone, Debug)]
struct CachedEval {
    logits: Vec<f32>,
    value: f32,
}

impl OnnxPolicyValueEval {
    pub fn new(policy: SharedPolicy) -> Self {
        Self {
            policy,
            cache: HashMap::new(),
        }
    }
}

impl PolicyValueEval for OnnxPolicyValueEval {
    type Error = String;

    fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<PolicyValueOutput, Self::Error> {
        let Some(policy) = self.policy.as_ref() else {
            return Ok(uniform_output(input.legal_moves.len()));
        };
        let state_key = input.position.key();
        let cached = if let Some(cached) = self.cache.get(&state_key) {
            cached.clone()
        } else {
            let out = {
                let mut net = policy.lock().map_err(|_| "policy 锁中毒".to_string())?;
                net.eval_history(input.history).map_err(|e| e.to_string())?
            };
            let cached = CachedEval {
                logits: out.logits,
                value: out.wdl.map(|wdl| (wdl[0] - wdl[2]).clamp(-1.0, 1.0)).unwrap_or(0.0),
            };
            self.cache.insert(state_key, cached.clone());
            cached
        };

        let mut priors = Vec::with_capacity(input.legal_moves.len());
        let black_to_move = input.position.side_to_move == Color::Black;
        for mv in input.legal_moves {
            let prior = px0_policy_index(*mv, black_to_move)
                .and_then(|idx| cached.logits.get(idx))
                .copied()
                .unwrap_or(0.0);
            priors.push(prior);
        }

        normalize_priors(&mut priors);

        Ok(PolicyValueOutput {
            priors,
            value: cached.value,
        })
    }

    fn evaluate_many(&mut self, tasks: &[PolicyValueTask]) -> Result<Vec<PolicyValueOutput>, Self::Error> {
        if tasks.is_empty() {
            return Ok(Vec::new());
        }
        let Some(policy) = self.policy.as_ref() else {
            return Ok(tasks
                .iter()
                .map(|task| uniform_output(task.legal_moves.len()))
                .collect());
        };

        let mut outputs = vec![PolicyValueOutput::default(); tasks.len()];
        let mut misses = Vec::new();
        for (idx, task) in tasks.iter().enumerate() {
            if let Some(cached) = self.cache.get(&task.position.key()).cloned() {
                outputs[idx] = output_from_cached(&cached, &task.position, &task.legal_moves);
            } else {
                misses.push(idx);
            }
        }
        if misses.is_empty() {
            return Ok(outputs);
        }

        let mut net = policy.lock().map_err(|_| "policy 锁中毒".to_string())?;
        let batch = net
            .eval_histories(misses.iter().map(|&idx| &tasks[idx].history))
            .map_err(|e| e.to_string())?;
        if batch.logits.len() != misses.len() {
            return Err(format!(
                "batched policy outputs mismatch: got {} expected {}",
                batch.logits.len(),
                misses.len()
            ));
        }

        for (slot, &task_idx) in misses.iter().enumerate() {
            let cached = CachedEval {
                logits: batch.logits[slot].clone(),
                value: batch
                    .wdl
                    .as_ref()
                    .and_then(|wdls| wdls.get(slot))
                    .map(|wdl| (wdl[0] - wdl[2]).clamp(-1.0, 1.0))
                    .unwrap_or(0.0),
            };
            self.cache.insert(tasks[task_idx].position.key(), cached.clone());
            outputs[task_idx] = output_from_cached(&cached, &tasks[task_idx].position, &tasks[task_idx].legal_moves);
        }

        Ok(outputs)
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

fn output_from_cached(cached: &CachedEval, position: &Position, legal_moves: &[Move]) -> PolicyValueOutput {
    let mut priors = Vec::with_capacity(legal_moves.len());
    let black_to_move = position.side_to_move == Color::Black;
    for mv in legal_moves {
        let prior = px0_policy_index(*mv, black_to_move)
            .and_then(|idx| cached.logits.get(idx))
            .copied()
            .unwrap_or(0.0);
        priors.push(prior);
    }
    normalize_priors(&mut priors);
    PolicyValueOutput {
        priors,
        value: cached.value,
    }
}
