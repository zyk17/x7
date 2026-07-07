use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use xiangqi_core::types::Move;
use xiangqi_core::Color;
use xiangqi_core::Position;

use crate::history::PositionHistory;
use crate::policy_onnx::PolicyOnnx;
use crate::px0_policy::px0_policy_index;

pub type SharedPolicy = Option<Arc<Mutex<PolicyOnnx>>>;
pub(crate) type SharedEvalCache = Arc<Mutex<HashMap<u64, CachedEval>>>;

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
    cache: SharedEvalCache,
    scratch_board: ndarray::Array4<f32>,
    scratch_batch: ndarray::Array4<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedEval {
    logits: Vec<f32>,
    value: f32,
}

impl OnnxPolicyValueEval {
    pub fn new(policy: SharedPolicy) -> Self {
        Self {
            policy,
            cache: Arc::new(Mutex::new(HashMap::new())),
            scratch_board: ndarray::Array4::<f32>::zeros(crate::fen_tensor::PX0_INPUT_SHAPE),
            scratch_batch: ndarray::Array4::<f32>::zeros((0, crate::fen_tensor::PX0_INPUT_SHAPE.1, 10, 9)),
        }
    }

    pub(crate) fn with_shared_cache(policy: SharedPolicy, cache: SharedEvalCache) -> Self {
        Self {
            policy,
            cache,
            scratch_board: ndarray::Array4::<f32>::zeros(crate::fen_tensor::PX0_INPUT_SHAPE),
            scratch_batch: ndarray::Array4::<f32>::zeros((0, crate::fen_tensor::PX0_INPUT_SHAPE.1, 10, 9)),
        }
    }

    pub(crate) fn shared_cache(&self) -> SharedEvalCache {
        Arc::clone(&self.cache)
    }

    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    pub(crate) fn evaluate_many_shared(
        &mut self,
        tasks: &[Arc<PolicyValueTask>],
    ) -> Result<Vec<PolicyValueOutput>, String> {
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
        {
            let cache = self.cache.lock().map_err(|_| "eval cache 锁中毒".to_string())?;
            for (idx, task) in tasks.iter().enumerate() {
                if let Some(cached) = cache.get(&task.history.input_cache_key()) {
                    outputs[idx] = output_from_cached(cached, &task.position, &task.legal_moves);
                } else {
                    misses.push(idx);
                }
            }
        }
        if misses.is_empty() {
            return Ok(outputs);
        }

        let mut unique_misses = Vec::new();
        let mut unique_slots = HashMap::<u64, usize>::new();
        let mut miss_to_unique = Vec::with_capacity(misses.len());
        for &task_idx in &misses {
            let key = tasks[task_idx].history.input_cache_key();
            let slot = if let Some(&slot) = unique_slots.get(&key) {
                slot
            } else {
                let slot = unique_misses.len();
                unique_misses.push(task_idx);
                unique_slots.insert(key, slot);
                slot
            };
            miss_to_unique.push(slot);
        }

        let history_refs = unique_misses.iter().map(|&idx| &tasks[idx].history).collect::<Vec<_>>();
        crate::fen_tensor::histories_to_planes_into(&history_refs, &mut self.scratch_batch)
            .map_err(|e| e.to_string())?;
        let mut net = policy.lock().map_err(|_| "policy 锁中毒".to_string())?;
        let batch = net.eval_boards(&self.scratch_batch).map_err(|e| e.to_string())?;
        if batch.logits.len() != unique_misses.len() {
            return Err(format!(
                "batched policy outputs mismatch: got {} expected {}",
                batch.logits.len(),
                unique_misses.len()
            ));
        }

        let mut cache = self.cache.lock().map_err(|_| "eval cache 锁中毒".to_string())?;
        let mut unique_cached = Vec::with_capacity(unique_misses.len());
        for (slot, &task_idx) in unique_misses.iter().enumerate() {
            let cached = CachedEval {
                logits: batch.logits[slot].clone(),
                value: batch
                    .wdl
                    .as_ref()
                    .and_then(|wdls| wdls.get(slot))
                    .map(|wdl| (wdl[0] - wdl[2]).clamp(-1.0, 1.0))
                    .unwrap_or(0.0),
            };
            cache.insert(tasks[task_idx].history.input_cache_key(), cached.clone());
            unique_cached.push(cached);
        }

        for (&task_idx, &unique_slot) in misses.iter().zip(miss_to_unique.iter()) {
            outputs[task_idx] = output_from_cached(
                &unique_cached[unique_slot],
                &tasks[task_idx].position,
                &tasks[task_idx].legal_moves,
            );
        }

        Ok(outputs)
    }
}

impl PolicyValueEval for OnnxPolicyValueEval {
    type Error = String;

    fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<PolicyValueOutput, Self::Error> {
        let Some(policy) = self.policy.as_ref() else {
            return Ok(uniform_output(input.legal_moves.len()));
        };
        let state_key = input.history.input_cache_key();
        if let Some(cached) = self
            .cache
            .lock()
            .map_err(|_| "eval cache 锁中毒".to_string())?
            .get(&state_key)
            .cloned()
        {
            return Ok(output_from_cached(&cached, &input.position, input.legal_moves));
        }

        crate::fen_tensor::history_to_planes_into(input.history, &mut self.scratch_board).map_err(|e| e.to_string())?;
        let cached = {
            let out = {
                let mut net = policy.lock().map_err(|_| "policy 锁中毒".to_string())?;
                net.eval_board(&self.scratch_board).map_err(|e| e.to_string())?
            };
            CachedEval {
                logits: out.logits,
                value: out.wdl.map(|wdl| (wdl[0] - wdl[2]).clamp(-1.0, 1.0)).unwrap_or(0.0),
            }
        };
        let out = output_from_cached(&cached, &input.position, input.legal_moves);
        self.cache
            .lock()
            .map_err(|_| "eval cache 锁中毒".to_string())?
            .insert(state_key, cached);
        Ok(out)
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
        {
            let cache = self.cache.lock().map_err(|_| "eval cache 锁中毒".to_string())?;
            for (idx, task) in tasks.iter().enumerate() {
                if let Some(cached) = cache.get(&task.history.input_cache_key()) {
                    outputs[idx] = output_from_cached(cached, &task.position, &task.legal_moves);
                } else {
                    misses.push(idx);
                }
            }
        }
        if misses.is_empty() {
            return Ok(outputs);
        }

        let mut unique_misses = Vec::new();
        let mut unique_slots = HashMap::<u64, usize>::new();
        let mut miss_to_unique = Vec::with_capacity(misses.len());
        for &task_idx in &misses {
            let key = tasks[task_idx].history.input_cache_key();
            let slot = if let Some(&slot) = unique_slots.get(&key) {
                slot
            } else {
                let slot = unique_misses.len();
                unique_misses.push(task_idx);
                unique_slots.insert(key, slot);
                slot
            };
            miss_to_unique.push(slot);
        }

        let history_refs = unique_misses.iter().map(|&idx| &tasks[idx].history).collect::<Vec<_>>();
        crate::fen_tensor::histories_to_planes_into(&history_refs, &mut self.scratch_batch)
            .map_err(|e| e.to_string())?;
        let mut net = policy.lock().map_err(|_| "policy 锁中毒".to_string())?;
        let batch = net.eval_boards(&self.scratch_batch).map_err(|e| e.to_string())?;
        if batch.logits.len() != unique_misses.len() {
            return Err(format!(
                "batched policy outputs mismatch: got {} expected {}",
                batch.logits.len(),
                unique_misses.len()
            ));
        }

        let mut cache = self.cache.lock().map_err(|_| "eval cache 锁中毒".to_string())?;
        let mut unique_cached = Vec::with_capacity(unique_misses.len());
        for (slot, &task_idx) in unique_misses.iter().enumerate() {
            let cached = CachedEval {
                logits: batch.logits[slot].clone(),
                value: batch
                    .wdl
                    .as_ref()
                    .and_then(|wdls| wdls.get(slot))
                    .map(|wdl| (wdl[0] - wdl[2]).clamp(-1.0, 1.0))
                    .unwrap_or(0.0),
            };
            cache.insert(tasks[task_idx].history.input_cache_key(), cached.clone());
            unique_cached.push(cached);
        }

        for (&task_idx, &unique_slot) in misses.iter().zip(miss_to_unique.iter()) {
            outputs[task_idx] = output_from_cached(
                &unique_cached[unique_slot],
                &tasks[task_idx].position,
                &tasks[task_idx].legal_moves,
            );
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
