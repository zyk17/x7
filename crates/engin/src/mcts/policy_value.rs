use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use xiangqi_core::types::Move;
use xiangqi_core::Color;
use xiangqi_core::Position;

use crate::history::PositionHistory;
use crate::policy_onnx::{PolicyOnnx, PolicySessionPool};
use crate::move_vocab::move_vocab_index;

pub type SharedPolicy = Option<Arc<PolicySessionPool>>;
pub(crate) type SharedEvalCache = Arc<EvalCache>;

trait TaskLike {
    fn position(&self) -> &Position;
    fn history(&self) -> &PositionHistory;
    fn legal_moves(&self) -> &[Move];
}

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

impl TaskLike for PolicyValueTask {
    fn position(&self) -> &Position {
        &self.position
    }

    fn history(&self) -> &PositionHistory {
        &self.history
    }

    fn legal_moves(&self) -> &[Move] {
        &self.legal_moves
    }
}

impl TaskLike for Arc<PolicyValueTask> {
    fn position(&self) -> &Position {
        &self.position
    }

    fn history(&self) -> &PositionHistory {
        &self.history
    }

    fn legal_moves(&self) -> &[Move] {
        &self.legal_moves
    }
}

/// 单次网络评估输出（fetch 后 wl 为 parent-move 视角）。
#[derive(Clone, Debug, Default)]
pub struct PolicyValueOutput {
    pub priors: Vec<f32>,
    pub wl: f32,
    pub d: f32,
    pub m: f32,
    /// 与 `wl` 相同，保留旧调用方兼容。
    pub value: f32,
}

impl PolicyValueOutput {
    pub fn from_stm_wdl(priors: Vec<f32>, wdl: [f32; 3]) -> Self {
        let wl_stm = (wdl[0] - wdl[2]).clamp(-1.0, 1.0);
        let wl = -wl_stm;
        let d = wdl[1].clamp(0.0, 1.0);
        Self {
            priors,
            wl,
            d,
            m: 0.0,
            value: wl,
        }
    }
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

    fn evaluate_many_arcs(&mut self, tasks: &[Arc<PolicyValueTask>]) -> Result<Vec<PolicyValueOutput>, Self::Error> {
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
    session: Option<Arc<Mutex<PolicyOnnx>>>,
    cache: SharedEvalCache,
    scratch_board: ndarray::Array4<f32>,
    scratch_batch: ndarray::Array4<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct CachedEval {
    logits: Vec<f32>,
    wdl: [f32; 3],
}

pub(crate) struct EvalCache {
    shards: Vec<Mutex<HashMap<u64, CachedEval>>>,
    lookup_hits: AtomicU64,
    lookup_misses: AtomicU64,
    miss_keys: AtomicU64,
}

impl EvalCache {
    fn new(shards: usize) -> Self {
        let shard_count = shards.max(1).next_power_of_two();
        let mut maps = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            maps.push(Mutex::new(HashMap::new()));
        }
        Self {
            shards: maps,
            lookup_hits: AtomicU64::new(0),
            lookup_misses: AtomicU64::new(0),
            miss_keys: AtomicU64::new(0),
        }
    }

    fn shard_index(&self, key: u64) -> usize {
        (key as usize) & (self.shards.len() - 1)
    }

    fn get(&self, key: u64) -> Option<CachedEval> {
        let shard = self.shards[self.shard_index(key)].lock().ok()?;
        let out = shard.get(&key).cloned();
        if out.is_some() {
            self.lookup_hits.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn record_lookup_miss(&self, n: u64) {
        if n > 0 {
            self.lookup_misses.fetch_add(n, Ordering::Relaxed);
        }
    }

    fn record_miss_keys(&self, n: u64) {
        if n > 0 {
            self.miss_keys.fetch_add(n, Ordering::Relaxed);
        }
    }

    fn insert(&self, key: u64, value: CachedEval) {
        if let Ok(mut shard) = self.shards[self.shard_index(key)].lock() {
            shard.insert(key, value);
        }
    }

    fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut map) = shard.lock() {
                map.clear();
            }
        }
        self.lookup_hits.store(0, Ordering::Relaxed);
        self.lookup_misses.store(0, Ordering::Relaxed);
        self.miss_keys.store(0, Ordering::Relaxed);
    }

    fn stats(&self) -> EvalCacheStats {
        let lookup_hits = self.lookup_hits.load(Ordering::Relaxed);
        let lookup_misses = self.lookup_misses.load(Ordering::Relaxed);
        let miss_keys = self.miss_keys.load(Ordering::Relaxed);
        EvalCacheStats {
            lookup_hits,
            lookup_misses,
            miss_keys,
            // 对外兼容：hits/misses 统一保持 lookup 口径（可直接比较）。
            hits: lookup_hits,
            misses: lookup_misses,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub lookup_hits: u64,
    pub lookup_misses: u64,
    pub miss_keys: u64,
}

impl OnnxPolicyValueEval {
    pub fn new(policy: SharedPolicy) -> Self {
        Self {
            session: policy.as_ref().map(|pool| pool.primary()),
            policy,
            cache: Arc::new(EvalCache::new(16)),
            scratch_board: ndarray::Array4::<f32>::zeros(crate::fen_tensor::PX0_INPUT_SHAPE),
            scratch_batch: ndarray::Array4::<f32>::zeros((0, crate::fen_tensor::PX0_INPUT_SHAPE.1, 10, 9)),
        }
    }

    pub(crate) fn with_shared_cache(policy: SharedPolicy, cache: SharedEvalCache) -> Self {
        Self {
            session: policy.as_ref().map(|pool| pool.primary()),
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
        self.cache.clear();
    }

    pub fn cache_stats(&self) -> EvalCacheStats {
        self.cache.stats()
    }

    fn evaluate_many_impl<T: TaskLike>(&mut self, tasks: &[T]) -> Result<Vec<PolicyValueOutput>, String> {
        if tasks.is_empty() {
            return Ok(Vec::new());
        }
        let Some(session) = self.session.as_ref() else {
            return Ok(tasks
                .iter()
                .map(|task| uniform_output(task.legal_moves().len()))
                .collect());
        };

        let mut outputs = vec![PolicyValueOutput::default(); tasks.len()];
        let mut misses = Vec::new();
        for (idx, task) in tasks.iter().enumerate() {
            if let Some(cached) = self.cache.get(task.history().input_cache_key()) {
                outputs[idx] = output_from_cached(&cached, task.position(), task.legal_moves());
            } else {
                misses.push(idx);
            }
        }
        self.cache.record_lookup_miss(misses.len() as u64);
        if misses.is_empty() {
            return Ok(outputs);
        }

        let mut unique_misses = Vec::new();
        let mut unique_slots = HashMap::<u64, usize>::new();
        let mut miss_to_unique = Vec::with_capacity(misses.len());
        for &task_idx in &misses {
            let key = tasks[task_idx].history().input_cache_key();
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
        self.cache.record_miss_keys(unique_misses.len() as u64);

        let history_refs = unique_misses
            .iter()
            .map(|&idx| tasks[idx].history())
            .collect::<Vec<_>>();
        crate::fen_tensor::histories_to_planes_into(&history_refs, &mut self.scratch_batch)
            .map_err(|e| e.to_string())?;
        let mut net = session.lock().map_err(|_| "policy 锁中毒".to_string())?;
        let batch = net.eval_boards(&self.scratch_batch).map_err(|e| e.to_string())?;
        if batch.logits.len() != unique_misses.len() {
            return Err(format!(
                "batched policy outputs mismatch: got {} expected {}",
                batch.logits.len(),
                unique_misses.len()
            ));
        }

        let mut unique_cached = Vec::with_capacity(unique_misses.len());
        for (slot, &task_idx) in unique_misses.iter().enumerate() {
            let wdl = batch
                .wdl
                .as_ref()
                .and_then(|wdls| wdls.get(slot))
                .copied()
                .unwrap_or([0.5, 0.0, 0.5]);
            let cached = CachedEval {
                logits: batch.logits[slot].clone(),
                wdl,
            };
            self.cache
                .insert(tasks[task_idx].history().input_cache_key(), cached.clone());
            unique_cached.push(cached);
        }

        for (&task_idx, &unique_slot) in misses.iter().zip(miss_to_unique.iter()) {
            outputs[task_idx] = output_from_cached(
                &unique_cached[unique_slot],
                tasks[task_idx].position(),
                tasks[task_idx].legal_moves(),
            );
        }

        Ok(outputs)
    }
}

impl PolicyValueEval for OnnxPolicyValueEval {
    type Error = String;

    fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<PolicyValueOutput, Self::Error> {
        let Some(session) = self.session.as_ref() else {
            return Ok(uniform_output(input.legal_moves.len()));
        };
        let state_key = input.history.input_cache_key();
        if let Some(cached) = self.cache.get(state_key) {
            return Ok(output_from_cached(&cached, input.position, input.legal_moves));
        }
        self.cache.record_lookup_miss(1);
        self.cache.record_miss_keys(1);

        crate::fen_tensor::history_to_planes_into(input.history, &mut self.scratch_board).map_err(|e| e.to_string())?;
        let cached = {
            let out = {
                let mut net = session.lock().map_err(|_| "policy 锁中毒".to_string())?;
                net.eval_board(&self.scratch_board).map_err(|e| e.to_string())?
            };
            CachedEval {
                logits: out.logits,
                wdl: out.wdl.unwrap_or([0.5, 0.0, 0.5]),
            }
        };
        let out = output_from_cached(&cached, input.position, input.legal_moves);
        self.cache.insert(state_key, cached);
        Ok(out)
    }

    fn evaluate_many(&mut self, tasks: &[PolicyValueTask]) -> Result<Vec<PolicyValueOutput>, Self::Error> {
        self.evaluate_many_impl(tasks)
    }

    fn evaluate_many_arcs(&mut self, tasks: &[Arc<PolicyValueTask>]) -> Result<Vec<PolicyValueOutput>, Self::Error> {
        self.evaluate_many_impl(tasks)
    }
}

fn uniform_output(len: usize) -> PolicyValueOutput {
    let p = if len == 0 { 0.0 } else { 1.0 / len as f32 };
    PolicyValueOutput {
        priors: vec![p; len],
        wl: 0.0,
        d: 0.0,
        m: 0.0,
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
        let prior = move_vocab_index(*mv, black_to_move)
            .and_then(|idx| cached.logits.get(idx))
            .copied()
            .unwrap_or(0.0);
        priors.push(prior);
    }
    normalize_priors(&mut priors);
    PolicyValueOutput::from_stm_wdl(priors, cached.wdl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_cache_lookup_hit_miss_stats() {
        let cache = EvalCache::new(4);
        let key = 42u64;
        assert!(cache.get(key).is_none());
        cache.record_lookup_miss(1);
        cache.record_miss_keys(1);
        let cached = CachedEval {
            logits: vec![0.1],
            wdl: [0.5, 0.0, 0.5],
        };
        cache.insert(key, cached.clone());
        assert!(cache.get(key).is_some());
        let stats = cache.stats();
        assert_eq!(stats.lookup_hits, 1);
        assert_eq!(stats.lookup_misses, 1);
        assert_eq!(stats.miss_keys, 1);
        assert_eq!(stats.hits, stats.lookup_hits);
        assert_eq!(stats.misses, stats.lookup_misses);
    }

    #[test]
    fn eval_cache_miss_keys_deduplicated_in_stats() {
        let cache = EvalCache::new(4);
        cache.record_lookup_miss(3);
        cache.record_miss_keys(1);
        let stats = cache.stats();
        assert_eq!(stats.lookup_misses, 3);
        assert_eq!(stats.miss_keys, 1);
    }
}
