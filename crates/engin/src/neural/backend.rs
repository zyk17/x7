//! NN backend 边界：属性、computation、cache 与评估结果。
//!
//! 接口形状历史上参考过 px0 `src/neural/backend.h`；正式推理走 ONNX，测试可用
//! `UniformBackend`。这不是 task-worker 时代的 backend 翻译层。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use xiangqi_core::{Move, MoveList, Position, PositionHistory};

use crate::EnginError;

use super::cache::{CachedEval, DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO, EvalCache};
use super::{
    BOARD_COLS, BOARD_ROWS, ENCODED_PLANE_FLOATS, INPUT_PLANES, InputPlanes, POLICY_SIZE, expand_input_planes,
};

/// Backend 能力与推荐 batch 大小。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendAttributes {
    pub has_mlh: bool,
    pub has_wdl: bool,
    pub runs_on_cpu: bool,
    pub suggested_num_search_threads: usize,
    pub recommended_batch_size: usize,
    pub maximum_batch_size: usize,
}

impl Default for BackendAttributes {
    fn default() -> Self {
        Self {
            has_mlh: false,
            has_wdl: true,
            runs_on_cpu: true,
            suggested_num_search_threads: 1,
            recommended_batch_size: 1,
            maximum_batch_size: 1,
        }
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

/// 批量原始网络输出：policy logits、WDL 概率、moves-left。
pub type EncodedInference = (Vec<f32>, Vec<f32>, Vec<f32>);

/// 一次评估请求：当前局面序列 + 合法着（仅编码用，不含规则裁决历史）。
///
/// computation 会跨 `compute_blocking()` 保存请求，因此使用 owned positions。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalPosition {
    pub positions: Vec<Position>,
    pub legal_moves: MoveList,
}

impl EvalPosition {
    /// 与 MCGS `NodeKey::board` 一致的 state key。NN cache 刻意只按当前棋盘复用，
    /// 不纳入完整 history 或 `Position::repetitions`；这是为了与 graph node 一致的取舍。
    pub(crate) fn board_key(&self) -> u64 {
        self.positions.last().map_or(0, |position| position.board().hash())
    }
}

/// `add_input` 的结果：入队评估，或立刻命中 cache。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddInputResult {
    EnqueuedForEval,
    FetchedImmediately,
}

/// 一个 computation 内的结果槽位。
///
/// Rust 使用 ticket 在 `compute_blocking()` 后取回所有权结果。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvalTicket(pub usize);

/// 一次可合批的 backend computation。
///
/// 可变 batch state 放在内部锁之后，不要求 `&mut self`。
pub trait BackendComputation: Send + Sync {
    fn used_batch_size(&self) -> usize;
    fn add_input(&self, position: EvalPosition) -> Result<(AddInputResult, EvalTicket), EnginError>;
    fn compute_blocking(&self) -> Result<(), EnginError>;
    fn take_result(&self, ticket: EvalTicket) -> Result<Arc<EvalResult>, EnginError>;
}

/// Backend 评估边界：属性、cache、encoded inference 与 computation。
pub trait Backend: Send + Sync {
    fn evaluate(&self, history: &PositionHistory, legal_moves: &[Move]) -> Arc<EvalResult>;

    fn attributes(&self) -> BackendAttributes;

    fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError>;

    fn cached_evaluation(&self, _position: &EvalPosition) -> Option<Arc<EvalResult>> {
        None
    }

    /// stream NN worker：只对已编码 NCHW planes 推理。依次返回 logits、WDL 概率、
    /// moves-left，形状为 `[batch * POLICY_SIZE]`、`[batch * 3]`、`[batch]`。
    fn infer_encoded(&self, _planes: &[f32], _batch: usize) -> Result<EncodedInference, EnginError> {
        Err(EnginError::PortIncomplete("backend has no encoded inference"))
    }

    /// 与 [`Self::infer_encoded`] 相同，但写入调用方缓冲（先 `clear` 再追加），供 NN worker 跨 batch 复用。
    fn infer_encoded_into(
        &self,
        planes: &[f32],
        batch: usize,
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError> {
        let (out_logits, out_wdl, out_moves_left) = self.infer_encoded(planes, batch)?;
        *logits = out_logits;
        *wdl = out_wdl;
        *moves_left = out_moves_left;
        Ok(())
    }

    /// stream NN worker：稀疏 `InputPlanes` 合批推理。
    ///
    /// 默认实现会先 expand 再走 [`Self::infer_encoded_into`]；ONNX 覆盖为直接写入
    /// session `input_scratch`（含 DirectML pad），去掉中间 dense `packed`。
    fn infer_input_planes_into(
        &self,
        samples: &[InputPlanes],
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError> {
        let batch = samples.len();
        let mut dense = vec![0.0; batch * ENCODED_PLANE_FLOATS];
        for (index, sample) in samples.iter().enumerate() {
            let offset = index * ENCODED_PLANE_FLOATS;
            expand_input_planes(sample, &mut dense[offset..offset + ENCODED_PLANE_FLOATS]);
        }
        self.infer_encoded_into(&dense, batch, logits, wdl, moves_left)
    }

    /// Eval 构造完整 `EvalResult` 后可选地写入 cache。
    fn store_evaluation(&self, _position: &EvalPosition, _result: Arc<EvalResult>) {}

    /// 非缓存 backend 的空实现。
    fn clear_cache(&self) {}

    /// 缓存容量为 `2^N` 个直映槽。
    fn set_cache_size_power_of_two(&self, _size_power_of_two: u8) {}
}

/// 带 NN cache 的 backend 包装：查找、batch miss 转发与插入由此层负责，不在 ONNX 内实现。
pub struct CachingBackend {
    wrapped: Arc<dyn Backend>,
    cache: Arc<EvalCache>,
}

impl CachingBackend {
    pub fn new(wrapped: Box<dyn Backend>) -> Self {
        Self::with_cache_size_power_of_two(wrapped, DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO)
    }

    pub fn with_cache_size_power_of_two(wrapped: Box<dyn Backend>, size_power_of_two: u8) -> Self {
        Self {
            wrapped: Arc::from(wrapped),
            cache: Arc::new(EvalCache::new(size_power_of_two)),
        }
    }

    fn cache_key(position: &EvalPosition) -> u64 {
        position.board_key()
    }

    fn cached(&self, position: &EvalPosition) -> Option<Arc<EvalResult>> {
        self.cache.get(Self::cache_key(position), position.legal_moves.len())
    }

    fn resize_cache(&self, size_power_of_two: u8) {
        self.cache.set_size_power_of_two(size_power_of_two);
    }
}

impl Backend for CachingBackend {
    fn evaluate(&self, history: &PositionHistory, legal_moves: &[Move]) -> Arc<EvalResult> {
        let computation = self.create_computation().expect("create caching computation");
        let (_, ticket) = computation
            .add_input(EvalPosition {
                positions: history.positions().to_vec(),
                legal_moves: legal_moves.to_vec(),
            })
            .expect("enqueue caching evaluation");
        computation.compute_blocking().expect("run caching evaluation");
        computation.take_result(ticket).expect("fetch caching evaluation")
    }

    fn attributes(&self) -> BackendAttributes {
        self.wrapped.attributes()
    }

    fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
        Ok(Box::new(CachingBackendComputation {
            wrapped: self.wrapped.create_computation()?,
            cache: Arc::clone(&self.cache),
            state: Mutex::new(CachingComputationState::default()),
        }))
    }

    fn cached_evaluation(&self, position: &EvalPosition) -> Option<Arc<EvalResult>> {
        self.cached(position)
    }

    fn infer_encoded(&self, planes: &[f32], batch: usize) -> Result<EncodedInference, EnginError> {
        self.wrapped.infer_encoded(planes, batch)
    }

    fn infer_encoded_into(
        &self,
        planes: &[f32],
        batch: usize,
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError> {
        self.wrapped.infer_encoded_into(planes, batch, logits, wdl, moves_left)
    }

    fn infer_input_planes_into(
        &self,
        samples: &[InputPlanes],
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError> {
        self.wrapped.infer_input_planes_into(samples, logits, wdl, moves_left)
    }

    fn store_evaluation(&self, position: &EvalPosition, result: Arc<EvalResult>) {
        self.cache.insert(
            Self::cache_key(position),
            CachedEval {
                result,
                num_moves: position.legal_moves.len(),
            },
        );
    }

    fn clear_cache(&self) {
        self.cache.clear();
        self.wrapped.clear_cache();
    }

    fn set_cache_size_power_of_two(&self, size_power_of_two: u8) {
        self.resize_cache(size_power_of_two);
        self.wrapped.set_cache_size_power_of_two(size_power_of_two);
    }
}

/// cache computation 中的一个待评估条目。
struct CacheEntry {
    outer_ticket: EvalTicket,
    inner_ticket: EvalTicket,
    key: u64,
    num_moves: usize,
}

#[derive(Default)]
struct CachingComputationState {
    entries: Vec<CacheEntry>,
    results: HashMap<usize, Arc<EvalResult>>,
    next_ticket: usize,
}

struct CachingBackendComputation {
    wrapped: Box<dyn BackendComputation>,
    cache: Arc<EvalCache>,
    state: Mutex<CachingComputationState>,
}

impl BackendComputation for CachingBackendComputation {
    fn used_batch_size(&self) -> usize {
        self.wrapped.used_batch_size()
    }

    fn add_input(&self, position: EvalPosition) -> Result<(AddInputResult, EvalTicket), EnginError> {
        let outer_ticket = {
            let mut state = self.state.lock().expect("caching computation lock");
            let ticket = EvalTicket(state.next_ticket);
            state.next_ticket += 1;
            ticket
        };
        let key = CachingBackend::cache_key(&position);
        if let Some(result) = self.cache.get(key, position.legal_moves.len()) {
            self.state
                .lock()
                .expect("caching computation lock")
                .results
                .insert(outer_ticket.0, result);
            return Ok((AddInputResult::FetchedImmediately, outer_ticket));
        }

        let num_moves = position.legal_moves.len();
        let (_, inner_ticket) = self.wrapped.add_input(position)?;
        self.state
            .lock()
            .expect("caching computation lock")
            .entries
            .push(CacheEntry {
                outer_ticket,
                inner_ticket,
                key,
                num_moves,
            });
        Ok((AddInputResult::EnqueuedForEval, outer_ticket))
    }

    fn compute_blocking(&self) -> Result<(), EnginError> {
        self.wrapped.compute_blocking()?;
        let entries = std::mem::take(&mut self.state.lock().expect("caching computation lock").entries);
        let mut results = Vec::with_capacity(entries.len());
        for entry in entries {
            let result = self.wrapped.take_result(entry.inner_ticket)?;
            self.cache.insert(
                entry.key,
                CachedEval {
                    result: Arc::clone(&result),
                    num_moves: entry.num_moves,
                },
            );
            results.push((entry.outer_ticket.0, result));
        }
        self.state
            .lock()
            .expect("caching computation lock")
            .results
            .extend(results);
        Ok(())
    }

    fn take_result(&self, ticket: EvalTicket) -> Result<Arc<EvalResult>, EnginError> {
        self.state
            .lock()
            .expect("caching computation lock")
            .results
            .remove(&ticket.0)
            .ok_or(EnginError::PortIncomplete("CachingBackendComputation missing result"))
    }
}

/// 测试用均匀 policy computation。
struct UniformBackendComputation {
    backend: UniformBackend,
    state: Mutex<UniformComputationState>,
}

struct UniformComputationState {
    pending: Vec<(EvalTicket, EvalPosition)>,
    results: HashMap<usize, Arc<EvalResult>>,
    next_ticket: usize,
}

impl UniformBackendComputation {
    fn new(backend: UniformBackend) -> Self {
        Self {
            backend,
            state: Mutex::new(UniformComputationState {
                pending: Vec::new(),
                results: HashMap::new(),
                next_ticket: 0,
            }),
        }
    }
}

impl BackendComputation for UniformBackendComputation {
    fn used_batch_size(&self) -> usize {
        self.state.lock().expect("uniform computation lock").pending.len()
    }

    fn add_input(&self, position: EvalPosition) -> Result<(AddInputResult, EvalTicket), EnginError> {
        let mut state = self.state.lock().expect("uniform computation lock");
        let ticket = EvalTicket(state.next_ticket);
        state.next_ticket += 1;
        if let Some(cached) = self.backend.cached_evaluation(&position) {
            state.results.insert(ticket.0, cached);
            return Ok((AddInputResult::FetchedImmediately, ticket));
        }
        state.pending.push((ticket, position));
        Ok((AddInputResult::EnqueuedForEval, ticket))
    }

    fn compute_blocking(&self) -> Result<(), EnginError> {
        let pending = std::mem::take(&mut self.state.lock().expect("uniform computation lock").pending);
        let mut results = Vec::with_capacity(pending.len());
        for (ticket, position) in pending {
            let history = PositionHistory::from_positions(position.positions.clone());
            let eval = self.backend.evaluate(&history, &position.legal_moves);
            self.backend.store_cache(&position, eval.clone());
            results.push((ticket, eval));
        }
        self.state
            .lock()
            .expect("uniform computation lock")
            .results
            .extend(results.into_iter().map(|(ticket, result)| (ticket.0, result)));
        Ok(())
    }

    fn take_result(&self, ticket: EvalTicket) -> Result<Arc<EvalResult>, EnginError> {
        self.state
            .lock()
            .expect("uniform computation lock")
            .results
            .remove(&ticket.0)
            .ok_or(EnginError::PortIncomplete("UniformBackendComputation missing result"))
    }
}

/// 测试用：均匀 policy + 固定 WDL，复用正式 NN cache 容器。
#[derive(Clone, Debug)]
pub struct UniformBackend {
    pub wl: f32,
    pub d: f32,
    pub plies_left: f32,
    cache: Arc<EvalCache>,
}

impl Default for UniformBackend {
    fn default() -> Self {
        Self {
            wl: 0.0,
            d: 0.0,
            plies_left: 0.0,
            cache: Arc::new(EvalCache::new(DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO)),
        }
    }
}

impl UniformBackend {
    fn cache_key(position: &EvalPosition) -> u64 {
        position.board_key()
    }

    pub fn with_wdl(wl: f32, d: f32, plies_left: f32) -> Self {
        Self {
            wl,
            d,
            plies_left,
            cache: Arc::new(EvalCache::new(DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO)),
        }
    }
}

impl Backend for UniformBackend {
    fn evaluate(&self, _history: &PositionHistory, legal_moves: &[Move]) -> Arc<EvalResult> {
        let count = legal_moves.len().max(1);
        let p = 1.0 / count as f32;
        Arc::new(EvalResult {
            wl: self.wl,
            d: self.d,
            plies_left: self.plies_left,
            policies: vec![p; legal_moves.len()],
        })
    }

    fn cached_evaluation(&self, position: &EvalPosition) -> Option<Arc<EvalResult>> {
        let key = Self::cache_key(position);
        let requested_moves = position.legal_moves.len();
        self.cache.get(key, requested_moves)
    }

    fn attributes(&self) -> BackendAttributes {
        BackendAttributes::default()
    }

    fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
        Ok(Box::new(UniformBackendComputation::new(self.clone())))
    }

    fn infer_encoded(&self, planes: &[f32], batch: usize) -> Result<EncodedInference, EnginError> {
        let plane_len = INPUT_PLANES * BOARD_ROWS * BOARD_COLS;
        if planes.len() != batch * plane_len {
            return Err(EnginError::PortIncomplete("uniform encoded planes length"));
        }
        // 相等 logits → Eval softmax 在合法着上均匀分布。
        let logits = vec![0.0; batch * POLICY_SIZE];
        let mut wdl = Vec::with_capacity(batch * 3);
        for _ in 0..batch {
            wdl.extend_from_slice(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        }
        Ok((logits, wdl, vec![self.plies_left; batch]))
    }

    fn store_evaluation(&self, position: &EvalPosition, result: Arc<EvalResult>) {
        self.store_cache(position, result);
    }

    fn clear_cache(&self) {
        self.cache.clear();
    }

    fn set_cache_size_power_of_two(&self, size_power_of_two: u8) {
        self.cache.set_size_power_of_two(size_power_of_two);
    }
}

impl UniformBackend {
    pub fn store_cache(&self, position: &EvalPosition, result: Arc<EvalResult>) {
        let key = Self::cache_key(position);
        self.cache.insert(
            key,
            CachedEval {
                result,
                num_moves: position.legal_moves.len(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use xiangqi_core::STARTPOS_FEN;

    use super::*;

    fn startpos_request(num_moves: usize) -> EvalPosition {
        EvalPosition {
            positions: vec![Position::from_fen(STARTPOS_FEN).expect("startpos")],
            legal_moves: vec![Move::NULL; num_moves],
        }
    }

    #[test]
    fn computation_accepts_concurrent_task_inputs() {
        let backend = UniformBackend::default();
        let computation: Arc<dyn BackendComputation> = Arc::from(backend.create_computation().expect("computation"));
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let computation = Arc::clone(&computation);
                thread::spawn(move || {
                    computation
                        .add_input(EvalPosition {
                            positions: Vec::new(),
                            legal_moves: Vec::new(),
                        })
                        .expect("add input")
                        .1
                })
            })
            .collect();
        let tickets: Vec<_> = handles.into_iter().map(|handle| handle.join().expect("task")).collect();

        assert_eq!(computation.used_batch_size(), 2);
        computation.compute_blocking().expect("compute");
        for ticket in tickets {
            assert!(computation.take_result(ticket).is_ok());
        }
    }

    #[test]
    fn encoded_uniform_inference_keeps_moves_left_per_position() {
        let backend = UniformBackend::with_wdl(0.0, 0.0, 17.0);
        let planes = vec![0.0; 2 * INPUT_PLANES * BOARD_ROWS * BOARD_COLS];
        let (_, _, moves_left) = backend.infer_encoded(&planes, 2).expect("infer");
        assert_eq!(moves_left, vec![17.0, 17.0]);
    }

    /// cache miss 入队，命中则立刻返回。
    /// （`memcache.cc:101-129`）：cache hit 跳过被包装的 batch；不同合法着数量是
    /// collision-safe miss。
    #[test]
    fn caching_backend_returns_hits_only_after_completed_batch_and_checks_move_count() {
        let backend = CachingBackend::new(Box::new(UniformBackend::default()));
        let first = backend.create_computation().expect("first computation");
        let (status, first_ticket) = first.add_input(startpos_request(2)).expect("first input");
        assert_eq!(status, AddInputResult::EnqueuedForEval);
        first.compute_blocking().expect("first compute");
        assert_eq!(first.take_result(first_ticket).expect("first result").policies.len(), 2);

        let hit = backend.create_computation().expect("hit computation");
        let (status, hit_ticket) = hit.add_input(startpos_request(2)).expect("hit input");
        assert_eq!(status, AddInputResult::FetchedImmediately);
        assert_eq!(hit.used_batch_size(), 0);
        assert_eq!(hit.take_result(hit_ticket).expect("cached result").policies.len(), 2);

        let collision_guard = backend.create_computation().expect("miss computation");
        let (status, _) = collision_guard
            .add_input(startpos_request(1))
            .expect("mismatched moves");
        assert_eq!(status, AddInputResult::EnqueuedForEval);
    }

    #[test]
    fn caching_backend_new_game_clear_boundary_removes_completed_entries() {
        let backend = CachingBackend::new(Box::new(UniformBackend::default()));
        let computation = backend.create_computation().expect("computation");
        let (_, ticket) = computation.add_input(startpos_request(1)).expect("input");
        computation.compute_blocking().expect("compute");
        computation.take_result(ticket).expect("result");
        assert!(backend.cached_evaluation(&startpos_request(1)).is_some());

        backend.clear_cache();
        assert!(backend.cached_evaluation(&startpos_request(1)).is_none());
    }

    #[test]
    fn caching_backend_can_use_a_single_cache_slot() {
        let backend = CachingBackend::new(Box::new(UniformBackend::default()));
        backend.set_cache_size_power_of_two(0);
        let computation = backend.create_computation().expect("computation");
        let (_, ticket) = computation.add_input(startpos_request(1)).expect("input");
        computation.compute_blocking().expect("compute");
        computation.take_result(ticket).expect("result");
        assert!(backend.cached_evaluation(&startpos_request(1)).is_some());
    }
}
