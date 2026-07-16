//! px0 `src/neural/backend.h:45-138` 的 P3/P4 后端边界（无 ONNX）。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use xiangqi_core::{Move, MoveList, Position, PositionHistory};

use crate::EnginError;

use super::cache::{CachedEval, EvalCache, DEFAULT_NN_CACHE_SIZE};

/// px0 `BackendAttributes` (`src/neural/backend.h:45-52`)。
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

/// px0 `EvalResult` / backend 评估输出子集。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvalResult {
    pub wl: f32,
    pub d: f32,
    pub m: f32,
    pub policies: Vec<f32>,
}

/// px0 `EvalPosition` (`src/neural/backend.h:62-65`) 的所有权版本。
///
/// P4 computation 会跨 `ComputeBlocking()` 保存请求；因此 Rust 不保存 C++ 的
/// `span` 借用。规则历史不属于 NN 后端输入，保持与 px0 一样只传 positions。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalPosition {
    pub positions: Vec<Position>,
    pub legal_moves: MoveList,
}

/// px0 `BackendComputation::AddInputResult` (`backend.h:69-73`)。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddInputResult {
    EnqueuedForEval,
    FetchedImmediately,
}

/// 一个 computation 内的结果槽位。
///
/// 对应 px0 由 `EvalResultPtr` 指向的外部结果内存；Rust 使用 ticket 在
/// `compute_blocking()` 后取回所有权结果。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvalTicket(pub usize);

/// px0 `BackendComputation` (`src/neural/backend.h:75-87`)。
///
/// `SearchWorker::ProcessPickedTask` may call `AddInput` from several px0 task
/// workers (`search.cc:1423-1462`). Rust implementations therefore own their
/// mutable batch state behind an internal lock instead of requiring `&mut self`.
pub trait BackendComputation: Send + Sync {
    fn used_batch_size(&self) -> usize;
    fn add_input(&self, position: EvalPosition) -> Result<(AddInputResult, EvalTicket), EnginError>;
    fn compute_blocking(&self) -> Result<(), EnginError>;
    fn take_result(&self, ticket: EvalTicket) -> Result<Arc<EvalResult>, EnginError>;
}

/// px0 `Backend` 评估边界（P3 单线程 + P4 batch）。
pub trait Backend: Send + Sync {
    fn evaluate(&self, history: &PositionHistory, legal_moves: &[Move]) -> Arc<EvalResult>;

    /// px0 `Backend::GetAttributes` (`src/neural/backend.h:85`)。
    fn attributes(&self) -> BackendAttributes;

    /// px0 `Backend::CreateComputation` (`src/neural/backend.h:86`)。
    fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError>;

    /// px0 `Backend::GetCachedEvaluation` (`src/neural/backend.h:95-97`)。
    fn cached_evaluation(&self, _position: &EvalPosition) -> Option<Arc<EvalResult>> {
        None
    }

    /// px0 `CachingBackend::ClearCache` (`src/neural/memcache.h:34-38`).
    /// Non-caching backends deliberately keep the no-op default.
    fn clear_cache(&self) {}

    /// px0 `CachingBackend::SetCacheSize` (`src/neural/memcache.h:36-38`).
    fn set_cache_size(&self, _size: usize) {}
}

/// px0 `CachingBackend` / `MemCache` (`src/neural/memcache.h:34-45`,
/// `memcache.cc:58-190`). This wrapper, rather than the ONNX implementation,
/// owns cache lookup, batch miss forwarding and post-compute insertion.
pub struct CachingBackend {
    wrapped: Arc<dyn Backend>,
    cache: Arc<EvalCache>,
}

impl CachingBackend {
    pub fn new(wrapped: Box<dyn Backend>) -> Self {
        Self {
            wrapped: Arc::from(wrapped),
            cache: Arc::new(EvalCache::new(DEFAULT_NN_CACHE_SIZE)),
        }
    }

    fn cache_key(position: &EvalPosition) -> u64 {
        // px0 `ComputeEvalPositionHash` (`memcache.cc:38-45`). History and
        // repetitions are intentionally not in this cache key yet.
        position.positions.last().map_or(0, Position::hash)
    }

    fn cached(&self, position: &EvalPosition) -> Option<Arc<EvalResult>> {
        self.cache.get(Self::cache_key(position), position.legal_moves.len())
    }

    fn resize_cache(&self, size: usize) {
        self.cache.set_capacity(size);
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

    fn clear_cache(&self) {
        self.cache.clear();
        self.wrapped.clear_cache();
    }

    fn set_cache_size(&self, size: usize) {
        self.resize_cache(size);
        self.wrapped.set_cache_size(size);
    }
}

/// One px0 `MemCacheComputation::Entry` (`memcache.cc:109-120`).
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
            self.cache.insert_if_absent(
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

/// px0 `UniformBackend` 的 P4 batch 实现（测试/对拍用）。
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

/// 测试用：均匀 policy + 固定 WDL，复用正式 px0 风格 NN cache 容器。
#[derive(Clone, Debug)]
pub struct UniformBackend {
    pub wl: f32,
    pub d: f32,
    pub m: f32,
    cache: Arc<EvalCache>,
}

impl Default for UniformBackend {
    fn default() -> Self {
        Self {
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            cache: Arc::new(EvalCache::new(DEFAULT_NN_CACHE_SIZE)),
        }
    }
}

impl UniformBackend {
    fn cache_key(position: &EvalPosition) -> u64 {
        position.positions.last().map(|p| p.hash()).unwrap_or(0)
    }

    pub fn with_wdl(wl: f32, d: f32, m: f32) -> Self {
        Self {
            wl,
            d,
            m,
            cache: Arc::new(EvalCache::new(DEFAULT_NN_CACHE_SIZE)),
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
            m: self.m,
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

    fn clear_cache(&self) {
        self.cache.clear();
    }

    fn set_cache_size(&self, size: usize) {
        self.cache.set_capacity(size);
    }
}

impl UniformBackend {
    pub fn store_cache(&self, position: &EvalPosition, result: Arc<EvalResult>) {
        let key = Self::cache_key(position);
        self.cache.insert_if_absent(
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

    /// px0 `MemCacheComputation::AddInput/ComputeBlocking`
    /// (`memcache.cc:101-129`): cache hits skip the wrapped batch, while a
    /// different legal move count is a collision-safe miss.
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
    fn caching_backend_respects_px0_zero_capacity() {
        let backend = CachingBackend::new(Box::new(UniformBackend::default()));
        backend.set_cache_size(0);
        let computation = backend.create_computation().expect("computation");
        let (_, ticket) = computation.add_input(startpos_request(1)).expect("input");
        computation.compute_blocking().expect("compute");
        computation.take_result(ticket).expect("result");
        assert!(backend.cached_evaluation(&startpos_request(1)).is_none());
    }
}
