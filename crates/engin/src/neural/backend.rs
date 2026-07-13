//! px0 `src/neural/backend.h:45-138` 的 P3/P4 后端边界（无 ONNX）。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use xiangqi_core::{Move, MoveList, Position, PositionHistory};

use crate::EnginError;

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
    fn take_result(&self, ticket: EvalTicket) -> Result<EvalResult, EnginError>;
}

/// px0 `Backend` 评估边界（P3 单线程 + P4 batch）。
pub trait Backend: Send + Sync {
    fn evaluate(&self, history: &PositionHistory, legal_moves: &[Move]) -> EvalResult;

    /// px0 `Backend::GetAttributes` (`src/neural/backend.h:85`)。
    fn attributes(&self) -> BackendAttributes;

    /// px0 `Backend::CreateComputation` (`src/neural/backend.h:86`)。
    fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError>;

    /// px0 `Backend::GetCachedEvaluation` (`src/neural/backend.h:95-97`)。
    fn cached_evaluation(&self, _position: &EvalPosition) -> Option<EvalResult> {
        None
    }
}

/// px0 `UniformBackend` 的 P4 batch 实现（测试/对拍用）。
struct UniformBackendComputation {
    backend: UniformBackend,
    state: Mutex<UniformComputationState>,
}

struct UniformComputationState {
    pending: Vec<(EvalTicket, EvalPosition)>,
    results: HashMap<usize, EvalResult>,
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

    fn take_result(&self, ticket: EvalTicket) -> Result<EvalResult, EnginError> {
        self.state
            .lock()
            .expect("uniform computation lock")
            .results
            .remove(&ticket.0)
            .ok_or(EnginError::PortIncomplete("UniformBackendComputation missing result"))
    }
}

/// 测试用：均匀 policy + 固定 WDL，带 px0 风格 NN cache。
#[derive(Clone, Debug)]
pub struct UniformBackend {
    pub wl: f32,
    pub d: f32,
    pub m: f32,
    cache: Arc<Mutex<HashMap<u64, CachedEval>>>,
}

/// px0 `neural/memcache.cc:50-55`：policy 缓存必须带合法着数量。
#[derive(Clone, Debug)]
struct CachedEval {
    result: EvalResult,
    num_moves: usize,
}

impl Default for UniformBackend {
    fn default() -> Self {
        Self {
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            cache: Arc::new(Mutex::new(HashMap::new())),
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
            cache: Arc::new(Mutex::new(HashMap::new())),
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

    fn cached_evaluation(&self, position: &EvalPosition) -> Option<EvalResult> {
        let key = Self::cache_key(position);
        let requested_moves = position.legal_moves.len();
        self.cache
            .lock()
            .expect("cache lock")
            .get(&key)
            .filter(|cached| requested_moves == 0 || cached.num_moves == requested_moves)
            .map(|cached| cached.result.clone())
    }

    fn attributes(&self) -> BackendAttributes {
        BackendAttributes::default()
    }

    fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
        Ok(Box::new(UniformBackendComputation::new(self.clone())))
    }
}

impl UniformBackend {
    pub fn store_cache(&self, position: &EvalPosition, result: EvalResult) {
        let key = Self::cache_key(position);
        self.cache.lock().expect("cache lock").insert(
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

    use super::*;

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
}
