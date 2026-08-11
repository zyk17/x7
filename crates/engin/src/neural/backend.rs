//! NN backend 边界：属性、cache 与批量评估结果。
//!
//! 接口形状历史上参考过 px0 `src/neural/backend.h`；正式推理走 ONNX，测试可用
//! `UniformBackend`。这不是 task-worker 时代的 backend 翻译层。

use std::sync::Arc;

use xiangqi_core::{MoveList, Position};

use crate::EnginError;

use super::cache::{CachedEval, DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO, EvalCache};
use super::{InputPlanes, POLICY_SIZE};

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

/// 一次评估请求：当前局面序列 + 合法着（仅编码用，不含规则裁决历史）。
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

/// Backend 评估边界：属性、cache 与 stream NN worker 的稀疏 batch 推理。
pub trait Backend: Send + Sync {
    fn attributes(&self) -> BackendAttributes;

    fn cached_evaluation(&self, _position: &EvalPosition) -> Option<Arc<EvalResult>> {
        None
    }

    /// stream NN worker：稀疏 `InputPlanes` 合批推理。
    fn infer_input_planes_into(
        &self,
        samples: &[InputPlanes],
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError>;

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
    fn attributes(&self) -> BackendAttributes {
        self.wrapped.attributes()
    }

    fn cached_evaluation(&self, position: &EvalPosition) -> Option<Arc<EvalResult>> {
        self.cached(position)
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
    fn cached_evaluation(&self, position: &EvalPosition) -> Option<Arc<EvalResult>> {
        let key = Self::cache_key(position);
        let requested_moves = position.legal_moves.len();
        self.cache.get(key, requested_moves)
    }

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
        for _ in 0..batch {
            wdl.extend_from_slice(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        }
        moves_left.clear();
        moves_left.resize(batch, self.plies_left);
        Ok(())
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

    use xiangqi_core::{Move, STARTPOS_FEN};

    use super::*;

    fn startpos_request(num_moves: usize) -> EvalPosition {
        EvalPosition {
            positions: vec![Position::from_fen(STARTPOS_FEN).expect("startpos")],
            legal_moves: vec![Move::NULL; num_moves],
        }
    }

    #[test]
    fn uniform_inference_keeps_moves_left_per_position() {
        let backend = UniformBackend::with_wdl(0.0, 0.0, 17.0);
        let samples = vec![[super::super::InputPlane::default(); super::super::INPUT_PLANES]; 2];
        let mut logits = Vec::new();
        let mut wdl = Vec::new();
        let mut moves_left = Vec::new();
        backend
            .infer_input_planes_into(&samples, &mut logits, &mut wdl, &mut moves_left)
            .expect("infer");
        assert_eq!(moves_left, vec![17.0, 17.0]);
    }

    /// cache miss 入队，命中则立刻返回。
    /// （`memcache.cc:101-129`）：cache hit 跳过被包装的 batch；不同合法着数量是
    /// collision-safe miss。
    #[test]
    fn caching_backend_returns_hits_only_after_completed_batch_and_checks_move_count() {
        let backend = CachingBackend::new(Box::new(UniformBackend::default()));
        let request = startpos_request(2);
        assert!(backend.cached_evaluation(&request).is_none());
        backend.store_evaluation(&request, Arc::new(EvalResult::default()));
        assert!(backend.cached_evaluation(&request).is_some());
        assert!(backend.cached_evaluation(&startpos_request(1)).is_none());
    }

    #[test]
    fn caching_backend_new_game_clear_boundary_removes_completed_entries() {
        let backend = CachingBackend::new(Box::new(UniformBackend::default()));
        let request = startpos_request(1);
        backend.store_evaluation(&request, Arc::new(EvalResult::default()));
        assert!(backend.cached_evaluation(&startpos_request(1)).is_some());

        backend.clear_cache();
        assert!(backend.cached_evaluation(&startpos_request(1)).is_none());
    }

    #[test]
    fn caching_backend_can_use_a_single_cache_slot() {
        let backend = CachingBackend::new(Box::new(UniformBackend::default()));
        backend.set_cache_size_power_of_two(0);
        let request = startpos_request(1);
        backend.store_evaluation(&request, Arc::new(EvalResult::default()));
        assert!(backend.cached_evaluation(&startpos_request(1)).is_some());
    }
}
