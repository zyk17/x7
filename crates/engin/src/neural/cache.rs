//! NN cache 存储。
//!
//! key 由 `EvalCacheKey::slot_key()` 提供：棋盘 hash 混入 `repetitions`；
//! 命中时再校验 `num_moves`（廉价碰撞护栏）。**不**纳入完整 history。
//! 容量与替换：固定 `2^N` 直映表，槽内新结果替换旧结果。

use std::sync::Arc;

use parking_lot::Mutex;
use xiangqi_core::Position;

use super::backend::EvalResult;

/// 默认 `2^20 = 1,048,576` 槽。
pub const DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO: u8 = 20;
/// 可设范围为 `0..=48`；超大值仍受实际可分配内存约束。
pub const MAX_NN_CACHE_SIZE_POWER_OF_TWO: u8 = 48;

/// NN cache 命中键：当前棋盘 + 合法着数 + `Position::repetitions`。
///
/// 树节点不按棋盘合并；cache 允许历史路径不同，但须区分编码平面中的 repetition 次数。
/// `num_moves` 用作 policy 长度的廉价护栏。为提高命中率，刻意不纳入完整 history 与
/// rule60；规则终局在读取 cache 前裁决。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EvalCacheKey {
    board_hash: u64,
    num_moves: usize,
    repetitions: u32,
}

impl EvalCacheKey {
    pub(crate) fn new(position: &Position, num_moves: usize) -> Self {
        Self { board_hash: position.board().hash(), num_moves, repetitions: position.repetitions() }
    }

    fn slot_key(self) -> u64 {
        if self.repetitions == 0 {
            self.board_hash
        } else {
            xiangqi_core::hashcat::hash_cat(self.board_hash, self.repetitions as u64)
        }
    }
}

/// cache 中保存的评估结果。
#[derive(Clone, Debug)]
pub(crate) struct CachedEval {
    pub result: Arc<EvalResult>,
    pub num_moves: usize,
}

#[derive(Debug)]
struct CacheEntry {
    key: u64,
    value: CachedEval,
}

#[derive(Debug, Default)]
struct CacheSlot(Mutex<Option<CacheEntry>>);

/// 直映 NN cache。每个槽只保留一个完整 key；不同 key 映射到同一槽时，
/// 后写结果替换先前结果。表大小只在 UCI option 改动时重建，查找只锁定目标槽。
#[derive(Debug)]
pub(crate) struct EvalCache {
    slots: Arc<[CacheSlot]>,
}

impl EvalCache {
    pub(crate) fn new(size_power_of_two: u8) -> Self {
        Self { slots: Self::allocate_slots(size_power_of_two) }
    }

    fn allocate_slots(size_power_of_two: u8) -> Arc<[CacheSlot]> {
        assert!(size_power_of_two <= MAX_NN_CACHE_SIZE_POWER_OF_TWO, "NN cache size power is out of range");
        let size = 1usize << size_power_of_two;
        let mut slots = Vec::with_capacity(size);
        slots.resize_with(size, CacheSlot::default);
        slots.into()
    }

    /// 查找 cache；key 冲突时校验完整 key / 合法着数。
    /// 空合法着列表可接受缓存结果；否则只有相同 policy 长度才安全。
    pub(crate) fn get(&self, key: u64, requested_moves: usize) -> Option<Arc<EvalResult>> {
        let slot = self.slots[key as usize & (self.slots.len() - 1)].0.lock();
        let entry = slot.as_ref()?;
        (entry.key == key && (requested_moves == 0 || entry.value.num_moves == requested_moves))
            .then(|| Arc::clone(&entry.value.result))
    }

    pub(crate) fn get_evaluation(&self, key: EvalCacheKey) -> Option<Arc<EvalResult>> {
        self.get(key.slot_key(), key.num_moves)
    }

    /// 同一槽内的新结果替换旧结果。旧 `Arc` 在离开锁后
    /// 才释放，避免析构占用槽锁。
    pub(crate) fn insert(&self, key: u64, value: CachedEval) {
        let previous = {
            let mut slot = self.slots[key as usize & (self.slots.len() - 1)].0.lock();
            slot.replace(CacheEntry { key, value })
        };
        drop(previous);
    }

    pub(crate) fn insert_evaluation(&self, key: EvalCacheKey, result: Arc<EvalResult>) {
        self.insert(key.slot_key(), CachedEval { result, num_moves: key.num_moves });
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.0.lock().is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_slot_replaces_an_older_key() {
        let result = Arc::new(EvalResult::default());
        let cache = EvalCache::new(0);
        cache.insert(1, CachedEval { result: Arc::clone(&result), num_moves: 1 });
        cache.insert(2, CachedEval { result: Arc::clone(&result), num_moves: 2 });
        assert!(cache.get(1, 1).is_none());
        assert!(cache.get(2, 2).is_some());
        assert!(Arc::ptr_eq(&cache.get(2, 2).expect("cached value"), &result));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_checks_the_full_key_inside_a_slot() {
        let cache = EvalCache::new(0);
        cache.insert(1, CachedEval { result: Arc::new(EvalResult::default()), num_moves: 1 });
        assert!(cache.get(3, 1).is_none());
        assert!(cache.get(1, 2).is_none());
        assert!(cache.get(1, 1).is_some());
        assert_eq!(cache.len(), 1);
    }
}
