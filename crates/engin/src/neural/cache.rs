//! NN cache 存储。
//!
//! key 与 MCGS 的 board node identity 同步。px0 `neural/memcache.cc:38-45` 同样不
//! hash 完整 history；本实现进一步不纳入 `Position::repetitions`，以复用同一 board node。
//! 容量与替换方式对齐 KataGo `neuralnet/nneval.cpp:1273-1339`：固定 `2^N` 直映表，
//! 索引碰撞时新结果直接替换旧结果。

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use super::backend::EvalResult;

/// KataGo GTP 默认值（`cpp/program/setup.cpp:248-255`）：`2^20 = 1,048,576` 槽。
pub const DEFAULT_NN_CACHE_SIZE_POWER_OF_TWO: u8 = 20;
/// KataGo `setup.cpp` 对该配置接受 `0..=48`。超大值仍受实际可分配内存约束。
pub const MAX_NN_CACHE_SIZE_POWER_OF_TWO: u8 = 48;

/// px0 `CachedValue` (`src/neural/memcache.cc:48-55`) 的 Rust 所有权版本。
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

/// KataGo 风格的直映 NN cache。每个槽只保留一个完整 key；不同 key 映射到同一槽时，
/// 后写结果替换先前结果。表大小只在 UCI option 改动时重建，查找只锁定目标槽。
#[derive(Debug)]
pub(crate) struct EvalCache {
    slots: RwLock<Arc<[CacheSlot]>>,
}

impl EvalCache {
    pub(crate) fn new(size_power_of_two: u8) -> Self {
        Self {
            slots: RwLock::new(Self::allocate_slots(size_power_of_two)),
        }
    }

    fn allocate_slots(size_power_of_two: u8) -> Arc<[CacheSlot]> {
        assert!(
            size_power_of_two <= MAX_NN_CACHE_SIZE_POWER_OF_TWO,
            "NN cache size power is out of range"
        );
        let size = 1usize << size_power_of_two;
        let mut slots = Vec::with_capacity(size);
        slots.resize_with(size, CacheSlot::default);
        slots.into()
    }

    fn slots(&self) -> Arc<[CacheSlot]> {
        Arc::clone(&self.slots.read())
    }

    /// px0 `MemCache::GetCachedEvaluation` / collision guard
    /// （`memcache.cc:130-150`）。空合法着列表可接受缓存结果；否则只有相同 policy
    /// 长度才安全。
    pub(crate) fn get(&self, key: u64, requested_moves: usize) -> Option<Arc<EvalResult>> {
        let slots = self.slots();
        let slot = slots[key as usize & (slots.len() - 1)].0.lock();
        let entry = slot.as_ref()?;
        (entry.key == key && (requested_moves == 0 || entry.value.num_moves == requested_moves))
            .then(|| Arc::clone(&entry.value.result))
    }

    /// KataGo `NNCacheTable::set`：同一槽内的新结果替换旧结果。旧 `Arc` 在离开锁后
    /// 才释放，避免析构占用槽锁。
    pub(crate) fn insert(&self, key: u64, value: CachedEval) {
        let slots = self.slots();
        let previous = {
            let mut slot = slots[key as usize & (slots.len() - 1)].0.lock();
            slot.replace(CacheEntry { key, value })
        };
        drop(previous);
    }

    /// 仅供 Engine 生命周期 option 调用。先分配新表再交换，已有查找继续持有旧表快照。
    pub(crate) fn set_size_power_of_two(&self, size_power_of_two: u8) {
        let slots = Self::allocate_slots(size_power_of_two);
        *self.slots.write() = slots;
    }

    /// 清空时直接交换同尺寸空表，避免逐槽加锁。
    pub(crate) fn clear(&self) {
        let size = self.slots.read().len();
        let mut slots = Vec::with_capacity(size);
        slots.resize_with(size, CacheSlot::default);
        *self.slots.write() = slots.into();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slots.read().iter().filter(|slot| slot.0.lock().is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_slot_replaces_an_older_key() {
        let result = Arc::new(EvalResult::default());
        let cache = EvalCache::new(0);
        cache.insert(
            1,
            CachedEval {
                result: Arc::clone(&result),
                num_moves: 1,
            },
        );
        cache.insert(
            2,
            CachedEval {
                result: Arc::clone(&result),
                num_moves: 2,
            },
        );
        assert!(cache.get(1, 1).is_none());
        assert!(cache.get(2, 2).is_some());
        assert!(Arc::ptr_eq(&cache.get(2, 2).expect("cached value"), &result));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_checks_the_full_key_inside_a_slot() {
        let cache = EvalCache::new(0);
        cache.insert(
            1,
            CachedEval {
                result: Arc::new(EvalResult::default()),
                num_moves: 1,
            },
        );
        assert!(cache.get(3, 1).is_none());
        assert!(cache.get(1, 2).is_none());
        assert!(cache.get(1, 1).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_resize_replaces_all_entries() {
        let cache = EvalCache::new(1);
        cache.insert(
            1,
            CachedEval {
                result: Arc::new(EvalResult::default()),
                num_moves: 0,
            },
        );
        cache.set_size_power_of_two(0);
        assert_eq!(cache.len(), 0);
    }
}
