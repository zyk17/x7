//! px0 `src/neural/memcache.cc:38-190` 的 FIFO cache 存储。
//!
//! px0 `HashKeyedCache` 不维护 LRU，插入已有 key 时不替换，并按 FIFO
//! 淘汰（`src/utils/cache.h:35-57,69-105,214-230`）。这里仅承载该通用
//! 容器语义；`CachingBackend` 负责 NN 请求/回填时序。

use std::collections::{HashMap, VecDeque};

use super::backend::EvalResult;

/// px0 `SharedBackendParams::kNNCacheSizeId` 的默认值
/// (`src/neural/shared_params.cc:63-82`)。
pub const DEFAULT_NN_CACHE_SIZE: usize = 2_000_000;

/// px0 `CachedValue` (`src/neural/memcache.cc:48-55`) 的 Rust 所有权版本。
#[derive(Clone, Debug)]
pub(crate) struct CachedEval {
    pub result: EvalResult,
    pub num_moves: usize,
}

/// px0 `HashKeyedCache<CachedValue>` 的最小 FIFO 语义。
#[derive(Debug)]
pub(crate) struct EvalCache {
    capacity: usize,
    values: HashMap<u64, CachedEval>,
    insertion_order: VecDeque<u64>,
}

impl EvalCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            values: HashMap::new(),
            insertion_order: VecDeque::new(),
        }
    }

    /// px0 `MemCache::GetCachedEvaluation` / collision guard
    /// (`memcache.cc:130-150`). An empty legal move list accepts a cached
    /// result; otherwise only an equal policy length is safe.
    pub(crate) fn get(&self, key: u64, requested_moves: usize) -> Option<EvalResult> {
        self.values
            .get(&key)
            .filter(|cached| requested_moves == 0 || cached.num_moves == requested_moves)
            .map(|cached| cached.result.clone())
    }

    /// px0 `HashKeyedCache::Insert` (`utils/cache.h:69-105`): preserve the
    /// first completed result for a key and evict oldest entries only.
    pub(crate) fn insert_if_absent(&mut self, key: u64, value: CachedEval) {
        if self.capacity == 0 || self.values.contains_key(&key) {
            return;
        }
        self.values.insert(key, value);
        self.insertion_order.push_back(key);
        while self.values.len() > self.capacity {
            let key = self.insertion_order.pop_front().expect("FIFO entry for cached value");
            self.values.remove(&key);
        }
    }

    /// px0 `HashKeyedCache::SetCapacity` (`src/utils/cache.h:143-167`).
    pub(crate) fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        while self.values.len() > self.capacity {
            let key = self.insertion_order.pop_front().expect("FIFO entry for cached value");
            self.values.remove(&key);
        }
    }

    /// px0 `HashKeyedCache::Clear` (`utils/cache.h:169-173`).
    pub(crate) fn clear(&mut self) {
        self.values.clear();
        self.insertion_order.clear();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_cache_preserves_first_value_and_evicts_oldest() {
        let result = EvalResult::default();
        let mut cache = EvalCache::new(2);
        cache.insert_if_absent(
            1,
            CachedEval {
                result: result.clone(),
                num_moves: 1,
            },
        );
        cache.insert_if_absent(
            2,
            CachedEval {
                result: result.clone(),
                num_moves: 2,
            },
        );
        cache.insert_if_absent(
            1,
            CachedEval {
                result: result.clone(),
                num_moves: 9,
            },
        );
        cache.insert_if_absent(3, CachedEval { result, num_moves: 3 });

        assert!(cache.get(1, 0).is_none());
        assert!(cache.get(2, 2).is_some());
        assert!(cache.get(3, 3).is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_capacity_shrinks_by_fifo_order() {
        let mut cache = EvalCache::new(2);
        for key in 1..=2 {
            cache.insert_if_absent(
                key,
                CachedEval {
                    result: EvalResult::default(),
                    num_moves: 0,
                },
            );
        }
        cache.set_capacity(1);
        assert!(cache.get(1, 0).is_none());
        assert!(cache.get(2, 0).is_some());
    }
}
