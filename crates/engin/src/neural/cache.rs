//! px0 `src/neural/memcache.cc:38-190` 的 NN cache 存储。
//!
//! px0 `HashKeyedCache` 不维护 LRU，插入已有 key 时不替换，并按 FIFO 淘汰
//! （`src/utils/cache.h:35-57,69-105,214-230`）。本项目改用 `quick_cache`
//! 的分片 S3-FIFO 容器，淘汰策略因此不再逐项等同 px0；key、collision guard
//! 与 completed-only 回填时序仍由 `CachingBackend` 保持。

use std::sync::Arc;

use super::backend::EvalResult;
use quick_cache::sync::{Cache, EntryAction, EntryResult};

/// px0 `SharedBackendParams::kNNCacheSizeId` 的默认值
/// (`src/neural/shared_params.cc:63-82`)。
pub const DEFAULT_NN_CACHE_SIZE: usize = 2_000_000;

/// px0 `CachedValue` (`src/neural/memcache.cc:48-55`) 的 Rust 所有权版本。
#[derive(Clone, Debug)]
pub(crate) struct CachedEval {
    pub result: Arc<EvalResult>,
    pub num_moves: usize,
}

/// 共享 NN 结果缓存。`quick_cache` 提供通用并发容器；wrapper 保留 px0 专用的 key 和
/// 合法着数量规则。
#[derive(Debug)]
pub(crate) struct EvalCache {
    values: Cache<u64, CachedEval>,
}

impl EvalCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            values: Cache::new(capacity),
        }
    }

    /// px0 `MemCache::GetCachedEvaluation` / collision guard
    /// （`memcache.cc:130-150`）。空合法着列表可接受缓存结果；否则只有相同 policy
    /// 长度才安全。
    pub(crate) fn get(&self, key: u64, requested_moves: usize) -> Option<Arc<EvalResult>> {
        self.values
            .get(&key)
            .filter(|cached| requested_moves == 0 || cached.num_moves == requested_moves)
            .map(|cached| cached.result)
    }

    /// 保留一个 key 的第一份完整结果。`quick_cache` 负责 S3-FIFO 淘汰策略；px0
    /// `HashKeyedCache::Insert` 只作为不替换缓存契约的参考（`utils/cache.h:69-105`）。
    pub(crate) fn insert_if_absent(&self, key: u64, value: CachedEval) {
        match self.values.entry(&key, None, |_, _| EntryAction::Retain(())) {
            EntryResult::Vacant(guard) => {
                let _ = guard.insert(value);
            }
            EntryResult::Retained(()) | EntryResult::Timeout => {}
            EntryResult::Removed(_, _) | EntryResult::Replaced(_, _) => unreachable!("retain-only cache entry"),
        }
    }

    /// px0 `HashKeyedCache::SetCapacity` (`src/utils/cache.h:143-167`).
    pub(crate) fn set_capacity(&self, capacity: usize) {
        self.values.set_capacity(capacity as u64);
    }

    /// px0 `HashKeyedCache::Clear` (`utils/cache.h:169-173`).
    pub(crate) fn clear(&self) {
        self.values.clear();
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
    fn cache_preserves_first_value_for_an_existing_key() {
        let result = Arc::new(EvalResult::default());
        let cache = EvalCache::new(2);
        cache.insert_if_absent(
            1,
            CachedEval {
                result: Arc::clone(&result),
                num_moves: 1,
            },
        );
        cache.insert_if_absent(
            2,
            CachedEval {
                result: Arc::clone(&result),
                num_moves: 2,
            },
        );
        cache.insert_if_absent(
            1,
            CachedEval {
                result: Arc::clone(&result),
                num_moves: 9,
            },
        );
        assert!(cache.get(1, 1).is_some());
        assert!(cache.get(1, 9).is_none());
        assert!(cache.get(2, 2).is_some());
        assert!(Arc::ptr_eq(&cache.get(1, 1).expect("cached value"), &result));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_capacity_shrinks_to_requested_limit() {
        let cache = EvalCache::new(2);
        for key in 1..=2 {
            cache.insert_if_absent(
                key,
                CachedEval {
                    result: Arc::new(EvalResult::default()),
                    num_moves: 0,
                },
            );
        }
        cache.set_capacity(1);
        assert!(cache.len() <= 1);
    }
}
