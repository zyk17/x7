//! 引擎内通用小工具（时间、哈希混合、确定性 PRNG）。

use std::time::{SystemTime, UNIX_EPOCH};

pub type TimePoint = i64;

/// 当前 UTC 毫秒时间戳。
pub fn now() -> TimePoint {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as TimePoint
}

/// boost::hash_combine 风格混合（用于杂凑种子）。
#[inline]
pub fn hash_combine(seed: &mut u64, v: u64) {
    *seed ^= v.wrapping_add(0x9e3779b9).wrapping_add(*seed << 6).wrapping_add(*seed >> 2);
}

// ── xorshift64* PRNG ───────────────────────────────────────────────────────────

/// xorshift64* 伪随机数生成器（Sebastiano Vigna, 公有领域）。
/// Zobrist 键需在多平台 **确定一致**，故不用 `std` 默认 PRNG。
pub struct PRNG {
    s: u64,
}

impl PRNG {
    pub fn new(seed: u64) -> Self {
        debug_assert!(seed != 0);
        PRNG { s: seed }
    }

    fn rand64(&mut self) -> u64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        self.s.wrapping_mul(2685821657736338717)
    }

    pub fn rand<T: From<u64>>(&mut self) -> T {
        T::from(self.rand64())
    }
}
