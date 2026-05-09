//! 置换表（Transposition Table）：Zobrist 键 + 分簇替换，供 Alpha-Beta 复用子树结果。
//! 结构参考 `pikafish-rust::tt`；条目存 **完整 64 位键** 以避免仅比较高 16 位的误命中。

use xiangqi_core::types::{Bound, Key, Move};

const CLUSTER_SIZE: usize = 4;

#[derive(Clone, Copy)]
struct TtEntry {
    key: Key,
    depth: u8,
    bound: u8,
    score: i32,
    best_raw: u16,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            key: 0,
            depth: 0,
            bound: Bound::None as u8,
            score: 0,
            best_raw: 0,
        }
    }
}

impl TtEntry {
    fn is_empty(self) -> bool {
        self.bound == Bound::None as u8
    }
}

/// 置换表（单线程搜索下由调用方持锁；与 `uci` 中 `Mutex` 组合使用）。
pub struct TranspositionTable {
    clusters: Vec<[TtEntry; CLUSTER_SIZE]>,
}

impl TranspositionTable {
    /// `size_mb`：近似内存上限（MiB），实际略小（对齐到整簇）。
    pub fn new(size_mb: usize) -> Self {
        let bytes = size_mb.saturating_mul(1024 * 1024);
        let cluster_bytes = std::mem::size_of::<[TtEntry; CLUSTER_SIZE]>().max(1);
        let n = (bytes / cluster_bytes).max(1);
        Self {
            clusters: vec![[TtEntry::default(); CLUSTER_SIZE]; n],
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.clusters {
            *c = [TtEntry::default(); CLUSTER_SIZE];
        }
    }

    #[inline]
    fn index(&self, key: Key) -> usize {
        key as usize % self.clusters.len()
    }

    /// 查找与 `key` 完全一致的条目；`depth_left` 为当前节点剩余层数（与存入时语义一致）。
    pub fn probe(&self, key: Key, depth_left: u32) -> TtProbe {
        let idx = self.index(key);
        let cluster = &self.clusters[idx];
        let mut out = TtProbe::default();
        for e in cluster {
            if e.is_empty() || e.key != key {
                continue;
            }
            let bm = Move::from_raw(e.best_raw);
            out.best_move = bm.is_ok().then_some(bm);
            if u32::from(e.depth) >= depth_left {
                out.usable_score = true;
                out.score = e.score;
                out.bound = bound_from_u8(e.bound);
            }
            break;
        }
        out
    }

    pub fn store(&mut self, key: Key, depth_left: u32, score: i32, bound: Bound, best: Move) {
        debug_assert!(bound != Bound::None);
        let d = depth_left.min(255) as u8;
        let idx = self.index(key);
        let cluster = &mut self.clusters[idx];

        let mut replace = 0usize;
        let mut found = false;
        for (i, e) in cluster.iter().enumerate() {
            if e.is_empty() {
                replace = i;
                found = true;
                break;
            }
            if e.key == key {
                replace = i;
                found = true;
                break;
            }
        }

        if !found {
            let mut worst_d = 255u8;
            for (i, e) in cluster.iter().enumerate() {
                if e.depth < worst_d {
                    worst_d = e.depth;
                    replace = i;
                }
            }
        }

        cluster[replace] = TtEntry {
            key,
            depth: d,
            bound: bound as u8,
            score,
            best_raw: if best.is_ok() { best.raw() } else { 0 },
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TtProbe {
    pub best_move: Option<Move>,
    pub usable_score: bool,
    pub score: i32,
    pub bound: Bound,
}

impl Default for TtProbe {
    fn default() -> Self {
        Self {
            best_move: None,
            usable_score: false,
            score: 0,
            bound: Bound::None,
        }
    }
}

impl TtProbe {
    /// 若可截断则返回应返回的分值。
    pub fn cutoff_score(self, alpha: i32, beta: i32) -> Option<i32> {
        if !self.usable_score {
            return None;
        }
        match self.bound {
            Bound::Exact => Some(self.score),
            Bound::Lower if self.score >= beta => Some(self.score),
            Bound::Upper if self.score <= alpha => Some(self.score),
            _ => None,
        }
    }
}

#[inline]
fn bound_from_u8(b: u8) -> Bound {
    match b {
        1 => Bound::Upper,
        2 => Bound::Lower,
        3 => Bound::Exact,
        _ => Bound::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::types::Square;

    #[test]
    fn store_probe_roundtrip() {
        let mut tt = TranspositionTable::new(1);
        let k: Key = 0x1234_5678_9abc_def0;
        let mv = Move::make(Square::SQ_A0, Square::SQ_A1);
        tt.store(k, 3, 42, Bound::Exact, mv);
        let p = tt.probe(k, 3);
        assert!(p.usable_score);
        assert_eq!(p.score, 42);
        assert_eq!(p.bound, Bound::Exact);
        assert_eq!(p.best_move, Some(mv));
    }

    #[test]
    fn shallow_entry_still_gives_move() {
        let mut tt = TranspositionTable::new(1);
        let k: Key = 0xaaaa_bbbb_cccc_dddd;
        let mv = Move::make(Square::SQ_B0, Square::SQ_B2);
        tt.store(k, 2, 100, Bound::Exact, mv);
        let p = tt.probe(k, 5);
        assert!(!p.usable_score);
        assert_eq!(p.best_move, Some(mv));
    }
}
