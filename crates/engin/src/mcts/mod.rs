//! MCTS 主线骨架。
//!
//! 这里定义当前项目搜索主线需要的稳定接口与基础数据结构。
//! lc0 classic 搜索基建；并发按 lc0 单写者树锁 + 本批 CancelCollisions。

mod backend;
mod config;
mod engine;
mod node;
mod policy_value;
mod search;
mod tree;
mod worker;

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

pub use config::{MctsBudget, MctsConfig};
pub use engine::{MctsEngine, MctsMoveStat, MctsSearchProgress, MctsSearchResult};
pub use node::{EdgeStats, MctsNode, MctsNodeId, TerminalKind};
pub use policy_value::{
    OnnxPolicyValueEval, PolicyValueEval, PolicyValueInput, PolicyValueOutput, PolicyValueTask, SharedPolicy,
};
pub use tree::MctsTree;

/// lc0 风格搜索统计：UCI nodes = total_playouts + initial_visits（search.cc:276,941）。
pub struct SearchStats {
    initial_visits: AtomicU32,
    total_playouts: AtomicU32,
    cum_depth: AtomicU64,
    max_depth: AtomicU32,
    nps_start: RwLock<Option<Instant>>,
    nns_started: AtomicBool,
    retry_without_playout: AtomicU64,
}

impl SearchStats {
    pub fn new(initial_visits: u32) -> Self {
        Self {
            initial_visits: AtomicU32::new(initial_visits),
            total_playouts: AtomicU32::new(0),
            cum_depth: AtomicU64::new(0),
            max_depth: AtomicU32::new(0),
            nps_start: RwLock::new(None),
            nns_started: AtomicBool::new(false),
            retry_without_playout: AtomicU64::new(0),
        }
    }

    pub fn total_playouts(&self) -> u32 {
        self.total_playouts.load(Ordering::Relaxed)
    }

    pub fn initial_visits(&self) -> u32 {
        self.initial_visits.load(Ordering::Relaxed)
    }

    pub(crate) fn subtract_initial_visits(&self, delta: u32) {
        self.initial_visits.fetch_sub(delta, Ordering::Relaxed);
    }

    pub fn uci_nodes(&self) -> usize {
        (self.total_playouts() + self.initial_visits()) as usize
    }

    pub fn depth(&self) -> u32 {
        let playouts = self.total_playouts();
        if playouts > 0 {
            (self.cum_depth.load(Ordering::Relaxed) / playouts as u64) as u32
        } else {
            0
        }
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth.load(Ordering::Relaxed)
    }

    pub fn nps_elapsed_ms(&self) -> u64 {
        if !self.nns_started.load(Ordering::Relaxed) {
            return 0;
        }
        self.nps_start
            .read()
            .ok()
            .and_then(|start| start.map(|t| t.elapsed().as_millis().min(u128::from(u64::MAX)) as u64))
            .unwrap_or(0)
    }

    pub fn retry_without_playout(&self) -> u64 {
        self.retry_without_playout.load(Ordering::Relaxed)
    }

    /// lc0 风格 nps：仅首批 NN 后开始计时的 playouts/s（search.cc:278-285）；未开始时返回 0。
    pub fn playouts_per_second(playouts: u32, nps_elapsed_ms: u64) -> u64 {
        if nps_elapsed_ms > 0 {
            (playouts as u128 * 1000 / u128::from(nps_elapsed_ms)) as u64
        } else {
            0
        }
    }

    /// lc0：首次 NN batch 后才开始计 nps（search.cc:915-917）。
    pub fn mark_first_batch(&self) {
        if !self.nns_started.swap(true, Ordering::Relaxed) {
            if let Ok(mut start) = self.nps_start.write() {
                *start = Some(Instant::now());
            }
        }
    }

    pub(crate) fn add_retry_without_playout(&self) {
        self.retry_without_playout.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_backup(&self, pending: &worker::PendingNode) {
        self.total_playouts
            .fetch_add(pending.multivisit, Ordering::Relaxed);
        let depth = worker::playout_depth(pending);
        self.max_depth.fetch_max(depth, Ordering::Relaxed);
        self.cum_depth.fetch_add(
            u64::from(depth) * u64::from(pending.multivisit),
            Ordering::Relaxed,
        );
    }

    pub(crate) fn commit_minibatch(&self, iteration: &worker::SearchIteration) {
        if iteration.seldepth > 0 {
            self.max_depth.fetch_max(iteration.seldepth, Ordering::Relaxed);
        }
    }

    /// 并行路径：backup 前检查预算（playout 在 `do_backup_single` 内提交）。
    pub(crate) fn try_commit_minibatch(&self, budget: &MctsBudget, iteration: &worker::SearchIteration) -> bool {
        let playouts = worker::pending_playouts(iteration);
        if playouts == 0 {
            return true;
        }
        worker::remaining_playout_budget(
            budget,
            self.total_playouts(),
            0,
            self.initial_visits(),
            playouts,
        ) >= playouts
    }
}

#[cfg(test)]
mod tests {
    use super::SearchStats;

    #[test]
    fn nps_elapsed_is_zero_before_first_nn_batch() {
        let stats = SearchStats::new(0);
        assert_eq!(stats.nps_elapsed_ms(), 0);
        stats.mark_first_batch();
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(stats.nps_elapsed_ms() > 0);
    }

    #[test]
    fn playouts_per_second_waits_for_nn_timing() {
        assert_eq!(SearchStats::playouts_per_second(128, 0), 0);
        assert_eq!(SearchStats::playouts_per_second(1000, 1000), 1000);
    }
}
