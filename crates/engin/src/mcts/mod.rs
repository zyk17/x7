//! MCTS 主线骨架。
//!
//! 这里定义当前项目搜索主线需要的稳定接口与基础数据结构。

mod backend;
mod config;
mod coordinator;
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

/// px0 风格搜索统计：UCI nodes = total_playouts + initial_visits。
pub struct SearchStats {
    initial_visits: AtomicU32,
    total_playouts: AtomicU32,
    cum_depth: AtomicU64,
    max_depth: AtomicU32,
    nps_start: RwLock<Instant>,
    nns_started: AtomicBool,
}

impl SearchStats {
    pub fn new(initial_visits: u32) -> Self {
        Self {
            initial_visits: AtomicU32::new(initial_visits),
            total_playouts: AtomicU32::new(0),
            cum_depth: AtomicU64::new(0),
            max_depth: AtomicU32::new(0),
            nps_start: RwLock::new(Instant::now()),
            nns_started: AtomicBool::new(false),
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
        self.nps_start
            .read()
            .map(|start| start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64)
            .unwrap_or(0)
    }

    /// px0：首次 NN batch 后才开始计 nps。
    pub fn mark_first_batch(&self) {
        if !self.nns_started.swap(true, Ordering::Relaxed) {
            if let Ok(mut start) = self.nps_start.write() {
                *start = Instant::now();
            }
        }
    }

    fn record_minibatch_depth(&self, iteration: &worker::SearchIteration) {
        let mut cum = 0u64;
        let mut max_d = 0u32;
        for pending in &iteration.pending {
            let depth = pending.path.len() as u32;
            cum += u64::from(depth) * u64::from(pending.multivisit);
            max_d = max_d.max(depth);
        }
        if cum > 0 {
            self.cum_depth.fetch_add(cum, Ordering::Relaxed);
        }
        if max_d > 0 {
            self.max_depth.fetch_max(max_d, Ordering::Relaxed);
        }
    }

    pub(crate) fn add_minibatch(&self, iteration: &worker::SearchIteration) {
        let playouts = iteration.playouts;
        if playouts == 0 {
            return;
        }
        self.total_playouts.fetch_add(playouts, Ordering::Relaxed);
        self.record_minibatch_depth(iteration);
    }

    /// 并行路径：CAS 提交 playout，避免超预算。
    pub(crate) fn try_add_minibatch(&self, budget: &MctsBudget, iteration: &worker::SearchIteration) -> bool {
        let playouts = iteration.playouts;
        if playouts == 0 {
            return true;
        }
        loop {
            let committed = self.total_playouts.load(Ordering::Acquire);
            if worker::remaining_playout_budget(budget, committed, 0, self.initial_visits(), playouts) < playouts {
                return false;
            }
            match self.total_playouts.compare_exchange(
                committed,
                committed.saturating_add(playouts),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.record_minibatch_depth(iteration);
                    return true;
                }
                Err(_) => continue,
            }
        }
    }

    pub(crate) fn rollback_minibatch(&self, iteration: &worker::SearchIteration) {
        let playouts = iteration.playouts;
        if playouts > 0 {
            self.total_playouts.fetch_sub(playouts, Ordering::Relaxed);
        }
    }
}
