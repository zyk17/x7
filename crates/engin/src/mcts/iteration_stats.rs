//! lc0 `PopulateCommonIterationStats` / `MaybeTriggerStop`（search.cc:930-1001,617-646）。

use std::sync::atomic::Ordering;
use std::time::Instant;

use super::config::{MctsBudget, MctsConfig};
use super::node::{EdgeStats, MctsNodeId};
use super::tree::MctsTree;
use super::worker::pick_edge_visits;
use super::SearchStats;

/// lc0 `IterationStats::TimeUsageHint`（timemgr.h:59-60）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TimeUsageHint {
    #[default]
    Normal,
    NeedMoreTime,
    ImmediateMove,
}

/// lc0 `IterationStats`（timemgr.h:43-61）。
#[derive(Debug, Clone, Default)]
pub struct IterationStats {
    pub time_since_movestart_ms: i64,
    pub time_since_first_batch_ms: i64,
    pub total_nodes: u64,
    pub nodes_since_movestart: u32,
    pub batches_since_movestart: u64,
    pub average_depth: u32,
    pub mate_depth: i32,
    pub edge_n: Vec<u32>,
    pub win_found: bool,
    pub may_resign: bool,
    pub num_losing_edges: i32,
    pub time_usage_hint: TimeUsageHint,
}

/// lc0 `StoppersHints` 子集（timemgr.h:68-83）。
#[derive(Debug, Clone, Default)]
pub struct StoppersHints {
    pub remaining_time_ms: i64,
    pub remaining_playouts: i64,
    pub estimated_nps: Option<f32>,
}

impl StoppersHints {
    pub fn reset(&mut self) {
        self.remaining_time_ms = 0;
        self.remaining_playouts = 0;
        self.estimated_nps = None;
    }
}

fn edge_q_for_stats(
    tree: &MctsTree,
    edge: &EdgeStats,
    default_q: f32,
    draw_score: f32,
) -> f32 {
    if let Some(child_id) = edge.child {
        if let Some(child) = tree.get(child_id) {
            if child.visits > 0 {
                return child.mean_value_with_draw(draw_score);
            }
        }
    }
    default_q
}

fn edge_wl_for_stats(tree: &MctsTree, edge: &EdgeStats) -> f32 {
    if let Some(child_id) = edge.child {
        if let Some(child) = tree.get(child_id) {
            if child.is_terminal() {
                return child.wl;
            }
            if child.visits > 0 {
                return child.wl;
            }
        }
    }
    0.0
}

/// lc0 `Search::PopulateCommonIterationStats`（search.cc:930-1001）。
pub fn populate_common_iteration_stats(
    tree: &MctsTree,
    root_id: MctsNodeId,
    config: MctsConfig,
    search_stats: &SearchStats,
) -> IterationStats {
    let mut stats = IterationStats {
        time_since_movestart_ms: search_stats.time_since_start_ms(),
        time_since_first_batch_ms: search_stats.nps_elapsed_ms() as i64,
        total_nodes: search_stats.uci_nodes() as u64,
        nodes_since_movestart: search_stats.total_playouts(),
        batches_since_movestart: search_stats.total_batches(),
        average_depth: search_stats.depth(),
        mate_depth: i32::MAX,
        edge_n: Vec::new(),
        win_found: false,
        may_resign: true,
        num_losing_edges: 0,
        time_usage_hint: TimeUsageHint::Normal,
    };

    let Some(root) = tree.get(root_id) else {
        return stats;
    };
    if root.visits == 0 {
        return stats;
    }

    let draw_score = config.draw_score;
    let default_q = -root.mean_value_with_draw(draw_score);
    let mut max_q_plus_m = -1000.0f32;
    let mut max_n = 0u32;
    let mut max_n_has_max_q_plus_m = true;

    for edge in &root.children {
        let n = pick_edge_visits(tree, edge);
        stats.edge_n.push(n);
        let q = edge_q_for_stats(tree, edge, default_q, draw_score);
        let m = edge.get_m(root.m);
        let q_plus_m = q + m;
        let wl = edge_wl_for_stats(tree, edge);

        if n > 0 {
            if let Some(child_id) = edge.child {
                if let Some(child) = tree.get(child_id) {
                    if child.is_terminal() && wl > 0.0 {
                        stats.win_found = true;
                    }
                    if child.is_terminal() && wl < 0.0 {
                        stats.num_losing_edges += 1;
                    }
                    if child.is_terminal() && wl == 1.0 {
                        let depth = (child.m.round() as i32) / 2 + 1;
                        stats.mate_depth = stats.mate_depth.min(depth);
                    }
                }
            }
            if q > -0.98 {
                stats.may_resign = false;
            }
        }

        if max_n < n {
            max_n = n;
            max_n_has_max_q_plus_m = false;
        }
        if max_q_plus_m <= q_plus_m {
            max_n_has_max_q_plus_m = max_n == n;
            max_q_plus_m = q_plus_m;
        }
    }

    if !max_n_has_max_q_plus_m {
        stats.time_usage_hint = TimeUsageHint::NeedMoreTime;
    }

    stats
}

/// lc0 `ChainedSearchStopper::ShouldStop` 的**子集**（stoppers.cc:39-152）。
/// 当前仅覆盖 mate/depth/nodes/playouts/deadline；不含 SmartPruning/KLD/TimeManager hints 回写。
fn chained_stoppers_should_stop(
    budget: &MctsBudget,
    stats: &IterationStats,
    hints: &mut StoppersHints,
) -> bool {
    if let Some(mate) = budget.max_mate {
        if stats.mate_depth <= mate as i32 {
            return true;
        }
    }
    if let Some(depth) = budget.max_depth {
        if stats.average_depth >= depth {
            return true;
        }
    }
    if let Some(limit) = budget.max_nodes {
        hints.remaining_playouts = i64::from(limit).saturating_sub(stats.total_nodes as i64);
        if stats.total_nodes >= u64::from(limit) {
            return true;
        }
    }
    if let Some(limit) = budget.max_playouts {
        hints.remaining_playouts = i64::from(limit).saturating_sub(i64::from(stats.nodes_since_movestart));
        if stats.nodes_since_movestart >= limit {
            return true;
        }
    }
    if let Some(deadline) = budget.deadline {
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .min(i64::MAX as u128) as i64;
        hints.remaining_time_ms = remaining;
        if Instant::now() >= deadline {
            return true;
        }
    }
    false
}

/// lc0 `Search::MaybeTriggerStop` 的 stopper 判定子集（search.cc:617-632）。
/// 尚未接入 lc0 `FireStopInternal` / bestmove 发送 / `stopper_->OnSearchDone`。
pub fn maybe_trigger_stop(
    budget: &MctsBudget,
    config: MctsConfig,
    iteration: &IterationStats,
    _search_stats: &SearchStats,
    hints: &mut StoppersHints,
) -> bool {
    hints.reset();
    if config.nps_limit > 0 {
        hints.estimated_nps = Some(config.nps_limit as f32);
    }
    if iteration.total_nodes == 0 {
        return false;
    }
    if chained_stoppers_should_stop(budget, iteration, hints) {
        return true;
    }
    if budget
        .stop
        .as_ref()
        .is_some_and(|stop| stop.load(Ordering::SeqCst))
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mate_stopper_uses_iteration_mate_depth() {
        let budget = MctsBudget {
            max_mate: Some(3),
            ..Default::default()
        };
        let iteration = IterationStats {
            total_nodes: 1,
            mate_depth: 2,
            ..Default::default()
        };
        let stats = SearchStats::new(0);
        let mut hints = StoppersHints::default();
        assert!(maybe_trigger_stop(
            &budget,
            MctsConfig::default(),
            &iteration,
            &stats,
            &mut hints
        ));
        let iteration = IterationStats {
            total_nodes: 1,
            mate_depth: i32::MAX,
            ..Default::default()
        };
        assert!(!maybe_trigger_stop(
            &budget,
            MctsConfig::default(),
            &iteration,
            &stats,
            &mut hints
        ));
    }

    #[test]
    fn depth_stopper_uses_average_depth() {
        let budget = MctsBudget {
            max_depth: Some(4),
            ..Default::default()
        };
        let iteration = IterationStats {
            total_nodes: 8,
            average_depth: 4,
            ..Default::default()
        };
        let stats = SearchStats::new(0);
        let mut hints = StoppersHints::default();
        assert!(maybe_trigger_stop(
            &budget,
            MctsConfig::default(),
            &iteration,
            &stats,
            &mut hints
        ));
    }
}
