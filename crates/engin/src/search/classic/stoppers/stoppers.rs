//! px0 `stoppers/stoppers.h`、`stoppers.cc:39-131`、`common.cc:118-165`。

use super::timemgr::{IterationStats, StoppersHints};
use crate::uci_loop::GoParams;

/// px0 `SearchStopper` (`timemgr.h:88-102`)。
pub trait SearchStopper: Send {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool;

    /// px0 `SearchStopper::OnSearchDone` (`stoppers.h:48-51`).
    fn on_search_done(&mut self, _stats: &IterationStats) {}
}

/// px0 `ChainedSearchStopper` (`stoppers.cc:39-53`)。
pub struct ChainedSearchStopper {
    stoppers: Vec<Box<dyn SearchStopper>>,
}

impl ChainedSearchStopper {
    pub fn new() -> Self {
        Self { stoppers: Vec::new() }
    }

    pub fn add(&mut self, stopper: Box<dyn SearchStopper>) {
        self.stoppers.push(stopper);
    }

    pub fn is_empty(&self) -> bool {
        self.stoppers.is_empty()
    }
}

impl Default for ChainedSearchStopper {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchStopper for ChainedSearchStopper {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool {
        self.stoppers
            .iter_mut()
            .any(|stopper| stopper.should_stop(stats, hints))
    }

    fn on_search_done(&mut self, stats: &IterationStats) {
        for stopper in &mut self.stoppers {
            stopper.on_search_done(stats);
        }
    }
}

/// px0 `VisitsStopper` (`stoppers.cc:59-70`)。
pub struct VisitsStopper {
    nodes_limit: i64,
    populate_remaining_playouts: bool,
}

impl VisitsStopper {
    pub const fn new(nodes_limit: i64, populate_remaining_playouts: bool) -> Self {
        Self {
            nodes_limit,
            populate_remaining_playouts,
        }
    }
}

impl SearchStopper for VisitsStopper {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool {
        if self.populate_remaining_playouts {
            hints.update_estimated_remaining_playouts(self.nodes_limit - stats.total_nodes);
        }
        stats.total_nodes >= self.nodes_limit
    }
}

/// px0 `PlayoutsStopper` (`stoppers.cc:76-88`)。
pub struct PlayoutsStopper {
    nodes_limit: i64,
    populate_remaining_playouts: bool,
}

impl PlayoutsStopper {
    pub const fn new(nodes_limit: i64, populate_remaining_playouts: bool) -> Self {
        Self {
            nodes_limit,
            populate_remaining_playouts,
        }
    }
}

impl SearchStopper for PlayoutsStopper {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool {
        if self.populate_remaining_playouts {
            hints.update_estimated_remaining_playouts(self.nodes_limit - stats.nodes_since_movestart);
        }
        stats.nodes_since_movestart >= self.nodes_limit
    }
}

/// px0 `TimeLimitStopper` (`stoppers.cc:117-129`)。
pub struct TimeLimitStopper {
    time_limit_ms: i64,
}

impl TimeLimitStopper {
    pub const fn new(time_limit_ms: i64) -> Self {
        Self { time_limit_ms }
    }

    /// px0 `TimeLimitStopper::GetTimeLimitMs` (`stoppers.cc:131`).
    pub const fn time_limit_ms(&self) -> i64 {
        self.time_limit_ms
    }
}

impl SearchStopper for TimeLimitStopper {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool {
        hints.update_estimated_remaining_time_ms(self.time_limit_ms - stats.time_since_movestart);
        stats.nodes_since_movestart >= 1 && stats.time_since_movestart >= self.time_limit_ms
    }
}

/// px0 `PopulateCommonUciStoppers` (`stoppers/common.cc:118-165`)。
pub fn build_search_stoppers(
    params: &GoParams,
    nodes_as_playouts: bool,
    move_overhead_ms: i64,
    time_manager_stopper: Option<Box<dyn SearchStopper>>,
) -> ChainedSearchStopper {
    let mut chain = ChainedSearchStopper::new();
    if let Some(stopper) = time_manager_stopper {
        chain.add(stopper);
    }
    if let Some(nodes) = params.nodes.filter(|&n| n > 0) {
        if nodes_as_playouts {
            chain.add(Box::new(PlayoutsStopper::new(nodes as i64, true)));
        } else {
            chain.add(Box::new(VisitsStopper::new(nodes as i64, true)));
        }
    }
    if let Some(movetime) = params.movetime.filter(|&t| t >= 0) {
        chain.add(Box::new(TimeLimitStopper::new(movetime - move_overhead_ms)));
    }
    chain
}
