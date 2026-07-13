//! px0 `stoppers/stoppers.h`、`stoppers.cc:39-131`、`common.cc:118-165`、`simple.cc:74-126`。

use crate::uci_loop::GoParams;
use xiangqi_core::PositionHistory;

use super::timemgr::{IterationStats, StoppersHints};

/// px0 `SearchStopper` (`timemgr.h:88-102`)。
pub trait SearchStopper: Send {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool;
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
}

impl SearchStopper for TimeLimitStopper {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool {
        hints.update_estimated_remaining_time_ms(self.time_limit_ms - stats.time_since_movestart);
        stats.nodes_since_movestart >= 1 && stats.time_since_movestart >= self.time_limit_ms
    }
}

/// px0 `PopulateCommonUciStoppers` + `SimpleTimeManager::GetStopper` 子集。
pub fn build_search_stoppers(
    params: &GoParams,
    history: &PositionHistory,
    nodes_as_playouts: bool,
) -> ChainedSearchStopper {
    let position = history.last();
    let mut chain = ChainedSearchStopper::new();
    if let Some(nodes) = params.nodes.filter(|&n| n > 0) {
        if nodes_as_playouts {
            chain.add(Box::new(PlayoutsStopper::new(nodes as i64, true)));
        } else {
            chain.add(Box::new(VisitsStopper::new(nodes as i64, true)));
        }
    }
    if let Some(movetime) = params.movetime.filter(|&t| t >= 0) {
        chain.add(Box::new(TimeLimitStopper::new(movetime)));
    }
    if !params.infinite && !params.ponder {
        let is_black = history.is_black_to_move();
        let time = if is_black { params.btime } else { params.wtime };
        if let Some(remaining) = time.filter(|&t| t > 0) {
            let inc = if is_black {
                params.binc.unwrap_or(0)
            } else {
                params.winc.unwrap_or(0)
            };
            let overhead = 50i64;
            let available = (remaining - overhead).max(0);
            let ply = position.game_ply() as f32;
            let pct = (0.014 + ply * 0.00049).min(0.5);
            let budget = (available as f32 * pct + inc as f32 * 0.5).round() as i64;
            chain.add(Box::new(TimeLimitStopper::new(budget.max(1))));
        }
    }
    chain
}
