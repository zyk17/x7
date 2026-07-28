//! Search stopper chain and individual stop conditions.
//!
//! Reference: px0 `stoppers/stoppers.h`, `stoppers.cc:39-131`, and
//! `common.cc:118-165`.

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
        // px0 checks only elapsed time here (`stoppers.cc:120-129`). Its
        // caller already guards an entirely unexpanded root, while a reused
        // root may legitimately stop without forcing one new playout.
        stats.time_since_movestart >= self.time_limit_ms
    }
}

/// px0 `PopulateCommonUciStoppers` (`stoppers/common.cc:118-165`)。
pub fn build_search_stoppers(
    params: &GoParams,
    nodes_as_playouts: bool,
    move_overhead_ms: i64,
    time_manager_stopper: Option<Box<dyn SearchStopper>>,
) -> ChainedSearchStopper {
    // px0 always installs a visit cap, even when UCI did not send `go nodes`,
    // to bound tree growth at 4_000_000_000 visits (`common.cc:133-145`).
    const PX0_MAX_VISITS: i64 = 4_000_000_000;

    let mut chain = ChainedSearchStopper::new();
    if let Some(stopper) = time_manager_stopper {
        chain.add(stopper);
    }
    let mut visit_limit = PX0_MAX_VISITS;
    if let Some(nodes) = params.nodes {
        if nodes_as_playouts {
            chain.add(Box::new(PlayoutsStopper::new(nodes as i64, true)));
        } else {
            visit_limit = nodes as i64;
        }
    }
    chain.add(Box::new(VisitsStopper::new(visit_limit, true)));
    // px0's `infinite` also covers ponder/mate; the Rust port rejects those
    // untranslated modes before building this chain. `go infinite movetime N`
    // must therefore keep searching until `stop`, not honor `movetime`.
    if !params.infinite {
        if let Some(movetime) = params.movetime {
            chain.add(Box::new(TimeLimitStopper::new(movetime - move_overhead_ms)));
        }
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::{build_search_stoppers, SearchStopper, TimeLimitStopper};
    use crate::search::stoppers::timemgr::{IterationStats, StoppersHints};
    use crate::uci_loop::GoParams;

    /// px0 `TimeLimitStopper::ShouldStop` depends only on elapsed time
    /// (`src/search/classic/stoppers/stoppers.cc:120-129`). A tree-reused
    /// root can therefore honor `go movetime 0` before a new playout starts.
    #[test]
    fn time_limit_stops_a_reused_tree_without_a_new_playout() {
        let mut stopper = TimeLimitStopper::new(0);
        let stats = IterationStats {
            total_nodes: 128,
            nodes_since_movestart: 0,
            ..IterationStats::default()
        };
        assert!(stopper.should_stop(&stats, &mut StoppersHints::default()));
    }

    /// px0 always appends `VisitsStopper(4_000_000_000)` even without a UCI
    /// `go nodes` request (`common.cc:133-145`).
    #[test]
    fn default_chain_keeps_the_px0_tree_visit_hard_cap() {
        let mut chain = build_search_stoppers(&GoParams::default(), false, 0, None);
        assert!(!chain.should_stop(
            &IterationStats {
                total_nodes: 3_999_999_999,
                ..IterationStats::default()
            },
            &mut StoppersHints::default(),
        ));
        assert!(chain.should_stop(
            &IterationStats {
                total_nodes: 4_000_000_000,
                ..IterationStats::default()
            },
            &mut StoppersHints::default(),
        ));
    }

    /// px0 passes parsed `go nodes` through unchanged, so zero is a valid
    /// immediate visit limit rather than a Rust-only parser rejection
    /// (`chess/uciloop.cc:230-237`, `stoppers/common.cc:133-145`).
    #[test]
    fn zero_nodes_keeps_px0_immediate_visit_limit() {
        let mut chain = build_search_stoppers(
            &GoParams {
                nodes: Some(0),
                ..GoParams::default()
            },
            false,
            0,
            None,
        );
        assert!(chain.should_stop(&IterationStats::default(), &mut StoppersHints::default()));
    }

    /// px0 excludes `go movetime` from the chain for an infinite search
    /// (`stoppers/common.cc:123,147-151`).
    #[test]
    fn infinite_go_ignores_movetime_like_px0() {
        let mut chain = build_search_stoppers(
            &GoParams {
                infinite: true,
                movetime: Some(0),
                ..GoParams::default()
            },
            false,
            0,
            None,
        );
        assert!(!chain.should_stop(
            &IterationStats {
                total_nodes: 1,
                time_since_movestart: 1_000,
                ..IterationStats::default()
            },
            &mut StoppersHints::default(),
        ));
    }
}
