//! Reusable stream search state.
//!
//! Reference: LC3 overview, "Search" / "WatchdogWorker":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>.
//! Tree replacement follows px0 `NodeTree::ResetToPosition`
//! (`src/search/classic/node.cc:484-520`).

use std::sync::Arc;
use std::time::Instant;

use xiangqi_core::PositionHistory;

use crate::callbacks::{ThinkingInfo, Wdl};
use crate::neural::backend::Backend;
use crate::EnginError;

use super::{
    best_mate, best_move_filtered, principal_variation_filtered, root_stats, GcStats, NodeKey, NodeRepository, Search,
    SearchConfig, SearchControl, SearchGeneration, SearchLimits, Stats, Tree,
};

/// Read-only root view owned by the watchdog, never by a search worker.
#[derive(Clone)]
pub(crate) struct WatchdogSnapshot {
    repository: Arc<NodeRepository>,
    root_key: NodeKey,
    root_is_black: bool,
    root_move_filter: Vec<xiangqi_core::Move>,
}

impl WatchdogSnapshot {
    pub fn thinking_info(&self, stats: Stats, started: Instant) -> ThinkingInfo {
        let time = started.elapsed().as_millis() as i64;
        let nodes = stats.completed_playouts as i64;
        let nps = if time == 0 { 0 } else { (nodes * 1000 / time) as i32 };
        let eps = if time == 0 {
            0
        } else {
            (stats.network_evaluations as i64 * 1000 / time) as i32
        };
        let Some(root) = root_stats(&self.repository, self.root_key) else {
            return ThinkingInfo {
                depth: stats.average_depth.min(i32::MAX as u64) as i32,
                seldepth: stats.max_depth.min(i32::MAX as u64) as i32,
                time,
                nodes,
                nps,
                eps,
                ..ThinkingInfo::default()
            };
        };

        // A root node stores the incoming-edge/mover perspective. UCI reports
        // the side to move, hence the sign flip (LC3 glossary: `v = w - l`).
        let wl = (-root.q).clamp(-1.0, 1.0);
        let draw = root.draw.clamp(0.0, 1.0);
        let win = ((1.0 - draw + wl) * 0.5).clamp(0.0, 1.0);
        let loss = ((1.0 - draw - wl) * 0.5).clamp(0.0, 1.0);
        let mate = best_mate(&self.repository, self.root_key, &self.root_move_filter);
        ThinkingInfo {
            depth: stats.average_depth.min(i32::MAX as u64) as i32,
            seldepth: stats.max_depth.min(i32::MAX as u64) as i32,
            time,
            nodes,
            nps,
            eps,
            mate,
            score: mate.is_none().then_some((wl * 1000.0).round() as i32),
            wdl: Some(Wdl {
                w: (win * 1000.0).round() as i32,
                d: (draw * 1000.0).round() as i32,
                l: (loss * 1000.0).round() as i32,
            }),
            pv: principal_variation_filtered(
                &self.repository,
                self.root_key,
                self.root_is_black,
                &self.root_move_filter,
            ),
            ..ThinkingInfo::default()
        }
    }
}

/// Completed stream result consumed by the outer Engine.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub stats: Stats,
    pub best_move: Option<xiangqi_core::Move>,
    pub principal_variation: Vec<xiangqi_core::Move>,
}

/// Reusable stream state: backend, retained tree, and search generation.
pub(crate) struct SearchState {
    backend: Arc<dyn Backend>,
    tree: Option<Tree>,
    next_generation: u64,
}

/// One started stream search; its owner runs and joins all workers.
pub(crate) struct RunningSearch {
    search: Search,
    root_is_black: bool,
    root_move_filter: Vec<xiangqi_core::Move>,
}

impl RunningSearch {
    pub fn control(&self) -> SearchControl {
        self.search.control()
    }

    pub fn watchdog_snapshot(&self) -> WatchdogSnapshot {
        WatchdogSnapshot {
            repository: Arc::clone(self.search.repository()),
            root_key: self.search.root_key(),
            root_is_black: self.root_is_black,
            root_move_filter: self.root_move_filter.clone(),
        }
    }

    pub fn run(mut self, limits: SearchLimits) -> Result<SearchResult, EnginError> {
        let stats = self.search.run_with_limits(limits)?;
        let best_move = best_move_filtered(
            self.search.repository(),
            self.search.root_key(),
            self.root_is_black,
            &self.root_move_filter,
        );
        let principal_variation = principal_variation_filtered(
            self.search.repository(),
            self.search.root_key(),
            self.root_is_black,
            &self.root_move_filter,
        );
        self.search.stop_and_join();
        Ok(SearchResult {
            stats,
            best_move,
            principal_variation,
        })
    }
}

impl SearchState {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            tree: None,
            next_generation: 0,
        }
    }

    /// Applies a complete UCI position history after any active search has
    /// stopped. Retained prefixes reuse the tree; unrelated lines rebuild it.
    pub fn set_position(&mut self, history: Arc<PositionHistory>) -> Result<GcStats, EnginError> {
        match self.tree.as_mut() {
            Some(tree) => tree.reset_to_history(history),
            None => {
                self.tree = Some(Tree::new(history));
                Ok(GcStats::default())
            }
        }
    }

    /// Starts one owned search. This state keeps the resulting tree
    /// only after all worker threads joined, which is the reservation boundary
    /// required before the next `set_position` can prune or rewind it.
    pub fn begin_search(&mut self, searchmoves: &[String]) -> Result<RunningSearch, EnginError> {
        let tree = self
            .tree
            .as_ref()
            .ok_or(EnginError::Uci("position is not configured".into()))?;
        // px0 `StringsToMovelist` (`src/search/classic/wrapper.cc:78-100`):
        // retain legal root requests and reject a non-empty list with none.
        let board = tree.root_history().last().board();
        let legal_moves = board.generate_legal_moves();
        let root_move_filter: Vec<_> = searchmoves
            .iter()
            .filter_map(|move_text| board.parse_move(move_text).ok())
            .filter(|mv| legal_moves.contains(mv))
            .collect();
        if !searchmoves.is_empty() && root_move_filter.is_empty() {
            return Err(EnginError::Uci("No legal searchmoves.".into()));
        }
        self.next_generation = self.next_generation.wrapping_add(1);
        let config = SearchConfig {
            root_move_filter: root_move_filter.clone(),
            ..SearchConfig::default()
        };
        let search = Search::new_with_tree(
            Arc::clone(&self.backend),
            SearchGeneration(self.next_generation),
            tree,
            config,
        );
        let root_is_black = tree.root_history().last().is_black_to_move();
        Ok(RunningSearch {
            search,
            root_is_black,
            root_move_filter,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use crate::neural::backend::UniformBackend;

    use super::{SearchLimits, SearchState};

    #[test]
    fn state_reuses_tree_between_completed_searches() {
        let mut state = SearchState::new(Arc::new(UniformBackend::default()));
        let start = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        state
            .set_position(Arc::new(PositionHistory::from_positions(start.positions())))
            .expect("set startpos");
        let first = state
            .begin_search(&[])
            .expect("start first")
            .run(SearchLimits {
                max_playouts: Some(8),
                deadline: None,
            })
            .expect("first search");
        let best = first.best_move.expect("best move");

        let next = GameState::from_fen_moves(STARTPOS_FEN, &[best.to_string()]).expect("played move");
        state
            .set_position(Arc::new(PositionHistory::from_positions(next.positions())))
            .expect("advance tree");
        let second = state
            .begin_search(&[])
            .expect("start second")
            .run(SearchLimits {
                max_playouts: Some(4),
                deadline: None,
            })
            .expect("second search");
        assert!(second.stats.completed_playouts >= 4);
        assert!(second.best_move.is_some());
    }
}
