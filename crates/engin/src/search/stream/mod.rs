//! LC3-style streaming search foundation.
//!
//! Design references (LC3 code is not published in the local lc0 checkout):
//! - <https://lczero.org/dev/lc0/search/lc3/overview/>
//! - <https://lczero.org/dev/lc0/search/lc3/policy/>
//! - <https://lczero.org/dev/lc0/search/lc3/glossary/>
//!
//! This module deliberately does **not** reuse `classic::node` or
//! `classic::worker`. Selection and final-move rules are classic-aligned
//! (px0 PUCT / N-Q-P ranking). UCI remains on classic until an explicit S3 switch.

use xiangqi_core::GameResult;

mod event;
mod extension;
mod policy;
mod repository;
mod search;
mod stats;

pub use event::{BackpropEvent, NodeEvent, SearchGeneration, Variation};
pub use policy::{select_edge, select_edge_from_node, ValueDelta};
pub use repository::{EdgeReservation, ExpansionState, NodeKey, NodeRepository, Edge, Node};
pub use search::{Search, SearchConfig, SearchLimits, Stats};
pub use stats::{
    best_move, best_move_filtered, principal_variation, principal_variation_filtered, root_settled,
    root_stats, RootEdgeStats, RootStats,
};

/// px0 `FetchSingleNodeResult`: `eval->q = -eval->q` (`search.cc:2129`).
/// Network WDL is side-to-move; node statistics use the incoming-edge / mover
/// perspective, matching classic `Node::wl_`.
pub(crate) fn network_wl_to_node(stm_wl: f32) -> f32 {
    -stm_wl
}

/// Absolute game result → mover-perspective `(wl, d)` for a terminal leaf.
///
/// Equivalent to converting to STM then applying the same negate as NN fetch,
/// which matches px0 writing `WHITE_WON` (+1) for a checkmated leaf
/// (`search.cc:1913-1919`, `node.cc:300-317`).
pub(crate) fn terminal_wl_for_node(result: GameResult, black_to_move: bool) -> (f32, f32) {
    let (stm_wl, draw) = match result {
        GameResult::WhiteWon => (if black_to_move { -1.0 } else { 1.0 }, 0.0),
        GameResult::BlackWon => (if black_to_move { 1.0 } else { -1.0 }, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("terminal stream evaluation requires a result"),
    };
    (-stm_wl, draw)
}
