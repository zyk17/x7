//! LC3-style streaming search foundation.
//!
//! Design references (LC3 code is not published in the local lc0 checkout):
//! - <https://lczero.org/dev/lc0/search/lc3/overview/>
//! - <https://lczero.org/dev/lc0/search/lc3/policy/>
//! - <https://lczero.org/dev/lc0/search/lc3/glossary/>
//!
//! This module deliberately does **not** reuse `classic::node` or
//! `classic::worker`. LC3's concurrent unit is an owned node event and its
//! repository owns independent nodes; classic's mutable DFS arena has a
//! different aliasing model. The first stream milestones are repository,
//! reservation, and event contracts. UCI remains on `classic` until a complete
//! Gather -> Eval -> Backprop pipeline has fixed-visits regression coverage.

use xiangqi_core::GameResult;

mod event;
mod pipeline;
mod policy;
mod repository;
mod search;
mod stats;
mod workers;

pub use event::{BackpropEvent, NodeEvent, SearchGeneration, Variation};
pub use pipeline::{StreamPipeline, StreamPipelineConfig};
pub use policy::{select_edge, ValueDelta};
pub use repository::{EdgeReservation, ExpansionState, NodeKey, NodeRepository, StreamEdge, StreamNode};
pub use search::{StreamOutcome, StreamSearch, StreamStats};
pub use stats::{root_stats, StreamRootEdgeStats, StreamRootStats};
pub use workers::{StreamWorkerConfig, StreamWorkerPipeline};

/// Converts an absolute Xiangqi terminal result to the current leaf side's
/// compact WDL value. Both serial and queued stream paths must use it.
pub(crate) fn terminal_value_for_side_to_move(result: GameResult, black_to_move: bool) -> (f32, f32) {
    match result {
        GameResult::WhiteWon => (if black_to_move { -1.0 } else { 1.0 }, 0.0),
        GameResult::BlackWon => (if black_to_move { 1.0 } else { -1.0 }, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("terminal stream evaluation requires a result"),
    }
}
