//! LC3 风格的流式搜索。
//!
//! 设计参考（本地没有 LC3 源码）：
//! - <https://lczero.org/dev/lc0/search/lc3/overview/>
//! - <https://lczero.org/dev/lc0/search/lc3/policy/>
//! - <https://lczero.org/dev/lc0/search/lc3/glossary/>
//!
//! 本模块拥有 tree 与 worker 生命周期。LC3 未公开公式时，选择和最终着法使用
//! 有文档的 px0 PUCT / N-Q-P 语义。

use xiangqi_core::GameResult;

mod event;
mod extension;
mod pipeline;
mod policy;
mod session;
mod state;
mod stats;
mod time;
mod tree;

pub use event::{BackpropEvent, NodeEvent, SearchGeneration, Variation};
pub(crate) use pipeline::WorkerPool;
pub use pipeline::{QueueStats, Search, SearchConfig, SearchControl, SearchLimits, Stats};
pub use policy::{select_edge, select_edge_from_node, SearchParams, ValueDelta};
pub(crate) use session::SearchSession;
pub use state::SearchResult;
pub(crate) use state::{SearchState, WatchdogSnapshot};
pub(crate) use stats::best_mate;
pub use stats::{
    best_move, best_move_filtered, principal_variation, principal_variation_filtered, root_stats, RootEdgeStats,
    RootStats,
};
pub use tree::{Edge, EdgeReservation, ExpansionState, GcStats, Node, NodeKey, NodeRepository, Tree};

/// px0 `FetchSingleNodeResult`：`eval->q = -eval->q`（`search.cc:2129`）。NN WDL
/// 按 side-to-move 表示；node 统计按 incoming-edge / 走子方视角表示，对齐 px0 `Node::wl_`。
pub(crate) fn network_wl_to_node(stm_wl: f32) -> f32 {
    -stm_wl
}

/// 将绝对 game result 转换为终局叶子的走子方视角 `(wl, d)`。
///
/// 等价于先转为 STM，再应用与 NN fetch 相同的取反；这对齐 px0 为被将死叶子写入
/// `WHITE_WON`（+1）的方式（`search.cc:1913-1919`、`node.cc:300-317`）。
pub(crate) fn terminal_wl_for_node(result: GameResult, black_to_move: bool) -> (f32, f32) {
    let (stm_wl, draw) = match result {
        GameResult::WhiteWon => (if black_to_move { -1.0 } else { 1.0 }, 0.0),
        GameResult::BlackWon => (if black_to_move { 1.0 } else { -1.0 }, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("terminal search evaluation requires a result"),
    };
    (-stm_wl, draw)
}
