//! LC3 风格的流式搜索。
//!
//! 设计参考（本地没有 LC3 源码）：
//! - <https://lczero.org/dev/lc0/search/lc3/overview/>
//! - <https://lczero.org/dev/lc0/search/lc3/policy/>
//! - <https://lczero.org/dev/lc0/search/lc3/glossary/>
//!
//! 本模块拥有 MCGS 图与 worker 生命周期。LC3 未公开公式时，选择和最终着法使用
//! 有文档的 px0 PUCT / N-Q-P 语义。

use xiangqi_core::GameResult;

mod event;
mod extension;
mod graph;
mod pipeline;
mod policy;
mod session;
mod state;
mod stats;
mod time;

pub use event::{BackpropEvent, NodeEvent, SearchGeneration, Variation};
pub use graph::{Edge, EdgeReservation, ExpansionState, GcStats, Node, NodeKey, NodeRepository, SearchGraph};
#[cfg(feature = "benchmark")]
pub use pipeline::QueueStats;
pub(crate) use pipeline::WorkerPool;
pub use pipeline::{Search, SearchConfig, SearchControl, SearchLimits, Stats};
pub use policy::{SearchParams, ValueDelta, select_edge, select_edge_from_node};
pub(crate) use session::SearchSession;
pub use state::SearchResult;
pub(crate) use state::{SearchState, WatchdogProgress, WatchdogSnapshot};
pub(crate) use stats::best_mate;
pub(crate) use stats::root_variations;
pub use stats::{
    RootEdgeStats, RootStats, best_move, best_move_filtered, principal_variation, principal_variation_filtered,
    root_stats,
};

/// px0 `FetchSingleNodeResult`：`eval->q = -eval->q`（`search.cc:2129`）。NN WDL
/// 按 side-to-move 表示；node 统计按 incoming-edge / 走子方视角表示，对齐 px0 `Node::wl_`。
pub(crate) fn network_wl_to_node(stm_wl: f32) -> f32 {
    -stm_wl
}

/// 将**绝对** game result（`compute_game_result`）转为终局叶子 incoming-edge `(wl, d)`。
///
/// 先换成 STM 视角，再取反，对齐 NN fetch 与将死快径 `WHITE_WON`→`+1`
///（`search.cc:1913-1919`、`node.cc:300-317`）。
///
/// 注意：`rule_judge()` 的返回值**不是**绝对胜负，不能走这里；应使用
/// [`rule_judge_wl_for_node`]。
pub(crate) fn terminal_wl_for_node(result: GameResult, black_to_move: bool) -> (f32, f32) {
    let (stm_wl, draw) = match result {
        GameResult::WhiteWon => (if black_to_move { -1.0 } else { 1.0 }, 0.0),
        GameResult::BlackWon => (if black_to_move { 1.0 } else { -1.0 }, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("terminal search evaluation requires a result"),
    };
    (-stm_wl, draw)
}

/// px0 `Node::MakeTerminal(RuleJudge())`：`WHITE_WON`→`+1`，`BLACK_WON`→`-1`。
///
/// `rule_judge` 的胜负枚举已是 node / incoming-edge 视角（与 `checkThem`/`checkUs`
/// 绑定），**不要**再按 `is_black_to_move` 当绝对颜色转换，否则白方行棋时的长将/长捉
/// 符号会反转。
pub(crate) fn rule_judge_wl_for_node(result: GameResult) -> (f32, f32) {
    match result {
        GameResult::WhiteWon => (1.0, 0.0),
        GameResult::BlackWon => (-1.0, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("rule-judge terminal requires a decided result"),
    }
}
