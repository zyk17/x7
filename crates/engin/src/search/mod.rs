//! X7 stream MCGS。
//!
//! 架构形状可参考 LC3 公开文档
//! - <https://lczero.org/dev/lc0/search/lc3/overview/>
//! - <https://lczero.org/dev/lc0/search/lc3/policy/>
//! - <https://lczero.org/dev/lc0/search/lc3/glossary/>
//!
//! 本模块拥有 MCGS 图与连续的 Gather / Eval / NN / Backprop 生命周期。Gather 每次采集
//! 一个叶子；实战用 pending reservation 上的 FPU virtual mean 分流。不把一次评估记成
//! 多次 visit：那会一次打入 K 份 FPU，破坏这份温和分流；后续若调分流，改 virtual mean /
//! virtual visit。没有 prefetch 或 tree-batch gather。

use xiangqi_core::GameResult;

mod event;
mod extension;
mod graph;
mod pipeline;
mod policy;
mod stats;
mod time;

pub use event::{BackpropEvent, NodeEvent, Variation};
pub use graph::{Edge, EdgeReservation, ExpansionState, Node, NodeKey, NodeRepository, SearchGraph};
#[cfg(feature = "benchmark")]
pub use pipeline::QueueStats;
pub(crate) use pipeline::WorkerPool;
pub use pipeline::{Search, SearchConfig, SearchControl, SearchLimits, Stats};
pub(crate) use policy::select_edge;
pub use policy::{SearchParams, ValueDelta};
pub use stats::{
    RootEdgeStats, RootStats, best_move, best_move_filtered, principal_variation, principal_variation_filtered,
    root_stats,
};
pub(crate) use stats::{
    best_mate_with_params, best_move_filtered_with_params, principal_variation_with_history_and_params,
    root_variations_with_history_and_params,
};
pub(crate) use time::{TimeBudget, TimeManager};

/// 将 NN 的 side-to-move WDL 转为 node / incoming-edge 视角：取反。
pub(crate) fn network_wl_to_node(stm_wl: f32) -> f32 {
    -stm_wl
}

/// 将**绝对** game result（`compute_game_result`）转为终局叶子 incoming-edge `(wl, d)`。
///
/// 先换成 STM 视角，再取反，对齐 NN fetch 与将死快径 `WHITE_WON`→`+1`
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

/// 将 `rule_judge` 结果转为 node / incoming-edge `(wl, d)`：
/// `WHITE_WON`→`+1`，`BLACK_WON`→`-1`。
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
