//! X7 stream MCGS。
//!
//! 架构形状可参考 LC3 公开文档（本地没有 LC3 源码）：
//! - <https://lczero.org/dev/lc0/search/lc3/overview/>
//! - <https://lczero.org/dev/lc0/search/lc3/policy/>
//! - <https://lczero.org/dev/lc0/search/lc3/glossary/>
//!
//! 本模块拥有 MCGS 图与 worker 生命周期。相对早期 px0/Lc0 stream 基线，这里已有
//! multivisit + 单轮 gather batch；当前实战还使用只在 pending reservation 期间生效的 FPU
//! virtual mean，但没有 prefetch。选择公式历史上参考过 px0 PUCT / N-Q-P，默认参数与图统计
//! 语义以本仓为准，不是 px0/LC3 等价实现。

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
///
/// 历史语义参考：px0 `FetchSingleNodeResult` 的 `eval->q = -eval->q`，以及
/// `Node::wl_` 的走子方视角约定。
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

/// 将 `rule_judge` 结果转为 node / incoming-edge `(wl, d)`：
/// `WHITE_WON`→`+1`，`BLACK_WON`→`-1`。
///
/// `rule_judge` 的胜负枚举已是 node / incoming-edge 视角（与 `checkThem`/`checkUs`
/// 绑定），**不要**再按 `is_black_to_move` 当绝对颜色转换，否则白方行棋时的长将/长捉
/// 符号会反转。历史语义参考：px0 `Node::MakeTerminal(RuleJudge())`。
pub(crate) fn rule_judge_wl_for_node(result: GameResult) -> (f32, f32) {
    match result {
        GameResult::WhiteWon => (1.0, 0.0),
        GameResult::BlackWon => (-1.0, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("rule-judge terminal requires a decided result"),
    }
}
