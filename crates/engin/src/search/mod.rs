//! X7 stream 树搜索。
//!
//! ## 模块分层
//!
//! | 模块 | 负责 |
//! |------|------|
//! | `select` / `expand` / `eval` / `backprop` | 算法方法（MCTS 实验改这里） |
//! | `workerpool` | 事件 + 线程池 + Gather/Eval/NN/Backprop 循环壳 |
//! | `pipeline` | `Shared` / `Stats` + Gather 树走组装 + `Search` API |
//! | `tree` | 树 / 节点 / 边 / Repo 数据结构 |
//! | `decision` | 搜后根选着 / PV / LCB |
//! | `param` / `time` | 参数与时钟 |
//!
//! 硬规则：
//! - 只有 **Gather**（`pipeline::process_gather_event`，由 `workerpool` 调度）可 `reserve_edge` / `descend`
//! - 只有 **Eval** 可把 Unexpanded claim 后变成 Expanded（`publish_edges`）或首次 NN 终局
//! - 只有 **Backprop** 可 `complete` reservation 与 `add_delta`
//!
//! MCGS：edge 一次性绑定 `NodeId`；历史重复仍由 variation 的 RuleJudge 裁决。
//! NN cache 按棋盘 + repetition + 合法着数。Gather 每次一个叶子；virtual mean 分流。
//! 无 prefetch / multivisit。

mod backprop;
mod decision;
mod eval;
mod expand;
mod observer;
mod param;
mod pipeline;
mod select;
mod time;
mod tree;
mod workerpool;

pub use decision::{RootEdgeStats, RootStats, best_move, principal_variation, root_stats};
pub(crate) use decision::{
    best_mate_with_params, best_move_filtered_with_params, principal_variation_with_params, root_variations,
};
pub use observer::{
    BenchObserver, BenchStats, InstantQueueStamp, NoQueueStamp, NoopObserver, QueueKind, QueueStamp, QueueStats,
    SearchObserver,
};
pub use param::{SearchConfig, SearchParams};
pub use pipeline::{Search, SearchControl, SearchLimits, Stats};
pub(crate) use time::{TimeBudget, TimeManager};
pub use tree::{Edge, EdgeReservation, ExpansionState, Node, NodeArena, NodeId, SearchTree, ValueDelta};
pub(crate) use workerpool::WorkerPool;
pub use workerpool::{BackpropEvent, PlayoutEvent, Variation};
