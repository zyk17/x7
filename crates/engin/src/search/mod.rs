//! X7 stream 树搜索。
//!
//! ## 模块分层
//!
//! | 模块 | 负责 |
//! |------|------|
//! | `select` / `expand` / `eval` / `backprop` | 算法方法（MCTS 实验改这里） |
//! | `decision` | 搜后根选着 / PV / LCB |
//! | `workerpool` | 事件 + 通用线程池 + 独立 NN worker 循环壳 |
//! | `pipeline` | `Shared` / `Stats` + Select 树走组装 + `Search` API |
//! | `tree` | 树 / 节点 / 边 / Repo 数据结构 |
//! | `param` / `time` | 参数与时钟 |
//!
//! 硬规则：
//! - 只有 **Select**（`pipeline::process_select_event`，由 `workerpool` 调度）可 `reserve_edge` / `descend`
//! - 只有 **Select** 可 claim Unexpanded；只有 **Expand** 裁决规则终局/合法着；只有 **Eval/Reply**
//!   可将普通叶子发布为 Expanded（`publish_edges`）
//! - 只有 **Backprop** 可 `complete` reservation 与 `add_delta`

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

pub use decision::{
    DecisionRule, RootEdgeStats, RootStats, best_move, best_move_with_params, principal_variation, root_stats,
};
pub(crate) use decision::{
    best_mate_with_params, best_move_filtered_with_params, principal_variation_with_params, root_variations,
};
pub use observer::{
    BenchObserver, BenchStats, ExecutionKind, ExecutionStats, InstantQueueStamp, NoQueueStamp, NoopObserver, QueueKind,
    QueueStamp, QueueStats, SearchObserver,
};
pub use param::{SearchConfig, SearchParams};
pub(crate) use pipeline::StopHandle;
pub use pipeline::{Search, SearchLimits, Stats};
pub use select::{compute_cpuct, variance_bonus_from_se};
pub(crate) use time::{TimeBudget, TimeManager};
pub use tree::{Edge, EdgeReservation, ExpansionState, Node, NodeArena, NodeId, SearchTree};
pub(crate) use workerpool::WorkerPool;
pub use workerpool::{BackpropEvent, Event, SelectEvent, Variation};
