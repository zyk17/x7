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

mod event;
mod extension;
mod graph;
mod pipeline;
mod policy;
mod stats;
mod time;

pub use event::{BackpropEvent, PlayoutEvent, Variation};
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
    root_variations,
};
pub(crate) use time::{TimeBudget, TimeManager};
