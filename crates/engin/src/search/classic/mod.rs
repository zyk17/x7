//! px0 `src/search/classic` 的 P3 文件边界。

mod factory;
pub mod node;
pub mod params;
pub mod search;
pub mod stoppers;
pub mod uct;
pub mod worker;

pub use factory::Factory;
pub use node::{Edge, Node, NodeArena, NodeTree, Terminal};
pub use params::{ContemptMode, ScoreType, SearchParams};
pub use search::{best_move, ClassicRootEdgeStats, ClassicRootStats, ClassicSearch};
pub use worker::{NodeToProcess, SearchWorker, WorkerSearchState};
