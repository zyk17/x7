//! px0 `src/search/classic` 的 P3 文件边界。

pub mod backend;
pub mod node;
pub mod params;
pub mod search;
pub mod stoppers;
pub mod uct;
pub mod worker;

pub use crate::neural::onnx::OnnxBackend;
pub use backend::{
    AddInputResult, Backend, BackendAttributes, BackendComputation, EvalPosition, EvalResult, EvalTicket,
    UniformBackend,
};
pub use node::{Edge, Node, NodeArena, NodeTree, Terminal};
pub use params::SearchParams;
pub use search::{best_move, ClassicSearch, SearchOutput};
pub use worker::{NodeToProcess, SearchWorker, WorkerSearchState};
