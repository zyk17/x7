//! px0 `src/search/classic` 的 P3 文件边界。

pub mod backend;
pub mod node;
pub mod params;
pub mod search;

pub use backend::{Backend, EvalResult, UniformBackend};
pub use node::{Edge, Node, NodeArena, NodeTree, Terminal};
pub use params::SearchParams;
pub use search::{ClassicSearch, SearchOutput, SearchSession};
