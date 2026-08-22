//! UCI 引擎入口：UCI 循环、ONNX backend 与 stream 路径树搜索。

pub mod engine;
pub mod error;
pub mod neural;
pub mod options;
pub mod search;
pub mod uci_loop;
pub mod utils;

pub use engine::Engine;
pub use error::EnginError;
pub use options::Options;
pub use uci_loop::{
    GoParams, UciLoop, contains_key, format_best_move, format_thinking_info, get_numeric, get_or_empty, parse_command,
};
