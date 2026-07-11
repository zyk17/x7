//! 用户侧引擎基础设施。
//!
//! 当前 crate 只保留 MCTS 主线所需能力：
//!
//! - MCTS 搜索骨架
//! - ONNX policy/value 消费
//! - UCI 对弈入口
//! - 基准与最小调试工具

use std::io;

pub mod benchmark;
pub mod eval;
pub mod fen_tensor;
pub mod history;
pub mod mcts;
pub mod policy_onnx;
pub mod move_vocab;
pub mod uci;

pub use benchmark::{default_benchmark_fen_strings, resolve_data_file, BenchJsonMeta, BenchSessionParams};
pub use eval::{material_stm, terminal_score};
pub use history::{HistoryDebugEntry, PositionHistory, PX0_HISTORY_LEN};
pub use mcts::{
    EdgeStats, MctsBudget, MctsConfig, MctsEngine, MctsMoveStat, MctsNode, MctsNodeId, MctsSearchResult, MctsTree,
    OnnxPolicyValueEval, PolicyValueEval, PolicyValueInput, PolicyValueOutput, SharedPolicy,
};
pub use uci::{parse_position_history_uci, parse_position_uci};

pub use policy_onnx::{BackendAttributes, PolicyOnnx, PolicyOutputs, PolicySessionPool, resolved_search_threads};
pub use xiangqi_core::START_FEN;

/// 自 stdin 读行并处理 UCI，应答至 stdout（实现位于 [`uci::run_uci_stdio`]）。
pub fn run_uci_stdio() -> io::Result<()> {
    uci::run_uci_stdio()
}
