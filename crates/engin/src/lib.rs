//! 用户侧引擎基础设施。
//!
//! 搜索核心正在按 lc0 classic 重建。此 crate 当前只保留可独立验证的
//! 象棋 history、网络输入和 ONNX 推理地基，避免旧 MCTS 语义继续参与运行。

use std::io;

pub mod fen_tensor;
pub mod history;
pub mod move_vocab;
pub mod policy_onnx;
pub mod uci;

pub use history::{HistoryDebugEntry, PositionHistory, PX0_HISTORY_LEN};
pub use uci::{parse_position_history_uci, parse_position_uci};

pub use policy_onnx::{resolved_search_threads, BackendAttributes, PolicyOnnx, PolicyOutputs, PolicySessionPool};
pub use xiangqi_core::START_FEN;

/// 自 stdin 读行并处理 UCI，应答至 stdout（实现位于 [`uci::run_uci_stdio`]）。
pub fn run_uci_stdio() -> io::Result<()> {
    uci::run_uci_stdio()
}
