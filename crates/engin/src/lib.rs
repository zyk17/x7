//! 用户侧引擎：规则核心 + ONNX 策略/价值推理 + **Alpha-Beta**（置换表、杀手着法、MVV-LVA、根着 policy 排序）。
//!
//! 数据标注与 XRSH 生成见 **`xiangqi_dataset`**，不在本 crate。

use std::io;

pub mod benchmark;
pub mod eval;
pub mod fen_tensor;
pub mod policy_onnx;
pub mod search;
pub mod tt;
pub mod uci;
pub mod value_probe;
pub mod vocab;

pub use benchmark::{default_benchmark_fen_strings, resolve_data_file, BenchJsonMeta, BenchSessionParams};
pub use value_probe::{
    markdown_table_off_vs_main, ValueProbeCase, ValueProbeTableArgs, VALUE_PROBE_CASES,
};
pub use eval::{NNLeafMode, NnEvalSession, NnEvalSite};
pub use search::{root_search_iterative, RootSearchShared, RootSearchResult, SearchAblation, SearchLimits};
pub use tt::TranspositionTable;
pub use uci::parse_position_uci;

pub use policy_onnx::{PolicyOnnx, PolicyOutputs};
pub use xiangqi_core::START_FEN;

/// 自 stdin 读行并处理 UCI，应答至 stdout（实现位于 [`uci::run_uci_stdio`]）。
pub fn run_uci_stdio() -> io::Result<()> {
    uci::run_uci_stdio()
}
