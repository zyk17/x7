//! 中国象棋核心：局面、规则与合法着生成。
//!
//! 规则与走子生成来自 **pikafish-rust**（与 Pikafish 对齐），经 `xiangqi_core` 打包为库 API。
//! Zobrist 使用全局 `OnceLock`（种子 `1070372`），与参考引擎一致。
//!
//! 以下为自 Pikafish 移植代码的常见告警抑制（transmute、区间判断风格等与上游一致）。
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::cast_abs_to_unsigned)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::new_without_default)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::if_same_then_else)]

pub mod board;
pub mod misc;
pub mod movegen;
pub mod types;
pub mod uci_format;

pub use board::{global_zobrist, Position, Zobrist};
pub use movegen::{generate, ExtMove, GenType};
pub use types::{
    Color, File, Key, Move, Piece, PieceType, Rank, Square, Value, MAX_MOVES as TYPES_MAX_MOVES,
};
pub use uci_format::{
    move_to_uci, parse_pyffish_uci, square_to_algebraic, uci_to_move, START_FEN,
};

/// 与 pyffish 一致的 **合法着** UCI 列表（已过滤将帅照面等）。
pub fn legal_moves_uci(pos: &Position) -> Vec<String> {
    let mut list = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; types::MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut list);
    list[..n].iter().map(|e| move_to_uci(e.mv)).collect()
}

/// 向后兼容占位类型别名（原 stub）。
pub type BoardState = Position;
