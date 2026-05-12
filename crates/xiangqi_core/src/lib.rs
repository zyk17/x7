//! 中国象棋核心：局面、规则与合法着生成。
//!
//! 实现上曾参考公开引擎常见写法与互操作约定；本 crate 为**独立整理**的库形态
//! Zobrist 使用全局 `OnceLock`（种子 `1070372`），与常见皮卡鱼族实现一致以便对拍。
//!
//! 以下为与参考实现风格接近处的告警抑制（如 `transmute`、区间判断写法等）。
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

pub use board::{global_zobrist, Position, UndoFrame, Zobrist};
pub use movegen::{generate, ExtMove, GenType};
pub use types::{Color, File, Key, Move, Piece, PieceType, Rank, Square, Value, MAX_MOVES as TYPES_MAX_MOVES};
pub use uci_format::{
    move_to_uci, parse_move_uci, square_to_algebraic, uci_to_move, write_move_uci_bytes, START_FEN,
};

/// **合法着** UCI 字符串列表（`a0`～`i9`，与常见皮卡鱼族引擎约定一致；已过滤将帅照面等）。
pub fn legal_moves_uci(pos: &Position) -> Vec<String> {
    let mut list = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; types::MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut list);
    list[..n].iter().map(|e| move_to_uci(e.mv)).collect()
}

/// 向后兼容的类型别名（历史上曾用 `BoardState` 指代局面）。
pub type BoardState = Position;
