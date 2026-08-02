//! px0 `src/chess` 的 Rust 翻译。
//!
//! 旧规则核心已删除。这里只接受可追溯到 px0 文件与行区间的实现。

pub mod bitboard;
pub mod board;
pub mod board_attacks;
pub mod board_masks;
pub mod gamestate;
pub mod hashcat;
pub mod magic_numbers;
pub mod position;
pub mod types;

pub use bitboard::BitBoard;
pub use board::{ChessBoard, FenState, STARTPOS_FEN, board_to_fen, startpos_board};
pub use board_attacks::initialize_magic_bitboards;
pub use gamestate::GameState;
pub use position::{GameResult, Position, PositionHistory};
pub use types::{File, Move, MoveList, PieceType, Rank, Square};

/// px0 C++ 侧通过异常报告无效 FEN 或不可达路径；Rust 侧统一为显式错误。
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreError {
    InvalidFen(String),
    PortIncomplete(&'static str),
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFen(message) => write!(f, "invalid FEN: {message}"),
            Self::PortIncomplete(name) => write!(f, "px0 port incomplete: {name}"),
        }
    }
}

impl std::error::Error for CoreError {}
