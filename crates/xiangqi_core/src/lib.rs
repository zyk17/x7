//! X7 规则核心：棋盘、合法着、FEN、历史与裁判。
//!
//! 本 crate 是 px0 `src/chess` 规则实现的 Rust 重写；现由本仓维护。

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
pub use types::{File, LegalMoveList, Move, MoveList, PieceType, Rank, Square};
