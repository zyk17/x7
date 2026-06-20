//! 搜索相关的轻量评估工具。

use xiangqi_core::types::{color_of, type_of, Color, Piece, PieceType, PIECE_VALUE, SQUARE_NB, VALUE_DRAW};
use xiangqi_core::Position;

/// 行棋方物质优势（不含位置因子）。
pub fn material_stm(pos: &Position) -> i32 {
    let mut white = 0i32;
    let mut black = 0i32;
    for sq in 0..SQUARE_NB {
        let pc = pos.board[sq];
        if pc == Piece::NO_PIECE {
            continue;
        }
        let pt = type_of(pc);
        if matches!(pt, PieceType::NoPieceType | PieceType::KnightTo | PieceType::PawnTo) {
            continue;
        }
        let value = PIECE_VALUE[pc.to_usize()];
        if color_of(pc) == Color::White {
            white += value;
        } else {
            black += value;
        }
    }

    match pos.side_to_move {
        Color::White => white - black,
        Color::Black => black - white,
    }
}

/// 无子可走时返回将死 / 困毙分。
pub fn terminal_score(pos: &Position) -> Option<i32> {
    use xiangqi_core::generate;
    use xiangqi_core::movegen::{ExtMove, GenType};
    use xiangqi_core::types::{Move, MAX_MOVES};

    let mut buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut buf);
    if n != 0 {
        return None;
    }

    if pos.checkers() != 0 {
        Some(-30_000)
    } else {
        Some(VALUE_DRAW)
    }
}
