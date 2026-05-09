//! 叶子局面评估：优先 **NN value**（[-1,1]），否则 **物质差**（`xiangqi_core::PIECE_VALUE`，行棋方视角）。

use xiangqi_core::types::{color_of, type_of, Color, Piece, PieceType, PIECE_VALUE, SQUARE_NB, VALUE_DRAW};
use xiangqi_core::Position;

use crate::policy_onnx::PolicyOnnx;

/// 行棋方物质优势（不含位置因子），与核心 `PIECE_VALUE` 一致。
pub fn material_stm(pos: &Position) -> i32 {
    let mut w = 0i32;
    let mut b = 0i32;
    for sq in 0..SQUARE_NB {
        let pc = pos.board[sq];
        if pc == Piece::NO_PIECE {
            continue;
        }
        let pt = type_of(pc);
        if matches!(pt, PieceType::NoPieceType | PieceType::KnightTo | PieceType::PawnTo) {
            continue;
        }
        let v = PIECE_VALUE[pc.to_usize()];
        if color_of(pc) == Color::White {
            w += v;
        } else {
            b += v;
        }
    }
    match pos.side_to_move {
        Color::White => w - b,
        Color::Black => b - w,
    }
}

/// `value` ∈ [-1,1] 映射到与物质同量级的 centipawn 尺度（启发式）。
const NN_VALUE_SCALE_CP: f32 = 4000.0;

/// 叶子分：有 ONNX `value` 则用其，否则 `material_stm`。
pub fn evaluate_leaf(pos: &Position, net: Option<&mut PolicyOnnx>) -> i32 {
    if let Some(n) = net {
        if let Ok(out) = n.eval_fen(&pos.fen()) {
            if let Some(v) = out.value {
                return (v.clamp(-1.0_f32, 1.0_f32) * NN_VALUE_SCALE_CP) as i32;
            }
        }
    }
    material_stm(pos)
}

/// 无子可走：将死 / 困毙。
pub fn terminal_score(pos: &Position) -> Option<i32> {
    use xiangqi_core::generate;
    use xiangqi_core::movegen::{ExtMove, GenType};
    use xiangqi_core::types::MAX_MOVES;
    let mut buf = [ExtMove {
        mv: xiangqi_core::types::Move::none(),
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
