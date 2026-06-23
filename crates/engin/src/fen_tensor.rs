//! Position → px0 classical 输入平面：`[1, 124, 10, 9]`，NCHW，float32。

use ndarray::{s, Array4};
use xiangqi_core::Position;
use xiangqi_core::types::{color_of, file_of, flip_rank, rank_of, type_of, Color, Piece, PieceType, Square};
use xiangqi_core::START_FEN;

const PLANES_PER_BOARD: usize = 15;
const HISTORY_PLANES: usize = 8;
const AUX_BASE: usize = PLANES_PER_BOARD * HISTORY_PLANES;
const TOTAL_PLANES: usize = AUX_BASE + 4;

fn piece_plane_offset(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Rook => Some(0),
        PieceType::Advisor => Some(1),
        PieceType::Cannon => Some(2),
        PieceType::Pawn => Some(3),
        PieceType::Knight => Some(4),
        PieceType::Bishop => Some(5),
        PieceType::King => Some(6),
        _ => None,
    }
}

fn plane_row(sq: Square, black_to_move: bool) -> usize {
    let base = 9usize.saturating_sub(rank_of(sq) as usize);
    if black_to_move {
        9usize.saturating_sub(base)
    } else {
        base
    }
}

fn plane_file(sq: Square) -> usize {
    file_of(sq) as usize
}

fn is_startpos_board(pos: &Position) -> bool {
    let start = Position::from_fen(START_FEN).expect("valid startpos");
    pos.board == start.board
}

fn encode_history_block(planes: &mut Array4<f32>, block: usize, pos: &Position) {
    let base = block * PLANES_PER_BOARD;
    let black_to_move = pos.side_to_move == Color::Black;
    for sq_u in 0u8..90 {
        let sq: Square = unsafe { std::mem::transmute(sq_u) };
        let pc = pos.piece_on(sq);
        if pc == Piece::NO_PIECE {
            continue;
        }
        let Some(offset) = piece_plane_offset(type_of(pc)) else {
            continue;
        };
        let ours = color_of(pc) == pos.side_to_move;
        let plane = base + if ours { offset } else { offset + 7 };
        let rel_sq = if black_to_move { flip_rank(sq) } else { sq };
        let row = plane_row(rel_sq, false);
        let col = plane_file(rel_sq);
        planes[[0, plane, row, col]] = 1.0;
    }
}

fn encode_position(pos: &Position) -> Array4<f32> {
    let mut planes = Array4::<f32>::zeros((1, TOTAL_PLANES, 10, 9));
    encode_history_block(&mut planes, 0, pos);
    if !is_startpos_board(pos) {
        for block in 1..HISTORY_PLANES {
            encode_history_block(&mut planes, block, pos);
        }
    }

    let stm_fill = if pos.side_to_move == Color::Black { 1.0 } else { 0.0 };
    planes.slice_mut(s![0, AUX_BASE, .., ..]).fill(stm_fill);
    let rule60 = pos.state.rule60.clamp(0, 119) as f32 / 119.0;
    planes.slice_mut(s![0, AUX_BASE + 1, .., ..]).fill(rule60);
    planes.slice_mut(s![0, AUX_BASE + 3, .., ..]).fill(1.0);
    planes
}

pub(crate) fn fen_to_planes(fen: &str) -> Result<Array4<f32>, String> {
    let pos = Position::from_fen(fen)?;
    Ok(encode_position(&pos))
}

pub(crate) fn position_to_planes(pos: &Position) -> Result<Array4<f32>, String> {
    Ok(encode_position(pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_fen_shape_and_aux_planes() {
        let t = fen_to_planes(START_FEN).unwrap();
        assert_eq!(t.shape(), [1, 124, 10, 9]);
        assert!((t[[0, AUX_BASE, 0, 0]]).abs() < 1e-6);
        assert!((t[[0, AUX_BASE + 3, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn start_fen_relative_piece_planes() {
        let t = fen_to_planes(START_FEN).unwrap();
        assert!((t[[0, 0, 9, 0]] - 1.0).abs() < 1e-6);
        assert!((t[[0, 7, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn black_to_move_mirrors_and_marks_stm() {
        let fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1";
        let t = fen_to_planes(fen).unwrap();
        assert!((t[[0, AUX_BASE, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((t[[0, 0, 9, 0]] - 1.0).abs() < 1e-6);
        assert!((t[[0, 7, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn position_to_planes_matches_fen_startpos() {
        let pos = Position::from_fen(START_FEN).unwrap();
        let a = fen_to_planes(START_FEN).unwrap();
        let b = position_to_planes(&pos).unwrap();
        assert_eq!(a.shape(), b.shape());
        let diff: f32 = (&a - &b).mapv(f32::abs).sum();
        assert!(diff < 1e-5, "diff={diff}");
    }
}
