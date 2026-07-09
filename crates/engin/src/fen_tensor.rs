//! px0 classical 输入平面：`[1, 124, 10, 9]`，NCHW，float32。

use ndarray::{s, Array4};
use xiangqi_core::types::{color_of, file_of, flip_rank, rank_of, type_of, Color, Piece, PieceType, Square};
use xiangqi_core::{Position, START_FEN};

use crate::history::{HistoryEntry, PositionHistory, PX0_HISTORY_LEN};

const PLANES_PER_BOARD: usize = 15;
const AUX_BASE: usize = PLANES_PER_BOARD * PX0_HISTORY_LEN;
const TOTAL_PLANES: usize = AUX_BASE + 4;
pub const PX0_INPUT_SHAPE: (usize, usize, usize, usize) = (1, TOTAL_PLANES, 10, 9);

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

fn is_startpos_board(pos: &Position) -> bool {
    pos.fen() == START_FEN
}

fn plane_coords(sq: Square, flip: bool) -> (usize, usize) {
    let rel_sq = if flip { flip_rank(sq) } else { sq };
    (rank_of(rel_sq) as usize, file_of(rel_sq) as usize)
}

fn encode_history_block(planes: &mut Array4<f32>, batch: usize, block: usize, pos: &Position, history_flip: bool) {
    let base = block * PLANES_PER_BOARD;
    let stm = pos.side_to_move;
    let rank_flip = (stm == Color::Black) ^ history_flip;
    for sq_u in 0u8..90 {
        let sq: Square = unsafe { std::mem::transmute(sq_u) };
        let pc = pos.piece_on(sq);
        if pc == Piece::NO_PIECE {
            continue;
        }
        let Some(offset) = piece_plane_offset(type_of(pc)) else {
            continue;
        };
        let ours = if history_flip {
            color_of(pc) != stm
        } else {
            color_of(pc) == stm
        };
        let plane = base + if ours { offset } else { offset + 7 };
        let (row, col) = plane_coords(sq, rank_flip);
        planes[[batch, plane, row, col]] = 1.0;
    }
}

fn encode_history_into(history: &PositionHistory, planes: &mut Array4<f32>) {
    prepare_single_board(planes);
    encode_history_into_slot(history, planes, 0);
}

fn prepare_single_board(planes: &mut Array4<f32>) {
    if planes.shape()
        != [
            PX0_INPUT_SHAPE.0,
            PX0_INPUT_SHAPE.1,
            PX0_INPUT_SHAPE.2,
            PX0_INPUT_SHAPE.3,
        ]
    {
        *planes = Array4::<f32>::zeros(PX0_INPUT_SHAPE);
    } else {
        planes.fill(0.0);
    }
}

fn encode_history_into_slot(history: &PositionHistory, planes: &mut Array4<f32>, batch: usize) {
    let current = history.current();
    let entries = history.entries();
    let mut flip = false;
    let mut history_idx = entries.len() as isize - 1;

    for block in 0..PX0_HISTORY_LEN {
        let entry: &HistoryEntry = if history_idx >= 0 {
            &entries[history_idx as usize]
        } else {
            let earliest = entries.first().expect("non-empty history");
            if is_startpos_board(&earliest.position) {
                break;
            }
            earliest
        };

        encode_history_block(planes, batch, block, &entry.position, flip);
        if entry.repeated {
            planes
                .slice_mut(s![batch, block * PLANES_PER_BOARD + 14, .., ..])
                .fill(1.0);
        }

        if history_idx > 0 {
            flip = !flip;
        }
        history_idx -= 1;
    }

    let stm_fill = if current.side_to_move == Color::Black { 1.0 } else { 0.0 };
    planes.slice_mut(s![batch, AUX_BASE, .., ..]).fill(stm_fill);
    planes
        .slice_mut(s![batch, AUX_BASE + 1, .., ..])
        .fill(current.state.rule60.max(0) as f32);
    planes.slice_mut(s![batch, AUX_BASE + 3, .., ..]).fill(1.0);
}

fn encode_history(history: &PositionHistory) -> Array4<f32> {
    let mut planes = Array4::<f32>::zeros(PX0_INPUT_SHAPE);
    encode_history_into(history, &mut planes);
    planes
}

pub fn history_to_planes(history: &PositionHistory) -> Result<Array4<f32>, String> {
    if history.is_empty() {
        return Err("history 不能为空".into());
    }
    Ok(encode_history(history))
}

pub fn history_to_planes_into(history: &PositionHistory, planes: &mut Array4<f32>) -> Result<(), String> {
    if history.is_empty() {
        return Err("history 不能为空".into());
    }
    encode_history_into(history, planes);
    Ok(())
}

pub fn histories_to_planes(histories: &[&PositionHistory]) -> Result<Array4<f32>, String> {
    if histories.is_empty() {
        return Err("histories 不能为空".into());
    }
    let mut planes = Array4::<f32>::zeros((histories.len(), PX0_INPUT_SHAPE.1, PX0_INPUT_SHAPE.2, PX0_INPUT_SHAPE.3));
    histories_to_planes_into(histories, &mut planes)?;
    Ok(planes)
}

pub fn histories_to_planes_into(histories: &[&PositionHistory], planes: &mut Array4<f32>) -> Result<(), String> {
    if histories.is_empty() {
        return Err("histories 不能为空".into());
    }
    let expected_shape = (histories.len(), PX0_INPUT_SHAPE.1, PX0_INPUT_SHAPE.2, PX0_INPUT_SHAPE.3);
    if planes.shape() != [expected_shape.0, expected_shape.1, expected_shape.2, expected_shape.3] {
        *planes = Array4::<f32>::zeros(expected_shape);
    } else {
        planes.fill(0.0);
    }
    for (batch, history) in histories.iter().enumerate() {
        if history.is_empty() {
            return Err("history 不能为空".into());
        }
        encode_history_into_slot(history, planes, batch);
    }
    Ok(())
}

pub fn fen_to_planes(fen: &str) -> Result<Array4<f32>, String> {
    let history = PositionHistory::from_fen(fen)?;
    history_to_planes(&history)
}

pub fn position_to_planes(pos: &Position) -> Result<Array4<f32>, String> {
    let history = PositionHistory::from_position(pos.clone_for_search());
    history_to_planes(&history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::uci_to_move;

    fn block_sum(t: &Array4<f32>, block: usize) -> f32 {
        t.slice(s![0, block * PLANES_PER_BOARD..(block + 1) * PLANES_PER_BOARD, .., ..])
            .sum()
    }

    #[test]
    fn start_fen_only_keeps_single_history_block() {
        let t = fen_to_planes(START_FEN).unwrap();
        assert_eq!(t.shape(), [1, 124, 10, 9]);
        assert!(block_sum(&t, 0) > 0.0);
        assert_eq!(block_sum(&t, 1), 0.0);
    }

    #[test]
    fn isolated_non_start_fen_reuses_earliest_block() {
        let mut pos = Position::from_fen(START_FEN).unwrap();
        let mv = uci_to_move(&pos, "h2e2").expect("legal move");
        pos.do_move(mv);
        let t = position_to_planes(&pos).unwrap();
        let diff: f32 = (&t.slice(s![0, 0..15, .., ..]).to_owned() - &t.slice(s![0, 15..30, .., ..]).to_owned())
            .mapv(f32::abs)
            .sum();
        assert!(diff < 1e-6, "diff={diff}");
    }

    #[test]
    fn start_board_with_non_start_state_still_uses_fen_only_fill() {
        let fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 10 6";
        let t = fen_to_planes(fen).unwrap();
        let diff: f32 = (&t.slice(s![0, 0..15, .., ..]).to_owned() - &t.slice(s![0, 15..30, .., ..]).to_owned())
            .mapv(f32::abs)
            .sum();
        assert!(diff < 1e-6, "diff={diff}");
    }

    #[test]
    fn real_history_keeps_previous_block_distinct() {
        let mut history = PositionHistory::new_startpos();
        let mv = uci_to_move(history.current(), "h2e2").expect("legal move");
        history.push_move(mv);
        let t = history_to_planes(&history).unwrap();
        assert!(block_sum(&t, 0) > 0.0);
        assert!(block_sum(&t, 1) > 0.0);
        let diff: f32 = (&t.slice(s![0, 0..15, .., ..]).to_owned() - &t.slice(s![0, 15..30, .., ..]).to_owned())
            .mapv(f32::abs)
            .sum();
        assert!(diff > 1e-3, "diff={diff}");
    }

    #[test]
    fn history_blocks_use_per_position_side_to_move_orientation() {
        let mut history = PositionHistory::new_startpos();
        let mv = uci_to_move(history.current(), "h2e2").expect("legal move");
        history.push_move(mv);
        let t = history_to_planes(&history).unwrap();

        let block0_ours_rook = 0usize;
        let block1_ours_rook = PLANES_PER_BOARD;
        assert_eq!(
            t[[0, block0_ours_rook, 0, 0]],
            1.0,
            "current black rook should be ours at bottom-left"
        );
        assert_eq!(
            t[[0, block1_ours_rook, 0, 0]],
            1.0,
            "previous position should also be reoriented into the root side perspective"
        );
    }

    #[test]
    fn black_to_move_marks_aux_plane() {
        let fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1";
        let t = fen_to_planes(fen).unwrap();
        assert!((t[[0, AUX_BASE, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((t[[0, AUX_BASE + 3, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn repeated_history_sets_repetition_plane() {
        let mut history = PositionHistory::new_startpos();
        for u in ["h0g2", "h9g7", "g2h0", "g7h9"] {
            let mv = uci_to_move(history.current(), u).expect("legal move");
            history.push_move(mv);
        }
        assert!(history.current_is_repeated());

        let t = history_to_planes(&history).unwrap();
        assert_eq!(
            t[[0, 14, 0, 0]],
            1.0,
            "current repeated position should mark repetition plane"
        );
    }

    #[test]
    fn isolated_fen_fallback_does_not_fake_repetition() {
        let history = PositionHistory::from_fen(START_FEN).unwrap();
        let t = history_to_planes(&history).unwrap();
        assert_eq!(t[[0, 14, 0, 0]], 0.0);
        assert_eq!(t[[0, 29, 0, 0]], 0.0);
    }
}
