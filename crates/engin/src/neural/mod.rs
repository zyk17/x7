//! px0 `src/neural/encoder.h` / `encoder.cc` 的 classical 编码。

use std::collections::HashMap;
use std::sync::OnceLock;

use xiangqi_core::{startpos_board, ChessBoard, Move, Position, PositionHistory};

/// px0 classical 输入平面数。
pub const INPUT_PLANES: usize = 124;
pub const BOARD_ROWS: usize = 10;
pub const BOARD_COLS: usize = 9;
/// px0 policy 输出维度。
pub const POLICY_SIZE: usize = 2062;
pub const MOVE_HISTORY: usize = 8;
pub const PLANES_PER_BOARD: usize = 15;
pub const AUX_PLANE_BASE: usize = MOVE_HISTORY * PLANES_PER_BOARD;

pub mod backend;
pub mod onnx;

/// px0 `src/neural/encoder.h:42`。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FillEmptyHistory {
    No,
    FenOnly,
    Always,
}

/// px0 `EncodePositionForNN` (`src/neural/encoder.cc:118-217`) 的
/// classical、非 canonical 分支。
///
/// 输出是 NCHW 中单个样本的连续 `[124][10][9]` 数据。搜索只传真实
/// `PositionHistory`；孤立 FEN 则由调用方构造长度为一的 history，再使用
/// `FenOnly` 保持 px0 的缺失 history 语义。
pub fn encode_position_for_nn(history: &PositionHistory, fill: FillEmptyHistory) -> Vec<f32> {
    assert!(history.len() > 0, "EncodePositionForNN requires a position");
    let mut planes = vec![0.0; INPUT_PLANES * BOARD_ROWS * BOARD_COLS];
    let current = history.last();
    if current.is_black_to_move() {
        fill_plane(&mut planes, AUX_PLANE_BASE, 1.0);
    }
    fill_plane(&mut planes, AUX_PLANE_BASE + 1, current.rule60_ply() as f32);
    fill_plane(&mut planes, AUX_PLANE_BASE + 3, 1.0);

    let mut flip = false;
    let mut history_idx = history.len() as isize - 1;
    for block in 0..MOVE_HISTORY {
        let position = history.get(history_idx.max(0) as usize);
        if history_idx < 0 {
            match fill {
                FillEmptyHistory::No => break,
                FillEmptyHistory::FenOnly if position.board() == startpos_board() => break,
                FillEmptyHistory::FenOnly | FillEmptyHistory::Always => {}
            }
        }

        let mut board = position.board().clone();
        if flip {
            board.mirror();
        }
        write_board_planes(&mut planes, block * PLANES_PER_BOARD, &board, position);
        if history_idx > 0 {
            flip = !flip;
        }
        history_idx -= 1;
    }
    planes
}

/// px0 `kPackedIdxToNNIdx` / `MoveToNNIndex`
/// (`src/neural/encoder.cc:229-481`)。表由该 C++ 字面量机械提取，禁止改排序。
pub fn move_to_nn_index(mv: Move) -> Option<usize> {
    static MOVE_INDEX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let index = MOVE_INDEX.get_or_init(|| {
        include_str!("px0_policy_moves.txt")
            .lines()
            .enumerate()
            .map(|(idx, uci)| (uci, idx))
            .collect()
    });
    index.get(mv.to_uci().as_str()).copied()
}

fn write_board_planes(planes: &mut [f32], base: usize, board: &ChessBoard, position: &Position) {
    let ours = board.ours();
    let theirs = board.theirs();
    let masks = [
        ours.intersection(board.rooks()).bits(),
        ours.intersection(board.advisors()).bits(),
        ours.intersection(board.cannons()).bits(),
        ours.intersection(board.pawns()).bits(),
        ours.intersection(board.knights()).bits(),
        ours.intersection(board.bishops()).bits(),
        ours.intersection(board.kings()).bits(),
        theirs.intersection(board.rooks()).bits(),
        theirs.intersection(board.advisors()).bits(),
        theirs.intersection(board.cannons()).bits(),
        theirs.intersection(board.pawns()).bits(),
        theirs.intersection(board.knights()).bits(),
        theirs.intersection(board.bishops()).bits(),
        theirs.intersection(board.kings()).bits(),
    ];
    for (plane, mask) in masks.into_iter().enumerate() {
        write_mask(planes, base + plane, mask);
    }
    if position.repetitions() >= 1 {
        fill_plane(planes, base + 14, 1.0);
    }
}

fn write_mask(planes: &mut [f32], plane: usize, mask: u128) {
    let offset = plane * BOARD_ROWS * BOARD_COLS;
    let mut bits = mask;
    while bits != 0 {
        let square = bits.trailing_zeros() as usize;
        planes[offset + square] = 1.0;
        bits &= bits - 1;
    }
}

fn fill_plane(planes: &mut [f32], plane: usize, value: f32) {
    let offset = plane * BOARD_ROWS * BOARD_COLS;
    planes[offset..offset + BOARD_ROWS * BOARD_COLS].fill(value);
}

#[cfg(test)]
mod tests {
    use xiangqi_core::Position;

    use super::*;

    #[test]
    fn startpos_fen_only_has_no_synthetic_history() {
        let history = PositionHistory::from_positions(vec![Position::from_fen(xiangqi_core::STARTPOS_FEN).unwrap()]);
        let planes = encode_position_for_nn(&history, FillEmptyHistory::FenOnly);
        assert!(planes[..PLANES_PER_BOARD * BOARD_ROWS * BOARD_COLS].contains(&1.0));
        assert!(
            planes[PLANES_PER_BOARD * BOARD_ROWS * BOARD_COLS..AUX_PLANE_BASE * BOARD_ROWS * BOARD_COLS]
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn px0_policy_table_is_complete_and_stable() {
        let move_count = include_str!("px0_policy_moves.txt").lines().count();
        assert_eq!(move_count, POLICY_SIZE);
        let mv = xiangqi_core::Move::new(
            xiangqi_core::Square::parse("a0").unwrap(),
            xiangqi_core::Square::parse("a1").unwrap(),
        );
        assert_eq!(move_to_nn_index(mv), Some(0));
    }
}
