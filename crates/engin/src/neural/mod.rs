//! classical NN 编码：`124 x 10 x 9` 输入平面与 `2062` policy 映射。
//!
//! 平面布局与 policy 表历史上源于 px0 classical encoder；本模块由 X7 维护。
//! 热路径保持 px0 同系稀疏 `InputPlane`（mask + value），进 ORT 前再 CPU expand 为 dense NCHW。

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use xiangqi_core::{ChessBoard, Move, Position, PositionHistory, startpos_board};

use crate::EnginError;

use self::backend::EvalResult;

/// classical 输入平面数（契约 `124x10x9`）。
pub const INPUT_PLANES: usize = 124;
pub const BOARD_ROWS: usize = 10;
pub const BOARD_COLS: usize = 9;
pub const BOARD_SQUARES: usize = BOARD_ROWS * BOARD_COLS;
/// policy 输出维度（契约 `2062`）。
pub const POLICY_SIZE: usize = 2062;
pub const MOVE_HISTORY: usize = 8;
pub const PLANES_PER_BOARD: usize = 15;
pub const AUX_PLANE_BASE: usize = MOVE_HISTORY * PLANES_PER_BOARD;
/// 单样本编码后的 f32 个数。
pub const ENCODED_PLANE_FLOATS: usize = INPUT_PLANES * BOARD_SQUARES;
/// 棋盘 90 格 bitmask；对齐 px0 `kAllSquares`（`src/neural/network.h`）。
pub const ALL_SQUARES: u128 = (1_u128 << BOARD_SQUARES) - 1;

/// 稀疏输入平面：mask 标出写入位置，value 为该平面统一取值。
///
/// 参考：px0 `InputPlane`（`src/neural/network.h`）。
#[derive(Clone, Copy, Debug)]
pub struct InputPlane {
    pub mask: u128,
    pub value: f32,
}

impl Default for InputPlane {
    fn default() -> Self {
        Self { mask: 0, value: 1.0 }
    }
}

impl InputPlane {
    pub fn set_all(&mut self) {
        self.mask = ALL_SQUARES;
    }

    pub fn fill(&mut self, value: f32) {
        self.set_all();
        self.value = value;
    }
}

/// 单样本 124 个稀疏平面（进 ORT 前再 expand）。
pub type InputPlanes = [InputPlane; INPUT_PLANES];

pub mod backend;
pub mod cache;
pub mod onnx;

/// 一次合批推理的原始输出：完整 policy logits / WDL / moves-left。
///
/// 搜索 pipeline 只负责搬运；合法着 softmax 与 `EvalResult` 组装见
/// [`eval_result_from_encoded_row`]。
pub struct EncodedBatch {
    pub logits: Vec<f32>,
    pub wdl: Vec<f32>,
    pub moves_left: Vec<f32>,
}

impl EncodedBatch {
    /// 从 NN worker 的可复用缓冲中取出本批结果（缓冲变空，容量随所有权移走）。
    pub fn take_from(logits: &mut Vec<f32>, wdl: &mut Vec<f32>, moves_left: &mut Vec<f32>) -> Self {
        Self {
            logits: std::mem::take(logits),
            wdl: std::mem::take(wdl),
            moves_left: std::mem::take(moves_left),
        }
    }

    pub fn ensure_batch_len(&self, batch: usize) -> Result<(), EnginError> {
        if self.logits.len() != batch * POLICY_SIZE || self.wdl.len() != batch * 3 || self.moves_left.len() != batch {
            return Err(EnginError::PortIncomplete("stream nn output shape"));
        }
        Ok(())
    }

    /// 为下一轮 `infer_encoded_into` 预留下一轮容量。
    pub fn reserve_scratch(logits: &mut Vec<f32>, wdl: &mut Vec<f32>, moves_left: &mut Vec<f32>, batch: usize) {
        logits.reserve(batch * POLICY_SIZE);
        wdl.reserve(batch * 3);
        moves_left.reserve(batch);
    }
}

/// 从完整 policy logits 中按合法着抽取子集。
fn gather_legal_logits(logits: &[f32], legal_moves: &[Move]) -> Result<Vec<f32>, EnginError> {
    let mut selected = Vec::with_capacity(legal_moves.len());
    for &mv in legal_moves {
        let index = move_to_nn_index(mv)
            .ok_or_else(|| EnginError::Onnx(format!("legal move absent from px0 policy table: {mv}")))?;
        if index >= logits.len() {
            return Err(EnginError::Onnx(format!(
                "policy logit index {index} out of range {}",
                logits.len()
            )));
        }
        selected.push(logits[index]);
    }
    Ok(selected)
}

/// 对已对齐合法着的 logit 子集做 in-place softmax。
fn softmax_inplace(logits: &mut [f32]) -> Result<(), EnginError> {
    let mut maximum = f32::NEG_INFINITY;
    for &value in logits.iter() {
        maximum = maximum.max(value);
    }
    let total: f32 = logits
        .iter_mut()
        .map(|value| {
            *value = (*value - maximum).exp();
            *value
        })
        .sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(EnginError::Onnx("legal policy softmax is invalid".into()));
    }
    for value in logits.iter_mut() {
        *value /= total;
    }
    Ok(())
}

/// 对合法着做 softmax，得到 policy 概率。
pub fn softmax_legal_policy(logits: &[f32], legal_moves: &[Move]) -> Result<Vec<f32>, EnginError> {
    let mut selected = gather_legal_logits(logits, legal_moves)?;
    softmax_inplace(&mut selected)?;
    Ok(selected)
}

/// 将合批原始输出的一行转为可缓存的正式 `EvalResult`。
///
/// 参考：px0 `BackendComputation` 的统一结果路径（`src/neural/backend.h:75-87`）。
pub fn eval_result_from_encoded_row(
    batch: &EncodedBatch,
    row: usize,
    legal_moves: &[Move],
) -> Result<Arc<EvalResult>, EnginError> {
    let logits = batch
        .logits
        .get(row * POLICY_SIZE..(row + 1) * POLICY_SIZE)
        .ok_or(EnginError::PortIncomplete("stream nn logits row"))?;
    let wdl = batch
        .wdl
        .get(row * 3..(row + 1) * 3)
        .ok_or(EnginError::PortIncomplete("stream nn wdl row"))?;
    let moves_left = *batch
        .moves_left
        .get(row)
        .ok_or(EnginError::PortIncomplete("stream nn moves_left row"))?;
    let policies = softmax_legal_policy(logits, legal_moves)?;
    if !wdl.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        || (wdl.iter().sum::<f32>() - 1.0).abs() > 1e-3
        || !moves_left.is_finite()
        || moves_left < 0.0
    {
        return Err(EnginError::Onnx("stream nn values are invalid".into()));
    }
    Ok(Arc::new(EvalResult {
        wl: wdl[0] - wdl[2],
        d: wdl[1],
        plies_left: moves_left,
        policies,
    }))
}

/// 历史平面填充策略。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FillEmptyHistory {
    No,
    FenOnly,
    Always,
}

/// 将 PositionHistory 编码为稀疏 `InputPlanes`。
///
/// 布局与填充语义对齐 px0 classical encoder（`src/neural/encoder.cc`）。
/// 搜索只传真实 `PositionHistory`；孤立 FEN 由调用方构造长度为一的 history，再用 `FenOnly`。
pub fn encode_position_input_planes(history: &PositionHistory, fill: FillEmptyHistory) -> InputPlanes {
    assert!(!history.is_empty(), "EncodePositionForNN requires a position");
    let mut planes = [InputPlane::default(); INPUT_PLANES];
    let current = history.last();
    if current.is_black_to_move() {
        planes[AUX_PLANE_BASE].set_all();
    }
    planes[AUX_PLANE_BASE + 1].fill(current.rule60_ply() as f32);
    planes[AUX_PLANE_BASE + 3].set_all();

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

/// 稀疏平面 → dense NCHW `[124][10][9]`。
///
/// 对齐 px0 ONNX 非 CUDA 分支的 CPU expand（`network_onnx.cc` `PrepareInputs`）。
pub fn expand_input_planes(planes: &[InputPlane], dest: &mut [f32]) {
    assert_eq!(planes.len(), INPUT_PLANES, "expand expects {INPUT_PLANES} planes");
    assert_eq!(dest.len(), ENCODED_PLANE_FLOATS, "expand dest must be NCHW floats");
    dest.fill(0.0);
    expand_input_planes_into_zeroed(planes, dest);
}

/// 与 [`expand_input_planes`] 相同，但假定 `dest` 已全零（供 ORT scratch 一次清零后批量 expand）。
pub fn expand_input_planes_into_zeroed(planes: &[InputPlane], dest: &mut [f32]) {
    assert_eq!(planes.len(), INPUT_PLANES, "expand expects {INPUT_PLANES} planes");
    assert_eq!(dest.len(), ENCODED_PLANE_FLOATS, "expand dest must be NCHW floats");
    for (plane, dest_plane) in planes.iter().zip(dest.chunks_exact_mut(BOARD_SQUARES)) {
        let mut bits = plane.mask;
        let value = plane.value;
        while bits != 0 {
            let square = bits.trailing_zeros() as usize;
            if square < BOARD_SQUARES {
                dest_plane[square] = value;
            }
            bits &= bits - 1;
        }
    }
}

/// 将 PositionHistory 编码为 dense classical NCHW planes（测试 / 兼容路径）。
pub fn encode_position_for_nn(history: &PositionHistory, fill: FillEmptyHistory) -> Vec<f32> {
    let sparse = encode_position_input_planes(history, fill);
    let mut dense = vec![0.0; ENCODED_PLANE_FLOATS];
    expand_input_planes(&sparse, &mut dense);
    dense
}

/// 着法到 policy 下标映射。
/// 表来自固定 2062 词表，禁止改排序。
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

fn write_board_planes(planes: &mut InputPlanes, base: usize, board: &ChessBoard, position: &Position) {
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
        planes[base + plane].mask = mask;
    }
    if position.repetitions() >= 1 {
        planes[base + 14].set_all();
    }
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{GameState, Position};

    use super::*;

    #[test]
    fn sparse_expand_matches_dense_encode() {
        let start = PositionHistory::from_positions(vec![Position::from_fen(xiangqi_core::STARTPOS_FEN).unwrap()]);
        let dense = encode_position_for_nn(&start, FillEmptyHistory::No);
        let sparse = encode_position_input_planes(&start, FillEmptyHistory::No);
        assert_eq!(sparse[0].mask, (1_u128 << 0) | (1 << 8));
        assert_eq!(sparse[AUX_PLANE_BASE + 3].mask, ALL_SQUARES);
        assert_eq!(sparse[AUX_PLANE_BASE + 3].value, 1.0);
        let mut expanded = vec![0.0; ENCODED_PLANE_FLOATS];
        expand_input_planes(&sparse, &mut expanded);
        assert_eq!(expanded, dense);
    }

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

    /// 直接移植 px0 `src/neural/encoder_test.cc:25-137` 的 classical 基线：
    /// square layout、黑方 auxiliary plane、rule60 与历史交替 mirror 都不能靠
    /// 当前实现自身生成期望值。
    #[test]
    fn px0_classical_startpos_and_two_ply_history() {
        fn mask(planes: &[f32], plane: usize) -> u128 {
            planes[plane * BOARD_ROWS * BOARD_COLS..(plane + 1) * BOARD_ROWS * BOARD_COLS]
                .iter()
                .enumerate()
                .filter_map(|(square, &value)| (value == 1.0).then_some(1_u128 << square))
                .sum()
        }

        fn is_filled_with(planes: &[f32], plane: usize, value: f32) -> bool {
            planes[plane * BOARD_ROWS * BOARD_COLS..(plane + 1) * BOARD_ROWS * BOARD_COLS]
                .iter()
                .all(|&cell| cell == value)
        }

        let start = PositionHistory::from_positions(vec![Position::from_fen(xiangqi_core::STARTPOS_FEN).unwrap()]);
        let planes = encode_position_for_nn(&start, FillEmptyHistory::No);
        assert_eq!(mask(&planes, 0), (1_u128 << 0) | (1 << 8));
        assert_eq!(mask(&planes, 1), (1_u128 << 3) | (1 << 5));
        assert_eq!(mask(&planes, 2), (1_u128 << 19) | (1 << 25));
        assert_eq!(
            mask(&planes, 3),
            (1_u128 << 27) | (1 << 29) | (1 << 31) | (1 << 33) | (1 << 35)
        );
        assert_eq!(mask(&planes, 4), (1_u128 << 1) | (1 << 7));
        assert_eq!(mask(&planes, 5), (1_u128 << 2) | (1 << 6));
        assert_eq!(mask(&planes, 6), 1_u128 << 4);
        assert_eq!(mask(&planes, 13), 1_u128 << 85);
        assert_eq!(mask(&planes, AUX_PLANE_BASE), 0);
        // px0 的 sparse plane 此时是“全 mask + value 0”；dense ONNX 输入中等价为全零。
        assert!(is_filled_with(&planes, AUX_PLANE_BASE + 1, 0.0));

        let game = GameState::from_fen_moves(xiangqi_core::STARTPOS_FEN, &["h2e2"]).unwrap();
        let history = PositionHistory::from_positions(game.positions());
        let planes = encode_position_for_nn(&history, FillEmptyHistory::No);
        assert_eq!(mask(&planes, AUX_PLANE_BASE), (1_u128 << 90) - 1);
        assert!(is_filled_with(&planes, AUX_PLANE_BASE + 1, 1.0));

        let game = GameState::from_fen_moves(xiangqi_core::STARTPOS_FEN, &["h2e2", "h9g7"]).unwrap();
        let history = PositionHistory::from_positions(game.positions());
        let planes = encode_position_for_nn(&history, FillEmptyHistory::No);
        assert_eq!(mask(&planes, AUX_PLANE_BASE), 0);
        assert!(is_filled_with(&planes, AUX_PLANE_BASE + 1, 2.0));
        assert_eq!(mask(&planes, PLANES_PER_BOARD), (1_u128 << 0) | (1 << 8));
    }

    #[test]
    fn softmax_is_limited_to_legal_policy_entries() {
        let a0 = xiangqi_core::Square::parse("a0").unwrap();
        let a1 = xiangqi_core::Square::parse("a1").unwrap();
        let a2 = xiangqi_core::Square::parse("a2").unwrap();
        let mut logits = vec![f32::NEG_INFINITY; POLICY_SIZE];
        logits[move_to_nn_index(xiangqi_core::Move::new(a0, a1)).unwrap()] = 0.0;
        logits[move_to_nn_index(xiangqi_core::Move::new(a0, a2)).unwrap()] = 1.0;
        let policy = softmax_legal_policy(
            &logits,
            &[xiangqi_core::Move::new(a0, a1), xiangqi_core::Move::new(a0, a2)],
        )
        .unwrap();
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(policy[1] > policy[0]);
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
