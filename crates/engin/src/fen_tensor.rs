//! FEN → CNN 输入平面，与 `nn/src/nn/fen_tensor.py` 一致：`[1, 15, 10, 9]`，NCHW，float32。

use ndarray::{s, Array4};
use xiangqi_core::board::PIECE_TO_CHAR;
use xiangqi_core::types::{file_of, rank_of, Color, Piece, Square};
use xiangqi_core::Position;

/// 红方 7 类棋子通道顺序（与 Python `RED_CHARS` 一致）。
const RED_PLANE: [char; 7] = ['R', 'N', 'B', 'A', 'K', 'C', 'P'];
/// 黑方 7 类（与 Python `BLACK_CHARS` 一致）。
const BLACK_PLANE: [char; 7] = ['r', 'n', 'b', 'a', 'k', 'c', 'p'];

fn expand_rank(rank_str: &str) -> Result<Vec<char>, String> {
    let mut cells = Vec::with_capacity(9);
    for ch in rank_str.chars() {
        if ch.is_ascii_digit() {
            let n = ch.to_digit(10).ok_or_else(|| format!("无效 FEN 数字: {ch:?}"))? as usize;
            cells.extend(std::iter::repeat_n('.', n));
        } else {
            cells.push(ch);
        }
    }
    Ok(cells)
}

fn plane_index(ch: char) -> Option<usize> {
    RED_PLANE
        .iter()
        .position(|&c| c == ch)
        .or_else(|| BLACK_PLANE.iter().position(|&c| c == ch).map(|i| i + RED_PLANE.len()))
}

/// 将完整 FEN 编码为 `float32[1, 15, 10, 9]`。
///
/// - 通道 0–6：红方棋子；7–13：黑方；14：轮到红走为全 1，否则全 0。
/// - `row` 0 对应 FEN 棋盘串的第一行（远离红方一侧，与训练一致）。
pub(crate) fn fen_to_planes(fen: &str) -> Result<Array4<f32>, String> {
    let mut parts = fen.split_whitespace();
    let board = parts.next().ok_or_else(|| "FEN 为空".to_string())?;
    let stm = parts.next().unwrap_or("w");

    let ranks: Vec<&str> = board.split('/').collect();
    if ranks.len() != 10 {
        return Err(format!("期望 10 行棋盘，得到 {}: {board}", ranks.len()));
    }

    let mut planes = Array4::<f32>::zeros((1, 15, 10, 9));

    for (ri, rank) in ranks.iter().enumerate() {
        let cells = expand_rank(rank)?;
        if cells.len() != 9 {
            return Err(format!("行 {ri} 长度应为 9，得到 {}: {rank:?}", cells.len()));
        }
        for (fi, ch) in cells.iter().enumerate() {
            if *ch == '.' {
                continue;
            }
            let c = plane_index(*ch).ok_or_else(|| format!("未知棋子符号: {ch:?} in {fen}"))?;
            planes[[0, c, ri, fi]] = 1.0;
        }
    }

    let stm_fill = if stm == "w" { 1.0 } else { 0.0 };
    planes.slice_mut(s![0, 14, .., ..]).fill(stm_fill);

    Ok(planes)
}

/// 由 [`Position`] 直接编码为 `float32[1, 15, 10, 9]`，避免 `FEN` 字符串分配（与 [`fen_to_planes`] 语义一致）。
pub(crate) fn position_to_planes(pos: &Position) -> Result<Array4<f32>, String> {
    let mut planes = Array4::<f32>::zeros((1, 15, 10, 9));
    for sq_u in 0u8..90 {
        let sq: Square = unsafe { std::mem::transmute(sq_u) };
        let pc = pos.piece_on(sq);
        if pc == Piece::NO_PIECE {
            continue;
        }
        let ch = PIECE_TO_CHAR.as_bytes()[pc.0 as usize] as char;
        if ch == ' ' {
            continue;
        }
        let c = plane_index(ch).ok_or_else(|| format!("未知棋子符号: {ch:?}"))?;
        let fi = file_of(sq) as usize;
        let r = rank_of(sq) as usize;
        let ri = 9usize.saturating_sub(r);
        planes[[0, c, ri, fi]] = 1.0;
    }
    let stm_fill = if pos.side_to_move == Color::White { 1.0 } else { 0.0 };
    planes.slice_mut(s![0, 14, .., ..]).fill(stm_fill);
    Ok(planes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::Position;
    use xiangqi_core::START_FEN;

    #[test]
    fn start_fen_stm_plane_red() {
        let t = fen_to_planes(START_FEN).unwrap();
        assert_eq!(t.shape(), [1, 15, 10, 9]);
        assert!((t[[0, 14, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn start_fen_corner_rooks() {
        let t = fen_to_planes(START_FEN).unwrap();
        // 顶行黑车 'r' → 通道 7，(0,0)
        assert!((t[[0, 7, 0, 0]] - 1.0).abs() < 1e-6);
        // 底行红车 'R' → 通道 0，(9,0)
        assert!((t[[0, 0, 9, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn black_to_move_zero_stm_plane() {
        let fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1";
        let t = fen_to_planes(fen).unwrap();
        assert!((t[[0, 14, 0, 0]]).abs() < 1e-6);
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
