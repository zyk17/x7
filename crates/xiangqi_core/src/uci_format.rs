//! UCI 坐标与着法串（**常见皮卡鱼族象棋 UCI**：`a0`～`i9`，纵坐标 **0～9** 对应内部 `rank_of` 0～9）。
//!
//! 说明：旧版曾使用纵坐标 **1～10**（与部分 Python `pyffish` 字符串习惯一致），与 **常见引擎 UCI（0～9）** 不兼容；现已统一为 0～9。

use crate::board::Position;
use crate::types::*;

pub const START_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

/// 转为 UCI 半串：`[a-i][0-9]`（与常见皮卡鱼族引擎一致）。
pub fn square_to_algebraic(s: Square) -> String {
    let f = (file_of(s) as u8 + b'a') as char;
    let r = rank_of(s) as u32;
    format!("{f}{r}")
}

pub fn move_to_uci(m: Move) -> String {
    format!("{}{}", square_to_algebraic(m.from_sq()), square_to_algebraic(m.to_sq()))
}

/// 将着法写成与 [`move_to_uci`] 相同的 **ASCII**（`a`～`i` + 纵坐标 `0`～`9`），写入 `buf`；单格 2 字节，着法通常 4 字节。
#[inline]
pub fn write_move_uci_bytes(m: Move, buf: &mut [u8; 8]) -> usize {
    #[inline]
    fn write_sq(s: Square, dst: &mut [u8]) -> usize {
        let f = file_of(s) as u8;
        let r = rank_of(s) as u8;
        dst[0] = b'a' + f;
        dst[1] = b'0' + r;
        2
    }
    let n0 = write_sq(m.from_sq(), &mut buf[..]);
    let n1 = write_sq(m.to_sq(), &mut buf[n0..]);
    n0 + n1
}

/// 从 **着法 UCI 串**（如 `a0a1`）解析 [`Move`]，**不** 校验是否合法；纵坐标 **0～9**（与常见引擎 UCI 一致）。
pub fn parse_move_uci(s: &str) -> Option<Move> {
    let s = s.trim().to_ascii_lowercase();
    let b = s.as_bytes();
    if b.len() < 4 {
        return None;
    }
    let (from, to, _) = parse_two_square_uci_move(b)?;
    Some(Move::make(from, to))
}

fn parse_half_square_uci(b: &[u8]) -> Option<(Square, usize)> {
    if b.is_empty() {
        return None;
    }
    let f = b[0];
    if !matches!(f, b'a'..=b'i') {
        return None;
    }
    let mut j = 1usize;
    let mut num: u32 = 0;
    while j < b.len() && b[j].is_ascii_digit() {
        num = num.saturating_mul(10).saturating_add((b[j] - b'0') as u32);
        j += 1;
    }
    if num > 9 {
        return None;
    }
    let rank_u8 = num as u8;
    let file_u8 = f - b'a';
    let sq = make_square(unsafe { std::mem::transmute(file_u8) }, unsafe {
        std::mem::transmute(rank_u8)
    });
    Some((sq, j))
}

fn parse_two_square_uci_move(b: &[u8]) -> Option<(Square, Square, usize)> {
    let (from, j1) = parse_half_square_uci(b)?;
    let (to, j2) = parse_half_square_uci(&b[j1..])?;
    Some((from, to, j1 + j2))
}

/// 若 `from` 上为当前行棋方棋子则解析为 [`Move`]（坐标 `a0`～`i9`）。
pub fn uci_to_move(pos: &Position, s: &str) -> Option<Move> {
    let m = parse_move_uci(s)?;
    let from = m.from_sq();
    if pos.piece_on(from) == Piece::NO_PIECE || color_of(pos.piece_on(from)) != pos.side_to_move {
        return None;
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Piece;

    #[test]
    fn write_move_uci_bytes_matches_move_to_uci() {
        let m = parse_move_uci("b1e1").expect("b1e1");
        let mut buf = [0u8; 8];
        let n = write_move_uci_bytes(m, &mut buf);
        assert_eq!(std::str::from_utf8(&buf[..n]).unwrap(), move_to_uci(m));
    }

    #[test]
    fn parse_top_rank_nine() {
        let m = parse_move_uci("i7i9").expect("i7i9");
        assert_eq!(square_to_algebraic(m.to_sq()), "i9");
    }

    #[test]
    fn roundtrip_start_legal_first() {
        let pos = Position::from_fen(START_FEN).unwrap();
        let u = crate::legal_moves_uci(&pos).into_iter().next().expect("legal");
        let mv = parse_move_uci(&u).expect("parse");
        assert!(pos.piece_on(mv.from_sq()) != Piece::NO_PIECE);
    }
}
