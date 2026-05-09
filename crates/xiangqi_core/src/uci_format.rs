//! UCI 坐标与着法串（与 pyffish / Pikafish 一致：`a0`～`i9`，纵坐标可为 **1～10** 对应盘面条纹 0～9）。

use crate::board::Position;
use crate::types::*;

pub const START_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

/// 转为 pyffish 半串：纵坐标为 **1～10**（对应内部条纹 0～9）。
pub fn square_to_algebraic(s: Square) -> String {
    let r = rank_of(s) as u8 as u32 + 1;
    format!("{}{}", (file_of(s) as u8 + b'a') as char, r)
}

pub fn move_to_uci(m: Move) -> String {
    format!("{}{}", square_to_algebraic(m.from_sq()), square_to_algebraic(m.to_sq()))
}

/// 从 pyffish 风格 UCI 串解析着法（**不** 校验局面）；纵坐标为 **1～10**（盘面条纹为 值−1）。
pub fn parse_pyffish_uci(s: &str) -> Option<Move> {
    let s = s.trim().to_ascii_lowercase();
    let b = s.as_bytes();
    if b.len() < 4 {
        return None;
    }
    let (from, to, _) = parse_two_squares_pyffish(b)?;
    Some(Move::make(from, to))
}

fn parse_half_pyffish(b: &[u8]) -> Option<(Square, usize)> {
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
    if !(1..=10).contains(&num) {
        return None;
    }
    let rank_u8 = (num - 1) as u8;
    let file_u8 = f - b'a';
    let sq = make_square(unsafe { std::mem::transmute(file_u8) }, unsafe {
        std::mem::transmute(rank_u8)
    });
    Some((sq, j))
}

fn parse_two_squares_pyffish(b: &[u8]) -> Option<(Square, Square, usize)> {
    let (from, j1) = parse_half_pyffish(b)?;
    let (to, j2) = parse_half_pyffish(&b[j1..])?;
    Some((from, to, j1 + j2))
}

/// 若 `from` 上为当前行棋方棋子则解析为 [`Move`]（单数字纵坐标兼容，与旧逻辑一致）。
pub fn uci_to_move(pos: &Position, s: &str) -> Option<Move> {
    let m = parse_pyffish_uci(s)?;
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
    fn parse_double_digit_rank_on_to_square() {
        // 纵坐标 10：后半「i10」共 4 字符
        let m = parse_pyffish_uci("i8i10").expect("i8i10");
        assert_eq!(square_to_algebraic(m.to_sq()), "i10");
    }

    #[test]
    fn roundtrip_start_legal_first() {
        let pos = Position::from_fen(START_FEN).unwrap();
        let u = crate::legal_moves_uci(&pos).into_iter().next().expect("legal");
        let mv = parse_pyffish_uci(&u).expect("parse");
        assert!(pos.piece_on(mv.from_sq()) != Piece::NO_PIECE);
    }
}
