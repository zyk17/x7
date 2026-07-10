//! px0 固定 2062 维 move head 词表：UCI 着法 ↔ logits 下标。

use std::sync::OnceLock;

use xiangqi_core::types::{flip_rank, Square};
use xiangqi_core::Move;

const MOVE_VOCAB: &str = include_str!("move_vocab.txt");

fn parse_square(bytes: &[u8]) -> Option<Square> {
    if bytes.len() != 2 {
        return None;
    }
    let file = bytes[0];
    let rank = bytes[1];
    if !(b'a'..=b'i').contains(&file) || !rank.is_ascii_digit() {
        return None;
    }
    let f = (file - b'a') as u16;
    let r = (rank - b'0') as u16;
    let idx = r * 9 + f;
    Some(unsafe { std::mem::transmute::<u8, Square>(idx as u8) })
}

fn packed_idx(mv: Move) -> usize {
    ((mv.from_sq().to_u8() as usize) << 7) | mv.to_sq().to_u8() as usize
}

fn index_table() -> &'static [i16; 128 * 128] {
    static TABLE: OnceLock<[i16; 128 * 128]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [-1i16; 128 * 128];
        for (idx, line) in MOVE_VOCAB.lines().enumerate() {
            let bytes = line.as_bytes();
            let Some(from) = parse_square(&bytes[0..2]) else {
                continue;
            };
            let Some(to) = parse_square(&bytes[2..4]) else {
                continue;
            };
            let mv = Move::make(from, to);
            table[packed_idx(mv)] = idx as i16;
        }
        table
    })
}

fn vocab_moves() -> &'static [&'static str] {
    static MOVES: OnceLock<Vec<&'static str>> = OnceLock::new();
    MOVES.get_or_init(|| MOVE_VOCAB.lines().collect()).as_slice()
}

pub fn move_vocab_index(mv: Move, black_to_move: bool) -> Option<usize> {
    let mapped = if black_to_move {
        Move::make(flip_rank(mv.from_sq()), flip_rank(mv.to_sq()))
    } else {
        mv
    };
    let idx = index_table()[packed_idx(mapped)];
    (idx >= 0).then_some(idx as usize)
}

pub fn move_vocab_move(index: usize, black_to_move: bool) -> Option<Move> {
    let uci = *vocab_moves().get(index)?;
    let bytes = uci.as_bytes();
    let from = parse_square(&bytes[0..2])?;
    let to = parse_square(&bytes[2..4])?;
    let mapped = if black_to_move {
        Move::make(flip_rank(from), flip_rank(to))
    } else {
        Move::make(from, to)
    };
    Some(mapped)
}
