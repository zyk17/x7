use std::sync::OnceLock;

use xiangqi_core::types::{flip_rank, Square};
use xiangqi_core::Move;

const PX0_POLICY_MOVES: &str = include_str!("px0_policy_moves.txt");

fn parse_square(bytes: &[u8]) -> Option<Square> {
    if bytes.len() != 2 {
        return None;
    }
    let file = bytes[0];
    let rank = bytes[1];
    if !(b'a'..=b'i').contains(&file) || !(b'0'..=b'9').contains(&rank) {
        return None;
    }
    let f = (file - b'a') as u16;
    let r = (rank - b'0') as u16;
    let idx = r * 9 + f;
    Some(unsafe { std::mem::transmute(idx as u8) })
}

fn packed_idx(mv: Move) -> usize {
    ((mv.from_sq().to_u8() as usize) << 7) | mv.to_sq().to_u8() as usize
}

fn px0_index_table() -> &'static [i16; 128 * 128] {
    static TABLE: OnceLock<[i16; 128 * 128]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [-1i16; 128 * 128];
        for (idx, line) in PX0_POLICY_MOVES.lines().enumerate() {
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

pub fn px0_policy_index(mv: Move, black_to_move: bool) -> Option<usize> {
    let mapped = if black_to_move {
        Move::make(flip_rank(mv.from_sq()), flip_rank(mv.to_sq()))
    } else {
        mv
    };
    let idx = px0_index_table()[packed_idx(mapped)];
    (idx >= 0).then_some(idx as usize)
}
