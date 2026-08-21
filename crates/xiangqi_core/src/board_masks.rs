//! 几何 mask 与距离辅助。来源：px0 board.cc。

use crate::bitboard::BitBoard;
use crate::types::{Direction, EAST, NORTH, SOUTH, Square, WEST, file_distance, rank_distance};

pub const PALACE: u128 = (0x0000_0000_0070_381Cu128 << 64) | 0x0000_0000_00E0_7038u128;
/// 士只能停在九宫的五个斜线交点；`PALACE` 还包含将可走、但士不可停的四个边点。
///
/// px0 的 FEN 校验只要求士在九宫内。x7 在合法着生成前额外区分士位，防止
/// `f1e0` 这类从九宫边点出发的非法士着进入 policy 与搜索。
pub const ADVISOR_SQUARES: u128 = (1u128 << 3)
    | (1u128 << 5)
    | (1u128 << 13)
    | (1u128 << 21)
    | (1u128 << 23)
    | (1u128 << 66)
    | (1u128 << 68)
    | (1u128 << 76)
    | (1u128 << 84)
    | (1u128 << 86);
pub const FILE_A_BB: u128 = (0x0000_0000_0002_0100u128 << 64) | 0x8040_2010_0804_0201u128;
pub const FILE_C_BB: u128 = FILE_A_BB << 2;
pub const FILE_E_BB: u128 = FILE_A_BB << 4;
pub const FILE_G_BB: u128 = FILE_A_BB << 6;
pub const FILE_I_BB: u128 = FILE_A_BB << 8;

pub const RANK0_BB: u128 = 0x1FF;
pub const RANK1_BB: u128 = RANK0_BB << 9;
pub const RANK2_BB: u128 = RANK0_BB << 18;
pub const RANK3_BB: u128 = RANK0_BB << 27;
pub const RANK4_BB: u128 = RANK0_BB << 36;
pub const RANK5_BB: u128 = RANK0_BB << 45;
pub const RANK6_BB: u128 = RANK0_BB << 54;
pub const RANK7_BB: u128 = RANK0_BB << 63;
pub const RANK8_BB: u128 = RANK0_BB << 72;
pub const RANK9_BB: u128 = RANK0_BB << 81;

pub const BISHOP_DIRECTIONS: [Direction; 4] = [Direction(2, 2), Direction(-2, 2), Direction(2, -2), Direction(-2, -2)];

pub const KNIGHT_DIRECTIONS: [Direction; 8] = [
    Direction(-2, -1),
    Direction(-2, 1),
    Direction(2, -1),
    Direction(2, 1),
    Direction(1, -2),
    Direction(1, 2),
    Direction(-1, -2),
    Direction(-1, 2),
];

pub fn bishop_bb() -> BitBoard {
    let file_mask = BitBoard::from_bits(FILE_A_BB | FILE_E_BB | FILE_I_BB);
    let file_cg = BitBoard::from_bits(FILE_C_BB | FILE_G_BB);
    let rank_27 = BitBoard::from_bits(RANK2_BB | RANK7_BB);
    let rank_0459 = BitBoard::from_bits(RANK0_BB | RANK4_BB | RANK5_BB | RANK9_BB);
    file_mask.intersection(rank_27).union(file_cg.intersection(rank_0459))
}

pub const PAWN_FILE_BB: u128 = FILE_A_BB | FILE_C_BB | FILE_E_BB | FILE_G_BB | FILE_I_BB;

pub const HALF_BB: [u128; 2] = [
    RANK0_BB | RANK1_BB | RANK2_BB | RANK3_BB | RANK4_BB,
    RANK5_BB | RANK6_BB | RANK7_BB | RANK8_BB | RANK9_BB,
];

pub fn pawn_bb(for_theirs: bool) -> BitBoard {
    let half = if for_theirs { HALF_BB[0] } else { HALF_BB[1] };
    let extra = if for_theirs {
        (RANK6_BB | RANK5_BB) & PAWN_FILE_BB
    } else {
        (RANK3_BB | RANK4_BB) & PAWN_FILE_BB
    };
    BitBoard::from_bits(half | extra)
}

pub fn rank_bb(rank: u8) -> BitBoard {
    BitBoard::from_bits(RANK0_BB << (9 * rank))
}

pub fn file_bb(file: u8) -> BitBoard {
    BitBoard::from_bits(FILE_A_BB << file)
}

pub fn distance(a: Square, b: Square) -> i32 {
    match (a.file(), a.rank(), b.file(), b.rank()) {
        (Some(af), Some(ar), Some(bf), Some(br)) => file_distance(af, bf).max(rank_distance(ar, br)),
        _ => i32::MAX,
    }
}

pub fn safe_destination(s: Square, step: Direction) -> BitBoard {
    let to = s.offset_by(step);
    if to.is_valid() && distance(s, to) <= 2 {
        BitBoard::from_square(to)
    } else {
        BitBoard::EMPTY
    }
}

pub fn shift(direction: Direction, board: BitBoard) -> BitBoard {
    let bits = board.bits();
    let shifted = match direction {
        NORTH => (bits & !RANK9_BB) << 9,
        SOUTH => bits >> 9,
        EAST => (bits & !FILE_I_BB) << 1,
        WEST => (bits & !FILE_A_BB) >> 1,
        _ => 0,
    };
    BitBoard::from_bits(shifted)
}
