//! px0 `board.cc:123-672` 攻击表与 magic bitboard 初始化。

use std::sync::OnceLock;

use crate::bitboard::BitBoard;
use crate::board_masks::{
    distance, file_bb, rank_bb, safe_destination, shift, BISHOP_DIRECTIONS, FILE_A_BB, FILE_I_BB, HALF_BB,
    KNIGHT_DIRECTIONS, PALACE, RANK0_BB, RANK9_BB,
};
use crate::magic_numbers::{KBISHOPMAGICNUMBERS, KKNIGHTMAGICNUMBERS, KKNIGHTTOMAGICNUMBERS, KROOKMAGICNUMBERS};
use crate::types::{
    file_distance, rank_distance, Direction, PieceType, Rank, Square, EAST, NORTH, NORTH_EAST, NORTH_WEST, SOUTH,
    SOUTH_EAST, SOUTH_WEST, WEST,
};

const PSEUDO_ATTACK_TYPES: usize = 10;

#[derive(Clone, Copy)]
struct MagicParams {
    mask: u128,
    attacks_table_offset: usize,
    magic_number: u128,
    shift_bits: u8,
}

struct AttackTables {
    rook_magic_params: [MagicParams; 90],
    cannon_magic_params: [MagicParams; 90],
    bishop_magic_params: [MagicParams; 90],
    knight_magic_params: [MagicParams; 90],
    knight_to_magic_params: [MagicParams; 90],
    rook_attacks_table: Vec<BitBoard>,
    cannon_attacks_table: Vec<BitBoard>,
    bishop_attacks_table: Vec<BitBoard>,
    knight_attacks_table: Vec<BitBoard>,
    knight_to_attacks_table: Vec<BitBoard>,
    pseudo_attacks: [[BitBoard; 90]; PSEUDO_ATTACK_TYPES],
    between_sq: [[Square; 90]; 90],
}

static ATTACK_TABLES: OnceLock<AttackTables> = OnceLock::new();

impl MagicParams {
    fn index(&self, occupied: u128) -> usize {
        self.attacks_table_offset + ((occupied & self.mask).wrapping_mul(self.magic_number) >> self.shift_bits) as usize
    }
}

fn sliding_attack(pt: PieceType, sq: Square, occupied: BitBoard) -> BitBoard {
    debug_assert!(pt == PieceType::Rook || pt == PieceType::Cannon);
    let mut attack = BitBoard::EMPTY;
    for direction in [NORTH, SOUTH, WEST, EAST] {
        let mut hurdle = false;
        let mut s = sq.offset_by(direction);
        while s.is_valid() && distance(s.offset_by(Direction(-direction.0, -direction.1)), s) == 1 {
            if pt == PieceType::Rook || hurdle {
                attack.set(s);
            }
            if occupied.contains(s) {
                if pt == PieceType::Cannon && !hurdle {
                    hurdle = true;
                } else {
                    break;
                }
            }
            s = s.offset_by(direction);
        }
    }
    attack
}

fn lame_leaper_path_one(pt: PieceType, d: Direction, mut s: Square) -> BitBoard {
    let mut to = s.offset_by(d);
    if !to.is_valid() || distance(s, to) >= 4 {
        return BitBoard::EMPTY;
    }

    if pt == PieceType::KnightTo {
        std::mem::swap(&mut s, &mut to);
        let d_inv = Direction(-d.0, -d.1);
        return lame_leaper_path_one(PieceType::Knight, d_inv, s);
    }

    let dr = Direction(if d.0 > 0 { 1 } else { -1 }, 0);
    let df = Direction(0, if d.1 > 0 { 1 } else { -1 });

    let diff =
        file_distance(to.file().unwrap(), s.file().unwrap()) - rank_distance(to.rank().unwrap(), s.rank().unwrap());
    if diff > 0 {
        s = s.offset_by(df);
    } else if diff < 0 {
        s = s.offset_by(dr);
    } else {
        s = s.offset_by(df);
        s = s.offset_by(dr);
    }

    let mut b = BitBoard::EMPTY;
    if s.is_valid() {
        b.set(s);
    }
    b
}

fn lame_leaper_path(pt: PieceType, s: Square) -> BitBoard {
    let directions = if pt == PieceType::Bishop {
        &BISHOP_DIRECTIONS[..]
    } else {
        &KNIGHT_DIRECTIONS[..]
    };
    let mut b = BitBoard::EMPTY;
    for d in directions {
        b = b.union(lame_leaper_path_one(pt, *d, s));
    }
    if pt == PieceType::Bishop {
        let half_idx = usize::from(s.rank().unwrap().index() > Rank::R4.index());
        b = b.intersection(BitBoard::from_bits(HALF_BB[half_idx]));
    }
    b
}

fn lame_leaper_attack(pt: PieceType, s: Square, occupied: BitBoard) -> BitBoard {
    let directions = if pt == PieceType::Bishop {
        &BISHOP_DIRECTIONS[..]
    } else {
        &KNIGHT_DIRECTIONS[..]
    };
    let mut b = BitBoard::EMPTY;
    for d in directions {
        let to = s.offset_by(*d);
        if to.is_valid() && distance(s, to) < 4 && lame_leaper_path_one(pt, *d, s).intersection(occupied).is_empty() {
            b.set(to);
        }
    }
    if pt == PieceType::Bishop {
        let half_idx = usize::from(s.rank().unwrap().index() > Rank::R4.index());
        b = b.intersection(BitBoard::from_bits(HALF_BB[half_idx]));
    }
    b
}

fn pawn_attacks_bb(s: Square) -> BitBoard {
    let b = BitBoard::from_square(s);
    let mut attack = shift(NORTH, b);
    if s.rank().unwrap().index() > Rank::R4.index() {
        attack = attack.union(shift(WEST, b)).union(shift(EAST, b));
    }
    attack
}

fn pawn_attacks_to_bb<const OURS: bool>(s: Square) -> BitBoard {
    let b = BitBoard::from_square(s);
    let direction = if OURS { NORTH } else { SOUTH };
    let mut attack = shift(direction, b);
    let rank = s.rank().unwrap().index();
    if (OURS && rank < Rank::R5.index()) || (!OURS && rank > Rank::R4.index()) {
        attack = attack.union(shift(WEST, b)).union(shift(EAST, b));
    }
    attack
}

fn build_attacks_table(
    pt: PieceType,
    magic_params: &mut [MagicParams; 90],
    attacks_table: &mut Vec<BitBoard>,
    rook_magic_params: Option<&[MagicParams; 90]>,
) {
    let mut table_offset = 0usize;
    for square in 0u8..90 {
        let b_sq = Square::from_idx(square).unwrap();
        let edges = BitBoard::from_bits(RANK0_BB | RANK9_BB)
            .difference(rank_bb(b_sq.rank().unwrap().index()))
            .union(BitBoard::from_bits(FILE_A_BB | FILE_I_BB).difference(file_bb(b_sq.file().unwrap().index())));

        let mask = match pt {
            PieceType::Rook => sliding_attack(pt, b_sq, BitBoard::EMPTY),
            PieceType::Cannon => BitBoard::from_bits(
                rook_magic_params.expect("cannon table requires rook magic params")[square as usize].mask,
            ),
            PieceType::Bishop | PieceType::Knight | PieceType::KnightTo => lame_leaper_path(pt, b_sq),
            _ => BitBoard::EMPTY,
        };
        let mask = if pt != PieceType::KnightTo {
            mask.difference(edges)
        } else {
            mask
        };

        let mask_bits = mask.bits();
        let mask_count = mask.count();
        let shift_bits = (128 - mask_count) as u8;

        magic_params[square as usize] = MagicParams {
            mask: mask_bits,
            attacks_table_offset: table_offset,
            magic_number: match pt {
                PieceType::Rook | PieceType::Cannon => KROOKMAGICNUMBERS[square as usize],
                PieceType::Bishop => KBISHOPMAGICNUMBERS[square as usize],
                PieceType::Knight => KKNIGHTMAGICNUMBERS[square as usize],
                PieceType::KnightTo => KKNIGHTTOMAGICNUMBERS[square as usize],
                _ => 0,
            },
            shift_bits,
        };

        let table_size = 1usize << mask_count;
        attacks_table.resize(table_offset + table_size, BitBoard::EMPTY);

        let mut b: u128 = 0;
        loop {
            let idx = magic_params[square as usize].index(b) - table_offset;
            let attacks = match pt {
                PieceType::Rook | PieceType::Cannon => sliding_attack(pt, b_sq, BitBoard::from_bits(b)),
                PieceType::Bishop | PieceType::Knight | PieceType::KnightTo => {
                    lame_leaper_attack(pt, b_sq, BitBoard::from_bits(b))
                }
                _ => BitBoard::EMPTY,
            };
            attacks_table[table_offset + idx] = attacks;
            b = (b.wrapping_sub(mask_bits)) & mask_bits;
            if b == 0 {
                break;
            }
        }

        table_offset += table_size;
    }
}

impl AttackTables {
    fn new() -> Self {
        let empty_magic = MagicParams {
            mask: 0,
            attacks_table_offset: 0,
            magic_number: 0,
            shift_bits: 0,
        };

        let mut rook_magic_params = [empty_magic; 90];
        let mut cannon_magic_params = [empty_magic; 90];
        let mut bishop_magic_params = [empty_magic; 90];
        let mut knight_magic_params = [empty_magic; 90];
        let mut knight_to_magic_params = [empty_magic; 90];

        let mut rook_attacks_table = Vec::new();
        let mut cannon_attacks_table = Vec::new();
        let mut bishop_attacks_table = Vec::new();
        let mut knight_attacks_table = Vec::new();
        let mut knight_to_attacks_table = Vec::new();

        build_attacks_table(PieceType::Rook, &mut rook_magic_params, &mut rook_attacks_table, None);
        build_attacks_table(
            PieceType::Cannon,
            &mut cannon_magic_params,
            &mut cannon_attacks_table,
            Some(&rook_magic_params),
        );
        build_attacks_table(
            PieceType::Bishop,
            &mut bishop_magic_params,
            &mut bishop_attacks_table,
            None,
        );
        build_attacks_table(
            PieceType::Knight,
            &mut knight_magic_params,
            &mut knight_attacks_table,
            None,
        );
        build_attacks_table(
            PieceType::KnightTo,
            &mut knight_to_magic_params,
            &mut knight_to_attacks_table,
            None,
        );

        let mut pseudo_attacks = [[BitBoard::EMPTY; 90]; PSEUDO_ATTACK_TYPES];
        let mut between_sq = [[Square::INVALID; 90]; 90];

        for square in 0u8..90 {
            let b_sq = Square::from_idx(square).unwrap();
            pseudo_attacks[PieceType::Pawn as usize][square as usize] = pawn_attacks_bb(b_sq);
            pseudo_attacks[PieceType::PawnToOurs as usize][square as usize] = pawn_attacks_to_bb::<true>(b_sq);
            pseudo_attacks[PieceType::PawnToTheirs as usize][square as usize] = pawn_attacks_to_bb::<false>(b_sq);

            if (PALACE & (1u128 << square)) != 0 {
                let mut king_attacks = BitBoard::EMPTY;
                for d in [NORTH, SOUTH, WEST, EAST] {
                    king_attacks = king_attacks.union(safe_destination(b_sq, d));
                }
                pseudo_attacks[PieceType::King as usize][square as usize] =
                    king_attacks.intersection(BitBoard::from_bits(PALACE));

                let mut advisor_attacks = BitBoard::EMPTY;
                for d in [NORTH_WEST, NORTH_EAST, SOUTH_WEST, SOUTH_EAST] {
                    advisor_attacks = advisor_attacks.union(safe_destination(b_sq, d));
                }
                pseudo_attacks[PieceType::Advisor as usize][square as usize] =
                    advisor_attacks.intersection(BitBoard::from_bits(PALACE));
            }

            pseudo_attacks[PieceType::Knight as usize][square as usize] =
                lame_leaper_attack(PieceType::Knight, b_sq, BitBoard::EMPTY);

            for square2 in 0u8..90 {
                let b_sq2 = Square::from_idx(square2).unwrap();
                if pseudo_attacks[PieceType::Knight as usize][square as usize].contains(b_sq2) {
                    let direction = Direction(
                        b_sq2.rank().unwrap().index() as i32 - b_sq.rank().unwrap().index() as i32,
                        b_sq2.file().unwrap().index() as i32 - b_sq.file().unwrap().index() as i32,
                    );
                    if let Some(blocker) = lame_leaper_path_one(PieceType::KnightTo, direction, b_sq).iter().next() {
                        between_sq[square as usize][square2 as usize] = blocker;
                    }
                }
            }
        }

        Self {
            rook_magic_params,
            cannon_magic_params,
            bishop_magic_params,
            knight_magic_params,
            knight_to_magic_params,
            rook_attacks_table,
            cannon_attacks_table,
            bishop_attacks_table,
            knight_attacks_table,
            knight_to_attacks_table,
            pseudo_attacks,
            between_sq,
        }
    }
}

fn tables() -> &'static AttackTables {
    ATTACK_TABLES
        .get()
        .expect("initialize_magic_bitboards must be called before using attacks")
}

/// px0 `board.cc:619-672`。
pub fn initialize_magic_bitboards() {
    ATTACK_TABLES.get_or_init(AttackTables::new);
}

/// px0 `board.cc:563-615`。
pub fn get_attacks(pt: PieceType, square: Square, pieces: BitBoard) -> BitBoard {
    let _ = tables();
    let s = square.index() as usize;
    let t = tables();
    match pt {
        PieceType::Rook => t.rook_attacks_table[t.rook_magic_params[s].index(pieces.bits())],
        PieceType::Cannon => t.cannon_attacks_table[t.cannon_magic_params[s].index(pieces.bits())],
        PieceType::Bishop => t.bishop_attacks_table[t.bishop_magic_params[s].index(pieces.bits())],
        PieceType::Knight => t.knight_attacks_table[t.knight_magic_params[s].index(pieces.bits())],
        PieceType::KnightTo => t.knight_to_attacks_table[t.knight_to_magic_params[s].index(pieces.bits())],
        _ => t.pseudo_attacks[pt as usize][s],
    }
}

pub fn between_sq(from: Square, to: Square) -> Square {
    let _ = tables();
    tables().between_sq[from.index() as usize][to.index() as usize]
}
