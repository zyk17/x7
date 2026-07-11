//! px0 规则判定（`position.cc:126-169`、`board.cc:825-948`、`board.cc:1072-1141`）。

use crate::board::{
    advisor_attacks, bishop_attacks, cannon_attacks, king_attacks, knight_attacks, pawn_attacks, piece_attacks,
    popcount, rook_attacks, square_bb, Position,
};
use crate::movegen::{generate, ExtMove, GenType};
use crate::types::*;

/// px0 `GameResult`（`types.h`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Undecided,
    Draw,
    WhiteWon,
    BlackWon,
}

const MAX_MOVES: usize = 128;

// px0 `HalfBB[1]`（board.cc:118-119）：rank 5–9。
const HALF_BB_UPPER: Bitboard = 0x1FF | (0x1FF << 9) | (0x1FF << 18) | (0x1FF << 27) | (0x1FF << 36);

/// px0 `ChessBoard::IsUnderCheck()` → `checkers().any()`.
pub fn is_under_check(pos: &Position) -> bool {
    pos.checkers() != 0
}

/// px0 `ChessBoard::RecapturesTo`（board.cc:825-843）。
pub fn recaptures_to(pos: &Position, sq: Square) -> Bitboard {
    let occupied = pos.occupancy();
    let mut attackers = 0u128;
    attackers |= rook_attacks(sq, occupied) & pos.piece_type_bb(PieceType::Rook);
    attackers |= advisor_attacks(sq) & pos.piece_type_bb(PieceType::Advisor);
    attackers |= cannon_attacks(sq, occupied) & pos.piece_type_bb(PieceType::Cannon);
    let capturer = pos.side_to_move;
    attackers |= pawn_attacks(sq, capturer) & pos.piece_type_bb(PieceType::Pawn);
    attackers |= knight_attacks(sq, occupied) & pos.piece_type_bb(PieceType::Knight);
    attackers |= bishop_attacks(sq, occupied) & pos.piece_type_bb(PieceType::Bishop);
    attackers |= king_attacks(sq) & pos.pieces_c_pt(capturer, PieceType::King);
    attackers & pos.color_bb(capturer)
}

fn checkers_to_px0(pos: &Position, our: bool, ksq: Square, occupied: Bitboard) -> Bitboard {
    let mut checkers = 0u128;
    checkers |= rook_attacks(ksq, occupied) & pos.piece_type_bb(PieceType::Rook);
    checkers |= cannon_attacks(ksq, occupied) & pos.piece_type_bb(PieceType::Cannon);
    let pawn_color = if our { pos.side_to_move } else { !pos.side_to_move };
    checkers |= pawn_attacks(ksq, pawn_color) & pos.piece_type_bb(PieceType::Pawn);
    checkers |= knight_attacks(ksq, occupied) & pos.piece_type_bb(PieceType::Knight);
    let attackers = if our { !pos.side_to_move } else { pos.side_to_move };
    checkers & pos.color_bb(attackers)
}

/// px0 `IsLegalMove<our>`（board.cc:845-871）。
fn is_legal_move_for(pos: &Position, m: Move, our: bool) -> bool {
    let stm = pos.side_to_move;
    let our_king = pos.king_square(if our { stm } else { !stm });
    let their_king = pos.king_square(if our { !stm } else { stm });

    let mut occupied = pos.occupancy();
    occupied ^= square_bb(m.from_sq());
    occupied |= square_bb(m.to_sq());

    let ksq = if our_king == m.from_sq() { m.to_sq() } else { our_king };
    if rook_attacks(ksq, occupied) & square_bb(their_king) != 0 {
        return false;
    }
    if ksq != our_king {
        return checkers_to_px0(pos, our, ksq, occupied) == 0;
    }
    let mut checkers = checkers_to_px0(pos, our, ksq, occupied);
    checkers &= !square_bb(m.to_sq());
    checkers == 0
}

fn make_chase(_pos: &Position, _us: Color, to: Square, id_board: &[u8; SQUARE_NB]) -> u16 {
    let id = id_board[to as usize];
    1u16 << id
}

fn build_id_board(pos: &Position, us: Color) -> [u8; SQUARE_NB] {
    let mut id_board = [0u8; SQUARE_NB];
    let mut our_id = 0u8;
    let mut their_id = 0u8;
    let mut bb = pos.occupancy();
    while bb != 0 {
        let sq: Square = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
        let pc = pos.piece_on(sq);
        if color_of(pc) == us {
            id_board[sq as usize] = our_id;
            our_id += 1;
        } else {
            id_board[sq as usize] = their_id;
            their_id += 1;
        }
        bb &= bb - 1;
    }
    id_board
}

/// px0 `ChessBoard::UsChased`（board.cc:879-942）。
pub fn us_chased(pos: &Position) -> u16 {
    let us = pos.side_to_move;
    let them = !us;
    let id_board = build_id_board(pos, us);
    let occupied = pos.occupancy();
    let kings = pos.pieces_c_pt(Color::White, PieceType::King) | pos.pieces_c_pt(Color::Black, PieceType::King);
    let mut chase = 0u16;

    let mut add_chase = |attacker_type: PieceType, attacker_bb: Bitboard| {
        let mut bb = attacker_bb & pos.color_bb(us);
        while bb != 0 {
            let from: Square = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            let mut attacks = piece_attacks(attacker_type, from, occupied, us) & pos.color_bb(them);
            attacks &= !(kings | (pos.piece_type_bb(PieceType::Pawn) & HALF_BB_UPPER));

            let mut candidates = 0u128;
            if attacker_type == PieceType::Knight || attacker_type == PieceType::Cannon {
                candidates = attacks & pos.piece_type_bb(PieceType::Rook);
            }
            if attacker_type == PieceType::Advisor || attacker_type == PieceType::Bishop {
                candidates = attacks
                    & (pos.piece_type_bb(PieceType::Rook)
                        | pos.piece_type_bb(PieceType::Knight)
                        | pos.piece_type_bb(PieceType::Cannon));
            }
            attacks &= !candidates;
            let mut cand = candidates;
            while cand != 0 {
                let to: Square = unsafe { std::mem::transmute(cand.trailing_zeros() as u8) };
                let m = Move::make(from, to);
                if is_legal_move_for(pos, m, true) {
                    chase |= make_chase(pos, us, to, &id_board);
                }
                cand &= cand - 1;
            }

            let mut att = attacks;
            while att != 0 {
                let to: Square = unsafe { std::mem::transmute(att.trailing_zeros() as u8) };
                let m = Move::make(from, to);
                if is_legal_move_for(pos, m, true) {
                    let mut true_chase = true;
                    let mut after = pos.clone_for_search();
                    after.do_move(m);
                    let mut recaptures = recaptures_to(&after, to);
                    while recaptures != 0 {
                        let s: Square = unsafe { std::mem::transmute(recaptures.trailing_zeros() as u8) };
                        let recapture = Move::make(s, to);
                        if after.legal(recapture) {
                            true_chase = false;
                            break;
                        }
                        recaptures &= recaptures - 1;
                    }
                    if true_chase {
                        if attacker_bb & square_bb(to) != 0 {
                            let knight_pin = attacker_type == PieceType::Knight
                                && knight_attacks(to, occupied) & square_bb(from) == 0;
                            let reverse = Move::make(to, from);
                            if knight_pin || !is_legal_move_for(pos, reverse, false) {
                                chase |= make_chase(pos, us, to, &id_board);
                            }
                        } else {
                            chase |= make_chase(pos, us, to, &id_board);
                        }
                    }
                }
                att &= att - 1;
            }
            bb &= bb - 1;
        }
    };

    add_chase(PieceType::Rook, pos.piece_type_bb(PieceType::Rook));
    add_chase(PieceType::Advisor, pos.piece_type_bb(PieceType::Advisor));
    add_chase(PieceType::Cannon, pos.piece_type_bb(PieceType::Cannon));
    add_chase(PieceType::Knight, pos.piece_type_bb(PieceType::Knight));
    add_chase(PieceType::Bishop, pos.piece_type_bb(PieceType::Bishop));
    chase
}

/// px0 `ChessBoard::ThemChased`（board.cc:944-948）：`Mirror` 后 `UsChased`。
pub fn them_chased(pos: &Position) -> u16 {
    us_chased(&pos.mirrored())
}

/// px0 `ChessBoard::HasMatingMaterial`（board.cc:1072-1141）。
pub fn has_mating_material(pos: &Position) -> bool {
    let pawns = pos.piece_type_bb(PieceType::Pawn);
    let rooks = pos.piece_type_bb(PieceType::Rook);
    let knights = pos.piece_type_bb(PieceType::Knight);
    if popcount(pawns) == 0 && popcount(rooks) == 0 && popcount(knights) == 0 {
        #[derive(PartialEq)]
        enum DrawLevel {
            NoDraw,
            DirectDraw,
            MateDraw,
        }

        let cannons = pos.piece_type_bb(PieceType::Cannon);
        let advisors = pos.piece_type_bb(PieceType::Advisor);
        let bishops = pos.piece_type_bb(PieceType::Bishop);

        let level = {
            if popcount(cannons) == 0 {
                DrawLevel::DirectDraw
            } else if popcount(cannons) == 1 {
                let mut cannon_side_occ = pos.color_bb(pos.side_to_move);
                let mut non_cannon_side_occ = pos.color_bb(!pos.side_to_move);
                if popcount(cannon_side_occ & cannons) == 0 {
                    std::mem::swap(&mut cannon_side_occ, &mut non_cannon_side_occ);
                }
                if popcount(advisors & cannon_side_occ) == 0 {
                    if popcount(advisors & non_cannon_side_occ) == 0 {
                        DrawLevel::DirectDraw
                    } else if popcount(advisors & non_cannon_side_occ) == 1 {
                        if popcount(bishops & cannon_side_occ) == 0 {
                            DrawLevel::DirectDraw
                        } else {
                            DrawLevel::MateDraw
                        }
                    } else if popcount(bishops & cannon_side_occ) == 0 {
                        DrawLevel::MateDraw
                    } else {
                        DrawLevel::NoDraw
                    }
                } else {
                    DrawLevel::NoDraw
                }
            } else if popcount(cannons & pos.color_bb(pos.side_to_move)) == 1
                && popcount(cannons & pos.color_bb(!pos.side_to_move)) == 1
                && popcount(advisors) == 0
            {
                if popcount(bishops) == 0 {
                    DrawLevel::DirectDraw
                } else {
                    DrawLevel::MateDraw
                }
            } else {
                DrawLevel::NoDraw
            }
        };

        if level != DrawLevel::NoDraw {
            if level == DrawLevel::MateDraw {
                let mut list = [ExtMove {
                    mv: Move::none(),
                    value: 0,
                }; MAX_MOVES];
                let n = generate(pos, GenType::Legal, &mut list);
                for em in &list[..n] {
                    let mut after = pos.clone_for_search();
                    after.do_move(em.mv);
                    let mirrored = after.mirrored();
                    let mut opp = [ExtMove {
                        mv: Move::none(),
                        value: 0,
                    }; MAX_MOVES];
                    if generate(&mirrored, GenType::Legal, &mut opp) == 0 {
                        return true;
                    }
                }
            }
            return false;
        }
    }
    true
}
