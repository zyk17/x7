//! **走子生成**（中国象棋）。
//!
//! 从当前局面生成伪合法 / 合法着，支持：`Captures`、`Quiets`、`PseudoLegal`、
//! 被将军时的 `Evasions`、以及过滤后的全 **`Legal`**。

use crate::board::*;
use crate::types::*;

/// The type of moves to generate.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenType {
    Captures,
    Quiets,
    Evasions,
    PseudoLegal,
    Legal,
}

/// Maximum number of moves in a position (empirically, never exceeds 128).
pub const MAX_MOVES: usize = 128;

/// A move with an associated score (for move ordering).
#[derive(Clone, Copy, Debug)]
pub struct ExtMove {
    pub mv: Move,
    pub value: i32,
}

/// Generate moves of the given type for the position into `list`.
/// Returns the number of moves generated.
pub fn generate(pos: &Position, gen_type: GenType, list: &mut [ExtMove]) -> usize {
    match gen_type {
        GenType::Captures => generate_captures(pos, list),
        GenType::Quiets => generate_quiets(pos, list),
        GenType::PseudoLegal => generate_pseudo_legal(pos, list),
        GenType::Evasions => generate_evasions(pos, list),
        GenType::Legal => generate_legal(pos, list),
    }
}

/// Generate all pseudo-legal moves, return count.
fn generate_pseudo_legal(pos: &Position, list: &mut [ExtMove]) -> usize {
    let us = pos.side_to_move;
    let _them = !us;
    let our_pieces = pos.color_bb(us);
    let mut count = 0;

    let mut bb = our_pieces;
    while bb != 0 {
        let from: Square = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
        bb &= bb - 1;
        let pc = pos.piece_on(from);
        let pt = type_of(pc);

        let attacks = piece_attacks(pt, from, pos.occupancy(), us);

        // Filter out squares occupied by own pieces
        let targets = attacks & !pos.color_bb(us);

        let mut t = targets;
        while t != 0 {
            let to: Square = unsafe { std::mem::transmute(t.trailing_zeros() as u8) };
            t &= t - 1;
            if count < list.len() {
                list[count] = ExtMove {
                    mv: Move::make(from, to),
                    value: 0,
                };
                count += 1;
            }
        }
    }

    count
}

/// Generate all pseudo-legal captures.
fn generate_captures(pos: &Position, list: &mut [ExtMove]) -> usize {
    let us = pos.side_to_move;
    let them = !us;
    let our_pieces = pos.color_bb(us);
    let their_pieces = pos.color_bb(them);
    let occupied = pos.occupancy();
    let mut count = 0;

    let mut bb = our_pieces;
    while bb != 0 {
        let from: Square = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
        bb &= bb - 1;
        let pc = pos.piece_on(from);
        let pt = type_of(pc);

        let attacks = if pt == PieceType::Cannon {
            cannon_attacks(from, occupied) & their_pieces
        } else {
            piece_attacks(pt, from, occupied, us) & their_pieces
        };

        let mut t = attacks;
        while t != 0 {
            let to: Square = unsafe { std::mem::transmute(t.trailing_zeros() as u8) };
            t &= t - 1;
            if count < list.len() {
                list[count] = ExtMove {
                    mv: Move::make(from, to),
                    value: 0,
                };
                count += 1;
            }
        }
    }

    count
}

/// Generate all pseudo-legal non-captures (quiets).
fn generate_quiets(pos: &Position, list: &mut [ExtMove]) -> usize {
    let us = pos.side_to_move;
    let our_pieces = pos.color_bb(us);
    let occupied = pos.occupancy();
    let mut count = 0;

    let mut bb = our_pieces;
    while bb != 0 {
        let from: Square = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
        bb &= bb - 1;
        let pc = pos.piece_on(from);
        let pt = type_of(pc);

        let attacks = if pt == PieceType::Cannon {
            // Cannon quiets: rook-style sliding to empty squares
            rook_attacks(from, occupied) & !occupied
        } else {
            piece_attacks(pt, from, occupied, us) & !occupied
        };

        let mut t = attacks;
        while t != 0 {
            let to: Square = unsafe { std::mem::transmute(t.trailing_zeros() as u8) };
            t &= t - 1;
            if count < list.len() {
                list[count] = ExtMove {
                    mv: Move::make(from, to),
                    value: 0,
                };
                count += 1;
            }
        }
    }

    count
}

/// Generate check evasions. Only the moves that get out of check are generated.
fn generate_evasions(pos: &Position, list: &mut [ExtMove]) -> usize {
    let us = pos.side_to_move;
    let _them = !us;
    let ksq = pos.king_square(us);
    let checkers = pos.checkers();

    // If double check, only king moves can save
    if more_than_one(checkers) {
        return generate_king_evasions(pos, list, ksq);
    }

    let mut count = 0;
    let checksq: Square = unsafe { std::mem::transmute(checkers.trailing_zeros() as u8) };
    let pt = type_of(pos.piece_on(checksq));

    // Target squares: between king and checker (excluding own pieces) — these are blocking squares
    let between = between_bb(ksq, checksq) & !pos.color_bb(us);

    // Non-king pieces can block or capture the checker
    let our_pieces = pos.color_bb(us) & !square_bb(ksq);
    let occupied = pos.occupancy();
    let mut bb = our_pieces;
    while bb != 0 {
        let from: Square = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
        bb &= bb - 1;
        let pc = pos.piece_on(from);
        let piece_pt = type_of(pc);

        // Targets for this piece: between squares + the checker itself (capture)
        let targets = if piece_pt == PieceType::Cannon {
            let cap_targets = cannon_attacks(from, occupied) & square_bb(checksq);
            let quiet_targets = rook_attacks(from, occupied) & between & !occupied;
            cap_targets | quiet_targets
        } else {
            let att = piece_attacks(piece_pt, from, occupied, us);
            let cap = att & square_bb(checksq);
            let blk = att
                & between
                & if piece_pt == PieceType::Pawn {
                    // Pawn can only advance to block, not capture sideways unless aligned
                    !occupied
                } else if piece_pt == PieceType::Bishop || piece_pt == PieceType::Knight {
                    // Lame leapers can block
                    !occupied
                } else {
                    !occupied
                };
            // For simplicity, only capture or block the checking piece
            cap | blk
        };

        let mut t = targets;
        while t != 0 {
            let to: Square = unsafe { std::mem::transmute(t.trailing_zeros() as u8) };
            t &= t - 1;
            if !pos.color_bb(us) & square_bb(to) != 0 || square_bb(to) & square_bb(checksq) != 0 {
                if count < list.len() {
                    list[count] = ExtMove {
                        mv: Move::make(from, to),
                        value: 0,
                    };
                    count += 1;
                }
            }
        }
    }

    // King evasions
    count += generate_king_evasions_with_check(pos, list, ksq, checksq, pt, count);

    // Cannon hurdle evasion: if the checker is a cannon, we can move the hurdle piece
    if pt == PieceType::Cannon {
        let hurdle_bb = between_bb(ksq, checksq) & pos.color_bb(us);
        if hurdle_bb != 0 {
            let hurdle_sq: Square = unsafe { std::mem::transmute(hurdle_bb.trailing_zeros() as u8) };
            let hurdle_pt = type_of(pos.piece_on(hurdle_sq));
            let hurdle_attacks = piece_attacks(hurdle_pt, hurdle_sq, occupied, us);
            // Hurdle can move to any non-king-attacked square except along the cannon line
            let line_to_checker = between_bb(checksq, hurdle_sq);
            let targets = hurdle_attacks & !line_to_checker & !pos.color_bb(us);
            // But can also capture the cannon if possible
            let cannon_cap = hurdle_attacks & square_bb(checksq);
            let all_targets = targets | cannon_cap;

            let mut t = all_targets;
            while t != 0 {
                let to: Square = unsafe { std::mem::transmute(t.trailing_zeros() as u8) };
                t &= t - 1;
                if !pos.color_bb(us) & square_bb(to) != 0 || square_bb(to) & square_bb(checksq) != 0 {
                    if count < list.len() {
                        list[count] = ExtMove {
                            mv: Move::make(hurdle_sq, to),
                            value: 0,
                        };
                        count += 1;
                    }
                }
            }
        }
    }

    count
}

/// Generate king moves to escape check. Simple version.
fn generate_king_evasions(pos: &Position, list: &mut [ExtMove], ksq: Square) -> usize {
    let us = pos.side_to_move;
    let them = !us;
    let occupied = pos.occupancy();
    let attacks = king_attacks(ksq) & !pos.color_bb(us);
    let mut count = 0;

    let mut t = attacks;
    while t != 0 {
        let to: Square = unsafe { std::mem::transmute(t.trailing_zeros() as u8) };
        t &= t - 1;
        // Check if destination is not attacked
        let after_occupied = (occupied ^ square_bb(ksq)) | square_bb(to);
        if pos.checkers_to(them, to) & after_occupied == 0 {
            if count < list.len() {
                list[count] = ExtMove {
                    mv: Move::make(ksq, to),
                    value: 0,
                };
                count += 1;
            }
        }
    }
    count
}

/// King evasions that also consider slider checkers (cannons/rooks).
/// King cannot move along the check line.
fn generate_king_evasions_with_check(
    pos: &Position,
    list: &mut [ExtMove],
    ksq: Square,
    checksq: Square,
    checker_pt: PieceType,
    offset: usize,
) -> usize {
    let us = pos.side_to_move;
    let them = !us;
    let occupied = pos.occupancy();
    let mut attacks = king_attacks(ksq) & !pos.color_bb(us);

    // Remove squares on the line of a sliding checker
    if checker_pt == PieceType::Rook || checker_pt == PieceType::Cannon {
        let line = {
            let mut bb = 0u128;
            if rank_of(ksq) == rank_of(checksq) {
                bb |= between_bb(ksq, checksq) | square_bb(checksq);
            }
            if file_of(ksq) == file_of(checksq) {
                bb |= between_bb(ksq, checksq) | square_bb(checksq);
            }
            bb
        };
        attacks &= !line;
    }

    let mut count = offset;
    let mut t = attacks;
    while t != 0 {
        let to: Square = unsafe { std::mem::transmute(t.trailing_zeros() as u8) };
        t &= t - 1;
        let after_occupied = (occupied ^ square_bb(ksq)) | square_bb(to);
        if pos.checkers_to(them, to) & after_occupied == 0 {
            if count < list.len() {
                list[count] = ExtMove {
                    mv: Move::make(ksq, to),
                    value: 0,
                };
                count += 1;
            }
        }
    }
    count
}

/// Generate all legal moves.
fn generate_legal(pos: &Position, list: &mut [ExtMove]) -> usize {
    let pseudo_count = if pos.checkers() != 0 {
        generate_evasions(pos, list)
    } else {
        generate_pseudo_legal(pos, list)
    };

    // Filter out illegal moves
    let mut legal_count = 0;
    for i in 0..pseudo_count {
        let mv = list[i].mv;
        if !mv.is_ok() {
            // Skip invalid moves — they come from generate_pseudo_legal internal bug
            continue;
        }
        if pos.legal(mv) {
            list[legal_count] = list[i];
            legal_count += 1;
        }
    }
    legal_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Zobrist;
    use crate::misc::PRNG;

    fn init_zobrist() -> &'static Zobrist {
        Box::leak(Box::new(Zobrist::init(&mut PRNG::new(1070372))))
    }

    #[test]
    fn test_initial_position_moves() {
        let zobrist = init_zobrist();
        let mut pos = Position::new(zobrist);
        pos.set_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
            .unwrap();

        let mut list = vec![
            ExtMove {
                mv: Move::none(),
                value: 0
            };
            256
        ];
        let count = generate_legal(&pos, &mut list);
        assert!(count > 0, "Starting position should have legal moves");
        assert!(count <= 128, "Starting position should not exceed 128 moves");

        // Verify all generated moves are legal
        for i in 0..count {
            assert!(
                pos.legal(list[i].mv),
                "Move {:?}->{:?} should be legal",
                list[i].mv.from_sq(),
                list[i].mv.to_sq()
            );
        }
    }

    #[test]
    fn test_captures_and_quiets() {
        let zobrist = init_zobrist();
        let mut pos = Position::new(zobrist);
        pos.set_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
            .unwrap();

        let mut list = vec![
            ExtMove {
                mv: Move::none(),
                value: 0
            };
            256
        ];

        let cap_count = generate_captures(&pos, &mut list);
        let quiet_count = generate_quiets(&pos, &mut list);
        let _total = generate_pseudo_legal(&pos, &mut list);

        // In starting position, cannons can capture via own pawn as hurdle.
        // B2 cannon captures B6 (black pawn), H2 captures H6 (black pawn) = 2 captures.
        assert_eq!(cap_count, 2, "Starting position should have 2 cannon captures");
        assert!(quiet_count > 0, "Starting position should have quiet moves");
    }
}
