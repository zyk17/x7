//! **走子生成**（中国象棋）。
//!
//! 从当前局面生成伪合法 / 合法着，支持：`Captures`、`Quiets`、`PseudoLegal`、
//! 被将军时的 `Evasions`、以及过滤后的全 **`Legal`**。

use crate::board::*;
use crate::types::*;

/// 要生成的着法类型。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenType {
    Captures,
    Quiets,
    Evasions,
    PseudoLegal,
    Legal,
}

/// 单局面着法数上限（经验上不超过 128）。
pub const MAX_MOVES: usize = 128;

/// 带着法分值的扩展着法（用于排序）。
#[derive(Clone, Copy, Debug)]
pub struct ExtMove {
    pub mv: Move,
    pub value: i32,
}

/// 向 `list` 生成指定类型的着法，返回生成数量。
pub fn generate(pos: &Position, gen_type: GenType, list: &mut [ExtMove]) -> usize {
    match gen_type {
        GenType::Captures => generate_captures(pos, list),
        GenType::Quiets => generate_quiets(pos, list),
        GenType::PseudoLegal => generate_pseudo_legal(pos, list),
        GenType::Evasions => generate_evasions(pos, list),
        GenType::Legal => generate_legal(pos, list),
    }
}

/// 生成全部伪合法着，返回数量。
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

        // 排除己方占据格
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

/// 生成全部伪合法吃子。
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

/// 生成全部伪合法非吃子（安静着）。
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
            // 炮的安静着：车式滑向空格
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

/// 生成应将着：仅包含能解除将军的着法。
fn generate_evasions(pos: &Position, list: &mut [ExtMove]) -> usize {
    let us = pos.side_to_move;
    let _them = !us;
    let ksq = pos.king_square(us);
    let checkers = pos.checkers();

    // 双将时仅将/帅可走
    if more_than_one(checkers) {
        return generate_king_evasions(pos, list, ksq);
    }

    let mut count = 0;
    let checksq: Square = unsafe { std::mem::transmute(checkers.trailing_zeros() as u8) };
    let pt = type_of(pos.piece_on(checksq));

    // 目标格：王与将军子之间（去掉己方占据）——垫将格
    let between = between_bb(ksq, checksq) & !pos.color_bb(us);

    // 非将子可垫将或吃去将军子
    let our_pieces = pos.color_bb(us) & !square_bb(ksq);
    let occupied = pos.occupancy();
    let mut bb = our_pieces;
    while bb != 0 {
        let from: Square = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
        bb &= bb - 1;
        let pc = pos.piece_on(from);
        let piece_pt = type_of(pc);

        // 该子目标：中间格 + 将军子所在格（吃）
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
                    // 兵垫将仅能向前，除非吃子与将军子共线才可横吃
                    !occupied
                } else if piece_pt == PieceType::Bishop || piece_pt == PieceType::Knight {
                    // 马、象可垫到空格
                    !occupied
                } else {
                    !occupied
                };
            // 简化：仅吃子或垫向将军子
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

    // 将/帅应将
    count += generate_king_evasions_with_check(pos, list, ksq, checksq, pt, count);

    // 炮架移动：若将军子为炮，可移动炮架子解将
    if pt == PieceType::Cannon {
        let hurdle_bb = between_bb(ksq, checksq) & pos.color_bb(us);
        if hurdle_bb != 0 {
            let hurdle_sq: Square = unsafe { std::mem::transmute(hurdle_bb.trailing_zeros() as u8) };
            let hurdle_pt = type_of(pos.piece_on(hurdle_sq));
            let hurdle_attacks = piece_attacks(hurdle_pt, hurdle_sq, occupied, us);
            // 炮架可走至任意不受王攻击的格，但不能沿炮线走入「炮盯王」线
            let line_to_checker = between_bb(checksq, hurdle_sq);
            let targets = hurdle_attacks & !line_to_checker & !pos.color_bb(us);
            // 若可吃炮则亦可
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

/// 将/帅简单应将（仅看目标格是否受攻）。
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
        // 目标格须不受对方攻击
        let after_occupied = (occupied ^ square_bb(ksq)) | square_bb(to);
        let captured_square = (pos.piece_on(to) != Piece::NO_PIECE).then_some(to);
        if pos.checkers_to_with_occupied(them, to, after_occupied, captured_square) == 0 {
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

/// 将/帅应将：考虑车/炮等滑子将军线，**不可沿将军线走入仍被将军的格**。
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

    // 滑子将军时去掉王与将军子射线上的格
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
        let captured_square = (pos.piece_on(to) != Piece::NO_PIECE).then_some(to);
        if pos.checkers_to_with_occupied(them, to, after_occupied, captured_square) == 0 {
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

/// 生成全部合法着。
fn generate_legal(pos: &Position, list: &mut [ExtMove]) -> usize {
    let pseudo_count = if pos.checkers() != 0 {
        generate_evasions(pos, list)
    } else {
        generate_pseudo_legal(pos, list)
    };

    // 过滤非法着
    let mut legal_count = 0;
    for i in 0..pseudo_count {
        let mv = list[i].mv;
        if !mv.is_ok() {
            // 跳过无效着（不应由伪合法生成产生）
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
        assert!(count > 0, "起始局面应有合法着");
        assert!(count <= 128, "起始局面着法数不应超过 128");

        // 校验生成的着法全部合法
        for i in 0..count {
            assert!(
                pos.legal(list[i].mv),
                "着法 {:?}->{:?} 应合法",
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

        // 起始局面炮可隔己方兵吃对方兵：B 线、H 线各一吃，共 2 个吃子。
        assert_eq!(cap_count, 2, "起始局面应有 2 个炮吃");
        assert!(quiet_count > 0, "起始局面应有安静着");
    }
}
