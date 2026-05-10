//! 与 `nn/src/nn/aux_pseudo_labels.py` 语义对齐的伪标签（仅用 `xiangqi_core`）。
//!
//! 设计见仓库根 `temp.md`：**danger / tactical / attack** 为可解释浅规则头，彼此不互为镜像；
//! 与 **value**（结局监督，在 Python 侧 `value_target_side_to_move`）分离。

use xiangqi_core::board::has_crossed_river;
use xiangqi_core::types::MAX_MOVES;
use xiangqi_core::types::{rank_of, type_of, PieceType, Square};
use xiangqi_core::{generate, Color, ExtMove, GenType, Move, Piece, Position};

fn material_red_black(board_field: &str) -> (f32, f32) {
    let mut red = 0.0_f32;
    let mut black = 0.0_f32;
    for ch in board_field.chars() {
        let v = match ch {
            'R' | 'r' => 9.0_f32,
            'N' | 'n' => 2.0,
            'B' | 'b' => 2.0,
            'A' | 'a' => 2.0,
            'C' | 'c' => 4.5,
            'P' | 'p' => 1.0,
            'K' | 'k' => 0.0,
            _ => continue,
        };
        if ch.is_uppercase() {
            red += v;
        } else {
            black += v;
        }
    }
    (red, black)
}

#[inline]
fn on_enemy_half(sq: Square, us: Color) -> bool {
    let r = rank_of(sq) as u8;
    match us {
        Color::White => r >= 5,
        Color::Black => r <= 4,
    }
}

/// **danger**：行棋方难受程度 ∈ [0,1] — 被将军、低机动、物质压力、将帅暴露。
/// **tactical**：强制/战术色彩 — 吃子占比、将军着占比、己方已被将军加成。
/// **attack**：主动施压 — 对敌方老将的威胁、过河兵、子力深入对方半场（弱化物质差）。
pub fn pseudo_aux_labels(pos: &Position) -> (f32, f32, f32) {
    let fen = pos.fen();
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let board_field = parts.first().copied().unwrap_or("");

    let mut list = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut list);
    if n == 0 {
        return (0.5, 1.0, 0.0);
    }

    let us = pos.side_to_move;
    let them = !us;

    // —— tactical ——
    let mut caps = 0_usize;
    let mut checks = 0_usize;
    for em in list.iter().take(n) {
        if pos.piece_on(em.mv.to_sq()) != Piece::NO_PIECE {
            caps += 1;
        }
        if pos.gives_check(em.mv) {
            checks += 1;
        }
    }
    let capture_ratio = caps as f32 / n as f32;
    let check_ratio = checks as f32 / n as f32;
    let in_check_bonus = if pos.checkers() != 0 { 1.0_f32 } else { 0.0_f32 };
    let tactical = (0.5_f32 * capture_ratio + 0.3_f32 * check_ratio + 0.2_f32 * in_check_bonus).clamp(0.0, 1.0);

    // —— danger ——
    let danger_check = if pos.checkers() != 0 { 1.0_f32 } else { 0.0_f32 };
    let mob_norm = (n as f32 / 48.0_f32).min(1.0);
    let low_mobility = 1.0_f32 - mob_norm;

    let stm_w = us == Color::White;
    let (red, black) = material_red_black(board_field);
    let adv = if stm_w { red - black } else { black - red };
    let material_stress = (0.5_f32 * (1.0_f32 + (-adv / 12.0_f32).tanh())).clamp(0.0, 1.0);

    let our_king = pos.king_square(us);
    let att_k = pos.attackers_to(our_king) & pos.color_bb(them);
    let nk = xiangqi_core::board::popcount(att_k) as f32;
    let king_exposure = (nk / 6.0_f32).min(1.0);

    let danger =
        (0.35_f32 * danger_check + 0.30_f32 * low_mobility + 0.20_f32 * material_stress + 0.15_f32 * king_exposure)
            .clamp(0.0, 1.0);

    // —— attack：对敌方老将威胁 + 过河兵比例 + 深入对方半场子力 ——
    let enemy_king = pos.king_square(them);
    let att_e = pos.attackers_to(enemy_king) & pos.color_bb(us);
    let ne = xiangqi_core::board::popcount(att_e) as f32;
    let threat_k = (ne / 8.0_f32).min(1.0);

    let mut pawn_total = 0_u32;
    let mut pawn_crossed = 0_u32;
    let mut bb = pos.pieces_c_pt(us, PieceType::Pawn);
    while bb != 0 {
        let psq = unsafe { std::mem::transmute::<u8, Square>(bb.trailing_zeros() as u8) };
        pawn_total += 1;
        if has_crossed_river(psq, us) {
            pawn_crossed += 1;
        }
        bb &= bb - 1;
    }
    let crossed_ratio = if pawn_total == 0 {
        0.0_f32
    } else {
        pawn_crossed as f32 / pawn_total as f32
    };

    let mut deep = 0_u32;
    bb = pos.color_bb(us);
    while bb != 0 {
        let psq = unsafe { std::mem::transmute::<u8, Square>(bb.trailing_zeros() as u8) };
        let pc = pos.piece_on(psq);
        if type_of(pc) == PieceType::King {
            bb &= bb - 1;
            continue;
        }
        if on_enemy_half(psq, us) {
            deep += 1;
        }
        bb &= bb - 1;
    }
    let half_norm = (deep as f32 / 12.0_f32).min(1.0);

    let attack = (0.45_f32 * threat_k + 0.35_f32 * crossed_ratio + 0.20_f32 * half_norm).clamp(0.0, 1.0);

    (attack, danger, tactical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::Color;
    use xiangqi_core::Position;
    use xiangqi_core::START_FEN;

    #[test]
    fn start_position_symmetric_attack_tactical() {
        let pos = Position::from_fen(START_FEN).unwrap();
        let (a, d, t) = pseudo_aux_labels(&pos);
        assert!(d < 0.35, "开局合法着多，danger 以低机动/物质项为主且应偏低 d={d}");
        assert!(t > 0.01 && t < 0.6, "开局吃子着占比通常较低但非零 t={t}");
        assert!(a < 0.45, "开局通常难以直接威胁对方老将 a={a}");
    }

    #[test]
    fn user_fen_black_down_high_danger() {
        let fen = "3Rka3/9/1cR6/p7p/1r7/6P2/4P3P/2N1C1N2/4A4/4KAB2 b - - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.side_to_move, Color::Black);
        let (a, d, t) = pseudo_aux_labels(&pos);
        assert!(d > 0.75, "黑方崩溃局面 danger 应高 d={d} a={a} t={t}");
        assert!(t > 0.35, "常有吃子/战术 t={t}");
        assert!(a < 0.55, "被动方 attack 不应虚高（非纯物质） a={a}");
    }
}
