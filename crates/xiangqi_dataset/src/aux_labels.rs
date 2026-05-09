//! 与 `nn/src/nn/aux_pseudo_labels.py` 对齐的伪标签，仅用 `xiangqi_core`（无 pyffish）。

use xiangqi_core::{
    generate, Color, ExtMove, GenType, Move, Piece, Position,
};
use xiangqi_core::types::MAX_MOVES;

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

/// 与 Python `pseudo_aux_labels_from_sample`（基于 Rust 合法表 + 局面吃子判定）一致。
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

    let mut caps = 0_usize;
    for i in 0..n {
        let mv = list[i].mv;
        let pc = pos.piece_on(mv.to_sq());
        if pc != Piece::NO_PIECE {
            caps += 1;
        }
    }
    let tactical = caps as f32 / n as f32;

    let stm_w = pos.side_to_move == Color::White;
    let (red, black) = material_red_black(board_field);
    let adv = if stm_w {
        red - black
    } else {
        black - red
    };

    let attack = 0.5 * (1.0 + (adv / 12.0).tanh());

    let mob_norm = (n as f32 / 48.0).min(1.0);
    let danger_from_moves = 1.0 - mob_norm;
    let mat_stress = 0.5 * (1.0 + (-adv / 12.0).tanh());
    let danger_raw = 0.55 * danger_from_moves + 0.45 * mat_stress;
    let danger = danger_raw.clamp(0.0, 1.0);

    (attack, danger, tactical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::START_FEN;
    use xiangqi_core::Position;

    #[test]
    fn start_position_attack_near_half() {
        let pos = Position::from_fen(START_FEN).unwrap();
        let (a, _d, _t) = pseudo_aux_labels(&pos);
        assert!((a - 0.5).abs() < 0.02, "a={a}");
    }
}
