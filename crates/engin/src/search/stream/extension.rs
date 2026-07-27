//! Position classification before NN / terminal marking.
//!
//! Mirrors classic `evaluate_extension` / px0 `search.cc:1913-1959` enough for
//! X7 stream: checkmate, repetitions, early two-fold, and draw-by-material/rule60.
//! Plies-left `m` is recorded on terminals for ranking; MultiPV/TB stay out of scope.

use xiangqi_core::{GameResult, PositionHistory};

use super::terminal_wl_for_node;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExtensionKind {
    /// NN evaluation + edge publish.
    Evaluate,
    /// Terminal leaf: mover-perspective `wl` / `draw`≡`d` / `plies_left`≡`m`.
    Terminal { wl: f32, draw: f32, plies_left: f32 },
}

/// Classifies the leaf at `history` for stream Gather/Eval.
///
/// `depth` is variation length from the search root (0 = root).
pub(crate) fn classify_extension(
    history: &PositionHistory,
    depth: usize,
) -> ExtensionKind {
    let is_root = depth == 0;
    let board = history.last().board();
    let legal_moves = board.generate_legal_moves();
    if legal_moves.is_empty() {
        // px0 always writes WHITE_WON (+1 mover view) for no-legal-move leaves.
        return ExtensionKind::Terminal {
            wl: 1.0,
            draw: 0.0,
            plies_left: 0.0,
        };
    }
    if !is_root {
        if history.last().repetitions() >= 2 {
            let result = history.rule_judge();
            let (wl, draw) = terminal_wl_for_node(result, history.last().is_black_to_move());
            return ExtensionKind::Terminal {
                wl,
                draw,
                plies_left: 0.0,
            };
        }
        // px0 `search.cc:1930-1959`: initial repetition may become TwoFold.
        if history.last().repetitions() == 1
            && depth.saturating_sub(1) >= 4
            && depth.saturating_sub(1) as u32 >= history.last().cycle_length()
        {
            let cycle_length = history.last().cycle_length() as f32;
            let result = history.rule_judge();
            if result == GameResult::Draw {
                let (wl, draw) = terminal_wl_for_node(result, history.last().is_black_to_move());
                return ExtensionKind::Terminal {
                    wl,
                    draw,
                    plies_left: cycle_length,
                };
            }
            if two_fold_chase_or_check_cycle(history) && history.last().rule60_ply() < 120 {
                let (wl, draw) = terminal_wl_for_node(result, history.last().is_black_to_move());
                return ExtensionKind::Terminal {
                    wl,
                    draw,
                    plies_left: cycle_length,
                };
            }
        }
        if !board.has_mating_material() || history.last().rule60_ply() >= 120 {
            return ExtensionKind::Terminal {
                wl: 0.0,
                draw: 1.0,
                plies_left: 0.0,
            };
        }
    } else {
        // Root still uses compute_game_result for checkmate / settled draws.
        match history.compute_game_result() {
            GameResult::Undecided => {}
            result => {
                let (wl, draw) = terminal_wl_for_node(result, history.last().is_black_to_move());
                return ExtensionKind::Terminal {
                    wl,
                    draw,
                    plies_left: 0.0,
                };
            }
        }
    }
    ExtensionKind::Evaluate
}

/// px0 two-fold chase/check cycle probe (`search.cc:1940-1958`).
fn two_fold_chase_or_check_cycle(history: &PositionHistory) -> bool {
    let mut idx = history.len() - 1;
    let mut idx2 = idx;
    while idx2 > 0 {
        idx2 -= 1;
        if history.get(idx2).board() == history.last().board() {
            break;
        }
    }
    if idx2 == 0 {
        return false;
    }
    if history.get(idx - 1).board() != history.get(idx2 - 1).board() {
        return false;
    }
    idx -= 1;
    while idx2 != idx {
        idx2 += 1;
        if history.get(idx2).repetitions() > 0 {
            break;
        }
    }
    idx2 == idx
}
