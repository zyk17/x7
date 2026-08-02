//! NN 前的局面分类与终局标记。
//!
//! 以 px0 `evaluate_extension`（`search.cc:1913-1959`）为参考，实现 X7 stream 所需的
//! 将死、重复、早期 two-fold、材料和 rule60 和棋。终局保存 plies-left `m` 用于排序；
//! MultiPV/TB 不在范围内。

use xiangqi_core::{GameResult, PositionHistory};

use super::{rule_judge_wl_for_node, terminal_wl_for_node};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExtensionKind {
    /// NN 评估并发布 edge。
    Evaluate,
    /// 终局叶子：incoming-edge 视角 `wl` / `draw`≡`d` / `plies_left`≡`m`。
    Terminal { wl: f32, draw: f32, plies_left: f32 },
}

/// 为 stream Gather/Eval 分类 `history` 的叶子。
///
/// `depth` 是自搜索 root 起的 variation 长度（0 即 root）。
pub(crate) fn classify_extension(history: &PositionHistory, depth: usize) -> ExtensionKind {
    let is_root = depth == 0;
    let board = history.last().board();
    let legal_moves = board.generate_legal_moves();
    if legal_moves.is_empty() {
        // `wl` 按 incoming edge / 上一走子方视角保存。因此无合法着的中国象棋局面对它
        // 总是胜利。对齐 px0 的 `WHITE_WON` canonical-board 快径
        // （`search.cc:1913-1919`、`node.cc:300-317`）。
        return ExtensionKind::Terminal {
            wl: 1.0,
            draw: 0.0,
            plies_left: 0.0,
        };
    }
    if !is_root {
        if history.last().repetitions() >= 2 {
            // 对齐 px0 `MakeTerminal(history->RuleJudge())`，勿经绝对颜色转换。
            let (wl, draw) = rule_judge_wl_for_node(history.rule_judge());
            return ExtensionKind::Terminal {
                wl,
                draw,
                plies_left: 0.0,
            };
        }
        // px0 `search.cc:1930-1959`：初始重复局面可能成为 TwoFold。
        if history.last().repetitions() == 1
            && depth.saturating_sub(1) >= 4
            && depth.saturating_sub(1) as u32 >= history.last().cycle_length()
        {
            let cycle_length = history.last().cycle_length() as f32;
            let result = history.rule_judge();
            if result == GameResult::Draw {
                let (wl, draw) = rule_judge_wl_for_node(result);
                return ExtensionKind::Terminal {
                    wl,
                    draw,
                    plies_left: cycle_length,
                };
            }
            if two_fold_chase_or_check_cycle(history) && history.last().rule60_ply() < 120 {
                let (wl, draw) = rule_judge_wl_for_node(result);
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
        // root 仍通过 compute_game_result 判断将死和已确定的和棋。
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

/// px0 的 two-fold chase/check cycle 探测（`search.cc:1940-1958`）。
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

#[cfg(test)]
mod tests {
    use xiangqi_core::{ChessBoard, GameResult, GameState, PositionHistory};

    use super::{ExtensionKind, classify_extension};
    use crate::search::{rule_judge_wl_for_node, terminal_wl_for_node};

    #[test]
    fn checkmated_side_to_move_is_a_terminal_win_for_the_incoming_edge() {
        // 黑方被将死。stream root terminal 测试也覆盖此局面；这里保护非 root 的
        // incoming-edge 价值契约。
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmate fen");
        let history = PositionHistory::from_positions(state.positions());

        assert_eq!(
            classify_extension(&history, 1),
            ExtensionKind::Terminal {
                wl: 1.0,
                draw: 0.0,
                plies_left: 0.0,
            }
        );
    }

    /// 白方长将、终局落在**白方行棋**时：`rule_judge` 不能再经绝对颜色转换，
    /// 否则长将方会被标成胜利（对方先变招）。
    #[test]
    fn white_perpetual_check_at_white_to_move_is_loss_for_checker() {
        let (board, _) = ChessBoard::from_fen("3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = PositionHistory::default();
        history.reset(board, 2, 30);
        // 两轮半循环，停在与首个白方行棋局面相同的位置（rep >= 1，白走）。
        for mv in ["d9e9", "d2e2", "e9d9", "e2d2", "d9e9"] {
            let parsed = history.last().board().parse_move(mv).expect(mv);
            history.append(parsed);
        }
        assert!(!history.last().is_black_to_move());
        assert!(history.last().repetitions() >= 1);

        let judged = history.rule_judge();
        // 仅对方被将 → RuleJudge::WHITE_WON；px0 MakeTerminal → node wl = +1
        //（incoming 是黑方逃将，长将方白应负）。
        assert_eq!(judged, GameResult::WhiteWon);
        assert_eq!(rule_judge_wl_for_node(judged), (1.0, 0.0));

        // 错误路径（把 rule_judge 当绝对胜负）会得到 -1，回传后长将方变“赢”。
        let wrong = terminal_wl_for_node(judged, history.last().is_black_to_move());
        assert_eq!(wrong, (-1.0, 0.0));

        let depth = history.len().saturating_sub(1).max(1);
        match classify_extension(&history, depth) {
            ExtensionKind::Terminal { wl, draw, .. } => {
                assert_eq!(draw, 0.0);
                assert_eq!(wl, 1.0, "escape from perpetual check must be winning for escaper");
            }
            other => panic!("expected terminal, got {other:?}"),
        }
    }
}
