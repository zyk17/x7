//! NN 前的局面分类与终局标记。
//!
//! 以 px0 `evaluate_extension`（`search.cc:1913-1959`）为参考，实现 X7 stream 所需的
//! 将死、重复、two-fold 闭环裁决、子力和 rule60 和棋。终局保存 plies-left `m` 用于排序；
//! MultiPV/TB 不在范围内。

use xiangqi_core::{GameResult, PositionHistory};

use super::{rule_judge_wl_for_node, terminal_wl_for_node};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExtensionKind {
    /// NN 评估并发布 edge。
    Evaluate,
    /// 只由棋盘决定的终局，可安全发布到共享 node。
    SharedTerminal { wl: f32, draw: f32, plies_left: f32 },
    /// 依赖当前 Variation 的终局。不得改变 board-key shared node；由实际入边保存这次
    /// 路径的裁决。参见 `MCGS.md` “环与重复”。
    PathTerminal { wl: f32, draw: f32, plies_left: f32 },
}

/// 为 stream Gather/Eval 分类 `history` 的叶子。
///
/// `depth` 是自搜索 root 起的 variation 长度（0 即 root）。
pub(crate) fn classify_extension(history: &PositionHistory, depth: usize) -> ExtensionKind {
    let board = history.last().board();
    let legal_moves = board.generate_legal_moves();
    if legal_moves.is_empty() {
        // `wl` 按 incoming edge / 上一走子方视角保存。因此无合法着的中国象棋局面对它
        // 总是胜利。对齐 px0 的 `WHITE_WON` canonical-board 快径
        // （`search.cc:1913-1919`、`node.cc:300-317`）。
        return ExtensionKind::SharedTerminal {
            wl: 1.0,
            draw: 0.0,
            plies_left: 0.0,
        };
    }
    if !board.has_mating_material() {
        return ExtensionKind::SharedTerminal {
            wl: 0.0,
            draw: 1.0,
            plies_left: 0.0,
        };
    }
    if let Some((wl, draw, plies_left)) = path_terminal_value(history, depth) {
        return ExtensionKind::PathTerminal { wl, draw, plies_left };
    }
    ExtensionKind::Evaluate
}

/// 只检查依赖 variation history 的终局。共享 node 已展开后再次被另一条 variation
/// 命中时，Gather 必须仍调用它；不能因 board node 已存在而跳过重复、长将/长捉或 rule60。
pub(crate) fn path_terminal_value(history: &PositionHistory, depth: usize) -> Option<(f32, f32, f32)> {
    let is_root = depth == 0;
    if !is_root {
        if history.last().repetitions() >= 2 {
            // 对齐 px0 `MakeTerminal(history->RuleJudge())`，勿经绝对颜色转换。
            let (wl, draw) = rule_judge_wl_for_node(history.rule_judge());
            return Some((wl, draw, 0.0));
        }
        // board-key 图不能在第一次闭环后继续扩展为带环图。这里是搜索截断，
        // 不是正式实战的重复门槛：普通重复按和棋，长将/长捉按 RuleJudge 裁决。
        if history.last().repetitions() == 1 {
            let plies_left = history.last().cycle_length() as f32;
            let result = history.rule_judge();
            if result == GameResult::Draw
                || matches!(result, GameResult::WhiteWon | GameResult::BlackWon)
                    && two_fold_chase_or_check_cycle(history)
            {
                let (wl, draw) = rule_judge_wl_for_node(result);
                return Some((wl, draw, plies_left));
            }
        }
        if history.last().rule60_ply() >= 120 {
            return Some((0.0, 1.0, 0.0));
        }
    } else {
        // root 仍通过 compute_game_result 判断将死和已确定的和棋。
        match history.compute_game_result() {
            GameResult::Undecided => {}
            result => {
                let (wl, draw) = terminal_wl_for_node(result, history.last().is_black_to_move());
                return Some((wl, draw, 0.0));
            }
        }
    }
    None
}

/// px0 `search.cc:1940-1958` 的 two-fold chase/check cycle 探测。它只用于严格
/// board-key DAG 的首次闭环搜索截断，不改变正式 `compute_game_result` 的重复规则。
fn two_fold_chase_or_check_cycle(history: &PositionHistory) -> bool {
    let mut idx = history.len() - 1;
    let mut idx2 = idx;
    while idx2 > 0 {
        idx2 -= 1;
        if history.get(idx2).board() == history.last().board() {
            break;
        }
    }
    if idx2 == 0 || history.get(idx - 1).board() != history.get(idx2 - 1).board() {
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

    #[test]
    fn checkmated_side_to_move_is_a_terminal_win_for_the_incoming_edge() {
        // 黑方被将死。stream root terminal 测试也覆盖此局面；这里保护非 root 的
        // incoming-edge 价值契约。
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmate fen");
        let history = PositionHistory::from_positions(state.positions());

        assert_eq!(
            classify_extension(&history, 1),
            ExtensionKind::SharedTerminal {
                wl: 1.0,
                draw: 0.0,
                plies_left: 0.0,
            }
        );
    }

    #[test]
    fn rule60_terminal_is_path_local() {
        let state =
            GameState::from_fen_moves("4k4/9/9/9/9/9/9/9/R8/4K4 w - - 120 1", &[] as &[&str]).expect("rule60 fen");
        let history = PositionHistory::from_positions(state.positions());

        assert!(matches!(
            classify_extension(&history, 0),
            ExtensionKind::PathTerminal {
                wl: 0.0,
                draw: 1.0,
                plies_left: 0.0,
            }
        ));
    }

    /// 严格 board-key DAG 在首次闭环时必须截断；长将方不能被当作和棋。
    #[test]
    fn first_perpetual_check_cycle_is_a_path_terminal() {
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

        let depth = history.len().saturating_sub(1).max(1);
        assert_eq!(history.rule_judge(), GameResult::WhiteWon);
        assert_eq!(
            classify_extension(&history, depth),
            ExtensionKind::PathTerminal {
                wl: 1.0,
                draw: 0.0,
                plies_left: 4.0,
            }
        );
    }

    #[test]
    fn repeated_check_cycle_from_game_history_is_terminal_at_root() {
        // 用户实战 PGN（2026-08-10）第 32 回合后：红马 h8-f7-f7-h8、黑将
        // e9-f9-f9-e9 构成四 ply 循环。px0 `PositionHistory::ComputeGameResult`
        // （position.cc:88-103）：第二次重复（repetitions >= 2）在 root 也须裁决。
        let moves = [
            "b2e2", "b9c7", "b0c2", "a9b9", "a0b0", "c6c5", "b0b6", "h9g7", "h0g2", "g6g5", "i0i1", "g9e7", "i1f1",
            "f9e8", "e3e4", "h7h3", "e4e5", "b7a7", "b6b9", "c7b9", "g3g4", "g5g4", "g2e3", "i9f9", "f1f9", "e9f9",
            "e3g4", "e6e5", "h2f2", "e5e4", "g4f6", "e4f4", "f6d5", "f4e4", "e2e1", "h3h5", "c0e2", "h5e5", "e1e4",
            "g7e6", "d0e1", "b9c7", "d5e3", "e6g5", "f2f5", "e5e3", "c2e3", "a7a3", "c3c4", "a3i3", "c4c5", "e7c5",
            "e3g4", "i3b3", "g4e5", "b3b5", "e5f7", "b5f5", "f7g5", "i6i5", "g5i6", "c7e6", "i6h8", "f9e9", "h8f7",
            "e9f9", "f7h8", "f9e9", "h8f7", "e9f9", "f7h8", "f9e9", "h8f7", "e9f9", "f7h8", "f9e9",
        ];
        let state = GameState::from_fen_moves(xiangqi_core::STARTPOS_FEN, &moves).expect("game history");
        let history = state.position_history();

        assert!(
            history.last().repetitions() >= 2,
            "second repetition must be visible in full UCI history"
        );
        assert!(matches!(
            classify_extension(&history, 0),
            ExtensionKind::PathTerminal { .. }
        ));
    }
}
