//! 叶子终局分类（不是 `publish_edges`，也不是 `ExpansionState::Expanded`）。
//!
//! 只回答「要不要 NN」：死/子力不足/重复/rule60 → Terminal，否则 Evaluate。
//! Eval 编码前调用；root 启动门禁复用 `path_terminal_value`。
//!
//! `mcts2`：`rep==1` 继续搜；`rep>=2` 才 RuleJudge。终局 `m` 用于排序。

use xiangqi_core::{GameResult, LegalMoveList, PositionHistory};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExpandKind {
    /// NN 评估并发布 edge；合法着在分类时已生成，避免 Eval 再算一遍。
    Evaluate { legal_moves: LegalMoveList },
    /// 终局叶子：`(wl, draw, plies_left)` 按 incoming edge / 上一走子方视角。
    Terminal { wl: f32, draw: f32, plies_left: f32 },
}

/// 为 stream Gather/Eval 分类 `history` 的叶子。
///
/// `depth` 是自搜索 root 起的 variation 长度（0 即 root）。
pub(crate) fn classify_expand(history: &PositionHistory, depth: usize) -> ExpandKind {
    let board = history.last().board();
    let legal_moves = board.generate_legal_moves();
    // `wl` 按 incoming edge / 上一走子方视角保存。
    if legal_moves.is_empty() {
        return ExpandKind::Terminal {
            wl: 1.0,
            draw: 0.0,
            plies_left: 0.0,
        };
    }
    if !board.has_mating_material() {
        return ExpandKind::Terminal {
            wl: 0.0,
            draw: 1.0,
            plies_left: 0.0,
        };
    }
    if let Some((wl, draw, plies_left)) = path_terminal_value(history, depth) {
        return ExpandKind::Terminal { wl, draw, plies_left };
    }
    ExpandKind::Evaluate { legal_moves }
}

/// 依赖完整 variation history 的终局：重复裁决与 rule60。
///
/// 当前路径树的 node key 已包含完整路径，故同一 node 的规则 history 不会变化；
/// 重复与 rule60 只在首次 Eval 分类，root 则在启动门禁分类。
pub(crate) fn path_terminal_value(history: &PositionHistory, depth: usize) -> Option<(f32, f32, f32)> {
    let is_root = depth == 0;
    if !is_root {
        if history.last().repetitions() >= 2 {
            // 语义参考自 px0 `MakeTerminal(history->RuleJudge())`，勿经绝对颜色转换。
            let (wl, draw) = rule_judge_wl_for_node(history.rule_judge());
            return Some((wl, draw, 0.0));
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

/// 将**绝对** game result（`compute_game_result`）转为终局叶子 incoming-edge `(wl, d)`。
///
/// 先换成 STM 视角，再取反，对齐 NN fetch 与将死快径 `WHITE_WON`→`+1`
///
/// 注意：`rule_judge()` 的返回值**不是**绝对胜负，不能走这里；应使用
/// [`rule_judge_wl_for_node`]。
fn terminal_wl_for_node(result: GameResult, black_to_move: bool) -> (f32, f32) {
    let (stm_wl, draw) = match result {
        GameResult::WhiteWon => (if black_to_move { -1.0 } else { 1.0 }, 0.0),
        GameResult::BlackWon => (if black_to_move { 1.0 } else { -1.0 }, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("terminal search evaluation requires a result"),
    };
    (-stm_wl, draw)
}

/// 将 `rule_judge` 结果转为 node / incoming-edge `(wl, d)`：
/// `WHITE_WON`→`+1`，`BLACK_WON`→`-1`。
///
/// `rule_judge` 的胜负枚举已是 node / incoming-edge 视角（与 `checkThem`/`checkUs`
/// 绑定），**不要**再按 `is_black_to_move` 当绝对颜色转换，否则白方行棋时的长将/长捉
/// 符号会反转。
fn rule_judge_wl_for_node(result: GameResult) -> (f32, f32) {
    match result {
        GameResult::WhiteWon => (1.0, 0.0),
        GameResult::BlackWon => (-1.0, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("rule-judge terminal requires a decided result"),
    }
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{ChessBoard, GameResult, GameState, PositionHistory};

    use super::{ExpandKind, classify_expand};

    #[test]
    fn checkmated_side_to_move_is_a_terminal_win_for_the_incoming_edge() {
        // 黑方被将死。stream root terminal 测试也覆盖此局面；这里保护非 root 的
        // incoming-edge 价值契约。
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmate fen");
        let history = PositionHistory::from_positions(state.positions());

        assert_eq!(
            classify_expand(&history, 1),
            ExpandKind::Terminal {
                wl: 1.0,
                draw: 0.0,
                plies_left: 0.0,
            }
        );
    }

    #[test]
    fn rule60_terminal_at_root() {
        let state =
            GameState::from_fen_moves("4k4/9/9/9/9/9/9/9/R8/4K4 w - - 120 1", &[] as &[&str]).expect("rule60 fen");
        let history = PositionHistory::from_positions(state.positions());

        assert!(matches!(
            classify_expand(&history, 0),
            ExpandKind::Terminal {
                wl: 0.0,
                draw: 1.0,
                plies_left: 0.0,
            }
        ));
    }

    /// 首次重复仍可继续搜索；只有第二次重复才由 RuleJudge 裁决。
    #[test]
    fn first_perpetual_check_cycle_remains_evaluable() {
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
        assert!(matches!(classify_expand(&history, depth), ExpandKind::Evaluate { .. }));
    }

    #[test]
    fn second_perpetual_check_cycle_is_rule_judge_terminal() {
        let (board, _) = ChessBoard::from_fen("3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = PositionHistory::default();
        history.reset(board, 2, 30);
        // 再走一轮半，使当前白走局面 repetitions >= 2。
        for mv in ["d9e9", "d2e2", "e9d9", "e2d2", "d9e9", "d2e2", "e9d9", "e2d2", "d9e9"] {
            let parsed = history.last().board().parse_move(mv).expect(mv);
            history.append(parsed);
        }
        assert!(!history.last().is_black_to_move());
        assert!(history.last().repetitions() >= 2);
        let depth = history.len().saturating_sub(1).max(1);
        let (wl, draw) = super::rule_judge_wl_for_node(history.rule_judge());
        assert_eq!(
            classify_expand(&history, depth),
            ExpandKind::Terminal {
                wl,
                draw,
                plies_left: 0.0,
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
        assert!(matches!(classify_expand(&history, 0), ExpandKind::Terminal { .. }));
    }
}
