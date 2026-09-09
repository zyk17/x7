//! 叶子终局分类（不是 `publish_edges`，也不是 `ExpansionState::Expanded`）。
//!
//! 只回答「要不要 NN」：死/子力不足/重复/rule60 → Terminal，否则 Evaluate。
//! Expand task 调用；root 启动门禁复用 `game_terminal_value`。
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

/// 为 stream Expand 分类 `history` 的叶子。
pub(crate) fn classify_expand(history: &PositionHistory) -> ExpandKind {
    let board = history.last().board();
    let legal_moves = board.generate_legal_moves();
    // `wl` 按 incoming edge / 上一走子方视角保存。
    if legal_moves.is_empty() {
        return ExpandKind::Terminal { wl: 1.0, draw: 0.0, plies_left: 0.0 };
    }
    if !board.has_mating_material() {
        return ExpandKind::Terminal { wl: 0.0, draw: 1.0, plies_left: 0.0 };
    }
    if let Some((wl, draw, plies_left)) = path_terminal_value(history) {
        return ExpandKind::Terminal { wl, draw, plies_left };
    }
    ExpandKind::Evaluate { legal_moves }
}

/// 依赖完整 variation history 的终局：重复裁决与 rule60。
///
/// 当前路径树的 node key 已包含完整路径，故同一 node 的规则 history 不会变化；
/// 重复与 rule60 只在首次 Eval 分类，root 则在启动门禁分类。
pub(crate) fn path_terminal_value(history: &PositionHistory) -> Option<(f32, f32, f32)> {
    let position = history.last();
    if position.repetitions() < 2 && position.rule60_ply() < 120 {
        return None;
    }
    game_terminal_value(history)
}

/// 将完整规则裁决转换为 incoming-edge value，供 root 启动门禁使用。
pub(crate) fn game_terminal_value(history: &PositionHistory) -> Option<(f32, f32, f32)> {
    let position = history.last();
    match history.compute_game_result() {
        GameResult::Undecided => None,
        result => {
            let (wl, draw) = terminal_wl_for_node(result, position.is_black_to_move());
            Some((wl, draw, 0.0))
        }
    }
}

/// 将**绝对** game result（`compute_game_result`）转为终局叶子 incoming-edge `(wl, d)`。
///
/// 先换成 STM 视角，再取反，对齐 NN fetch 与将死快径 `WHITE_WON`→`+1`
///
fn terminal_wl_for_node(result: GameResult, black_to_move: bool) -> (f32, f32) {
    let (stm_wl, draw) = match result {
        GameResult::WhiteWon => (if black_to_move { -1.0 } else { 1.0 }, 0.0),
        GameResult::BlackWon => (if black_to_move { 1.0 } else { -1.0 }, 0.0),
        GameResult::Draw => (0.0, 1.0),
        GameResult::Undecided => unreachable!("terminal search evaluation requires a result"),
    };
    (-stm_wl, draw)
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

        assert_eq!(classify_expand(&history), ExpandKind::Terminal { wl: 1.0, draw: 0.0, plies_left: 0.0 });
    }

    #[test]
    fn rule60_terminal_at_root() {
        let state =
            GameState::from_fen_moves("4k4/9/9/9/9/9/9/9/R8/4K4 w - - 120 1", &[] as &[&str]).expect("rule60 fen");
        let history = PositionHistory::from_positions(state.positions());

        assert!(matches!(classify_expand(&history), ExpandKind::Terminal { wl: 0.0, draw: 1.0, plies_left: 0.0 }));
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

        assert_eq!(history.compute_game_result(), GameResult::Undecided);
        assert!(matches!(classify_expand(&history), ExpandKind::Evaluate { .. }));
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
        let result = history.compute_game_result();
        let (wl, draw) = super::terminal_wl_for_node(result, history.last().is_black_to_move());
        assert_eq!(classify_expand(&history), ExpandKind::Terminal { wl, draw, plies_left: 0.0 });
    }
}
