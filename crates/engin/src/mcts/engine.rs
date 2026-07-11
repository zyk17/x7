use std::sync::Arc;
use std::time::Duration;

use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Move, MAX_MOVES};
use xiangqi_core::{generate, Position};

use crate::history::PositionHistory;

use super::search::{run_parallel_with_progress, SearchSession};
use super::node::TerminalKind;
use super::worker::SelectionScratch;
use super::{
    EdgeStats, MctsBudget, MctsConfig, MctsNode, MctsNodeId, MctsTree, OnnxPolicyValueEval, PolicyValueEval,
    PolicyValueInput, SearchStats,
};

/// MCTS 单次搜索结果。
#[derive(Clone, Debug)]
pub struct MctsMoveStat {
    pub mv: Move,
    pub prior: f32,
    pub visits: u32,
    pub q: f32,
}

/// MCTS 单次搜索结果。
#[derive(Clone, Debug, Default)]
pub struct MctsSearchResult {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    pub pv_lines: Vec<PvLineInfo>,
    pub multi_pv: u32,
    pub playouts: u32,
    pub root_visits: u32,
    pub nodes: usize,
    pub tree_nodes: usize,
    pub depth: u32,
    pub seldepth: u32,
    pub root_value: f32,
    pub best_value: f32,
    pub best_mate: Option<i32>,
    pub nps_elapsed_ms: u64,
    /// 预算未耗尽时 gather 返回 0 playout 的重试次数。
    pub retry_without_playout: u64,
    pub moves: Vec<MctsMoveStat>,
}

/// 单条 UCI 主变线（lc0 `SendUciInfo` 每条 `ThinkingInfo`）。
#[derive(Clone, Debug, Default)]
pub struct PvLineInfo {
    pub multipv: u32,
    pub best_value: f32,
    pub best_mate: Option<i32>,
    pub pv: Vec<Move>,
}

#[derive(Clone, Debug, Default)]
pub struct MctsSearchProgress {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    /// `multi_pv > 1` 时填充；`multi_pv == 1` 时为空，沿用上方单线字段。
    pub pv_lines: Vec<PvLineInfo>,
    pub multi_pv: u32,
    pub playouts: u32,
    pub root_visits: u32,
    pub nodes: usize,
    pub tree_nodes: usize,
    pub depth: u32,
    pub seldepth: u32,
    pub root_value: f32,
    pub best_value: f32,
    pub best_mate: Option<i32>,
    pub nps_elapsed_ms: u64,
    pub retry_without_playout: u64,
    pub moves: Vec<MctsMoveStat>,
}

/// MCTS 引擎最小实现。
pub struct MctsEngine<E> {
    pub config: MctsConfig,
    pub evaluator: E,
    pub tree: MctsTree,
    pub(crate) root_id: Option<MctsNodeId>,
    pub(crate) root_history: Option<PositionHistory>,
}

impl<E> MctsEngine<E> {
    pub fn new(config: MctsConfig, evaluator: E) -> Self {
        Self {
            config,
            evaluator,
            tree: MctsTree::new(),
            root_id: None,
            root_history: None,
        }
    }
}

impl<E> MctsEngine<E>
where
    E: PolicyValueEval<Error = String>,
{
    pub fn search_root(&mut self, pos: &Position, budget: MctsBudget) -> Result<MctsSearchResult, E::Error> {
        let history = PositionHistory::from_position(pos.clone_for_search());
        self.search_root_history(&history, budget)
    }

    pub fn search_root_with_progress<F>(
        &mut self,
        pos: &Position,
        budget: MctsBudget,
        info_interval: Duration,
        on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let history = PositionHistory::from_position(pos.clone_for_search());
        self.search_root_history_with_progress(&history, budget, info_interval, None, on_progress)
    }

    pub fn search_root_history(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
    ) -> Result<MctsSearchResult, E::Error> {
        self.search_root_history_with_progress(history, budget, Duration::ZERO, None, |_| {})
    }

    pub fn search_root_history_with_progress<F>(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
        info_interval: Duration,
        root_moves: Option<&[Move]>,
        on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let Some(root_id) = self.prepare_root(history)? else {
            return Ok(MctsSearchResult::default());
        };
        if let Some(moves) = root_moves {
            if !moves.is_empty() {
                self.filter_root_children(root_id, moves).map_err(|e| e.to_string())?;
            }
        }

        let initial_visits = self.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
        let batch_limit = self.batch_limit();
        let mut session = SearchSession {
            tree: &mut self.tree,
            config: self.config,
            batch_limit,
            root_id,
            root_history: history.clone_for_search(),
            budget,
            stats: Arc::new(SearchStats::new(initial_visits)),
            eval: &mut self.evaluator,
            selection_scratch: SelectionScratch::default(),
        };
        session.run_with_progress(info_interval, on_progress)
    }

    fn batch_limit(&self) -> usize {
        let attrs = self.evaluator.backend_attributes();
        self.config.effective_minibatch_size(attrs.as_ref())
    }

    fn prepare_root(&mut self, history: &PositionHistory) -> Result<Option<MctsNodeId>, E::Error> {
        let start_key = history.game_start_key();
        if self.tree.gamebegin_id().is_some() && start_key != self.tree.gamebegin_start_key() {
            self.tree.clear();
            self.root_id = None;
            self.root_history = None;
        }

        if self.tree.gamebegin_id().is_none() {
            let start_history =
                PositionHistory::from_position(history.game_start().clone_for_search());
            let Some(gamebegin_id) =
                self.initialize_root_at(history.game_start(), &start_history)?
            else {
                return Ok(None);
            };
            self.tree.set_gamebegin(gamebegin_id, start_key);
        }

        let old_head = self.root_id;
        let position_keys = position_keys_for_history(history);
        let (new_root, _) = self.tree.reset_to_position(
            start_key,
            history.game_moves(),
            &position_keys,
            old_head,
        );
        let expected_key = history.current().key();
        if self.tree.get(new_root).is_none_or(|node| {
            node.state_key != expected_key
                || (!node.is_terminal() && node.children.is_empty())
        }) {
            let Some(fresh_root) = self.initialize_root_at(history.current(), history)? else {
                return Ok(None);
            };
            self.root_id = Some(fresh_root);
        } else {
            self.root_id = Some(new_root);
        }
        self.root_history = Some(history.clone_for_search());
        if let Some(root_id) = self.root_id {
            self.root_id = Some(self.tree.compact_if_bloated(root_id));
        }
        Ok(self.root_id)
    }

    fn initialize_root_at(
        &mut self,
        pos: &Position,
        history: &PositionHistory,
    ) -> Result<Option<MctsNodeId>, E::Error> {
        let mut buf = [ExtMove {
            mv: Move::none(),
            value: 0,
        }; MAX_MOVES];
        let n = generate(pos, GenType::Legal, &mut buf);
        if n == 0 {
            return Ok(None);
        }

        let legal: Vec<Move> = buf[..n].iter().map(|e| e.mv).collect();
        let out = self.evaluator.evaluate(PolicyValueInput {
            position: pos,
            history,
            legal_moves: &legal,
        })?;

        let mut root = MctsNode {
            state_key: pos.key(),
            visits: 0,
            in_flight: 0,
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::with_capacity(legal.len()),
        };
        for (i, mv) in legal.iter().copied().enumerate() {
            root.children.push(EdgeStats {
                mv,
                prior: out.priors.get(i).copied().unwrap_or(0.0),
                visits: 0,
                in_flight: 0,
                wl: 0.0,
                d: 0.0,
                m: 0.0,
                child: None,
            });
        }
        Ok(Some(self.tree.add_node(root)))
    }

    fn filter_root_children(&mut self, root_id: MctsNodeId, allowed: &[Move]) -> Result<(), String> {
        if let Some(root) = self.tree.get_mut(root_id) {
            root.children.retain(|edge| allowed.contains(&edge.mv));
            if root.children.is_empty() {
                return Err("searchmoves 过滤后无合法根着法".into());
            }
        }
        Ok(())
    }

    pub fn ponder_move_for(&self, best_move: Move) -> Option<Move> {
        let root_id = self.root_id?;
        super::worker::ponder_move_from_tree(&self.tree, root_id, best_move)
    }
}

fn position_keys_for_history(history: &PositionHistory) -> Vec<u64> {
    let mut pos = history.game_start().clone_for_search();
    history
        .game_moves()
        .iter()
        .map(|&mv| {
            pos.do_move(mv);
            pos.key()
        })
        .collect()
}

impl MctsEngine<OnnxPolicyValueEval> {
    pub fn search_root_history_parallel_with_progress<F>(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
        threads: usize,
        info_interval: Duration,
        root_moves: Option<&[Move]>,
        on_progress: F,
    ) -> Result<MctsSearchResult, String>
    where
        F: FnMut(&MctsSearchProgress) + Send,
    {
        if threads <= 1 {
            return self
                .search_root_history_with_progress(history, budget, info_interval, root_moves, on_progress)
                .map_err(|e| e.to_string());
        }

        let Some(root_id) = self.prepare_root(history).map_err(|e| e.to_string())? else {
            return Ok(MctsSearchResult::default());
        };
        if let Some(moves) = root_moves {
            if !moves.is_empty() {
                self.filter_root_children(root_id, moves)
                    .map_err(|e| e.to_string())?;
            }
        }
        self.root_history = Some(history.clone_for_search());

        run_parallel_with_progress(
            &mut self.tree,
            self.config,
            &mut self.evaluator,
            history,
            root_id,
            budget,
            threads,
            info_interval,
            on_progress,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::PositionHistory;
    use crate::mcts::PolicyValueOutput;

    #[derive(Default)]
    struct StubEval;

    impl PolicyValueEval for StubEval {
        type Error = String;

        fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<PolicyValueOutput, Self::Error> {
            let p = 1.0 / input.legal_moves.len().max(1) as f32;
            Ok(PolicyValueOutput {
                priors: vec![p; input.legal_moves.len()],
                wl: 0.0,
                d: 0.0,
                m: 0.0,
                value: 0.0,
            })
        }
    }

    #[test]
    fn search_smoke_returns_legal_move() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let pos = Position::from_fen(xiangqi_core::START_FEN).expect("start");
        let result = engine
            .search_root(
                &pos,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    max_depth: None,
                    max_mate: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("search ok");
        assert!(result.best_move.is_some());
    }

    #[test]
    fn repeated_search_reuses_same_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let budget = MctsBudget {
            max_playouts: Some(8),
            max_nodes: None,
            max_depth: None,
            max_mate: None,
            deadline: None,
            stop: None,
        };
        engine.search_root_history(&history, budget.clone()).expect("first");
        let root_after_first = engine.root_id;
        engine.search_root_history(&history, budget).expect("second");
        assert_eq!(engine.root_id, root_after_first);
    }
}
