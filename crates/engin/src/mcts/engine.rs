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

#[derive(Clone, Debug, Default)]
pub struct MctsSearchProgress {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
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
    E: PolicyValueEval,
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
        self.search_root_history_with_progress(&history, budget, info_interval, on_progress)
    }

    pub fn search_root_history(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
    ) -> Result<MctsSearchResult, E::Error> {
        self.search_root_history_with_progress(history, budget, Duration::ZERO, |_| {})
    }

    pub fn search_root_history_with_progress<F>(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
        info_interval: Duration,
        on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let Some(root_id) = self.prepare_root(history)? else {
            return Ok(MctsSearchResult::default());
        };

        let initial_visits = self.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
        let mut session = SearchSession {
            tree: &mut self.tree,
            config: self.config,
            root_id,
            root_history: history.clone_for_search(),
            budget,
            stats: Arc::new(SearchStats::new(initial_visits)),
            eval: &mut self.evaluator,
            selection_scratch: SelectionScratch::default(),
        };
        session.run_with_progress(info_interval, on_progress)
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
        on_progress: F,
    ) -> Result<MctsSearchResult, String>
    where
        F: FnMut(&MctsSearchProgress) + Send,
    {
        if threads <= 1 {
            return self
                .search_root_history_with_progress(history, budget, info_interval, on_progress)
                .map_err(|e| e.to_string());
        }

        let Some(root_id) = self.prepare_root(history).map_err(|e| e.to_string())? else {
            return Ok(MctsSearchResult::default());
        };
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
    use crate::mcts::backend::BackendComputation;
    use crate::mcts::search::execute_one_iteration;
    use crate::mcts::PolicyValueOutput;
    use crate::mcts::worker::{
        apply_minibatch, cancel_minibatch, gather_minibatch, progress_from_tree, pv_summary_from_tree,
        result_from_tree, select_edge, total_in_flight_in_tree, GatherParams, PathStep, PendingKey, PendingKind,
        PendingNode, SearchIteration, SelectionScratch,
    };
    use super::TerminalKind;
    use super::*;
    use xiangqi_core::uci_to_move;
    use xiangqi_core::Position;

    #[derive(Default)]
    struct StubEval;

    impl PolicyValueEval for StubEval {
        type Error = String;

        fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<PolicyValueOutput, Self::Error> {
            let p = 1.0 / input.legal_moves.len() as f32;
            Ok(PolicyValueOutput {
                priors: vec![p; input.legal_moves.len()],
                wl: 0.0,
                d: 0.0,
                m: 0.0,
                value: 0.0,
            })
        }
    }

    fn apply_minibatch_with_eval(
        engine: &mut MctsEngine<StubEval>,
        root_id: MctsNodeId,
        history: &PositionHistory,
        budget: &MctsBudget,
        batch_limit: usize,
    ) -> Result<(), String> {
        let mut session = SearchSession {
            tree: &mut engine.tree,
            config: engine.config,
            root_id,
            root_history: history.clone_for_search(),
            budget: budget.clone(),
            stats: Arc::new(SearchStats::new(0)),
            eval: &mut engine.evaluator,
            selection_scratch: SelectionScratch::default(),
        };
        execute_one_iteration(&mut session, batch_limit)?;
        Ok(())
    }

    fn make_gather_params<'a>(
        config: MctsConfig,
        budget: &'a MctsBudget,
        root_id: MctsNodeId,
        root_visits: u32,
        batch_limit: usize,
        base_playouts: u32,
        stats: Option<&'a SearchStats>,
    ) -> GatherParams<'a> {
        GatherParams {
            config,
            budget,
            base_playouts,
            in_flight_playouts: 0,
            initial_visits: stats.map(SearchStats::initial_visits).unwrap_or(0),
            batch_limit,
            stats,
            root_id,
            root_visits,
            thread_count: 1,
            backend_waiting: 0,
        }
    }

    #[test]
    fn gather_minibatch_respects_batch_limit() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine
            .initialize_root_at(history.current(), &history)
            .expect("init ok")
            .expect("root");

        let batch_size = engine.config.search_batch_size;
        let budget = MctsBudget::default();
        let root_visits = engine.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
        let params = make_gather_params(
            engine.config,
            &budget,
            root_id,
            root_visits,
            batch_size,
            0,
            None,
        );
        let mut scratch = SelectionScratch::default();
        let iteration = gather_minibatch(&mut engine.tree, &history, &params, &mut scratch);
        assert!(!iteration.pending.is_empty());
        assert!(iteration.pending.len() <= batch_size);
        assert!(iteration.playouts >= iteration.pending.len() as u32);
    }

    #[test]
    fn select_edge_avoids_expanding_new_edge_when_idle_alternative_exists() {
        let mut node = MctsNode {
            state_key: 1,
            visits: 32,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        };
        node.children.push(EdgeStats {
            mv: Move::make(xiangqi_core::types::Square::SQ_A0, xiangqi_core::types::Square::SQ_A1),
            prior: 0.9,
            visits: 0,
            in_flight: 1,
            wl: 0.0, d: 0.0, m: 0.0,
            child: None,
        });
        node.children.push(EdgeStats {
            mv: Move::make(xiangqi_core::types::Square::SQ_B0, xiangqi_core::types::Square::SQ_B1),
            prior: 0.1,
            visits: 0,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            child: None,
        });
        assert_eq!(select_edge(&node, MctsConfig::default(), true, 0, u32::MAX, None), 0);
    }

    #[test]
    fn select_edge_keeps_root_focus_even_when_top_edge_is_inflight() {
        let mut node = MctsNode {
            state_key: 1,
            visits: 8,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        };
        node.children.push(EdgeStats {
            mv: Move::make(xiangqi_core::types::Square::SQ_A0, xiangqi_core::types::Square::SQ_A1),
            prior: 0.9,
            visits: 0,
            in_flight: 1,
            wl: 0.0, d: 0.0, m: 0.0,
            child: Some(MctsNodeId(1)),
        });
        node.children.push(EdgeStats {
            mv: Move::make(xiangqi_core::types::Square::SQ_B0, xiangqi_core::types::Square::SQ_B1),
            prior: 0.35,
            visits: 0,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            child: Some(MctsNodeId(2)),
        });
        assert_eq!(select_edge(&node, MctsConfig::default(), true, 0, u32::MAX, None), 0);
        assert_eq!(select_edge(&node, MctsConfig::default(), false, 0, u32::MAX, None), 0);
    }

    #[test]
    fn apply_minibatch_clears_inflight_and_updates_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine
            .initialize_root_at(history.current(), &history)
            .expect("init ok")
            .expect("root");

        apply_minibatch_with_eval(&mut engine, root_id, &history, &MctsBudget::default(), 4).expect("apply ok");

        let root = engine.tree.get(root_id).expect("root");
        assert_eq!(root.visits, 4);
        assert!(root.children.iter().all(|edge| edge.in_flight == 0));
    }

    #[test]
    fn cancel_minibatch_clears_inflight_flags() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine
            .initialize_root_at(history.current(), &history)
            .expect("init ok")
            .expect("root");


        let budget = MctsBudget::default();
        let root_visits = engine.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
        let params = make_gather_params(engine.config, &budget, root_id, root_visits, 4, 0, None);
        let mut scratch = SelectionScratch::default();
        let iteration = gather_minibatch(&mut engine.tree, &history, &params, &mut scratch);
        assert!(iteration.pending.iter().any(|pending| {
            matches!(pending.kind, PendingKind::NewTerminal { .. } | PendingKind::Expand { .. })
        }));

        cancel_minibatch(&mut engine.tree, iteration);

        let root = engine.tree.get(root_id).expect("root");
        assert!(root.children.iter().all(|edge| edge.in_flight == 0));
    }

    #[test]
    fn collision_merges_existing_terminal_leaf_multivisit() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let mv = uci_to_move(pos, "h2e2").expect("legal move");

        engine.tree.clear();
        let leaf_id = engine.tree.add_node(MctsNode {
            state_key: 42,
            visits: 0,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: Some(1.0),
            children: Vec::new(),
        });
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 0,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: vec![EdgeStats {
                mv,
                prior: 1.0,
                visits: 0,
                in_flight: 0,
                wl: 0.0, d: 0.0, m: 0.0,
                child: Some(leaf_id),
            }],
        });


        let budget = MctsBudget::default();
        let root_visits = engine.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
        let params = make_gather_params(engine.config, &budget, root_id, root_visits, 4, 0, None);
        let mut scratch = SelectionScratch::default();
        let iteration = gather_minibatch(&mut engine.tree, &history, &params, &mut scratch);
        assert_eq!(iteration.pending.len(), 1);
        assert_eq!(iteration.pending[0].multivisit, 4);
        apply_minibatch(&mut engine.tree, iteration, &[], None);
        let root = engine.tree.get(root_id).expect("root");
        let leaf = engine.tree.get(leaf_id).expect("leaf");
        assert_eq!(root.visits, 4);
        assert_eq!(root.children[0].visits, 4);
        assert_eq!(root.children[0].in_flight, 0);
        assert_eq!(leaf.visits, 4);
    }

    #[test]
    fn pv_ignores_inflight_only_edges() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine
            .initialize_root_at(history.current(), &history)
            .expect("init ok")
            .expect("root");

        let _iteration = {
            let budget = MctsBudget::default();
            let root_visits = engine.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
            let params = make_gather_params(engine.config, &budget, root_id, root_visits, 4, 0, None);
            let mut scratch = SelectionScratch::default();
            gather_minibatch(&mut engine.tree, &history, &params, &mut scratch)
        };
        assert!(pv_summary_from_tree(&engine.tree, root_id).pv.is_empty());
    }

    #[test]
    fn pv_can_follow_expanded_child_chain_without_extra_visits() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let root_mv = uci_to_move(pos, "b2e2").expect("legal move");

        let mut next = pos.clone();
        next.do_move(root_mv);
        let reply_mv = uci_to_move(&next, "b9c7").expect("legal move");

        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 8,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        });
        let child_id = engine.tree.add_node(MctsNode {
            state_key: next.key(),
            visits: 1,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        });
        let leaf_id = engine.tree.add_node(MctsNode {
            state_key: 0,
            visits: 1,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: false,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        });

        engine.tree.get_mut(root_id).expect("root").children = vec![EdgeStats {
            mv: root_mv,
            prior: 1.0,
            visits: 8,
            in_flight: 0,
            wl: 1.0, d: 0.0, m: 0.0,
            child: Some(child_id),
        }];
        engine.tree.get_mut(child_id).expect("child").children = vec![EdgeStats {
            mv: reply_mv,
            prior: 1.0,
            visits: 1,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            child: Some(leaf_id),
        }];

        assert_eq!(
            pv_summary_from_tree(&engine.tree, root_id)
                .pv
                .into_iter()
                .map(xiangqi_core::move_to_uci)
                .collect::<Vec<_>>(),
            vec!["b2e2".to_string(), "b9c7".to_string()]
        );
    }

    #[test]
    fn expanding_new_edge_is_not_queued_twice_before_backup() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let mv = uci_to_move(pos, "h2e2").expect("legal move");
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 0,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: vec![EdgeStats {
                mv,
                prior: 1.0,
                visits: 0,
                in_flight: 0,
                wl: 0.0, d: 0.0, m: 0.0,
                child: None,
            }],
        });

        let budget = MctsBudget::default();
        let root_visits = engine.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
        let params = make_gather_params(engine.config, &budget, root_id, root_visits, 4, 0, None);
        let mut scratch = SelectionScratch::default();
        let first = gather_minibatch(&mut engine.tree, &history, &params, &mut scratch);
        assert_eq!(first.pending.len(), 1);
        assert_eq!(first.pending[0].multivisit, 4);
        let second = gather_minibatch(&mut engine.tree, &history, &params, &mut scratch);
        assert_eq!(second.pending.len(), 1);
        assert_eq!(second.playouts, 1);

        let mut backend = BackendComputation::new(&mut engine.evaluator);
        for pending in &first.pending {
            if let PendingKind::Expand { task } = &pending.kind {
                backend.add_input(task);
            }
        }
        let outputs = backend.compute_blocking().expect("eval ok");
        apply_minibatch(&mut engine.tree, first, &outputs, None);

        let third = gather_minibatch(&mut engine.tree, &history, &params, &mut scratch);
        assert!(third.playouts > 0);
    }

    #[test]
    fn pv_summary_uses_best_edge_q_and_detects_terminal_mate() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let best_mv = uci_to_move(pos, "h2e2").expect("legal move");
        let alt_mv = uci_to_move(pos, "b2e2").expect("legal move");

        let mate_leaf = engine.tree.add_node(MctsNode {
            state_key: 1,
            visits: 1,
            in_flight: 0,
            wl: 1.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: Some(1.0),
            children: Vec::new(),
        });
        let alt_leaf = engine.tree.add_node(MctsNode {
            state_key: 2,
            visits: 10,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        });
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 11,
            in_flight: 0,
            wl: -0.1, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: vec![
                EdgeStats {
                    mv: best_mv,
                    prior: 0.5,
                    visits: 1,
                    in_flight: 0,
                    wl: 1.0, d: 0.0, m: 0.0,
                    child: Some(mate_leaf),
                },
                EdgeStats {
                    mv: alt_mv,
                    prior: 0.5,
                    visits: 10,
                    in_flight: 0,
                    wl: 0.0, d: 0.0, m: 0.0,
                    child: Some(alt_leaf),
                },
            ],
        });

        let summary = pv_summary_from_tree(&engine.tree, root_id);
        assert_eq!(summary.best_move, Some(best_mv));
        assert_eq!(summary.best_value, 1.0);

        let root = engine.tree.get_mut(root_id).expect("root");
        root.children[0].visits = 12;
        root.children[0].wl = 1.0;
        let summary = pv_summary_from_tree(&engine.tree, root_id);
        assert_eq!(summary.best_move, Some(best_mv));
        assert_eq!(summary.best_value, 1.0);
    }

    #[test]
    fn pv_summary_does_not_report_mate_for_unsolved_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let best_mv = uci_to_move(pos, "h2e2").expect("legal move");
        let reply_mv = uci_to_move(pos, "b2e2").expect("legal move");

        let terminal_leaf = engine.tree.add_node(MctsNode {
            state_key: 11,
            visits: 1,
            in_flight: 0,
            wl: -1.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: Some(-1.0),
            children: Vec::new(),
        });
        let reply_node = engine.tree.add_node(MctsNode {
            state_key: 12,
            visits: 8,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: vec![EdgeStats {
                mv: reply_mv,
                prior: 1.0,
                visits: 8,
                in_flight: 0,
                wl: 0.0, d: 0.0, m: 0.0,
                child: Some(terminal_leaf),
            }],
        });
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 16,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: vec![EdgeStats {
                mv: best_mv,
                prior: 1.0,
                visits: 16,
                in_flight: 0,
                wl: 0.8, d: 0.0, m: 0.0,
                child: Some(reply_node),
            }],
        });

        let summary = pv_summary_from_tree(&engine.tree, root_id);
        assert_eq!(summary.best_move, Some(best_mv));
        assert!((summary.best_value - 0.8).abs() < 1e-6);
        assert_eq!(summary.best_mate, None);
    }

    #[test]
    fn large_batch_clears_inflight_after_apply() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine
            .initialize_root_at(history.current(), &history)
            .expect("init ok")
            .expect("root");

        let mut session = SearchSession {
            tree: &mut engine.tree,
            config: engine.config,
            root_id,
            root_history: history.clone_for_search(),
            budget: MctsBudget {
                max_playouts: Some(2048),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                stop: None,
            },
            stats: Arc::new(SearchStats::new(0)),
            eval: &mut engine.evaluator,
            selection_scratch: SelectionScratch::default(),
        };
        execute_one_iteration(&mut session, engine.config.search_batch_size).expect("iteration ok");
        assert_eq!(total_in_flight_in_tree(session.tree), 0);
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
            deadline: None,
                    stop: None,
                },
            )
            .expect("search ok");
        assert!(result.best_move.is_some());
    }

    #[test]
    fn gather_minibatch_respects_tight_playout_budget() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine
            .initialize_root_at(history.current(), &history)
            .expect("init ok")
            .expect("root");

        let budget = MctsBudget {
            max_playouts: Some(3),
            max_nodes: None,
            max_depth: None,
            deadline: None,
            stop: None,
        };
        let root_visits = engine.tree.get(root_id).map(|root| root.visits).unwrap_or(0);
        let params = make_gather_params(
            engine.config,
            &budget,
            root_id,
            root_visits,
            engine.config.search_batch_size,
            0,
            None,
        );
        let mut scratch = SelectionScratch::default();
        let iteration = gather_minibatch(&mut engine.tree, &history, &params, &mut scratch);
        assert_eq!(iteration.playouts, 3);
    }

    #[test]
    fn repeated_search_reuses_same_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        assert_eq!(first.root_visits, 16);

        let second = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(8),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        assert_eq!(second.playouts, 8);
        assert!(
            second.root_visits >= 24,
            "root visits should continue from prior search"
        );
    }

    #[test]
    fn child_position_reuses_subtree_as_new_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(32),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        let best = first.best_move.expect("best move");

        let mut next_history = history.clone_for_search();
        next_history.push_move(best);
        let second = engine
            .search_root_history(
                &next_history,
                MctsBudget {
                    max_playouts: Some(4),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        assert!(
            second.root_visits > second.playouts,
            "new root should inherit child visits"
        );
    }

    #[test]
    fn appended_two_move_history_reuses_deeper_subtree_as_new_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        let mv1 = first.best_move.expect("best move");

        let mut history1 = history.clone_for_search();
        history1.push_move(mv1);
        let second = engine
            .search_root_history(
                &history1,
                MctsBudget {
                    max_playouts: Some(16),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        let mv2 = second.best_move.expect("reply");

        let mut history2 = history.clone_for_search();
        history2.push_move(mv1);
        history2.push_move(mv2);
        let third = engine
            .search_root_history(
                &history2,
                MctsBudget {
                    max_playouts: Some(1),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("third");
        assert!(
            third.root_visits > third.playouts,
            "deeper appended path should retain subtree visits"
        );
    }

    #[test]
    fn transposed_history_does_not_reuse_exact_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);

        let mut history_a = PositionHistory::new_startpos();
        for u in ["h0g2", "h9g7", "b0c2", "b9c7"] {
            let mv = uci_to_move(history_a.current(), u).expect("legal move");
            history_a.push_move(mv);
        }
        let first = engine
            .search_root_history(
                &history_a,
                MctsBudget {
                    max_playouts: Some(16),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        assert_eq!(first.root_visits, 16);

        let mut history_b = PositionHistory::new_startpos();
        for u in ["b0c2", "b9c7", "h0g2", "h9g7"] {
            let mv = uci_to_move(history_b.current(), u).expect("legal move");
            history_b.push_move(mv);
        }
        assert_eq!(history_a.current().fen(), history_b.current().fen());
        assert!(!history_a.same_input_window(&history_b));

        let second = engine
            .search_root_history(
                &history_b,
                MctsBudget {
                    max_playouts: Some(1),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        assert_eq!(second.root_visits, 1, "different history window must rebuild root");
    }

    #[test]
    fn repeated_leaf_is_scored_as_draw() {
        let mut history = PositionHistory::new_startpos();
        for u in ["h0g2", "h9g7", "g2h0", "g7h9"] {
            let mv = uci_to_move(history.current(), u).expect("legal move");
            history.push_move(mv);
        }
        assert!(
            history.current_is_repeated(),
            "history should already contain repeated root"
        );

        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let result = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("search");
        assert!(result.best_move.is_some());
        assert!(result.moves.iter().all(|mv| mv.q.abs() <= 1.0));
    }

    #[test]
    fn reused_tree_does_not_inflate_reported_seldepth() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let root_id = engine.tree.add_node(MctsNode {
            state_key: 1,
            visits: 32,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: vec![EdgeStats {
                mv: Move::make(xiangqi_core::types::Square::SQ_A0, xiangqi_core::types::Square::SQ_A1),
                prior: 1.0,
                visits: 32,
                in_flight: 0,
                wl: 0.0, d: 0.0, m: 0.0,
                child: None,
            }],
        });
        let reused_child = engine.tree.add_node(MctsNode {
            state_key: 2,
            visits: 32,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: vec![EdgeStats {
                mv: Move::make(xiangqi_core::types::Square::SQ_A1, xiangqi_core::types::Square::SQ_A2),
                prior: 1.0,
                visits: 32,
                in_flight: 0,
                wl: 0.0, d: 0.0, m: 0.0,
                child: None,
            }],
        });
        let reused_leaf = engine.tree.add_node(MctsNode {
            state_key: 3,
            visits: 32,
            in_flight: 0,
            wl: 0.0, d: 0.0, m: 0.0,
            expanded: true,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: Some(0.0),
            children: Vec::new(),
        });
        engine.tree.get_mut(root_id).expect("root").children[0].child = Some(reused_child);
        engine.tree.get_mut(reused_child).expect("child").children[0].child = Some(reused_leaf);

        let stats = SearchStats::new(32);
        let iteration = SearchIteration {
            pending: vec![PendingNode {
                key: PendingKey::ExistingLeaf(reused_leaf),
                path: vec![PathStep {
                    node_id: root_id,
                    edge_idx: 0,
                }],
                kind: PendingKind::ExistingTerminal {
                    leaf_id: reused_leaf,
                    wl: 0.0,
                    d: 1.0,
                    m: 0.0,
                },
                multivisit: 1,
            }],
            playouts: 1,
            seldepth: 3,
        };
        stats.add_minibatch(&iteration);
        let progress = progress_from_tree(&engine.tree, root_id, &stats);
        let result = result_from_tree(&engine.tree, root_id, &stats);
        assert_eq!(progress.seldepth, 1);
        assert_eq!(result.seldepth, 1);
        assert_eq!(progress.depth, 1);
    }

    #[test]
    fn fresh_search_root_visits_match_playouts() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let result = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(24),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("search");
        assert_eq!(result.playouts, 24);
        assert_eq!(result.root_visits, 24);
        assert!(result.nodes >= 24);
    }

    #[test]
    fn non_appended_history_does_not_reuse_root_subtree() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        let mv1 = first.best_move.expect("best move");

        let mut history_a = history.clone_for_search();
        history_a.push_move(mv1);
        let _ = engine
            .search_root_history(
                &history_a,
                MctsBudget {
                    max_playouts: Some(8),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("second");

        let mut history_b = PositionHistory::new_startpos();
        let alt = uci_to_move(history_b.current(), "c3c4").expect("alt legal");
        history_b.push_move(alt);
        let third = engine
            .search_root_history(
                &history_b,
                MctsBudget {
                    max_playouts: Some(4),
            max_nodes: None,
            max_depth: None,
            deadline: None,
                    stop: None,
                },
            )
            .expect("third");
        assert_eq!(third.root_visits, 4, "non-appended history should rebuild root");
    }
}
