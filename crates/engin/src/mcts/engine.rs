use std::collections::HashMap;
use std::time::{Duration, Instant};

use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Move, MAX_MOVES};
use xiangqi_core::{generate, Position};

use crate::history::PositionHistory;

use super::{
    MctsBudget, MctsConfig, MctsNode, MctsNodeId, MctsTree, PolicyValueEval, PolicyValueInput, PolicyValueTask,
};

const SEARCH_BATCH_SIZE_CAP: usize = 64;
const SEARCH_COLLISION_PLAYOUT_FACTOR: usize = 4;

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
    pub depth: u32,
    pub seldepth: u32,
    pub root_value: f32,
    pub moves: Vec<MctsMoveStat>,
}

#[derive(Clone, Debug, Default)]
pub struct MctsSearchProgress {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    pub playouts: u32,
    pub root_visits: u32,
    pub nodes: usize,
    pub depth: u32,
    pub seldepth: u32,
    pub root_value: f32,
}

/// MCTS 引擎最小实现。
///
/// 当前版本提供：
/// - PUCT 选边
/// - 单线程 iteration + minibatch 搜索流水线
/// - in-flight / collision / multivisit
/// - value 回传
/// - 以 visit 为主的根着法选择
pub struct MctsEngine<E> {
    pub config: MctsConfig,
    pub evaluator: E,
    pub tree: MctsTree,
    root_id: Option<MctsNodeId>,
    root_history: Option<PositionHistory>,
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
        mut on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let Some(root_id) = self.prepare_root(history)? else {
            return Ok(MctsSearchResult::default());
        };

        let root_history = history.clone_for_search();
        let mut playouts = 0u32;
        let mut seldepth = 0u32;
        let mut next_report_at = if info_interval.is_zero() {
            None
        } else {
            Some(Instant::now() + info_interval)
        };

        while !budget_exhausted(&budget, playouts, self.tree.len()) {
            let batch_limit = remaining_batch_capacity(&budget, playouts, self.tree.len(), self.config.search_batch_size)
                .unwrap_or(self.config.search_batch_size)
                .max(1);
            let iteration = self.gather_iteration(root_id, &root_history, &budget, playouts, batch_limit);
            if iteration.playouts == 0 {
                break;
            }

            let iter_playouts = iteration.playouts;
            seldepth = seldepth.max(iteration.seldepth);
            self.apply_iteration(iteration)?;
            debug_assert_eq!(self.total_in_flight(), 0);

            playouts = playouts.saturating_add(iter_playouts);

            if let Some(deadline) = next_report_at {
                let now = Instant::now();
                if now >= deadline {
                    on_progress(&self.progress_from_root(root_id, playouts, seldepth));
                    next_report_at = Some(now + info_interval);
                }
            }
        }

        Ok(self.result_from_root(root_id, playouts, seldepth))
    }

    fn prepare_root(&mut self, history: &PositionHistory) -> Result<Option<MctsNodeId>, E::Error> {
        if self.try_reuse_exact_root(history) {
            return Ok(self.root_id);
        }
        if self.try_reuse_single_move_child(history) {
            return Ok(self.root_id);
        }
        self.tree.clear();
        self.root_id = None;
        self.root_history = None;
        let root_id = self.initialize_root(history)?;
        if root_id.is_some() {
            self.root_id = root_id;
            self.root_history = Some(history.clone_for_search());
        }
        Ok(root_id)
    }

    fn try_reuse_exact_root(&mut self, history: &PositionHistory) -> bool {
        matches!(
            (self.root_id, self.root_history.as_ref()),
            (Some(_), Some(old_history))
                if old_history.same_input_window(history)
        )
    }

    fn try_reuse_single_move_child(&mut self, history: &PositionHistory) -> bool {
        let (Some(root_id), Some(old_history)) = (self.root_id, self.root_history.as_ref()) else {
            return false;
        };
        let Some(mv) = appended_history_move(old_history, history) else {
            return false;
        };
        let Some(root) = self.tree.get(root_id) else {
            return false;
        };
        let Some(child_id) = root
            .children
            .iter()
            .find(|edge| edge.mv == mv)
            .and_then(|edge| edge.child)
        else {
            return false;
        };
        let (new_tree, new_root_id) = self.tree.copy_subtree(child_id);
        self.tree = new_tree;
        self.root_id = Some(new_root_id);
        self.root_history = Some(history.clone_for_search());
        true
    }

    fn initialize_root(&mut self, history: &PositionHistory) -> Result<Option<MctsNodeId>, E::Error> {
        let pos = history.current();
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
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: Vec::with_capacity(legal.len()),
        };
        for (i, mv) in legal.iter().copied().enumerate() {
            root.children.push(super::EdgeStats {
                mv,
                prior: out.priors.get(i).copied().unwrap_or(0.0),
                visits: 0,
                in_flight: 0,
                value_sum: 0.0,
                child: None,
            });
        }
        Ok(Some(self.tree.add_node(root)))
    }

    fn gather_iteration(
        &mut self,
        root_id: MctsNodeId,
        root_history: &PositionHistory,
        budget: &MctsBudget,
        base_playouts: u32,
        batch_limit: usize,
    ) -> SearchIteration {
        let mut iteration = SearchIteration::default();
        let mut slots = HashMap::<PendingKey, usize>::new();
        let playout_limit = batch_limit
            .saturating_mul(SEARCH_COLLISION_PLAYOUT_FACTOR)
            .max(batch_limit);
        while iteration.pending.len() < batch_limit && iteration.playouts < playout_limit as u32 {
            if budget_exhausted(
                budget,
                base_playouts.saturating_add(iteration.playouts),
                self.tree.len().saturating_add(iteration.pending_new_nodes()),
            ) {
                break;
            }
            let pending = self.select_pending(root_id, root_history);
            iteration.playouts = iteration.playouts.saturating_add(1);

            if let Some(&slot) = slots.get(&pending.key) {
                iteration.pending[slot].multivisit = iteration.pending[slot].multivisit.saturating_add(1);
                continue;
            }

            let slot = iteration.pending.len();
            slots.insert(pending.key.clone(), slot);
            iteration.seldepth = iteration.seldepth.max(pending.path.len() as u32);
            iteration.pending.push(pending);
        }

        iteration
    }

    fn select_pending(&mut self, root_id: MctsNodeId, root_history: &PositionHistory) -> PendingNode {
        let mut path = Vec::<PathStep>::new();
        let mut node_id = root_id;
        let mut pos = root_history.current().clone_for_search();
        let mut history = root_history.clone_for_search();

        loop {
            let node = self.tree.get(node_id).expect("selected node must exist");
            if let Some(value) = node.terminal_value {
                return PendingNode {
                    key: PendingKey::ExistingLeaf(node_id),
                    path,
                    kind: PendingKind::ExistingTerminal {
                        leaf_id: node_id,
                        value,
                    },
                    multivisit: 1,
                };
            }

            let edge_idx = select_edge(node, self.config, path.is_empty());
            let mv = node.children[edge_idx].mv;
            let child_id = node.children[edge_idx].child;

            let parent_id = node_id;
            let edge = &mut self.tree.get_mut(parent_id).expect("selected node must exist").children[edge_idx];
            edge.in_flight = edge.in_flight.saturating_add(1);

            pos.do_move(mv);
            history.push_search_position(pos.clone_for_search());
            path.push(PathStep {
                node_id: parent_id,
                edge_idx,
            });

            if history.current_is_repeated() {
                return PendingNode {
                    key: PendingKey::NewEdge(parent_id, edge_idx),
                    path,
                    kind: PendingKind::NewTerminal {
                        state_key: pos.key(),
                        value: 0.0,
                    },
                    multivisit: 1,
                };
            }

            if let Some(child_id) = child_id {
                node_id = child_id;
                continue;
            }

            let mut buf = [ExtMove {
                mv: Move::none(),
                value: 0,
            }; MAX_MOVES];
            let n = generate(&pos, GenType::Legal, &mut buf);
            if n == 0 {
                let value = if pos.checkers() != 0 { -1.0 } else { 0.0 };
                return PendingNode {
                    key: PendingKey::NewEdge(parent_id, edge_idx),
                    path,
                    kind: PendingKind::NewTerminal {
                        state_key: pos.key(),
                        value,
                    },
                    multivisit: 1,
                };
            }

            let legal_moves = buf[..n].iter().map(|e| e.mv).collect::<Vec<_>>();
            return PendingNode {
                key: PendingKey::NewEdge(parent_id, edge_idx),
                path,
                kind: PendingKind::Expand {
                    task: Box::new(PolicyValueTask {
                        position: pos,
                        history,
                        legal_moves,
                    }),
                },
                multivisit: 1,
            };
        }
    }

    fn apply_iteration(&mut self, iteration: SearchIteration) -> Result<(), E::Error> {
        let mut tasks = Vec::new();
        for pending in &iteration.pending {
            if let PendingKind::Expand { task } = &pending.kind {
                tasks.push((**task).clone());
            }
        }
        let outputs = self.evaluator.evaluate_many(&tasks)?;
        let mut eval_cursor = 0usize;

        for pending in iteration.pending {
            match pending.kind {
                PendingKind::ExistingTerminal { leaf_id, value } => {
                    self.add_leaf_visit(leaf_id, value, pending.multivisit);
                    self.backup_path(&pending.path, value, pending.multivisit);
                }
                PendingKind::NewTerminal { state_key, value } => {
                    let parent = pending.path.last().expect("new terminal must have parent");
                    self.add_terminal_child(parent.node_id, parent.edge_idx, state_key, value, pending.multivisit);
                    self.backup_path(&pending.path, value, pending.multivisit);
                }
                PendingKind::Expand { task } => {
                    let out = outputs.get(eval_cursor).expect("batched eval must match task count");
                    eval_cursor += 1;
                    let parent = pending.path.last().expect("expanded leaf must have parent");
                    self.add_expanded_child(parent.node_id, parent.edge_idx, &task, out, pending.multivisit);
                    self.backup_path(&pending.path, out.value, pending.multivisit);
                }
            }
        }

        Ok(())
    }

    fn add_expanded_child(
        &mut self,
        parent_id: MctsNodeId,
        edge_idx: usize,
        task: &PolicyValueTask,
        out: &super::PolicyValueOutput,
        multivisit: u32,
    ) -> MctsNodeId {
        let mut node = MctsNode {
            state_key: task.position.key(),
            visits: multivisit,
            value_sum: out.value * multivisit as f32,
            expanded: true,
            terminal_value: None,
            children: Vec::with_capacity(task.legal_moves.len()),
        };
        for (i, mv) in task.legal_moves.iter().copied().enumerate() {
            node.children.push(super::EdgeStats {
                mv,
                prior: out.priors.get(i).copied().unwrap_or(0.0),
                visits: 0,
                in_flight: 0,
                value_sum: 0.0,
                child: None,
            });
        }
        let child_id = self.tree.add_node(node);
        self.tree.get_mut(parent_id).expect("parent node must exist").children[edge_idx].child = Some(child_id);
        child_id
    }

    fn add_terminal_child(
        &mut self,
        parent_id: MctsNodeId,
        edge_idx: usize,
        state_key: u64,
        value: f32,
        multivisit: u32,
    ) -> MctsNodeId {
        let child_id = self.tree.add_node(MctsNode {
            state_key,
            visits: multivisit,
            value_sum: value * multivisit as f32,
            expanded: true,
            terminal_value: Some(value),
            children: Vec::new(),
        });
        self.tree.get_mut(parent_id).expect("parent node must exist").children[edge_idx].child = Some(child_id);
        child_id
    }

    fn add_leaf_visit(&mut self, leaf_id: MctsNodeId, value: f32, multivisit: u32) {
        if multivisit == 0 {
            return;
        }
        let leaf = self.tree.get_mut(leaf_id).expect("leaf node must exist");
        leaf.visits = leaf.visits.saturating_add(multivisit);
        leaf.value_sum += value * multivisit as f32;
    }

    fn backup_path(&mut self, path: &[PathStep], mut value: f32, multivisit: u32) {
        for step in path.iter().rev() {
            value = -value;
            let signed_delta = value * multivisit as f32;
            let node = self.tree.get_mut(step.node_id).expect("path node must exist");
            node.visits = node.visits.saturating_add(multivisit);
            node.value_sum += signed_delta;
            let edge = &mut node.children[step.edge_idx];
            edge.visits = edge.visits.saturating_add(multivisit);
            edge.value_sum += signed_delta;
            edge.in_flight = edge.in_flight.saturating_sub(multivisit);
        }
    }

    fn progress_from_root(&self, root_id: MctsNodeId, playouts: u32, seldepth: u32) -> MctsSearchProgress {
        let root = self.tree.get(root_id).expect("root must exist");
        let pv = self.extract_pv(root_id);
        let best_move = pv.first().copied();
        MctsSearchProgress {
            best_move,
            pv,
            playouts,
            root_visits: root.visits,
            nodes: self.tree.len(),
            depth: self.depth_from_root(root_id),
            seldepth,
            root_value: root.mean_value(),
        }
    }

    fn result_from_root(&self, root_id: MctsNodeId, playouts: u32, seldepth: u32) -> MctsSearchResult {
        let root = self.tree.get(root_id).expect("root must exist");
        let progress = self.progress_from_root(root_id, playouts, seldepth);
        MctsSearchResult {
            best_move: progress.best_move,
            pv: progress.pv,
            playouts: progress.playouts,
            root_visits: progress.root_visits,
            nodes: progress.nodes,
            depth: progress.depth,
            seldepth: progress.seldepth,
            root_value: progress.root_value,
            moves: root
                .children
                .iter()
                .map(|edge| MctsMoveStat {
                    mv: edge.mv,
                    prior: edge.prior,
                    visits: edge.visits,
                    q: edge.mean_q(),
                })
                .collect(),
        }
    }

    fn extract_pv(&self, root_id: MctsNodeId) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut node_id = root_id;
        while let Some(node) = self.tree.get(node_id) {
            let Some(edge) = node.children.iter().max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| a.mean_q().partial_cmp(&b.mean_q()).unwrap_or(std::cmp::Ordering::Equal))
            }) else {
                break;
            };
            if edge.visits == 0 {
                break;
            }
            pv.push(edge.mv);
            let Some(child_id) = edge.child else {
                break;
            };
            node_id = child_id;
        }
        pv
    }

    fn total_in_flight(&self) -> u32 {
        let mut total = 0u32;
        for idx in 0..self.tree.len() {
            let Some(node) = self.tree.get(MctsNodeId(idx)) else {
                continue;
            };
            total = total.saturating_add(node.children.iter().map(|edge| edge.in_flight).sum::<u32>());
        }
        total
    }

    fn depth_from_root(&self, root_id: MctsNodeId) -> u32 {
        self.extract_pv(root_id).len() as u32
    }

}

fn budget_exhausted(budget: &MctsBudget, playouts: u32, nodes: usize) -> bool {
    if let Some(target_playouts) = budget.max_playouts {
        if playouts >= target_playouts.max(1) {
            return true;
        }
    }
    if let Some(target_nodes) = budget.max_nodes {
        if nodes >= target_nodes as usize {
            return true;
        }
    }
    if let Some(deadline) = budget.deadline {
        if Instant::now() >= deadline {
            return true;
        }
    }
    if let Some(stop) = budget.stop.as_ref() {
        if stop.load(std::sync::atomic::Ordering::SeqCst) {
            return true;
        }
    }
    false
}

fn remaining_batch_capacity(budget: &MctsBudget, playouts: u32, nodes: usize, batch_size: usize) -> Option<usize> {
    let remaining_playouts = budget
        .max_playouts
        .map(|limit| limit.saturating_sub(playouts) as usize)
        .unwrap_or(batch_size);
    let remaining_nodes = budget
        .max_nodes
        .map(|limit| limit.saturating_sub(nodes as u32) as usize)
        .unwrap_or(batch_size);
    let cap = remaining_playouts
        .min(remaining_nodes)
        .min(batch_size.min(SEARCH_BATCH_SIZE_CAP));
    if cap == 0 {
        None
    } else {
        Some(cap)
    }
}

fn appended_history_move(old_history: &PositionHistory, new_history: &PositionHistory) -> Option<Move> {
    let old_positions = old_history.positions().map(Position::fen).collect::<Vec<_>>();
    let new_positions = new_history.positions().map(Position::fen).collect::<Vec<_>>();
    if new_positions.len() < 2 {
        return None;
    }
    let suffix_matches = if new_positions.len() == old_positions.len() + 1 {
        old_positions.iter().zip(new_positions.iter()).all(|(a, b)| a == b)
    } else if new_positions.len() == old_positions.len() && old_positions.len() >= 2 {
        old_positions[1..]
            .iter()
            .zip(new_positions[..new_positions.len() - 1].iter())
            .all(|(a, b)| a == b)
    } else {
        false
    };
    if !suffix_matches {
        return None;
    }

    let old_pos = old_history.current();
    let target_fen = new_history.current().fen();
    let mut buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(old_pos, GenType::Legal, &mut buf);
    for edge in &buf[..n] {
        let mv = edge.mv;
        let mut next = old_pos.clone_for_search();
        next.do_move(mv);
        if next.fen() == target_fen {
            return Some(mv);
        }
    }
    None
}

fn select_edge(node: &MctsNode, config: MctsConfig, is_root: bool) -> usize {
    let parent_effective_visits = node
        .visits
        .saturating_add(node.children.iter().map(|edge| edge.in_flight).sum::<u32>())
        .max(1) as f32;
    let sqrt_parent = parent_effective_visits.sqrt();
    let parent_q = node.mean_value();
    let cpuct = config.cpuct_for(is_root, node.visits);
    let fpu_reduction = config.fpu_for(is_root);
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for (idx, edge) in node.children.iter().enumerate() {
        let q = if edge.visits == 0 {
            (parent_q - fpu_reduction).clamp(-1.0, 1.0)
        } else {
            edge.mean_q()
        };
        let effective_visits = edge.visits.saturating_add(edge.in_flight) as f32;
        let u = cpuct * edge.prior * sqrt_parent / (1.0 + effective_visits);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    best_idx
}

#[derive(Clone, Copy, Debug)]
struct PathStep {
    node_id: MctsNodeId,
    edge_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PendingKey {
    ExistingLeaf(MctsNodeId),
    NewEdge(MctsNodeId, usize),
}

#[derive(Clone)]
enum PendingKind {
    ExistingTerminal { leaf_id: MctsNodeId, value: f32 },
    NewTerminal { state_key: u64, value: f32 },
    Expand { task: Box<PolicyValueTask> },
}

#[derive(Clone)]
struct PendingNode {
    key: PendingKey,
    path: Vec<PathStep>,
    kind: PendingKind,
    multivisit: u32,
}

#[derive(Default)]
struct SearchIteration {
    pending: Vec<PendingNode>,
    playouts: u32,
    seldepth: u32,
}

impl SearchIteration {
    fn pending_new_nodes(&self) -> usize {
        self.pending
            .iter()
            .filter(|pending| {
                matches!(
                    pending.kind,
                    PendingKind::NewTerminal { .. } | PendingKind::Expand { .. }
                )
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::uci_to_move;
    use xiangqi_core::Position;

    #[derive(Default)]
    struct StubEval;

    impl PolicyValueEval for StubEval {
        type Error = String;

        fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<super::super::PolicyValueOutput, Self::Error> {
            Ok(super::super::PolicyValueOutput {
                priors: vec![1.0 / input.legal_moves.len() as f32; input.legal_moves.len()],
                value: 0.0,
            })
        }
    }

    #[test]
    fn gather_iteration_spreads_root_edges() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let batch_size = engine.config.search_batch_size;
        let iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, batch_size);
        assert_eq!(iteration.playouts, batch_size as u32);

        let mut unique = std::collections::HashSet::new();
        for pending in &iteration.pending {
            unique.insert(pending.path.first().expect("root edge").edge_idx);
        }
        assert!(unique.len() > 1, "in-flight should spread root selections");
    }

    #[test]
    fn apply_iteration_clears_inflight_and_updates_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        engine.apply_iteration(iteration).expect("apply ok");

        let root = engine.tree.get(root_id).expect("root");
        assert_eq!(root.visits, 4);
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
            value_sum: 0.0,
            expanded: true,
            terminal_value: Some(1.0),
            children: Vec::new(),
        });
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 0,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv,
                prior: 1.0,
                visits: 0,
                in_flight: 0,
                value_sum: 0.0,
                child: Some(leaf_id),
            }],
        });

        let iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert_eq!(iteration.pending.len(), 1);
        assert_eq!(
            iteration.pending[0].multivisit,
            (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32
        );

        engine.apply_iteration(iteration).expect("apply ok");
        let root = engine.tree.get(root_id).expect("root");
        let leaf = engine.tree.get(leaf_id).expect("leaf");
        assert_eq!(root.visits, (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32);
        assert_eq!(root.children[0].visits, (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32);
        assert_eq!(root.children[0].in_flight, 0);
        assert_eq!(leaf.visits, (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32);
    }

    #[test]
    fn pv_ignores_inflight_only_edges() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let _iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert!(engine.extract_pv(root_id).is_empty());
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
                    deadline: None,
                    stop: None,
                },
            )
            .expect("search ok");
        assert!(result.best_move.is_some());
    }

    #[test]
    fn gather_iteration_respects_tight_playout_budget() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let budget = MctsBudget {
            max_playouts: Some(3),
            max_nodes: None,
            deadline: None,
            stop: None,
        };
        let iteration = engine.gather_iteration(root_id, &history, &budget, 0, engine.config.search_batch_size);
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
        assert!(history.current_is_repeated(), "history should already contain repeated root");

        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let result = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
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
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv: Move::make(xiangqi_core::types::Square::SQ_A0, xiangqi_core::types::Square::SQ_A1),
                prior: 1.0,
                visits: 32,
                in_flight: 0,
                value_sum: 0.0,
                child: None,
            }],
        });
        let reused_child = engine.tree.add_node(MctsNode {
            state_key: 2,
            visits: 32,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv: Move::make(xiangqi_core::types::Square::SQ_A1, xiangqi_core::types::Square::SQ_A2),
                prior: 1.0,
                visits: 32,
                in_flight: 0,
                value_sum: 0.0,
                child: None,
            }],
        });
        let reused_leaf = engine.tree.add_node(MctsNode {
            state_key: 3,
            visits: 32,
            value_sum: 0.0,
            expanded: true,
            terminal_value: Some(0.0),
            children: Vec::new(),
        });
        engine.tree.get_mut(root_id).expect("root").children[0].child = Some(reused_child);
        engine.tree.get_mut(reused_child).expect("child").children[0].child = Some(reused_leaf);

        let progress = engine.progress_from_root(root_id, 1, 1);
        let result = engine.result_from_root(root_id, 1, 1);
        assert_eq!(progress.seldepth, 1);
        assert_eq!(result.seldepth, 1);
        assert_eq!(progress.depth, 2, "pv depth may still reflect reused subtree");
    }
}
