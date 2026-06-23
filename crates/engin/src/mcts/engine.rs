use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Move, MAX_MOVES};
use xiangqi_core::{generate, Position};

use super::{MctsBudget, MctsConfig, MctsNode, MctsTree, PolicyValueEval, PolicyValueInput};
use std::time::{Duration, Instant};

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
    pub playouts: u32,
    pub root_visits: u32,
    pub nodes: usize,
    pub root_value: f32,
    pub moves: Vec<MctsMoveStat>,
}

#[derive(Clone, Debug, Default)]
pub struct MctsSearchProgress {
    pub best_move: Option<Move>,
    pub playouts: u32,
    pub root_visits: u32,
    pub nodes: usize,
    pub root_value: f32,
}

/// MCTS 引擎最小实现。
///
/// 当前版本提供：
/// - PUCT 选边
/// - 叶子扩展
/// - value 回传
/// - 以 visit 为主的根着法选择
pub struct MctsEngine<E> {
    pub config: MctsConfig,
    pub evaluator: E,
    pub tree: MctsTree,
}

impl<E> MctsEngine<E> {
    pub fn new(config: MctsConfig, evaluator: E) -> Self {
        Self {
            config,
            evaluator,
            tree: MctsTree::new(),
        }
    }
}

impl<E> MctsEngine<E>
where
    E: PolicyValueEval,
{
    pub fn search_root(&mut self, pos: &Position, budget: MctsBudget) -> Result<MctsSearchResult, E::Error> {
        self.search_root_with_progress(pos, budget, Duration::ZERO, |_| {})
    }

    pub fn search_root_with_progress<F>(
        &mut self,
        pos: &Position,
        budget: MctsBudget,
        info_interval: Duration,
        mut on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let mut buf = [ExtMove {
            mv: Move::none(),
            value: 0,
        }; MAX_MOVES];
        let n = generate(pos, GenType::Legal, &mut buf);
        if n == 0 {
            self.tree.clear();
            return Ok(MctsSearchResult::default());
        }

        self.tree.clear();
        let legal: Vec<Move> = buf[..n].iter().map(|e| e.mv).collect();
        let out = self.evaluator.evaluate(PolicyValueInput {
            position: pos,
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
            let prior = out.priors.get(i).copied().unwrap_or(0.0);
            root.children.push(super::EdgeStats {
                mv,
                prior,
                visits: 0,
                value_sum: 0.0,
                child: None,
            });
        }
        let root_id = self.tree.add_node(root);

        let mut work = pos.clone();
        let mut playouts = 0u32;
        let mut next_report_at = if info_interval.is_zero() {
            None
        } else {
            Some(Instant::now() + info_interval)
        };
        while !budget_exhausted(&budget, playouts, self.tree.len()) {
            self.simulate(root_id, &mut work)?;
            playouts = playouts.saturating_add(1);
            if let Some(deadline) = next_report_at {
                let now = Instant::now();
                if now >= deadline {
                    on_progress(&self.progress_from_root(root_id, playouts));
                    next_report_at = Some(now + info_interval);
                }
            }
        }

        Ok(self.result_from_root(root_id, playouts))
    }

    fn simulate(&mut self, root_id: super::MctsNodeId, pos: &mut Position) -> Result<f32, E::Error> {
        let mut path: Vec<(super::MctsNodeId, usize, Move)> = Vec::new();
        let mut node_id = root_id;

        loop {
            if let Some(v) = terminal_value(pos) {
                return self.backup(path, v, pos);
            }

            let edge_idx = {
                let node = self.tree.get(node_id).expect("selected node must exist");
                if let Some(v) = node.terminal_value {
                    return self.backup(path, v, pos);
                }
                select_edge(node, self.config.cpuct)
            };

            let mv = self.tree.get(node_id).expect("selected node must exist").children[edge_idx].mv;
            pos.do_move(mv);
            path.push((node_id, edge_idx, mv));

            let child_id = self.tree.get(node_id).expect("selected node must exist").children[edge_idx].child;
            if let Some(child_id) = child_id {
                node_id = child_id;
                continue;
            }

            let child_node = self.expand_node(pos)?;
            let child_value = child_node.mean_value();
            let child_id = self.tree.add_node(child_node);
            self.tree.get_mut(node_id).expect("parent node must exist").children[edge_idx].child = Some(child_id);
            return self.backup(path, child_value, pos);
        }
    }

    fn expand_node(&mut self, pos: &Position) -> Result<MctsNode, E::Error> {
        if let Some(v) = terminal_value(pos) {
            return Ok(MctsNode {
                state_key: pos.key(),
                visits: 1,
                value_sum: v,
                expanded: true,
                terminal_value: Some(v),
                children: Vec::new(),
            });
        }

        let mut buf = [ExtMove {
            mv: Move::none(),
            value: 0,
        }; MAX_MOVES];
        let n = generate(pos, GenType::Legal, &mut buf);
        let legal: Vec<Move> = buf[..n].iter().map(|e| e.mv).collect();
        let out = self.evaluator.evaluate(PolicyValueInput {
            position: pos,
            legal_moves: &legal,
        })?;

        let mut node = MctsNode {
            state_key: pos.key(),
            visits: 1,
            value_sum: out.value,
            expanded: true,
            terminal_value: None,
            children: Vec::with_capacity(legal.len()),
        };
        for (i, mv) in legal.iter().copied().enumerate() {
            node.children.push(super::EdgeStats {
                mv,
                prior: out.priors.get(i).copied().unwrap_or(0.0),
                visits: 0,
                value_sum: 0.0,
                child: None,
            });
        }
        Ok(node)
    }

    fn backup(
        &mut self,
        mut path: Vec<(super::MctsNodeId, usize, Move)>,
        mut value: f32,
        pos: &mut Position,
    ) -> Result<f32, E::Error> {
        while let Some((node_id, edge_idx, mv)) = path.pop() {
            pos.undo_move(mv);
            value = -value;
            let node = self.tree.get_mut(node_id).expect("path node must exist");
            node.visits = node.visits.saturating_add(1);
            node.value_sum += value;
            let edge = &mut node.children[edge_idx];
            edge.visits = edge.visits.saturating_add(1);
            edge.value_sum += value;
        }
        Ok(value)
    }

    fn progress_from_root(&self, root_id: super::MctsNodeId, playouts: u32) -> MctsSearchProgress {
        let root = self.tree.get(root_id).expect("root must exist");
        let best_move = root
            .children
            .iter()
            .max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| a.mean_q().partial_cmp(&b.mean_q()).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|edge| edge.mv);
        MctsSearchProgress {
            best_move,
            playouts,
            root_visits: root.visits,
            nodes: self.tree.len(),
            root_value: root.mean_value(),
        }
    }

    fn result_from_root(&self, root_id: super::MctsNodeId, playouts: u32) -> MctsSearchResult {
        let root = self.tree.get(root_id).expect("root must exist");
        let progress = self.progress_from_root(root_id, playouts);
        MctsSearchResult {
            best_move: progress.best_move,
            playouts: progress.playouts,
            root_visits: progress.root_visits,
            nodes: progress.nodes,
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
        if std::time::Instant::now() >= deadline {
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

fn select_edge(node: &MctsNode, cpuct: f32) -> usize {
    let parent_visits = node.visits.max(1) as f32;
    let sqrt_parent = parent_visits.sqrt();
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for (idx, edge) in node.children.iter().enumerate() {
        let q = edge.mean_q();
        let u = cpuct * edge.prior * sqrt_parent / (1.0 + edge.visits as f32);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    best_idx
}

fn terminal_value(pos: &Position) -> Option<f32> {
    let mut buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut buf);
    if n != 0 {
        return None;
    }
    if pos.checkers() != 0 {
        Some(-1.0)
    } else {
        Some(0.0)
    }
}
