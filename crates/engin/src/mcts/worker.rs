use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Move, MAX_MOVES};
use xiangqi_core::generate;
use xiangqi_core::Position;

use crate::history::PositionHistory;

use super::coordinator::{calculate_collisions_left, SharedCollisions};
use super::node::{cancel_score_update, terminal_wdl, MctsNode, TerminalKind};
use super::{
    EdgeStats, MctsBudget, MctsConfig, MctsMoveStat, MctsNodeId, MctsSearchProgress, MctsSearchResult,
    MctsTree, PolicyValueOutput, PolicyValueTask, SearchStats,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PathStep {
    pub node_id: MctsNodeId,
    pub edge_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PendingKey {
    ExistingLeaf(MctsNodeId),
    NewEdge(MctsNodeId, usize),
}

#[derive(Clone)]
pub(crate) enum PendingKind {
    ExistingTerminal {
        leaf_id: MctsNodeId,
        wl: f32,
        d: f32,
        m: f32,
    },
    NewTerminal {
        state_key: u64,
        wl: f32,
        d: f32,
        m: f32,
        terminal_kind: TerminalKind,
    },
    Expand { task: Arc<PolicyValueTask> },
    /// px0 collision：同一边/节点已有 in-flight 扩展。
    Collision,
}

#[derive(Clone)]
pub(crate) struct PendingNode {
    pub key: PendingKey,
    pub path: Vec<PathStep>,
    pub kind: PendingKind,
    pub multivisit: u32,
}

#[derive(Default)]
pub(crate) struct SearchIteration {
    pub pending: Vec<PendingNode>,
    pub playouts: u32,
    pub seldepth: u32,
}

pub(crate) struct GatherParams<'a> {
    pub config: MctsConfig,
    pub budget: &'a MctsBudget,
    pub base_playouts: u32,
    pub in_flight_playouts: u32,
    pub initial_visits: u32,
    pub batch_limit: usize,
    pub stats: Option<&'a SearchStats>,
    pub root_id: MctsNodeId,
    pub root_visits: u32,
    pub thread_count: usize,
    pub backend_waiting: i32,
}

#[derive(Default)]
pub(crate) struct PvSummary {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    pub best_value: f32,
    pub best_mate: Option<i32>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeRank {
    TerminalLoss,
    NonTerminal,
    TerminalWin,
}

pub(crate) fn worker_batch_limit(config: MctsConfig, threads: usize) -> usize {
    (config.search_batch_size / threads.max(1)).max(1)
}

/// px0：PlayoutsStopper 看本次 playouts；VisitsStopper 看 playouts + initial_visits。
pub(crate) fn remaining_playout_budget(
    budget: &MctsBudget,
    session_playouts: u32,
    in_flight_playouts: u32,
    initial_visits: u32,
    want: u32,
) -> u32 {
    let mut cap = want;
    let scheduled = session_playouts.saturating_add(in_flight_playouts);
    if let Some(max_playouts) = budget.max_playouts {
        cap = cap.min(max_playouts.saturating_sub(scheduled));
    }
    if let Some(max_nodes) = budget.max_nodes {
        let total_nodes = scheduled.saturating_add(initial_visits);
        if total_nodes >= max_nodes {
            return 0;
        }
        cap = cap.min(max_nodes.saturating_sub(total_nodes));
    }
    cap
}

/// px0 `GatherMinibatch`：batch 未满则继续 `PickNodesToExtend`。
pub(crate) fn gather_minibatch(
    tree: &mut MctsTree,
    root_history: &PositionHistory,
    params: &GatherParams<'_>,
    scratch: &mut SelectionScratch,
) -> SearchIteration {
    let mut iteration = SearchIteration::default();
    let mut slots = HashMap::<PendingKey, usize>::new();
    let remaining_n = remaining_playout_budget(
        params.budget,
        params.base_playouts,
        params.in_flight_playouts,
        params.initial_visits,
        u32::MAX,
    ) as i64;
    let mut collisions_left = calculate_collisions_left(
        i64::from(params.root_visits).min(remaining_n),
        params.config,
    );
    let mut minibatch_non_collision = 0usize;
    let mut needs_nn = false;

    while iteration.pending.len() < params.batch_limit && iteration.playouts < params.batch_limit as u32 {
        if budget_exhausted(
            params.budget,
            params.base_playouts.saturating_add(iteration.playouts),
            params.in_flight_playouts,
            params.initial_visits,
            params.stats,
        ) {
            break;
        }
        // P2.3：当 backend 已拥堵且当前 minibatch 已有可消费工作时，尽快 flush，
        // 避免 worker 在 gather 里继续无意义等待导致吞吐抖动。
        if params.thread_count > 1
            && minibatch_non_collision > 0
            && needs_nn
            && params.backend_waiting > params.config.thread_idling_threshold
            && minibatch_non_collision >= params.config.idling_minimum_work as usize
        {
            break;
        }
        if collisions_left <= 0 && iteration.pending.iter().any(|p| matches!(p.kind, PendingKind::Collision)) {
            break;
        }

        let Some(pending) = select_pending(
            tree,
            params.config,
            params.root_id,
            root_history,
            params.stats,
            params.budget,
            params.base_playouts.saturating_add(iteration.playouts),
            scratch,
        ) else {
            break;
        };
        let is_collision = matches!(pending.kind, PendingKind::Collision);
        iteration.playouts = iteration.playouts.saturating_add(1);
        if is_collision {
            collisions_left = collisions_left.saturating_sub(pending.multivisit as i32);
        } else if matches!(pending.kind, PendingKind::Expand { .. }) {
            needs_nn = true;
            minibatch_non_collision += 1;
        } else {
            minibatch_non_collision += 1;
        }

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

pub(crate) struct SelectionScratch {
    path_positions: Vec<Position>,
    path_key_counts: HashMap<u64, usize>,
}

impl Default for SelectionScratch {
    fn default() -> Self {
        Self {
            path_positions: Vec::with_capacity(16),
            path_key_counts: HashMap::new(),
        }
    }
}

impl SelectionScratch {
    fn reset(&mut self) {
        self.path_positions.clear();
        self.path_key_counts.clear();
    }
}

pub(crate) fn select_pending(
    tree: &mut MctsTree,
    config: MctsConfig,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    stats: Option<&SearchStats>,
    budget: &MctsBudget,
    session_playouts: u32,
    scratch: &mut SelectionScratch,
) -> Option<PendingNode> {
    scratch.reset();
    let mut path = Vec::<PathStep>::new();
    let mut node_id = root_id;
    let mut pos = root_history.current().clone_for_search();
    let base_key_counts = root_history.key_counts();
    let remaining_playouts = remaining_playout_budget(budget, session_playouts, 0, stats.map(|s| s.initial_visits()).unwrap_or(0), u32::MAX);

    loop {
        let node = tree.get(node_id).expect("selected node must exist");
        if let Some(value) = node.terminal_value {
            let (wl, d, m) = terminal_wdl(value);
            return Some(PendingNode {
                key: PendingKey::ExistingLeaf(node_id),
                path,
                kind: PendingKind::ExistingTerminal {
                    leaf_id: node_id,
                    wl,
                    d,
                    m,
                },
                multivisit: 1,
            });
        }

        if node.children.is_empty() {
            return None;
        }

        let is_root = path.is_empty();
        let depth_from_root = path.len() as u32;
        let edge_idx = select_edge(
            node,
            config,
            is_root,
            depth_from_root,
            remaining_playouts,
            stats,
        );
        let (mv, child_id, edge_visits, edge_in_flight) = {
            let n = tree.get(node_id).expect("selected node must exist");
            let edge = &n.children[edge_idx];
            (edge.mv, edge.child, edge.visits, edge.in_flight)
        };

        if child_id.is_none() && edge_visits == 0 && edge_in_flight > 0 {
            return Some(PendingNode {
                key: PendingKey::NewEdge(node_id, edge_idx),
                path,
                kind: PendingKind::Collision,
                multivisit: 1,
            });
        }

        {
            let edge = &mut tree
                .get_mut(node_id)
                .expect("selected node must exist")
                .children[edge_idx];
            if child_id.is_none() {
                if !edge.try_start_score_update() {
                    return Some(PendingNode {
                        key: PendingKey::NewEdge(node_id, edge_idx),
                        path,
                        kind: PendingKind::Collision,
                        multivisit: 1,
                    });
                }
            } else {
                edge.in_flight = edge.in_flight.saturating_add(1);
            }
        }

        let parent_id = node_id;
        pos.do_move(mv);
        let repeated = PositionHistory::push_search_path_position(
            base_key_counts,
            &mut scratch.path_key_counts,
            &pos,
        );
        scratch.path_positions.push(pos.clone_for_search());
        path.push(PathStep {
            node_id: parent_id,
            edge_idx,
        });

        if repeated {
            let (wl, d, m) = (0.0, 1.0, path.len() as f32);
            return Some(PendingNode {
                key: PendingKey::NewEdge(parent_id, edge_idx),
                path,
                kind: PendingKind::NewTerminal {
                    state_key: pos.key(),
                    wl,
                    d,
                    m,
                    terminal_kind: TerminalKind::TwoFold,
                },
                multivisit: 1,
            });
        }

        if let Some(child_id) = child_id {
            if let Some(stats) = stats {
                ensure_node_twofold_correct_for_depth(tree, stats, child_id, &path);
            }
            node_id = child_id;
            continue;
        }

        let mut buf = [ExtMove {
            mv: Move::none(),
            value: 0,
        }; MAX_MOVES];
        let n = generate(&pos, GenType::Legal, &mut buf);
        if n == 0 {
            let (wl, d, m) = terminal_wdl(-1.0);
            return Some(PendingNode {
                key: PendingKey::NewEdge(parent_id, edge_idx),
                path,
                kind: PendingKind::NewTerminal {
                    state_key: pos.key(),
                    wl,
                    d,
                    m,
                    terminal_kind: TerminalKind::Generic,
                },
                multivisit: 1,
            });
        }

        let legal_moves = buf[..n].iter().map(|e| e.mv).collect::<Vec<_>>();
        let history = root_history.extended_with_search_path(&scratch.path_positions);
        return Some(PendingNode {
            key: PendingKey::NewEdge(parent_id, edge_idx),
            path,
            kind: PendingKind::Expand {
                task: Arc::new(PolicyValueTask {
                    position: pos,
                    history,
                    legal_moves,
                }),
            },
            multivisit: 1,
        });
    }
}

/// px0 `FetchMinibatchResults` + `DoBackupUpdate`。
pub(crate) fn apply_minibatch(
    tree: &mut MctsTree,
    iteration: SearchIteration,
    outputs: &[PolicyValueOutput],
    shared_collisions: Option<&SharedCollisions>,
) {
    let mut eval_cursor = 0usize;
    let mut had_work = false;
    for pending in iteration.pending {
        match pending.kind {
            PendingKind::Collision => {
                // px0：collision 不在本 batch backup；由其他 worker 完成扩展后统一 CancelSharedCollisions。
            }
            PendingKind::ExistingTerminal { leaf_id, wl, d, m } => {
                do_backup_from_leaf(tree, leaf_id, &pending.path, wl, d, m, pending.multivisit);
                had_work = true;
            }
            PendingKind::NewTerminal {
                state_key,
                wl,
                d,
                m,
                terminal_kind,
            } => {
                let parent = pending.path.last().expect("new terminal must have parent");
                let child_id = add_terminal_child(
                    tree,
                    parent.node_id,
                    parent.edge_idx,
                    state_key,
                    wl,
                    d,
                    m,
                    terminal_kind,
                );
                do_backup_from_leaf(tree, child_id, &pending.path, wl, d, m, pending.multivisit);
                had_work = true;
            }
            PendingKind::Expand { task } => {
                let out = outputs.get(eval_cursor).expect("batched eval must match task count");
                eval_cursor += 1;
                let parent = pending.path.last().expect("expanded leaf must have parent");
                let child_id = add_expanded_child(tree, parent.node_id, parent.edge_idx, task.as_ref(), out);
                do_backup_from_leaf(tree, child_id, &pending.path, out.wl, out.d, out.m, pending.multivisit);
                had_work = true;
            }
        }
    }
    if had_work {
        if let Some(shared) = shared_collisions {
            shared.cancel_all(tree);
        }
    }
}

/// px0 `DoBackupUpdateSingleNode`：从叶子向根 backup，`v=-v`, `m++`。
pub(crate) fn do_backup_from_leaf(
    tree: &mut MctsTree,
    leaf_id: MctsNodeId,
    path: &[PathStep],
    mut wl: f32,
    d: f32,
    mut m: f32,
    multivisit: u32,
) {
    if let Some(leaf) = tree.get(leaf_id) {
        if leaf.is_terminal() {
            wl = leaf.wl;
            m = leaf.m;
        }
    }
    if let Some(leaf) = tree.get_mut(leaf_id) {
        leaf.finalize_score_update(wl, d, m, multivisit);
    }
    for step in path.iter().rev() {
        wl = -wl;
        m += 1.0;
        let node = tree.get_mut(step.node_id).expect("path node must exist");
        node.finalize_score_update(wl, d, m, multivisit);
        let edge = &mut node.children[step.edge_idx];
        edge.finalize_score_update(wl, d, m, multivisit);
    }
}

pub(crate) fn cancel_collision_path(tree: &mut MctsTree, path: &[PathStep], multivisit: u32) {
    for step in path.iter().rev() {
        let node = tree.get_mut(step.node_id).expect("path node must exist");
        let edge = &mut node.children[step.edge_idx];
        cancel_score_update(&mut edge.in_flight, multivisit);
    }
}

pub(crate) fn cancel_pending(tree: &mut MctsTree, pending: &PendingNode) {
    if matches!(pending.kind, PendingKind::Collision) {
        return;
    }
    for step in pending.path.iter().rev() {
        let node = tree.get_mut(step.node_id).expect("path node must exist");
        let edge = &mut node.children[step.edge_idx];
        cancel_score_update(&mut edge.in_flight, pending.multivisit);
    }
}

pub(crate) fn cancel_minibatch(tree: &mut MctsTree, iteration: SearchIteration) {
    for pending in iteration.pending {
        cancel_pending(tree, &pending);
    }
}

pub(crate) fn progress_from_tree(tree: &MctsTree, root_id: MctsNodeId, stats: &SearchStats) -> MctsSearchProgress {
    let root = tree.get(root_id).expect("root must exist");
    let summary = pv_summary_from_tree(tree, root_id);
    MctsSearchProgress {
        best_move: summary.best_move,
        pv: summary.pv,
        playouts: stats.total_playouts(),
        root_visits: root.visits,
        nodes: stats.uci_nodes(),
        tree_nodes: tree.reachable_len(),
        depth: stats.depth(),
        seldepth: stats.max_depth(),
        root_value: root.mean_value(),
        best_value: summary.best_value,
        best_mate: summary.best_mate,
        nps_elapsed_ms: stats.nps_elapsed_ms(),
        retry_without_playout: stats.retry_without_playout(),
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

pub(crate) fn result_from_tree(tree: &MctsTree, root_id: MctsNodeId, stats: &SearchStats) -> MctsSearchResult {
    let root = tree.get(root_id).expect("root must exist");
    let progress = progress_from_tree(tree, root_id, stats);
    MctsSearchResult {
        best_move: progress.best_move,
        pv: progress.pv,
        playouts: progress.playouts,
        root_visits: progress.root_visits,
        nodes: progress.nodes,
        tree_nodes: progress.tree_nodes,
        depth: progress.depth,
        seldepth: progress.seldepth,
        root_value: progress.root_value,
        best_value: progress.best_value,
        best_mate: progress.best_mate,
        nps_elapsed_ms: progress.nps_elapsed_ms,
        retry_without_playout: progress.retry_without_playout,
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

fn ensure_node_twofold_correct_for_depth(
    tree: &mut MctsTree,
    stats: &SearchStats,
    child_id: MctsNodeId,
    path: &[PathStep],
) {
    let depth = path.len() as u32;
    let child = tree.get(child_id).expect("child");
    if !child.is_twofold_terminal() || depth >= child.m as u32 {
        return;
    }
    let wl = child.wl;
    let d = child.d;
    let m = child.m;
    let terminal_visits = child.visits;
    if let Some(child) = tree.get_mut(child_id) {
        child.revert_terminal_visits(wl, d, m, terminal_visits);
        child.make_not_terminal();
    }
    stats.subtract_initial_visits(terminal_visits);
    let mut depth_counter = 0u32;
    for step in path.iter().rev() {
        depth_counter += 1;
        if depth_counter > depth {
            break;
        }
        let node = tree.get_mut(step.node_id).expect("ancestor");
        node.revert_terminal_visits(wl, d, m + depth_counter as f32, terminal_visits);
        let edge = &mut node.children[step.edge_idx];
        edge.revert_terminal_visits(wl, d, m + depth_counter as f32, terminal_visits);
    }
}

fn edge_rank(edge: &EdgeStats, child: Option<&MctsNode>) -> EdgeRank {
    if edge.visits == 0 {
        return EdgeRank::NonTerminal;
    }
    let Some(child) = child else {
        return EdgeRank::NonTerminal;
    };
    let Some(tv) = child.terminal_value else {
        return EdgeRank::NonTerminal;
    };
    if tv > 0.0 {
        EdgeRank::TerminalWin
    } else if tv < 0.0 {
        EdgeRank::TerminalLoss
    } else {
        EdgeRank::NonTerminal
    }
}

fn edge_px0_cmp(a: &EdgeStats, a_child: Option<&MctsNode>, b: &EdgeStats, b_child: Option<&MctsNode>) -> std::cmp::Ordering {
    let a_rank = edge_rank(a, a_child);
    let b_rank = edge_rank(b, b_child);
    if a_rank != b_rank {
        return a_rank.cmp(&b_rank);
    }
    if a.visits != b.visits {
        return a.visits.cmp(&b.visits);
    }
    if a.visits == 0 {
        return a.prior.partial_cmp(&b.prior).unwrap_or(std::cmp::Ordering::Equal);
    }
    if a_rank == EdgeRank::NonTerminal {
        return a
            .mean_q()
            .partial_cmp(&b.mean_q())
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if a_rank == EdgeRank::TerminalWin {
        return a.m.partial_cmp(&b.m).unwrap_or(std::cmp::Ordering::Equal);
    }
    b.m.partial_cmp(&a.m).unwrap_or(std::cmp::Ordering::Equal)
}

/// px0 `GetBestChildrenNoTemperature`（无 TB / MLH 简化版）。
pub(crate) fn pv_summary_from_tree(tree: &MctsTree, root_id: MctsNodeId) -> PvSummary {
    let mut pv = Vec::new();
    let mut node_id = root_id;
    let mut best_value = tree.get(root_id).map(MctsNode::mean_value).unwrap_or(0.0);
    let mut best_mate = None;
    let mut _ply = 0usize;

    while let Some(node) = tree.get(node_id) {
        if node.visits == 0 {
            break;
        }
        let Some((edge_idx, _)) = node
            .children
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| {
                let a_child = a.child.and_then(|id| tree.get(id));
                let b_child = b.child.and_then(|id| tree.get(id));
                edge_px0_cmp(a, a_child, b, b_child).then_with(|| ai.cmp(bi))
            }) else {
            break;
        };
        let edge = &node.children[edge_idx];
        if edge.visits == 0 && edge.child.is_none() {
            break;
        }
        let Some(child_id) = edge.child else {
            if pv.is_empty() && edge.visits > 0 {
                best_value = edge.mean_q();
            }
            pv.push(edge.mv);
            break;
        };
        if pv.is_empty() && edge.visits > 0 {
            best_value = edge.mean_q();
            if let Some(child) = tree.get(child_id) {
                if child.is_terminal() && edge.wl.abs() > f32::EPSILON {
                    let mate = (edge.get_m(0.0).round() as i32) / 2 + 1;
                    best_mate = Some(if edge.wl > 0.0 { mate } else { -mate });
                }
            }
        }
        pv.push(edge.mv);
        _ply += 1;
        if tree.get(child_id).is_some_and(|child| child.is_terminal()) {
            break;
        }
        node_id = child_id;
    }
    PvSummary {
        best_move: pv.first().copied(),
        pv,
        best_value,
        best_mate,
    }
}

pub(crate) fn total_in_flight_in_tree(tree: &MctsTree) -> u32 {
    let mut total = 0u32;
    tree.for_each_reachable(|node_id| {
        let Some(node) = tree.get(node_id) else {
            return;
        };
        total = total.saturating_add(node.children.iter().map(|edge| edge.in_flight).sum::<u32>());
    });
    total
}

pub(crate) fn budget_exhausted(
    budget: &MctsBudget,
    session_playouts: u32,
    in_flight_playouts: u32,
    initial_visits: u32,
    stats: Option<&SearchStats>,
) -> bool {
    if let Some(target_depth) = budget.max_depth {
        if let Some(stats) = stats {
            if stats.total_playouts() > 0 && stats.max_depth() >= target_depth {
                return true;
            }
        }
    }
    if let Some(target_playouts) = budget.max_playouts {
        if session_playouts.saturating_add(in_flight_playouts) >= target_playouts.max(1) {
            return true;
        }
    }
    if let Some(target_nodes) = budget.max_nodes {
        if session_playouts
            .saturating_add(in_flight_playouts)
            .saturating_add(initial_visits)
            >= target_nodes
        {
            return true;
        }
    }
    if let Some(deadline) = budget.deadline {
        if Instant::now() >= deadline {
            return true;
        }
    }
    if let Some(stop) = budget.stop.as_ref() {
        if stop.load(Ordering::SeqCst) {
            return true;
        }
    }
    false
}

/// px0 PUCT：`cpuct * sqrt(GetChildrenVisits()) * P / (1 + N_started) + Q/FPU`。
pub(crate) fn select_edge(
    node: &MctsNode,
    config: MctsConfig,
    is_root: bool,
    depth_from_root: u32,
    remaining_playouts: u32,
    stats: Option<&SearchStats>,
) -> usize {
    let draw_score = if depth_from_root % 2 == 0 {
        config.draw_score
    } else {
        -config.draw_score
    };
    let sqrt_parent = (node.children_visits().max(1)) as f32;
    let sqrt_parent = sqrt_parent.sqrt();
    let parent_q = node.mean_value_with_draw(draw_score);
    let cpuct = config.cpuct_for(is_root, node.visits);
    let visited_policy = node
        .children
        .iter()
        .filter(|edge| edge.n_started() > 0)
        .map(|edge| edge.prior)
        .sum::<f32>()
        .clamp(0.0, 1.0);
    let fpu_reduction = config.fpu_for(is_root) * visited_policy.sqrt();
    let best_root_visits = if is_root && config.smart_pruning_factor > 0.0 {
        node.children.iter().map(|edge| edge.visits).max().unwrap_or(0)
    } else {
        0
    };
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, edge) in node.children.iter().enumerate() {
        if is_root
            && config.smart_pruning_factor > 0.0
            && best_root_visits > 0
            && edge.visits < best_root_visits
            && remaining_playouts < best_root_visits.saturating_sub(edge.visits)
        {
            continue;
        }
        let q = if edge.visits == 0 {
            (parent_q - fpu_reduction).clamp(-1.0, 1.0)
        } else {
            edge.mean_q_with_draw(draw_score)
        };
        let started = edge_started_for_selection(edge, is_root, config);
        let u = cpuct * edge.prior * sqrt_parent / (1.0 + started);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    let _ = stats;
    best_idx
}

#[inline]
fn edge_started_for_selection(edge: &EdgeStats, is_root: bool, config: MctsConfig) -> f32 {
    if is_root {
        edge.visits as f32 + edge.in_flight as f32 * config.root_inflight_fraction.clamp(0.0, 1.0)
    } else {
        edge.n_started() as f32
    }
}

fn add_expanded_child(
    tree: &mut MctsTree,
    parent_id: MctsNodeId,
    edge_idx: usize,
    task: &PolicyValueTask,
    out: &PolicyValueOutput,
) -> MctsNodeId {
    let mut node = MctsNode {
        state_key: task.position.key(),
        visits: 0,
        in_flight: 0,
        wl: 0.0,
        d: 0.0,
        m: 0.0,
        expanded: true,
        terminal_kind: TerminalKind::NonTerminal,
        terminal_value: None,
        children: Vec::with_capacity(task.legal_moves.len()),
    };
    for (i, mv) in task.legal_moves.iter().copied().enumerate() {
        node.children.push(EdgeStats {
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
    let child_id = tree.add_node(node);
    let edge = &mut tree.get_mut(parent_id).expect("parent node must exist").children[edge_idx];
    edge.child = Some(child_id);
    child_id
}

fn add_terminal_child(
    tree: &mut MctsTree,
    parent_id: MctsNodeId,
    edge_idx: usize,
    state_key: u64,
    wl: f32,
    d: f32,
    m: f32,
    terminal_kind: TerminalKind,
) -> MctsNodeId {
    let tv = if wl > 0.0 { 1.0 } else if wl < 0.0 { -1.0 } else { 0.0 };
    let child_id = tree.add_node(MctsNode {
        state_key,
        visits: 0,
        in_flight: 0,
        wl,
        d,
        m,
        expanded: true,
        terminal_kind,
        terminal_value: Some(tv),
        children: Vec::new(),
    });
    let edge = &mut tree.get_mut(parent_id).expect("parent node must exist").children[edge_idx];
    edge.child = Some(child_id);
    child_id
}

#[cfg(test)]
mod selection_tests {
    use super::edge_started_for_selection;
    use crate::mcts::{EdgeStats, MctsConfig};

    #[test]
    fn root_inflight_fraction_reduces_started_penalty() {
        let edge = EdgeStats {
            in_flight: 4,
            ..EdgeStats::default()
        };
        let config = MctsConfig::default();
        let non_root = edge_started_for_selection(&edge, false, config);
        let root = edge_started_for_selection(&edge, true, config);
        assert!(root < non_root);
        assert_eq!(root, 2.0);
        assert_eq!(non_root, 4.0);
    }
}
