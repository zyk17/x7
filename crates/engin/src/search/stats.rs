//! 只读的 stream root 统计、bestmove 与 principal variation。

use std::cmp::{Ordering, Reverse};
use std::collections::HashSet;
use std::sync::Arc;

use xiangqi_core::Move;

use super::{Edge, ExpansionState, Node, NodeKey, NodeRepository, SearchParams};

/// 一个 root edge 快照。`started_visits` 包含 in-flight；Q 只使用 completed 值。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootEdgeStats {
    pub mv: Move,
    pub completed_visits: u32,
    pub started_visits: u32,
    pub q: f32,
    pub prior: f32,
}

/// root node 快照（不是全局原子 graph view）。
#[derive(Clone, Debug, PartialEq)]
pub struct RootStats {
    pub completed_visits: u32,
    pub q: f32,
    pub draw: f32,
    pub edges: Vec<RootEdgeStats>,
}

/// 一条根候选及其从当前行棋方视角得到的统计值。
///
/// MultiPV 仅重排并展示已存在的根边，不改变搜索。
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootVariation {
    pub wl: f32,
    pub draw: f32,
    pub mate: Option<i32>,
    pub pv: Vec<Move>,
}

pub fn root_stats(repository: &NodeRepository, root_key: NodeKey) -> Option<RootStats> {
    let root = repository.get(root_key)?;
    Some(RootStats {
        completed_visits: root.completed_visits(),
        q: root.q(),
        draw: root.draw(),
        edges: root
            .edges()
            .iter()
            .map(|edge| RootEdgeStats {
                mv: edge.mv(),
                completed_visits: edge.completed_visits(),
                started_visits: edge.visits(),
                q: edge_value(repository, edge),
                prior: edge.prior(),
            })
            .collect(),
    })
}

/// MCGS 的 action Q 混合 shared child 的当前值与该 edge 实际遇到的路径终局样本。
/// 未完成 edge 的 Q 不参与最终统计。
fn edge_value(repository: &NodeRepository, edge: &Edge) -> f32 {
    let stats = edge.completed_stats();
    if stats.visits == 0 {
        return 0.0;
    }
    let propagated = stats.visits.saturating_sub(stats.local_leaf.visits);
    let child_value = edge
        .child_key()
        .and_then(|key| repository.get(key))
        .map_or(0.0, |node| node.q() * propagated as f32);
    (stats.local_leaf.wl_sum + child_value) / stats.visits as f32
}

/// root edge 当前 action value 的一、二阶矩。shared child 的二阶矩遵循与 Q 相同的
/// MCGS 幂等重算；edge N 仍严格只算实际从该 root action 走入的次数。
fn edge_q_moments(repository: &NodeRepository, edge: &Edge) -> Option<(f32, f32, u32)> {
    let stats = edge.completed_stats();
    if stats.visits == 0 {
        return None;
    }
    let propagated = stats.visits.saturating_sub(stats.local_leaf.visits);
    let (child_q, child_q_sq) = if propagated == 0 {
        (0.0, 0.0)
    } else {
        let child = edge.child_key().and_then(|key| repository.get(key))?;
        let (q, _, _, q_sq) = child.value_moments_snapshot();
        (q, q_sq)
    };
    let visits = stats.visits as f32;
    Some((
        (stats.local_leaf.wl_sum + child_q * propagated as f32) / visits,
        (stats.local_leaf.wl_sq_sum + child_q_sq * propagated as f32) / visits,
        stats.visits,
    ))
}

/// X7 的最小 root LCB：以 edge 的实际完成次数为样本量，并对极小样本加入有界
/// utility 方差先验。形状参考 KataGo
/// `cpp/search/searchhelpers.cpp::getSelfUtilityLCBAndRadius`，但不复制其围棋 utility、
/// 权重或 play-selection 机制。
fn edge_lcb(repository: &NodeRepository, edge: &Edge, stdevs: f32) -> Option<f32> {
    let (q, q_sq, visits) = edge_q_moments(repository, edge)?;
    let weight = visits as f32;
    let prior_weight = 1.0 / (weight * weight);
    let adjusted_sq = (q_sq * weight + (q_sq.max(q * q) + 1.0) * prior_weight) / (weight + prior_weight);
    let effective_samples = (weight + prior_weight).powi(2) / (weight + prior_weight * prior_weight);
    let variance = (adjusted_sq - q * q).max(0.0);
    Some(q - stdevs * (variance / effective_samples).sqrt())
}

/// 根边的 LCB 只有明确优于当前 N-first 候选时才翻盘。网络把多条边同时压到
/// `Q≈±1` 时，万分位的方差差不是足以改变着法的 Evidence；保留 N-first 能避免
/// tree reuse 中旧样本的微小差异来回切换 MultiPV。
const LCB_OVERRIDE_MIN_ADVANTAGE: f32 = 0.01;

fn edge_child(repository: &NodeRepository, edge: &Edge) -> Option<Arc<Node>> {
    edge.child_key().and_then(|key| repository.get(key))
}

fn orient_move(mv: Move, flip: bool) -> Move {
    if flip && !mv.is_null() { mv.flip() } else { mv }
}

/// bestmove 结果分组（px0 `EdgeRank`，`search.cc:737-756`）。
///
/// `NonTerminal` 表示未决：普通变化或终局和棋（`wl`/`q == 0`）。和棋不另设 rank，
/// 而是通过 N→Q→P（及下方终局和棋的 `m` 决胜）参与竞争，不会总是压过未证明变化。
/// 只有已证明的胜/负才有硬优先级。
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum BestEdgeRank {
    TerminalLoss,
    NonTerminal,
    TerminalWin,
}

fn best_edge_rank(edge: &Edge, child: Option<&Node>) -> BestEdgeRank {
    let Some(child) = child else {
        return BestEdgeRank::NonTerminal;
    };
    if child.expansion_state() != ExpansionState::Terminal || edge.completed_visits() == 0 {
        return BestEdgeRank::NonTerminal;
    }
    let q = child.terminal_wl().expect("terminal stream wl").0;
    if q > 0.0 {
        BestEdgeRank::TerminalWin
    } else if q < 0.0 {
        BestEdgeRank::TerminalLoss
    } else {
        BestEdgeRank::NonTerminal
    }
}

fn edge_q_for_ranking(repository: &NodeRepository, edge: &Edge) -> f32 {
    edge_value(repository, edge)
}

fn terminal_plies(child: &Node) -> f32 {
    child.terminal_plies_left().expect("terminal node has terminal plies")
}

fn is_visited_terminal(edge: &Edge, child: Option<&Node>) -> bool {
    child.is_some_and(|node| node.expansion_state() == ExpansionState::Terminal && edge.completed_visits() > 0)
}

/// 一次性快照排名键，保证 `sort` 全序；避免并发回写 N/Q 时 `sort_by` 比较器不一致而 panic。
#[derive(Clone, Copy)]
struct EdgeRankKey {
    rank: BestEdgeRank,
    is_draw_terminal: bool,
    plies: f32,
    visits: u32,
    q: f32,
    prior: f32,
}

impl PartialEq for EdgeRankKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for EdgeRankKey {}

impl Ord for EdgeRankKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.rank.cmp(&other.rank) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
        if self.rank == BestEdgeRank::NonTerminal && self.is_draw_terminal && other.is_draw_terminal {
            match other.plies.total_cmp(&self.plies) {
                Ordering::Equal => {}
                non_eq => return non_eq,
            }
        }
        if self.rank == BestEdgeRank::NonTerminal {
            return self
                .visits
                .cmp(&other.visits)
                .then_with(|| self.q.total_cmp(&other.q))
                .then_with(|| self.prior.total_cmp(&other.prior));
        }
        if self.rank == BestEdgeRank::TerminalWin {
            other.plies.total_cmp(&self.plies)
        } else {
            self.plies.total_cmp(&other.plies)
        }
    }
}

impl PartialOrd for EdgeRankKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn edge_rank_key(repository: &NodeRepository, edge: &Arc<Edge>, child: Option<&Node>) -> EdgeRankKey {
    let rank = best_edge_rank(edge, child);
    let is_draw_terminal = rank == BestEdgeRank::NonTerminal && is_visited_terminal(edge, child);
    let plies = match rank {
        BestEdgeRank::TerminalWin | BestEdgeRank::TerminalLoss => terminal_plies(child.expect("terminal child")),
        BestEdgeRank::NonTerminal if is_draw_terminal => terminal_plies(child.expect("visited terminal child")),
        BestEdgeRank::NonTerminal => 0.0,
    };
    EdgeRankKey {
        rank,
        is_draw_terminal,
        plies,
        visits: edge.completed_visits(),
        q: edge_q_for_ranking(repository, edge),
        prior: edge.prior(),
    }
}

fn first_filtered_move(edges: &[Arc<Edge>], root_move_filter: &[Move]) -> Option<Move> {
    if !root_move_filter.is_empty() {
        return root_move_filter
            .iter()
            .find(|mv| edges.iter().any(|edge| edge.mv() == **mv && !edge.topology_pruned()))
            .copied();
    }
    edges.iter().find(|edge| !edge.topology_pruned()).map(|edge| edge.mv())
}

fn ranked_root_edges(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_move_filter: &[Move],
    params: &SearchParams,
) -> Vec<(Arc<Edge>, Option<Arc<Node>>)> {
    // px0 `GetBestChildrenNoTemperature`（`search.cc:241`）：复用 bestmove 的根边
    // 排名，只改变展示顺序，不重新选择或分配 visit。
    let Some(root) = repository.get(root_key) else {
        return Vec::new();
    };
    let mut candidates: Vec<_> = root
        .edges()
        .iter()
        .filter(|edge| !edge.topology_pruned())
        .filter(|edge| root_move_filter.is_empty() || root_move_filter.contains(&edge.mv()))
        .map(|edge| {
            let child = edge_child(repository, edge);
            let key = edge_rank_key(repository, edge, child.as_deref());
            (Arc::clone(edge), child, key)
        })
        .collect();
    if root.completed_visits() > 0 {
        candidates.sort_unstable_by_key(|candidate| Reverse(candidate.2));
    }
    if params.lcb_stdevs > 0.0 {
        let Some((_, _, best_key)) = candidates.first() else {
            return Vec::new();
        };
        if best_key.rank == BestEdgeRank::NonTerminal {
            let min_visits = best_key.visits as f32 * params.lcb_min_visit_fraction;
            let incumbent_lcb = edge_lcb(repository, &candidates[0].0, params.lcb_stdevs);
            let mut best_lcb: Option<(usize, f32)> = None;
            for (index, (edge, _child, key)) in candidates.iter().enumerate() {
                if key.rank != BestEdgeRank::NonTerminal
                    || key.is_draw_terminal
                    || key.visits == 0
                    || (key.visits as f32) < min_visits
                {
                    continue;
                }
                let Some(lcb) = edge_lcb(repository, edge, params.lcb_stdevs) else {
                    continue;
                };
                if best_lcb.is_none_or(|(_, current)| lcb > current) {
                    best_lcb = Some((index, lcb));
                }
            }
            if let (Some((index, lcb)), Some(incumbent_lcb)) = (best_lcb, incumbent_lcb)
                && index != 0
                && lcb >= incumbent_lcb + LCB_OVERRIDE_MIN_ADVANTAGE
            {
                let selected = candidates.remove(index);
                candidates.insert(0, selected);
            }
        }
    }
    candidates.into_iter().map(|(edge, child, _)| (edge, child)).collect()
}

fn mate_from_terminal(child: &Node) -> Option<i32> {
    // px0 `SendUciInfo` 的 `edge.IsTerminal()` 分支（`search.cc:283-288`）。
    let (wl, _, m) = child.terminal_value()?;
    (wl != 0.0).then(|| {
        let distance = m.round() as i32 / 2 + 1;
        if wl > 0.0 { distance } else { -distance }
    })
}

fn principal_variation_from_root_edge(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_is_black: bool,
    first_move: Move,
) -> Vec<Move> {
    // px0 `SendUciInfo` 的 PV 遍历（`search.cc:343-350`）。
    let mut pv = vec![orient_move(first_move, root_is_black)];
    let Some(first_edge) = repository
        .get(root_key)
        .and_then(|root| root.edges().iter().find(|edge| edge.mv() == first_move).cloned())
    else {
        return pv;
    };
    let Some(mut key) = first_edge.child_key() else {
        return pv;
    };
    let mut seen = HashSet::from([root_key]);
    let mut flip = !root_is_black;
    while let Some(node) = repository.get(key) {
        if !seen.insert(key) {
            break;
        }
        if node.completed_visits() == 0 || node.expansion_state() != ExpansionState::Expanded {
            break;
        }
        let Some(abs_mv) = best_edge_absolute(repository, key, &[]) else {
            break;
        };
        if abs_mv.is_null() {
            break;
        }
        pv.push(orient_move(abs_mv, flip));
        let Some(next) = node
            .edges()
            .iter()
            .find(|edge| edge.mv() == abs_mv)
            .and_then(|edge| edge.child_key())
        else {
            break;
        };
        key = next;
        flip = !flip;
    }
    pv
}

pub(crate) fn root_variations_with_params(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_is_black: bool,
    root_move_filter: &[Move],
    max_pv: usize,
    params: &SearchParams,
) -> Vec<RootVariation> {
    // px0 `SendUciInfo` 的 root-edge WDL/Q 默认值（`search.cc:269-341`）。
    let Some(root) = repository.get(root_key) else {
        return Vec::new();
    };
    let default_wl = (-root.q()).clamp(-1.0, 1.0);
    let default_draw = root.draw().clamp(0.0, 1.0);
    ranked_root_edges(repository, root_key, root_move_filter, params)
        .into_iter()
        .take(max_pv)
        .map(|(edge, child)| {
            let visited = edge.completed_visits() > 0;
            let wl = if visited {
                edge_value(repository, &edge)
            } else {
                default_wl
            };
            let draw = child
                .as_ref()
                .filter(|_| visited)
                .map_or(default_draw, |node| node.draw());
            let mate = child.as_deref().filter(|_| visited).and_then(mate_from_terminal);
            RootVariation {
                wl,
                draw,
                mate,
                pv: principal_variation_from_root_edge(repository, root_key, root_is_black, edge.mv()),
            }
        })
        .collect()
}

/// 在当前根节点的边中，按内部棋盘坐标和搜索排名选出最佳的那条边；尚未转换为 UCI 坐标。
fn best_edge_absolute(repository: &NodeRepository, root_key: NodeKey, root_move_filter: &[Move]) -> Option<Move> {
    let root = repository.get(root_key)?;
    let edges = root.edges();
    if edges.is_empty() {
        return if root.expansion_state() == ExpansionState::Terminal || root.completed_visits() > 0 {
            Some(Move::NULL)
        } else {
            None
        };
    }
    // 尚无 completed visit 时，edge list 已从合法着生成。返回已验证的 searchmove，
    // 或第一条未被 topology prune 的合法 edge。
    if root.completed_visits() == 0 {
        return first_filtered_move(&edges, root_move_filter);
    }
    ranked_root_edges(
        repository,
        root_key,
        root_move_filter,
        &SearchParams {
            lcb_stdevs: 0.0,
            ..SearchParams::default()
        },
    )
    .into_iter()
    .next()
    .map(|(edge, _)| edge.mv())
    .or_else(|| first_filtered_move(&edges, root_move_filter))
}

fn best_root_edge_absolute(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_move_filter: &[Move],
    params: &SearchParams,
) -> Option<Move> {
    let root = repository.get(root_key)?;
    if root.edges().is_empty() || root.completed_visits() == 0 {
        return best_edge_absolute(repository, root_key, root_move_filter);
    }
    ranked_root_edges(repository, root_key, root_move_filter, params)
        .into_iter()
        .next()
        .map(|(edge, _)| edge.mv())
}

/// 带可选 `go searchmoves` filter 的 bestmove（px0 `root_move_filter_`）。
///
/// `root_is_black` 在 root ply 应用 UCI 坐标方向。
pub fn best_move_filtered(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_is_black: bool,
    root_move_filter: &[Move],
) -> Option<Move> {
    best_move_filtered_with_params(
        repository,
        root_key,
        root_is_black,
        root_move_filter,
        &SearchParams::default(),
    )
}

pub(crate) fn best_move_filtered_with_params(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_is_black: bool,
    root_move_filter: &[Move],
    params: &SearchParams,
) -> Option<Move> {
    best_root_edge_absolute(repository, root_key, root_move_filter, params).map(|mv| orient_move(mv, root_is_black))
}

pub fn best_move(repository: &NodeRepository, root_key: NodeKey, root_is_black: bool) -> Option<Move> {
    best_move_filtered(repository, root_key, root_is_black, &[])
}

/// 所选 root edge 的已证明 mate score。`m` 是从该 child 起的 ply 数。
pub(crate) fn best_mate_with_params(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_move_filter: &[Move],
    params: &SearchParams,
) -> Option<i32> {
    let root = repository.get(root_key)?;
    if let Some((wl, _, m)) = root.terminal_value() {
        return (wl != 0.0).then(|| {
            let distance = m.round() as i32 / 2 + 1;
            if wl < 0.0 { distance } else { -distance }
        });
    }
    let mv = best_root_edge_absolute(repository, root_key, root_move_filter, params)?;
    let child = repository
        .get(root_key)?
        .edges()
        .iter()
        .find(|edge| edge.mv() == mv)
        .and_then(|edge| edge_child(repository, edge))?;
    mate_from_terminal(&child)
}

/// principal variation（UCI `pv` 行），不是 policy/value。
///
/// 按 px0 `SendUciInfo`（`search.cc:345-350`）遍历：parent `N > 0` 时取最佳 child；
/// 零访问 child edge 可能出现一次（悬挂），随后停止。
pub fn principal_variation(repository: &NodeRepository, root_key: NodeKey, root_is_black: bool) -> Vec<Move> {
    principal_variation_filtered(repository, root_key, root_is_black, &[])
}

pub fn principal_variation_filtered(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_is_black: bool,
    root_move_filter: &[Move],
) -> Vec<Move> {
    principal_variation_filtered_with_params(
        repository,
        root_key,
        root_is_black,
        root_move_filter,
        &SearchParams::default(),
    )
}

pub(crate) fn principal_variation_filtered_with_params(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_is_black: bool,
    root_move_filter: &[Move],
    params: &SearchParams,
) -> Vec<Move> {
    let mut pv = Vec::new();
    let mut key = root_key;
    let mut seen = HashSet::new();
    let mut flip = root_is_black;
    while let Some(node) = repository.get(key) {
        if !seen.insert(key) {
            break;
        }
        if node.completed_visits() == 0 {
            break;
        }
        if node.expansion_state() != ExpansionState::Expanded {
            break;
        }
        let first = pv.is_empty();
        let filter = if first { root_move_filter } else { &[] };
        let selected = if first {
            best_root_edge_absolute(repository, key, filter, params)
        } else {
            best_edge_absolute(repository, key, filter)
        };
        let Some(abs_mv) = selected else {
            break;
        };
        if abs_mv.is_null() {
            break;
        }
        pv.push(orient_move(abs_mv, flip));
        let Some(next) = node
            .edges()
            .iter()
            .find(|edge| edge.mv() == abs_mv)
            .and_then(|edge| edge.child_key())
        else {
            break;
        };
        key = next;
        flip = !flip;
    }
    pv
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{
        LCB_OVERRIDE_MIN_ADVANTAGE, best_mate_with_params, best_move, best_move_filtered,
        best_move_filtered_with_params, edge_lcb, edge_value, principal_variation, root_stats,
        root_variations_with_params,
    };
    use crate::neural::backend::UniformBackend;
    use crate::search::{NodeKey, NodeRepository, Search, SearchConfig, SearchParams, ValueDelta};

    #[test]
    fn root_snapshot_reports_completed_and_in_flight_visits_separately() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            31,
            history,
            SearchConfig::default(),
        );
        pipeline.run_playouts(16).expect("playouts");

        let stats = root_stats(pipeline.repository(), pipeline.root_key()).expect("root snapshot");
        assert_eq!(stats.completed_visits, 16);
        assert!(stats.edges.iter().any(|edge| edge.completed_visits > 0));
        assert!(
            stats
                .edges
                .iter()
                .all(|edge| edge.started_visits == edge.completed_visits)
        );
        let mv = best_move(pipeline.repository(), pipeline.root_key(), root_is_black).expect("bestmove");
        assert!(!mv.is_null());
        let pv = principal_variation(pipeline.repository(), pipeline.root_key(), root_is_black);
        assert_eq!(pv.first().copied(), Some(mv));
        pipeline.stop_and_finish();
    }

    #[test]
    fn terminal_root_bestmove_is_null_not_none() {
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmate fen");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        let mut pipeline = Search::new(Arc::new(UniformBackend::default()), 2, history, SearchConfig::default());
        pipeline.run_playouts(1).expect("terminal");
        assert_eq!(
            best_move(pipeline.repository(), pipeline.root_key(), root_is_black),
            Some(Move::NULL)
        );
        assert!(principal_variation(pipeline.repository(), pipeline.root_key(), root_is_black).is_empty());
        pipeline.stop_and_finish();
    }

    #[test]
    fn root_lcb_can_choose_a_well_visited_higher_q_challenger() {
        let repo = NodeRepository::default();
        let root_key = NodeKey::board(1);
        let root = repo.get_or_insert(root_key);
        let first = Move::new(Square::parse("a0").unwrap(), Square::parse("a1").unwrap());
        let challenger = Move::new(Square::parse("b0").unwrap(), Square::parse("b1").unwrap());
        assert!(root.try_begin_evaluation());
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(first, 0.8), (challenger, 0.2)]);

        let edges = root.edges();
        for (index, (key, value, visits)) in [(NodeKey::board(2), 0.20, 100), (NodeKey::board(3), 0.25, 20)]
            .into_iter()
            .enumerate()
        {
            let child = repo.get_or_insert(key);
            child.set_graph_value(ValueDelta::one(value, 0.0));
            repo.recompute_graph_node(key);
            edges[index].bind_child_key(key);
            for _ in 0..visits {
                root.reserve_edge(index).unwrap().complete();
            }
        }
        repo.recompute_graph_node(root_key);

        let params = SearchParams::default();
        assert_eq!(
            best_move_filtered_with_params(&repo, root_key, false, &[], &params),
            Some(challenger),
            "LCB may overturn N-first only after the challenger reaches the visit threshold"
        );

        let n_first = SearchParams {
            lcb_stdevs: 0.0,
            ..params
        };
        assert_eq!(
            best_move_filtered_with_params(&repo, root_key, false, &[], &n_first),
            Some(first),
            "zero LCB stdevs retains N-first"
        );
        let too_few_visits = SearchParams {
            lcb_min_visit_fraction: 0.21,
            ..params
        };
        assert_eq!(
            best_move_filtered_with_params(&repo, root_key, false, &[], &too_few_visits),
            Some(first),
            "a challenger below the minimum visit fraction cannot replace N-first"
        );
    }

    #[test]
    fn root_lcb_keeps_n_first_when_its_advantage_is_too_small() {
        let repo = NodeRepository::default();
        let root_key = NodeKey::board(4);
        let root = repo.get_or_insert(root_key);
        let first = Move::new(Square::parse("a0").unwrap(), Square::parse("a1").unwrap());
        let challenger = Move::new(Square::parse("b0").unwrap(), Square::parse("b1").unwrap());
        assert!(root.try_begin_evaluation());
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(first, 0.8), (challenger, 0.2)]);

        let edges = root.edges();
        for (index, (key, value, visits)) in [(NodeKey::board(41), 0.998, 100), (NodeKey::board(42), 0.9999, 80)]
            .into_iter()
            .enumerate()
        {
            let child = repo.get_or_insert(key);
            child.set_graph_value(ValueDelta::one(value, 0.0));
            repo.recompute_graph_node(key);
            edges[index].bind_child_key(key);
            for _ in 0..visits {
                root.reserve_edge(index).unwrap().complete();
            }
        }
        repo.recompute_graph_node(root_key);

        let first_lcb = edge_lcb(&repo, &edges[0], 5.0).expect("first LCB");
        let challenger_lcb = edge_lcb(&repo, &edges[1], 5.0).expect("challenger LCB");
        assert!(challenger_lcb > first_lcb);
        assert!(challenger_lcb - first_lcb < LCB_OVERRIDE_MIN_ADVANTAGE);
        assert_eq!(
            best_move_filtered_with_params(&repo, root_key, false, &[], &SearchParams::default()),
            Some(first),
            "tiny saturated-value LCB differences must not overturn the N-first candidate"
        );
    }

    #[test]
    fn root_lcb_penalizes_an_unstable_action_value() {
        let repo = NodeRepository::default();
        let root_key = NodeKey::board(10);
        let root = repo.get_or_insert(root_key);
        let noisy = Move::new(Square::parse("a0").unwrap(), Square::parse("a1").unwrap());
        let stable = Move::new(Square::parse("b0").unwrap(), Square::parse("b1").unwrap());
        assert!(root.try_begin_evaluation());
        root.set_graph_value(ValueDelta::one(0.0, 0.0));
        root.publish_edges(vec![(noisy, 0.8), (stable, 0.2)]);

        let edges = root.edges();
        for index in 0..100 {
            let value = if index % 2 == 0 { 0.9 } else { -0.3 };
            root.reserve_edge(0)
                .unwrap()
                .complete_local_leaf(ValueDelta::one(value, 0.0));
        }
        for _ in 0..20 {
            root.reserve_edge(1)
                .unwrap()
                .complete_local_leaf(ValueDelta::one(0.25, 0.0));
        }
        repo.recompute_graph_node(root_key);

        assert!(edge_value(&repo, &edges[0]) > edge_value(&repo, &edges[1]));
        assert_eq!(
            best_move_filtered_with_params(&repo, root_key, false, &[], &SearchParams::default()),
            Some(stable),
            "LCB uses wl_sq_sum to prefer the lower-variance action"
        );
    }

    #[test]
    fn terminal_win_outranks_higher_n_terminal_loss() {
        use crate::search::{ExpansionState, NodeKey, NodeRepository};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repo = NodeRepository::default();
        let root_key = NodeKey::board(7);
        let root = repo.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![
            (mv("a0", "a1"), 0.10),
            (mv("b0", "b1"), 0.20),
            (mv("c0", "c1"), 0.70),
        ]);
        root.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        let edges = root.edges();
        let idx = |target: Move| edges.iter().position(|edge| edge.mv() == target).expect("edge");
        let loss_idx = idx(mv("a0", "a1"));
        let win_idx = idx(mv("b0", "b1"));
        let other_idx = idx(mv("c0", "c1"));

        {
            let child_key = NodeKey::board(71);
            edges[loss_idx].bind_child_key(child_key);
            let child = repo.get_or_insert(child_key);
            assert!(child.try_begin_evaluation());
            child.mark_terminal(-1.0, 0.0, 0.0);
            assert_eq!(child.expansion_state(), ExpansionState::Terminal);
            for _ in 0..50 {
                root.reserve_edge(loss_idx).expect("res").complete();
            }
        }
        {
            let child_key = NodeKey::board(72);
            edges[win_idx].bind_child_key(child_key);
            let child = repo.get_or_insert(child_key);
            assert!(child.try_begin_evaluation());
            child.mark_terminal(1.0, 0.0, 0.0);
            for _ in 0..5 {
                root.reserve_edge(win_idx).expect("res").complete();
            }
        }
        let other_key = NodeKey::board(73);
        edges[other_idx].bind_child_key(other_key);
        let other = repo.get_or_insert(other_key);
        other.set_graph_value(crate::search::ValueDelta::one(0.1, 0.0));
        repo.recompute_graph_node(other_key);
        for _ in 0..20 {
            root.reserve_edge(other_idx).expect("res").complete();
        }
        repo.recompute_graph_node(root_key);

        assert_eq!(
            best_move(&repo, root_key, false).expect("best"),
            mv("b0", "b1"),
            "proven terminal win must beat higher-N loss and non-terminal"
        );
    }

    #[test]
    fn best_mate_reports_proven_terminal_child_distance() {
        use crate::search::{NodeKey, NodeRepository};

        let repository = NodeRepository::default();
        let root_key = NodeKey::board(99);
        let root = repository.get_or_insert(root_key);
        let mv = Move::new(Square::parse("b2").expect("from"), Square::parse("b3").expect("to"));
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(mv, 1.0)]);
        let child_key = NodeKey::board(1000);
        root.edges()[0].bind_child_key(child_key);
        let child = repository.get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.mark_terminal(1.0, 0.0, 3.0);

        assert_eq!(
            best_mate_with_params(&repository, root_key, &[], &SearchParams::default()),
            Some(2)
        );
    }

    #[test]
    fn best_mate_reports_a_checkmated_root() {
        use crate::search::{NodeKey, NodeRepository};

        let repository = NodeRepository::default();
        let root = repository.get_or_insert(NodeKey::board(100));
        assert!(root.try_begin_evaluation());
        root.mark_terminal(1.0, 0.0, 0.0);

        assert_eq!(
            best_mate_with_params(&repository, NodeKey::board(100), &[], &SearchParams::default()),
            Some(-1)
        );
    }

    #[test]
    fn root_move_filter_applies_before_visits() {
        use crate::search::{NodeKey, NodeRepository};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repo = NodeRepository::default();
        let root_key = NodeKey::board(5);
        let root = repo.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(mv("a0", "a1"), 0.9), (mv("b0", "b1"), 0.1)]);
        let filter = [mv("b0", "b1")];
        assert_eq!(
            best_move_filtered(&repo, root_key, false, &filter).expect("fallback"),
            mv("b0", "b1")
        );
    }

    #[test]
    fn root_output_skips_topology_pruned_edges() {
        use crate::search::{NodeKey, NodeRepository};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repo = NodeRepository::default();
        let root_key = NodeKey::board(5);
        let child_key = NodeKey::board(6);
        let root = repo.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        let pruned = mv("a0", "a1");
        let usable = mv("b0", "b1");
        root.publish_edges(vec![(pruned, 0.9), (usable, 0.1)]);

        let child = repo.get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.publish_edges(vec![(mv("c0", "c1"), 1.0)]);
        child.edges()[0].bind_child_key(root_key);

        let root_edges = root.edges();
        let pruned_edge = root_edges.iter().find(|edge| edge.mv() == pruned).expect("pruned edge");
        assert!(matches!(
            repo.bind_child_or_cut_cycle(root_key, pruned_edge, child_key),
            crate::search::graph::ChildLink::TopologyPruned
        ));

        assert_eq!(best_move(&repo, root_key, false), Some(usable));
        let variations = root_variations_with_params(&repo, root_key, false, &[], 2, &SearchParams::default());
        assert_eq!(variations.len(), 1);
        assert_eq!(variations[0].pv, vec![usable]);
    }

    #[test]
    fn root_variations_use_bestmove_ranking_and_edge_values() {
        use crate::search::{NodeKey, NodeRepository};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repository = NodeRepository::default();
        let root_key = NodeKey::board(19);
        let root = repository.get_or_insert(root_key);
        let first = mv("a0", "a1");
        let second = mv("b0", "b1");
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(first, 0.9), (second, 0.1)]);
        root.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        for (index, (key, value)) in [(0, (NodeKey::board(191), 0.1)), (1, (NodeKey::board(192), 0.2))] {
            root.edges()[index].bind_child_key(key);
            let child = repository.get_or_insert(key);
            child.set_graph_value(crate::search::ValueDelta::one(value, 0.0));
            repository.recompute_graph_node(key);
        }
        for _ in 0..3 {
            root.reserve_edge(0).expect("first reservation").complete();
        }
        for _ in 0..5 {
            root.reserve_edge(1).expect("second reservation").complete();
        }
        repository.recompute_graph_node(root_key);

        let variations = root_variations_with_params(&repository, root_key, false, &[], 2, &SearchParams::default());
        assert_eq!(variations.len(), 2);
        assert_eq!(variations[0].pv, vec![second]);
        assert_eq!(variations[1].pv, vec![first]);
        assert!((variations[0].wl - 0.2).abs() < f32::EPSILON);
        assert!((variations[1].wl - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn shorter_terminal_draw_outranks_longer_draw_at_equal_n() {
        use crate::search::{NodeKey, NodeRepository};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repo = NodeRepository::default();
        let root_key = NodeKey::board(11);
        let root = repo.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(mv("a0", "a1"), 0.5), (mv("b0", "b1"), 0.5)]);
        let edges = root.edges();
        let short_idx = edges.iter().position(|e| e.mv() == mv("a0", "a1")).unwrap();
        let long_idx = edges.iter().position(|e| e.mv() == mv("b0", "b1")).unwrap();

        {
            let child_key = NodeKey::board(111);
            edges[short_idx].bind_child_key(child_key);
            let child = repo.get_or_insert(child_key);
            assert!(child.try_begin_evaluation());
            child.mark_terminal(0.0, 1.0, 2.0);
            root.reserve_edge(short_idx).unwrap().complete();
        }
        {
            let child_key = NodeKey::board(112);
            edges[long_idx].bind_child_key(child_key);
            let child = repo.get_or_insert(child_key);
            assert!(child.try_begin_evaluation());
            child.mark_terminal(0.0, 1.0, 8.0);
            root.reserve_edge(long_idx).unwrap().complete();
        }

        assert_eq!(
            best_move(&repo, root_key, false).expect("best"),
            mv("a0", "a1"),
            "equal-N terminal draws prefer shorter m"
        );
    }

    #[test]
    fn principal_variation_stops_at_a_graph_cycle() {
        use crate::search::{NodeKey, NodeRepository};

        let repository = NodeRepository::default();
        let root_key = NodeKey::board(400);
        let root = repository.get_or_insert(root_key);
        let mv = Move::new(Square::parse("a0").unwrap(), Square::parse("a1").unwrap());
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(mv, 1.0)]);
        root.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        root.edges()[0].bind_child_key(root_key);
        root.reserve_edge(0).unwrap().complete();
        repository.recompute_graph_node(root_key);

        assert_eq!(principal_variation(&repository, root_key, false), vec![mv]);
    }
}
