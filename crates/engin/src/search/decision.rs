//! 搜后根决策：bestmove、MultiPV、LCB、PV。不参与搜索过程。

use std::cmp::{Ordering, Reverse};
use std::sync::Arc;

use xiangqi_core::{Move, PositionHistory};

use super::param::SearchParams;
use super::{Edge, ExpansionState, Node, NodeKey, NodeRepository};

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
                q: edge_value(edge),
                prior: edge.prior(),
            })
            .collect(),
    })
}

/// 树搜索：action Q 就是 edge 自己的完成样本。
fn edge_value(edge: &Edge) -> f32 {
    edge.q()
}

/// root edge 当前 action value 的一、二阶矩；第三个返回值是 completed N。
fn edge_q_moments(edge: &Edge) -> Option<(f32, f32, u32)> {
    let stats = edge.stats();
    if stats.visits == 0 {
        return None;
    }
    let visits = stats.visits as f32;
    Some((stats.wl_sum / visits, stats.wl_sq_sum / visits, stats.visits))
}

/// root LCB：以 edge 的 completed N 为样本量，对极小样本加入有界 utility 方差先验。
/// 形状参考 KataGo `cpp/search/searchhelpers.cpp::getSelfUtilityLCBAndRadius`。
fn edge_lcb(edge: &Edge, stdevs: f32) -> Option<f32> {
    let (q, q_sq, visits) = edge_q_moments(edge)?;
    let effective_samples = visits.max(1) as f32;
    let prior_weight = 1.0 / (effective_samples * effective_samples);
    let adjusted_sq =
        (q_sq * effective_samples + (q_sq.max(q * q) + 1.0) * prior_weight) / (effective_samples + prior_weight);
    let lcb_samples = (effective_samples + prior_weight).powi(2) / (effective_samples + prior_weight * prior_weight);
    let variance = (adjusted_sq - q * q).max(0.0);
    let standard_error = (variance / lcb_samples).sqrt();
    Some(q - stdevs * standard_error)
}

/// 根边的 LCB 只有明确优于当前 N-first 候选时才翻盘。网络把多条边同时压到
/// `Q≈±1` 时，万分位的方差差不是足以改变着法的 Evidence；保留 N-first 能避免
/// tree reuse 中旧样本的微小差异来回切换 MultiPV。
const LCB_OVERRIDE_MIN_ADVANTAGE: f32 = 0.01;

fn edge_child(repository: &NodeRepository, parent: NodeKey, edge: &Edge) -> Option<Arc<Node>> {
    repository.get(parent.child(edge.mv()))
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
    let q = child.terminal_wl().expect("terminal wl").0;
    if q > 0.0 {
        BestEdgeRank::TerminalWin
    } else if q < 0.0 {
        BestEdgeRank::TerminalLoss
    } else {
        BestEdgeRank::NonTerminal
    }
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

fn edge_rank_key(edge: &Arc<Edge>, child: Option<&Node>) -> EdgeRankKey {
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
        q: edge_value(edge),
        prior: edge.prior(),
    }
}

fn first_filtered_move(edges: &[Arc<Edge>], root_move_filter: &[Move]) -> Option<Move> {
    if !root_move_filter.is_empty() {
        return root_move_filter
            .iter()
            .find(|mv| edges.iter().any(|edge| edge.mv() == **mv))
            .copied();
    }
    edges.first().map(|edge| edge.mv())
}

fn ranked_root_edges(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_move_filter: &[Move],
    params: &SearchParams,
) -> Vec<(Arc<Edge>, Option<Arc<Node>>)> {
    let Some(root) = repository.get(root_key) else {
        return Vec::new();
    };
    let mut candidates: Vec<_> = root
        .edges()
        .iter()
        .filter(|edge| root_move_filter.is_empty() || root_move_filter.contains(&edge.mv()))
        .map(|edge| {
            let child = edge_child(repository, root_key, edge);
            let key = edge_rank_key(edge, child.as_deref());
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
            let incumbent_lcb = edge_lcb(&candidates[0].0, params.lcb_stdevs);
            let mut best_lcb: Option<(usize, f32)> = None;
            for (index, (edge, _child, key)) in candidates.iter().enumerate() {
                if key.rank != BestEdgeRank::NonTerminal
                    || key.is_draw_terminal
                    || key.visits == 0
                    || (key.visits as f32) < min_visits
                {
                    continue;
                }
                let Some(lcb) = edge_lcb(edge, params.lcb_stdevs) else {
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
    let (wl, _, m) = child.terminal_value()?;
    (wl != 0.0).then(|| {
        let distance = m.round() as i32 / 2 + 1;
        if wl > 0.0 { distance } else { -distance }
    })
}

/// 从指定 root edge 出发的 PV。树 key 无环，沿 `parent.child(mv)` 下行。
fn principal_variation_from_root_edge(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_is_black: bool,
    first_move: Move,
    _root_history: Option<&PositionHistory>,
) -> Vec<Move> {
    let mut pv = vec![orient_move(first_move, root_is_black)];
    let mut key = root_key.child(first_move);
    let mut flip = !root_is_black;
    while let Some(node) = repository.get(key) {
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
        key = key.child(abs_mv);
        flip = !flip;
    }
    pv
}

pub(crate) fn root_variations(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_history: Option<&PositionHistory>,
    root_is_black: bool,
    root_move_filter: &[Move],
    max_pv: usize,
    params: &SearchParams,
) -> Vec<RootVariation> {
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
            let wl = if visited { edge_value(&edge) } else { default_wl };
            let draw = child
                .as_ref()
                .filter(|_| visited)
                .map_or(default_draw, |node| node.draw());
            let mate = child.as_deref().filter(|_| visited).and_then(mate_from_terminal);
            RootVariation {
                wl,
                draw,
                mate,
                pv: principal_variation_from_root_edge(repository, root_key, root_is_black, edge.mv(), root_history),
            }
        })
        .collect()
}

/// 在当前根节点的边中，按内部棋盘坐标和搜索排名选出最佳的那条边；尚未转换为 UCI 坐标。
fn best_edge_absolute(repository: &NodeRepository, root_key: NodeKey, root_move_filter: &[Move]) -> Option<Move> {
    let root = repository.get(root_key)?;
    let edges = root.edges();
    // 空边终局（将死或三次循环叶子当根）没有可走的 root edge。
    // 禁止用 a0a0 冒充 bestmove；UCI 有合法着就回退合法着，将死才是空着。
    if edges.is_empty() {
        return None;
    }
    // 尚无 completed visit 时，edge list 已从合法着生成。返回已验证的 searchmove，
    // 或第一条合法 edge。
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
        .and_then(|edge| edge_child(repository, root_key, edge))?;
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
    let mut flip = root_is_black;
    while let Some(node) = repository.get(key) {
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
        key = key.child(abs_mv);
        flip = !flip;
    }
    pv
}

pub(crate) fn principal_variation_with_history_and_params(
    repository: &NodeRepository,
    root_key: NodeKey,
    root_history: &PositionHistory,
    root_is_black: bool,
    root_move_filter: &[Move],
    params: &SearchParams,
) -> Vec<Move> {
    let Some(first_move) = best_root_edge_absolute(repository, root_key, root_move_filter, params) else {
        return Vec::new();
    };
    principal_variation_from_root_edge(repository, root_key, root_is_black, first_move, Some(root_history))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use super::{best_move, principal_variation, root_stats};
    use crate::neural::backend::UniformBackend;
    use crate::search::{Search, SearchConfig};

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
    fn terminal_root_has_no_best_edge() {
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmate fen");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        let mut pipeline = Search::new(Arc::new(UniformBackend::default()), 2, history, SearchConfig::default());
        pipeline.run_playouts(1).expect("terminal");
        assert_eq!(
            best_move(pipeline.repository(), pipeline.root_key(), root_is_black),
            None
        );
        assert!(principal_variation(pipeline.repository(), pipeline.root_key(), root_is_black).is_empty());
        pipeline.stop_and_finish();
    }
}
