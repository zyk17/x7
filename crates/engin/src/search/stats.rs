//! 只读的 stream root 统计、bestmove 与 principal variation。

use std::cmp::Ordering;
use std::sync::Arc;

use xiangqi_core::Move;

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

/// root node 快照（不是全局原子 tree view）。
#[derive(Clone, Debug, PartialEq)]
pub struct RootStats {
    pub completed_visits: u32,
    pub q: f32,
    pub draw: f32,
    pub edges: Vec<RootEdgeStats>,
}

/// 一条根候选及其从当前行棋方视角得到的统计值。
///
/// 对齐 px0 `GetBestChildrenNoTemperature` / `SendUciInfo`
/// （`search.cc:241-341`）：MultiPV 仅重排并展示已存在的根边，不改变搜索。
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
                q: edge.q(),
                prior: edge.prior(),
            })
            .collect(),
    })
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

fn edge_q_for_ranking(edge: &Edge) -> f32 {
    if edge.completed_visits() > 0 { edge.q() } else { 0.0 }
}

fn terminal_plies(child: &Node) -> f32 {
    child.terminal_plies_left().expect("terminal node has terminal plies")
}

fn is_visited_terminal(edge: &Edge, child: Option<&Node>) -> bool {
    child.is_some_and(|node| node.expansion_state() == ExpansionState::Terminal && edge.completed_visits() > 0)
}

fn edge_is_better(left: &Arc<Edge>, left_child: Option<&Node>, right: &Arc<Edge>, right_child: Option<&Node>) -> bool {
    let left_rank = best_edge_rank(left, left_child);
    let right_rank = best_edge_rank(right, right_child);
    if left_rank != right_rank {
        return left_rank > right_rank;
    }
    // 如果和棋找最短的
    if left_rank == BestEdgeRank::NonTerminal
        && is_visited_terminal(left, left_child)
        && is_visited_terminal(right, right_child)
    {
        let left_m = terminal_plies(left_child.expect("visited terminal child"));
        let right_m = terminal_plies(right_child.expect("visited terminal child"));
        if (left_m - right_m).abs() > f32::EPSILON {
            return left_m < right_m;
        }
    }
    if left_rank == BestEdgeRank::NonTerminal {
        let left_n = left.completed_visits();
        let right_n = right.completed_visits();
        if left_n != right_n {
            return left_n > right_n;
        }
        let left_q = edge_q_for_ranking(left);
        let right_q = edge_q_for_ranking(right);
        if (left_q - right_q).abs() > f32::EPSILON {
            return left_q > right_q;
        }
        return left.prior() > right.prior();
    }
    if left_rank == BestEdgeRank::TerminalWin {
        terminal_plies(left_child.expect("terminal winning child"))
            < terminal_plies(right_child.expect("terminal winning child"))
    } else {
        terminal_plies(left_child.expect("terminal losing child"))
            > terminal_plies(right_child.expect("terminal losing child"))
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
) -> Vec<(Arc<Edge>, Option<Arc<Node>>)> {
    // px0 `GetBestChildrenNoTemperature`（`search.cc:241`）：复用 bestmove 的根边
    // 排名，只改变展示顺序，不重新选择或分配 visit。
    let Some(root) = repository.get(root_key) else {
        return Vec::new();
    };
    let mut candidates: Vec<_> = root
        .edges()
        .iter()
        .filter(|edge| root_move_filter.is_empty() || root_move_filter.contains(&edge.mv()))
        .map(|edge| (Arc::clone(edge), repository.get(root_key.child(edge.mv()))))
        .collect();
    if root.completed_visits() > 0 {
        candidates.sort_by(|(left, left_child), (right, right_child)| {
            if edge_is_better(left, left_child.as_deref(), right, right_child.as_deref()) {
                Ordering::Less
            } else if edge_is_better(right, right_child.as_deref(), left, left_child.as_deref()) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
    }
    candidates
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
    root_is_black: bool,
    root_move_filter: &[Move],
    max_pv: usize,
) -> Vec<RootVariation> {
    // px0 `SendUciInfo` 的 root-edge WDL/Q 默认值（`search.cc:269-341`）。
    let Some(root) = repository.get(root_key) else {
        return Vec::new();
    };
    let default_wl = (-root.q()).clamp(-1.0, 1.0);
    let default_draw = root.draw().clamp(0.0, 1.0);
    ranked_root_edges(repository, root_key, root_move_filter)
        .into_iter()
        .take(max_pv)
        .map(|(edge, child)| {
            let visited = edge.completed_visits() > 0;
            let wl = if visited { edge.q() } else { default_wl };
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
    // 或第一条合法 edge。
    if root.completed_visits() == 0 {
        return first_filtered_move(&edges, root_move_filter);
    }
    ranked_root_edges(repository, root_key, root_move_filter)
        .into_iter()
        .next()
        .map(|(edge, _)| edge.mv())
        .or_else(|| first_filtered_move(&edges, root_move_filter))
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
    best_edge_absolute(repository, root_key, root_move_filter).map(|mv| orient_move(mv, root_is_black))
}

pub fn best_move(repository: &NodeRepository, root_key: NodeKey, root_is_black: bool) -> Option<Move> {
    best_move_filtered(repository, root_key, root_is_black, &[])
}

/// 所选 root edge 的已证明 mate score。`m` 是从该 child 起的 ply 数，对齐 px0
/// `SendUciInfo`（`search.cc:249-336`）。
pub(crate) fn best_mate(repository: &NodeRepository, root_key: NodeKey, root_move_filter: &[Move]) -> Option<i32> {
    let root = repository.get(root_key)?;
    if let Some((wl, _, m)) = root.terminal_value() {
        return (wl != 0.0).then(|| {
            let distance = m.round() as i32 / 2 + 1;
            if wl < 0.0 { distance } else { -distance }
        });
    }
    let mv = best_edge_absolute(repository, root_key, root_move_filter)?;
    let child = repository.get(root_key.child(mv))?;
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
        let filter = if pv.is_empty() { root_move_filter } else { &[] };
        let Some(abs_mv) = best_edge_absolute(repository, key, filter) else {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{best_mate, best_move, best_move_filtered, principal_variation, root_stats, root_variations};
    use crate::neural::backend::UniformBackend;
    use crate::search::{Search, SearchConfig, SearchGeneration};

    #[test]
    fn root_snapshot_reports_completed_and_in_flight_visits_separately() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(31),
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
        let mut pipeline = Search::new(
            Arc::new(UniformBackend::default()),
            SearchGeneration(2),
            history,
            SearchConfig::default(),
        );
        pipeline.run_playouts(1).expect("terminal");
        assert_eq!(
            best_move(pipeline.repository(), pipeline.root_key(), root_is_black),
            Some(Move::NULL)
        );
        assert!(principal_variation(pipeline.repository(), pipeline.root_key(), root_is_black).is_empty());
        pipeline.stop_and_finish();
    }

    #[test]
    fn terminal_win_outranks_higher_n_terminal_loss() {
        use crate::search::{ExpansionState, NodeKey, NodeRepository, ValueDelta};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repo = NodeRepository::default();
        let root_key = NodeKey::root(7);
        let root = repo.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![
            (mv("a0", "a1"), 0.10),
            (mv("b0", "b1"), 0.20),
            (mv("c0", "c1"), 0.70),
        ]);
        let edges = root.edges();
        let idx = |target: Move| edges.iter().position(|edge| edge.mv() == target).expect("edge");
        let loss_idx = idx(mv("a0", "a1"));
        let win_idx = idx(mv("b0", "b1"));
        let other_idx = idx(mv("c0", "c1"));

        {
            let child = repo.get_or_insert(root_key.child(mv("a0", "a1")));
            assert!(child.try_begin_evaluation());
            child.mark_terminal(-1.0, 0.0, 0.0);
            assert_eq!(child.expansion_state(), ExpansionState::Terminal);
            for _ in 0..50 {
                root.reserve_edge(loss_idx).expect("res").complete(-1.0);
                root.add_delta(ValueDelta::one(-1.0, 0.0));
            }
        }
        {
            let child = repo.get_or_insert(root_key.child(mv("b0", "b1")));
            assert!(child.try_begin_evaluation());
            child.mark_terminal(1.0, 0.0, 0.0);
            for _ in 0..5 {
                root.reserve_edge(win_idx).expect("res").complete(1.0);
                root.add_delta(ValueDelta::one(1.0, 0.0));
            }
        }
        for _ in 0..20 {
            root.reserve_edge(other_idx).expect("res").complete(0.1);
            root.add_delta(ValueDelta::one(0.1, 0.0));
        }

        assert_eq!(
            best_move(&repo, root_key, false).expect("best"),
            mv("b0", "b1"),
            "proven terminal win must beat higher-N loss and non-terminal"
        );
    }

    #[test]
    fn best_mate_reports_proven_terminal_child_distance() {
        use crate::search::{NodeKey, NodeRepository, ValueDelta};

        let repository = NodeRepository::default();
        let root_key = NodeKey::root(99);
        let root = repository.get_or_insert(root_key);
        let mv = Move::new(Square::parse("b2").expect("from"), Square::parse("b3").expect("to"));
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(mv, 1.0)]);
        root.add_delta(ValueDelta::one(0.0, 0.0));
        let child = repository.get_or_insert(root_key.child(mv));
        assert!(child.try_begin_evaluation());
        child.mark_terminal(1.0, 0.0, 3.0);

        assert_eq!(best_mate(&repository, root_key, &[]), Some(2));
    }

    #[test]
    fn best_mate_reports_a_checkmated_root() {
        use crate::search::{NodeKey, NodeRepository};

        let repository = NodeRepository::default();
        let root = repository.get_or_insert(NodeKey::root(100));
        assert!(root.try_begin_evaluation());
        root.mark_terminal(1.0, 0.0, 0.0);

        assert_eq!(best_mate(&repository, NodeKey::root(100), &[]), Some(-1));
    }

    #[test]
    fn root_move_filter_applies_before_visits() {
        use crate::search::{NodeKey, NodeRepository};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repo = NodeRepository::default();
        let root_key = NodeKey::root(5);
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
    fn root_variations_use_bestmove_ranking_and_edge_values() {
        use crate::search::{NodeKey, NodeRepository, ValueDelta};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repository = NodeRepository::default();
        let root_key = NodeKey::root(19);
        let root = repository.get_or_insert(root_key);
        let first = mv("a0", "a1");
        let second = mv("b0", "b1");
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(first, 0.9), (second, 0.1)]);
        for _ in 0..3 {
            root.reserve_edge(0).expect("first reservation").complete(0.1);
            root.add_delta(ValueDelta::one(0.1, 0.0));
        }
        for _ in 0..5 {
            root.reserve_edge(1).expect("second reservation").complete(0.2);
            root.add_delta(ValueDelta::one(0.2, 0.0));
        }

        let variations = root_variations(&repository, root_key, false, &[], 2);
        assert_eq!(variations.len(), 2);
        assert_eq!(variations[0].pv, vec![second]);
        assert_eq!(variations[1].pv, vec![first]);
        assert!((variations[0].wl - 0.2).abs() < f32::EPSILON);
        assert!((variations[1].wl - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn shorter_terminal_draw_outranks_longer_draw_at_equal_n() {
        use crate::search::{NodeKey, NodeRepository, ValueDelta};

        fn mv(from: &str, to: &str) -> Move {
            Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
        }

        let repo = NodeRepository::default();
        let root_key = NodeKey::root(11);
        let root = repo.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(mv("a0", "a1"), 0.5), (mv("b0", "b1"), 0.5)]);
        let edges = root.edges();
        let short_idx = edges.iter().position(|e| e.mv() == mv("a0", "a1")).unwrap();
        let long_idx = edges.iter().position(|e| e.mv() == mv("b0", "b1")).unwrap();

        {
            let child = repo.get_or_insert(root_key.child(mv("a0", "a1")));
            assert!(child.try_begin_evaluation());
            child.mark_terminal(0.0, 1.0, 2.0);
            root.reserve_edge(short_idx).unwrap().complete(0.0);
            root.add_delta(ValueDelta::one(0.0, 1.0));
        }
        {
            let child = repo.get_or_insert(root_key.child(mv("b0", "b1")));
            assert!(child.try_begin_evaluation());
            child.mark_terminal(0.0, 1.0, 8.0);
            root.reserve_edge(long_idx).unwrap().complete(0.0);
            root.add_delta(ValueDelta::one(0.0, 1.0));
        }

        assert_eq!(
            best_move(&repo, root_key, false).expect("best"),
            mv("a0", "a1"),
            "equal-N terminal draws prefer shorter m"
        );
    }
}
