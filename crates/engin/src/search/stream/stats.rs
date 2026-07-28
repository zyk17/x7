//! Read-only stream root statistics, bestmove, and principal variation.
//!
//! Bestmove / PV ranking follows px0 `GetBestChildrenNoTemperature`
//! (`search.cc:705-808`) and classic `search.rs`, with stream `draw_score = 0`.
//! Returned moves are UCI-oriented (px0 `GetMove(flip)`): flipped when the
//! side to move at that PV ply is black.

use std::sync::Arc;

use xiangqi_core::Move;

use super::{Edge, ExpansionState, Node, NodeKey, NodeRepository};

/// One root edge snapshot. `started_visits` includes in-flight; Q uses completed only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootEdgeStats {
    pub mv: Move,
    pub completed_visits: u32,
    pub started_visits: u32,
    pub q: f32,
    pub prior: f32,
}

/// Root node snapshot (not a globally atomic tree view).
#[derive(Clone, Debug, PartialEq)]
pub struct RootStats {
    pub completed_visits: u32,
    pub q: f32,
    pub draw: f32,
    pub edges: Vec<RootEdgeStats>,
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

/// px0 `EdgeAndNode::GetMove` orientation for UCI (`node.h` / `search.cc` PV loop).
fn orient_move(mv: Move, flip: bool) -> Move {
    if flip && !mv.is_null() {
        mv.flip()
    } else {
        mv
    }
}

/// Bestmove outcome bucket (px0 `EdgeRank`, `search.cc:737-756`).
///
/// `NonTerminal` means non-decisive: ordinary lines **or terminal draws**
/// (`wl`/`q == 0`). Draws are not a separate rank so they compete via N→Q→P
/// (and the terminal-draw `m` tie-break below) instead of always beating
/// unproven lines. Only proven wins/losses get hard priority.
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
    // Terminal nodes are proofs. Use their exact incoming-edge value rather
    // than an edge average that may still contain pre-proof in-flight visits.
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
    if edge.completed_visits() > 0 {
        edge.q()
    } else {
        0.0
    }
}

fn terminal_plies(child: Option<&Node>) -> f32 {
    child.and_then(Node::terminal_plies_left).unwrap_or(0.0)
}

fn is_visited_terminal(edge: &Edge, child: Option<&Node>) -> bool {
    child.is_some_and(|node| node.expansion_state() == ExpansionState::Terminal && edge.completed_visits() > 0)
}

/// px0 `GetBestChildrenNoTemperature` (`search.cc:776-795`): lexicographic
/// compare, not a weighted blend of N/Q/P.
///
/// `true` when `left` is strictly better than `right`.
fn edge_is_better(left: &Arc<Edge>, left_child: Option<&Node>, right: &Arc<Edge>, right_child: Option<&Node>) -> bool {
    let left_rank = best_edge_rank(left, left_child);
    let right_rank = best_edge_rank(right, right_child);
    if left_rank != right_rank {
        return left_rank > right_rank;
    }
    // px0: both NonTerminal and both terminal (draws) → shorter m.
    // Tablebase preference omitted (stream has no TB).
    if left_rank == BestEdgeRank::NonTerminal
        && is_visited_terminal(left, left_child)
        && is_visited_terminal(right, right_child)
    {
        let left_m = terminal_plies(left_child);
        let right_m = terminal_plies(right_child);
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
    if left_rank > BestEdgeRank::NonTerminal {
        terminal_plies(left_child) < terminal_plies(right_child)
    } else {
        terminal_plies(left_child) > terminal_plies(right_child)
    }
}

fn first_filtered_move(edges: &[Arc<Edge>], root_move_filter: &[Move]) -> Option<Move> {
    if !root_move_filter.is_empty() {
        return root_move_filter
            .iter()
            .find(|mv| edges.iter().any(|edge| edge.mv() == **mv))
            .copied()
            .or_else(|| edges.first().map(|edge| edge.mv()));
    }
    edges.first().map(|edge| edge.mv())
}

/// Absolute (board) best edge, before UCI orientation.
fn best_edge_absolute(repository: &NodeRepository, root_key: NodeKey, root_move_filter: &[Move]) -> Option<Move> {
    let node = repository.get(root_key)?;
    let edges = node.edges();
    if edges.is_empty() {
        return if node.expansion_state() == ExpansionState::Terminal || node.completed_visits() > 0 {
            Some(Move::NULL)
        } else {
            None
        };
    }
    // Parent N == 0: px0 `GetBestChildrenNoTemperature` returns empty
    // (`search.cc:710`). Classic Rust then falls back to searchmoves / first
    // edge so UCI still gets a legal reply (`classic/search.rs:31-43`).
    if node.completed_visits() == 0 {
        return first_filtered_move(&edges, root_move_filter);
    }
    let mut best: Option<(Arc<Edge>, Option<Arc<Node>>)> = None;
    for edge in edges.iter() {
        if !root_move_filter.is_empty() && !root_move_filter.contains(&edge.mv()) {
            continue;
        }
        let child = repository.get(root_key.child(edge.mv()));
        let better = match &best {
            None => true,
            Some((best_edge, best_child)) => edge_is_better(edge, child.as_deref(), best_edge, best_child.as_deref()),
        };
        if better {
            best = Some((Arc::clone(edge), child));
        }
    }
    best.map(|(edge, _)| edge.mv())
        .or_else(|| first_filtered_move(&edges, root_move_filter))
}

/// Bestmove with optional `go searchmoves` filter (px0 `root_move_filter_`).
///
/// `root_is_black` applies UCI orientation for the root ply.
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

/// Proven mate score for the selected root edge. `m` is in plies from that
/// child, matching px0 `SendUciInfo` (`search.cc:249-336`).
pub(crate) fn best_mate(repository: &NodeRepository, root_key: NodeKey, root_move_filter: &[Move]) -> Option<i32> {
    let root = repository.get(root_key)?;
    if let Some((wl, _, m)) = root.terminal_value() {
        return (wl != 0.0).then(|| {
            let distance = m.round() as i32 / 2 + 1;
            if wl < 0.0 {
                distance
            } else {
                -distance
            }
        });
    }
    let mv = best_edge_absolute(repository, root_key, root_move_filter)?;
    let child = repository.get(root_key.child(mv))?;
    let (wl, _, m) = child.terminal_value()?;
    (wl != 0.0).then(|| {
        let distance = m.round() as i32 / 2 + 1;
        if wl > 0.0 {
            distance
        } else {
            -distance
        }
    })
}

/// Principal variation (UCI `pv` line). Not policy/value.
///
/// Walks like px0 `SendUciInfo` (`search.cc:345-350`): best child while parent
/// `N > 0`; zero-visit child edges may appear once (dangling), then stop.
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

/// True when every root edge has `started == completed` (no in-flight).
pub fn root_settled(repository: &NodeRepository, root_key: NodeKey) -> bool {
    repository
        .get(root_key)
        .is_some_and(|root| root.edges().iter().all(|edge| edge.visits() == edge.completed_visits()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, Square, STARTPOS_FEN};

    use super::{best_mate, best_move, best_move_filtered, principal_variation, root_settled, root_stats};
    use crate::neural::backend::UniformBackend;
    use crate::search::stream::{Search, SearchConfig, SearchGeneration};

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
        assert!(root_settled(pipeline.repository(), pipeline.root_key()));
        let mv = best_move(pipeline.repository(), pipeline.root_key(), root_is_black).expect("bestmove");
        assert!(!mv.is_null());
        let pv = principal_variation(pipeline.repository(), pipeline.root_key(), root_is_black);
        assert_eq!(pv.first().copied(), Some(mv));
        pipeline.stop_and_join();
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
        pipeline.stop_and_join();
    }

    #[test]
    fn terminal_win_outranks_higher_n_terminal_loss() {
        use crate::search::stream::{ExpansionState, NodeKey, NodeRepository, ValueDelta};

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
        use crate::search::stream::{NodeKey, NodeRepository, ValueDelta};

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
        use crate::search::stream::{NodeKey, NodeRepository};

        let repository = NodeRepository::default();
        let root = repository.get_or_insert(NodeKey::root(100));
        assert!(root.try_begin_evaluation());
        root.mark_terminal(1.0, 0.0, 0.0);

        assert_eq!(best_mate(&repository, NodeKey::root(100), &[]), Some(-1));
    }

    #[test]
    fn root_move_filter_applies_before_visits() {
        use crate::search::stream::{NodeKey, NodeRepository};

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
    fn shorter_terminal_draw_outranks_longer_draw_at_equal_n() {
        use crate::search::stream::{NodeKey, NodeRepository, ValueDelta};

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
