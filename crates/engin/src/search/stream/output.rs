//! Read-only final-move and PV ranking for the stream repository.
//!
//! LC3's public policy page leaves final move selection TBD. X7 therefore
//! explicitly adopts px0 classic's no-temperature ranking for its first
//! stream output policy: terminal result, completed visits, Q, then prior.
//! Reference: px0 `src/search/classic/search.cc:705-808`.

use std::cmp::Ordering;

use xiangqi_core::Move;

use super::{ExpansionState, NodeKey, NodeRepository, StreamEdge, StreamNode};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamPrincipalVariation {
    pub moves: Vec<Move>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum EdgeRank {
    TerminalLoss,
    NonTerminal,
    TerminalWin,
}

fn child_node(repository: &NodeRepository, parent: NodeKey, edge: &StreamEdge) -> Option<std::sync::Arc<StreamNode>> {
    repository.get(parent.child(edge.mv()))
}

fn edge_rank(repository: &NodeRepository, parent: NodeKey, edge: &StreamEdge) -> EdgeRank {
    let Some(child) = child_node(repository, parent, edge) else {
        return EdgeRank::NonTerminal;
    };
    if edge.completed_visits() == 0 || child.expansion_state() != ExpansionState::Terminal || edge.q() == 0.0 {
        return EdgeRank::NonTerminal;
    }
    if edge.q() > 0.0 {
        EdgeRank::TerminalWin
    } else {
        EdgeRank::TerminalLoss
    }
}

fn edge_is_better(repository: &NodeRepository, parent: NodeKey, left: &StreamEdge, right: &StreamEdge) -> bool {
    let left_rank = edge_rank(repository, parent, left);
    let right_rank = edge_rank(repository, parent, right);
    if left_rank != right_rank {
        return left_rank > right_rank;
    }

    if left_rank == EdgeRank::NonTerminal {
        let left_is_terminal = child_node(repository, parent, left)
            .is_some_and(|node| node.expansion_state() == ExpansionState::Terminal && left.completed_visits() > 0);
        let right_is_terminal = child_node(repository, parent, right)
            .is_some_and(|node| node.expansion_state() == ExpansionState::Terminal && right.completed_visits() > 0);
        if left_is_terminal && right_is_terminal {
            let left_m = child_node(repository, parent, left)
                .expect("checked terminal child")
                .moves_left();
            let right_m = child_node(repository, parent, right)
                .expect("checked terminal child")
                .moves_left();
            return left_m < right_m;
        }
        if left.completed_visits() != right.completed_visits() {
            return left.completed_visits() > right.completed_visits();
        }
        if left.q() != right.q() {
            return left.q() > right.q();
        }
        return left.prior() > right.prior();
    }

    let left_m = child_node(repository, parent, left)
        .expect("terminal child")
        .moves_left();
    let right_m = child_node(repository, parent, right)
        .expect("terminal child")
        .moves_left();
    if left_rank == EdgeRank::TerminalWin {
        left_m < right_m
    } else {
        left_m > right_m
    }
}

/// px0 `Search::GetBestChildrenNoTemperature` without root move filtering or
/// tablebase ranks. Stream has neither searchmoves filtering nor tablebases
/// yet; those are UCI/terminal capabilities, not a hidden ranking heuristic.
pub fn best_children_no_temperature(repository: &NodeRepository, parent: NodeKey, count: usize) -> Vec<Move> {
    let Some(node) = repository.get(parent) else {
        return Vec::new();
    };
    if node.completed_visits() == 0 {
        return Vec::new();
    }
    let mut edges = node.edges().to_vec();
    edges.sort_unstable_by(|left, right| {
        if edge_is_better(repository, parent, left, right) {
            Ordering::Less
        } else if edge_is_better(repository, parent, right, left) {
            Ordering::Greater
        } else {
            left.mv().raw().cmp(&right.mv().raw())
        }
    });
    edges.into_iter().take(count).map(|edge| edge.mv()).collect()
}

pub fn principal_variation(repository: &NodeRepository, root: NodeKey) -> StreamPrincipalVariation {
    let mut moves = Vec::new();
    let mut parent = root;
    while let Some(mv) = best_children_no_temperature(repository, parent, 1).into_iter().next() {
        moves.push(mv);
        parent = parent.child(mv);
    }
    StreamPrincipalVariation { moves }
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{Move, Square};

    use super::{best_children_no_temperature, principal_variation};
    use crate::search::stream::{NodeKey, NodeRepository, ValueDelta};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    #[test]
    fn terminal_win_beats_more_visited_non_terminal_edge() {
        let repository = NodeRepository::default();
        let root_key = NodeKey::root(1);
        let root = repository.get_or_insert(root_key);
        let win = mv("b2", "b3");
        let ordinary = mv("c3", "c4");
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(win, 0.1), (ordinary, 0.9)]);
        root.add_value(ValueDelta::one(0.0, 0.0, 0.0));

        let terminal = repository.get_or_insert(root_key.child(win));
        assert!(terminal.try_begin_evaluation());
        terminal.mark_terminal(-1.0, 0.0, 0.0);
        terminal.add_value(ValueDelta::one(-1.0, 0.0, 0.0));
        root.reserve_edge(0).expect("win edge").complete(1.0);

        let child = repository.get_or_insert(root_key.child(ordinary));
        child.add_value(ValueDelta::one(-0.2, 0.0, 4.0));
        for _ in 0..8 {
            root.reserve_edge(1).expect("ordinary edge").complete(0.2);
        }

        assert_eq!(best_children_no_temperature(&repository, root_key, 1), vec![win]);
    }

    #[test]
    fn pv_follows_no_temperature_child_order() {
        let repository = NodeRepository::default();
        let root_key = NodeKey::root(2);
        let root = repository.get_or_insert(root_key);
        let first = mv("b2", "b3");
        let second = mv("c3", "c4");
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(first, 0.6), (second, 0.4)]);
        root.add_value(ValueDelta::one(0.0, 0.0, 0.0));
        root.reserve_edge(0).expect("root edge").complete(0.1);

        let child_key = root_key.child(first);
        let child = repository.get_or_insert(child_key);
        assert!(child.try_begin_evaluation());
        child.publish_edges(vec![(second, 1.0)]);
        child.add_value(ValueDelta::one(-0.1, 0.0, 3.0));
        child.reserve_edge(0).expect("child edge").complete(0.2);

        assert_eq!(principal_variation(&repository, root_key).moves, vec![first, second]);
    }

    #[test]
    fn non_terminal_order_is_visits_then_q_then_prior() {
        let repository = NodeRepository::default();
        let root_key = NodeKey::root(3);
        let root = repository.get_or_insert(root_key);
        let high_n = mv("b2", "b3");
        let high_q = mv("c3", "c4");
        let high_p = mv("h2", "h3");
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(high_n, 0.1), (high_q, 0.2), (high_p, 0.9)]);
        root.add_value(ValueDelta::one(0.0, 0.0, 0.0));

        for _ in 0..2 {
            root.reserve_edge(0).expect("high N").complete(0.0);
        }
        root.reserve_edge(1).expect("high Q").complete(0.8);
        root.reserve_edge(2).expect("high P").complete(0.8);
        assert_eq!(
            best_children_no_temperature(&repository, root_key, 3),
            vec![high_n, high_p, high_q]
        );
    }
}
