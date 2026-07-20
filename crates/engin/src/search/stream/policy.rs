//! Fixed LC3-style policy primitives for the first stream implementation.
//!
//! Reference: LC3 policy, "Search Policy" and the functions
//! `GetNumEdgesToFetch`, `NodeEventToValueDelta`, `MoveNodeUpdateToParent`,
//! and `MakeEdgeDelta`:
//! <https://lczero.org/dev/lc0/search/lc3/policy/>

use super::StreamEdge;

/// Compact WDL update used by stream backpropagation.
///
/// LC3 stores `v = W - L` and keeps draw separately where needed. Moving the
/// update to a parent flips v but preserves draw probability.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValueDelta {
    pub visits: u32,
    pub value_sum: f32,
    pub draw_sum: f32,
}

impl ValueDelta {
    pub fn one(value: f32, draw: f32) -> Self {
        assert!((-1.0..=1.0).contains(&value), "WDL value must be normalized");
        assert!((0.0..=1.0).contains(&draw), "WDL draw must be normalized");
        Self {
            visits: 1,
            value_sum: value,
            draw_sum: draw,
        }
    }

    /// LC3 `MoveNodeUpdateToParent`: parent sees the opponent's value.
    pub fn for_parent(self) -> Self {
        Self {
            visits: self.visits,
            value_sum: -self.value_sum,
            draw_sum: self.draw_sum,
        }
    }

    pub fn q(self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

/// Selects the highest PUCT edge from a single repository node.
///
/// `StreamEdge::visits()` includes in-flight visits, as required by LC3's
/// node structure. This makes collisions naturally less attractive while an
/// Eval worker owns the leaf. Ties keep the first policy order deterministically.
pub fn select_edge(edges: &[std::sync::Arc<StreamEdge>], parent_visits: u32, cpuct: f32) -> Option<usize> {
    assert!(
        cpuct.is_finite() && cpuct >= 0.0,
        "cpuct must be finite and non-negative"
    );
    let exploration = cpuct * (parent_visits.max(1) as f32).sqrt();
    let mut best: Option<(usize, f32)> = None;
    for (index, edge) in edges.iter().enumerate() {
        let score = edge.q() + exploration * edge.prior() / (1 + edge.visits()) as f32;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{Move, Square};

    use super::{select_edge, ValueDelta};
    use crate::search::stream::{NodeKey, NodeRepository};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    #[test]
    fn parent_delta_flips_value_not_draw() {
        let leaf = ValueDelta::one(0.6, 0.2);
        assert_eq!(leaf.for_parent(), ValueDelta::one(-0.6, 0.2));
    }

    #[test]
    fn in_flight_visit_deflects_selection_without_changing_completed_q() {
        let node = NodeRepository::default().get_or_insert(NodeKey::root(9));
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("b2", "b3"), 0.6), (mv("c3", "c4"), 0.4)]);
        let edges = node.edges();
        assert_eq!(select_edge(&edges, 1, 1.0), Some(0));

        let reservation = node.reserve_edge(0).expect("first edge");
        assert_eq!(select_edge(&edges, 1, 1.0), Some(1));
        reservation.cancel();
        assert_eq!(edges[0].completed_visits(), 0);
    }
}
