//! Stream search policy (classic-aligned selection + WDL deltas).
//!
//! Selection mirrors classic `ComputeCpuct` / `GetFpu` / PUCT edge scoring
//! (`crates/engin/src/search/classic/uct.rs`, px0 `search.cc:408-433`) with
//! `draw_score` fixed at 0 (Q is raw mover-perspective `wl`). Edge visit counts
//! include in-flight reservations (LC3 node structure).

use std::sync::Arc;

use xiangqi_core::Move;

use super::{Edge, Node};
use super::params::compute_cpuct;
use super::SearchParams;

/// Compact WDL update used by stream backpropagation.
///
/// - `wl_sum` ↔ px0 `wl_` (mover / incoming-edge perspective, not raw NN STM)
/// - `draw_sum` ↔ px0 `d_`
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValueDelta {
    pub visits: u32,
    pub wl_sum: f32,
    pub draw_sum: f32,
}

impl ValueDelta {
    pub fn one(wl: f32, draw: f32) -> Self {
        assert!((-1.0..=1.0).contains(&wl), "WDL wl must be normalized");
        assert!((0.0..=1.0).contains(&draw), "WDL draw must be normalized");
        Self {
            visits: 1,
            wl_sum: wl,
            draw_sum: draw,
        }
    }

    pub fn for_parent(self) -> Self {
        Self {
            visits: self.visits,
            wl_sum: -self.wl_sum,
            draw_sum: self.draw_sum,
        }
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            visits: self.visits + other.visits,
            wl_sum: self.wl_sum + other.wl_sum,
            draw_sum: self.draw_sum + other.draw_sum,
        }
    }

    pub fn q(self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.wl_sum / self.visits as f32
        }
    }
}

/// Classic `Node::GetVisitedPolicy` over stream edges.
pub fn visited_policy(edges: &[Arc<Edge>]) -> f32 {
    edges
        .iter()
        .filter(|edge| edge.completed_visits() > 0)
        .map(|edge| edge.prior())
        .sum()
}

/// Classic `GetFpu` (`search.cc:408-424`) with `draw_score = 0`.
pub fn get_fpu(
    params: &SearchParams,
    parent_wl: f32,
    edges: &[Arc<Edge>],
) -> f32 {
    -parent_wl - params.fpu_reduction * visited_policy(edges).sqrt()
}

/// Visited-edge Q, or FPU when the edge has no completed visit.
pub fn edge_utility(edge: &Edge, fpu: f32) -> f32 {
    if edge.completed_visits() == 0 {
        fpu
    } else {
        edge.q()
    }
}

fn root_filter_allows(is_root: bool, filter: &[Move], mv: Move) -> bool {
    !is_root || filter.is_empty() || filter.contains(&mv)
}

/// Selects the highest classic-style PUCT edge.
///
/// `root_move_filter` mirrors px0 `Search::root_move_filter_` from UCI
/// `go searchmoves` (`search.cc:53-58,721-724,1668-1739`): empty means no filter.
pub fn select_edge(
    edges: &[Arc<Edge>],
    parent_completed_visits: u32,
    parent_wl: f32,
    depth: usize,
    params: &SearchParams,
    root_move_filter: &[Move],
) -> Option<usize> {
    if edges.is_empty() {
        return None;
    }
    let is_root = depth == 0;
    let children_visits = if parent_completed_visits > 0 {
        parent_completed_visits - 1
    } else {
        0
    };
    let cpuct = compute_cpuct(*params, parent_completed_visits);
    let u_coeff = cpuct * (children_visits.max(1) as f32).sqrt();
    let fpu = get_fpu(params, parent_wl, edges);
    let mut best: Option<(usize, f32)> = None;
    for (index, edge) in edges.iter().enumerate() {
        if !root_filter_allows(is_root, root_move_filter, edge.mv()) {
            continue;
        }
        let q = edge_utility(edge, fpu);
        let score = q + u_coeff * edge.prior() / (1 + edge.visits()) as f32;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
}

/// Convenience for Gather call sites that already hold the parent node.
pub fn select_edge_from_node(
    node: &Node,
    depth: usize,
    params: &SearchParams,
    root_move_filter: &[Move],
) -> Option<usize> {
    let edges = node.edges();
    select_edge(
        &edges,
        node.completed_visits(),
        node.q(),
        depth,
        params,
        root_move_filter,
    )
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{Move, Square};

    use super::{edge_utility, select_edge, ValueDelta};
    use crate::search::stream::{NodeKey, NodeRepository, SearchParams};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    #[test]
    fn parent_delta_flips_wl_not_draw() {
        let leaf = ValueDelta::one(0.6, 0.2);
        assert_eq!(leaf.for_parent(), ValueDelta::one(-0.6, 0.2));
    }

    #[test]
    fn in_flight_visit_deflects_selection_without_changing_completed_q() {
        let repo = NodeRepository::default();
        let key = NodeKey::root(9);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("b2", "b3"), 0.6), (mv("c3", "c4"), 0.4)]);
        let edges = node.edges();
        let params = SearchParams::default();
        assert_eq!(select_edge(&edges, 0, 0.0, 0, &params, &[]), Some(0));

        let reservation = node.reserve_edge(0).expect("first edge");
        assert_eq!(select_edge(&edges, 0, 0.0, 0, &params, &[]), Some(1));
        reservation.cancel();
        assert_eq!(edges[0].completed_visits(), 0);
    }

    #[test]
    fn root_move_filter_restricts_selection() {
        let repo = NodeRepository::default();
        let key = NodeKey::root(9);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("b2", "b3"), 0.9), (mv("c3", "c4"), 0.1)]);
        let edges = node.edges();
        let params = SearchParams::default();
        let filter = [mv("c3", "c4")];
        assert_eq!(select_edge(&edges, 0, 0.0, 0, &params, &filter), Some(1));
    }

    #[test]
    fn visited_edge_utility_is_raw_q() {
        let repo = NodeRepository::default();
        let key = NodeKey::root(3);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("a0", "a1"), 1.0)]);
        let edge = node.edges()[0].clone();
        node.reserve_edge(0).expect("res").complete(0.5);
        assert!((edge_utility(&edge, 0.0) - 0.5).abs() < 1e-6);
    }
}
