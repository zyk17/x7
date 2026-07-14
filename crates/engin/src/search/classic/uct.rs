//! px0 `search.cc` 中 PUCT/FPU 辅助（`ComputeCpuct`、`GetFpu`、边打分）。

use super::node::{Node, NodeArena};
use super::params::SearchParams;
use crate::utils::fastmath::fast_log;

/// px0 `ComputeCpuct` (`src/search/classic/search.cc:426-433`), including
/// its `FastLog` approximation from `src/utils/fastmath.h:81-83`.
pub fn compute_cpuct(params: &SearchParams, n: u32, is_root: bool) -> f32 {
    let init = params.cpuct(is_root);
    let k = params.cpuct_factor(is_root);
    let base = params.cpuct_base(is_root);
    if k == 0.0 {
        init
    } else {
        init + k * fast_log((n as f32 + base) / base)
    }
}

/// px0 `GetFpu` (`src/search/classic/search.cc:408-424`)。
pub fn get_fpu(params: &SearchParams, node: &Node, arena: &NodeArena, is_root: bool, draw_score: f32) -> f32 {
    let visited_pol = node.visited_policy(arena);
    let value = params.fpu_value(is_root);
    if params.fpu_absolute(is_root) {
        value
    } else {
        -node.q(-draw_score) - value * visited_pol.sqrt()
    }
}

/// px0 PUCT 边打分（`search.cc` 中 `current_score` 语义）。
pub fn edge_score(
    parent: &Node,
    edge_idx: usize,
    child: Option<&Node>,
    arena: &NodeArena,
    params: &SearchParams,
    is_root: bool,
    draw_score: f32,
) -> f32 {
    let edge = parent.edge(edge_idx);
    let cpuct = compute_cpuct(params, parent.n(), is_root);
    let u_coeff = cpuct * (parent.children_visits().max(1) as f32).sqrt();
    let fpu = get_fpu(params, parent, arena, is_root, draw_score);
    let q = child
        .filter(|node| node.n() > 0)
        .map(|node| node.q(draw_score))
        .unwrap_or(fpu);
    q + u_coeff * edge.get_p() / (1.0 + child.map(|node| node.n_started()).unwrap_or(0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::{Move, Square};

    #[test]
    fn get_fpu_matches_px0_draw_score_and_visited_policy() {
        let params = SearchParams {
            fpu_value: 0.5,
            ..SearchParams::default()
        };

        let a0 = Square::parse("a0").expect("a0");
        let a1 = Square::parse("a1").expect("a1");
        let mut arena = NodeArena::default();
        let root = arena.alloc(Node::default());
        let root_node = arena.get_mut(root).expect("root");
        root_node.create_single_child_node(Move::new(a0, a1));
        root_node.edge_mut(0).set_p(0.25);
        assert!(root_node.try_start_score_update());
        root_node.finalize_score_update(0.2, 0.4, 0.0, 1);

        let child = arena.spawn_child(root, 0);
        let child_node = arena.get_mut(child).expect("child");
        assert!(child_node.try_start_score_update());
        child_node.finalize_score_update(0.0, 0.0, 0.0, 1);

        let actual = get_fpu(&params, arena.get(root).expect("root"), &arena, true, 0.1);
        let visited_policy = arena.get(root).expect("root").visited_policy(&arena);
        let expected = -arena.get(root).expect("root").q(-0.1) - 0.5 * visited_policy.sqrt();

        assert!((actual - expected).abs() < 1e-6, "actual={actual} expected={expected}");
        assert!((actual + 0.41).abs() < 0.02, "actual={actual}");
    }
}
