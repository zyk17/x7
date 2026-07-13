//! px0 `search.cc` 中 PUCT/FPU 辅助（`ComputeCpuct`、`GetFpu`、边打分）。

use super::node::{Node, NodeArena};
use super::params::SearchParams;

/// px0 `ComputeCpuct` (`search.cc`)。
pub fn compute_cpuct(params: &SearchParams, n: u32, is_root: bool) -> f32 {
    let init = params.cpuct(is_root);
    let k = params.cpuct_factor(is_root);
    let base = params.cpuct_base(is_root);
    if k == 0.0 {
        init
    } else {
        init + k * ((n as f32 + base) / base).ln()
    }
}

/// px0 `GetFpu` (`search.cc`)。
pub fn get_fpu(params: &SearchParams, node: &Node, arena: &NodeArena, is_root: bool, draw_score: f32) -> f32 {
    let visited_pol = if is_root {
        1.0
    } else {
        node.visited_policy(arena).max(1.0)
    };
    let value = params.fpu_value(is_root);
    if params.fpu_absolute(is_root) {
        value
    } else {
        -node.q(draw_score) - value * visited_pol.sqrt()
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
