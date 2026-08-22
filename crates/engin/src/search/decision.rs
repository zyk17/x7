//! 搜后根决策：bestmove、MultiPV、LCB、PV。不参与搜索过程。

use xiangqi_core::Move;

use super::param::SearchParams;
use super::{Edge, ExpansionState, Node, NodeArena, NodeId};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootEdgeStats {
    pub mv: Move,
    pub completed_visits: u32,
    pub started_visits: u32,
    pub q: f32,
    pub prior: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootStats {
    pub completed_visits: u32,
    pub q: f32,
    pub draw: f32,
    pub edges: Vec<RootEdgeStats>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootVariation {
    pub wl: f32,
    pub draw: f32,
    pub mate: Option<i32>,
    pub pv: Vec<Move>,
}

pub fn root_stats(arena: &NodeArena, root: NodeId) -> Option<RootStats> {
    let root = arena.get(root)?;
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

fn child<'a>(arena: &'a NodeArena, edge: &Edge) -> Option<&'a Node> {
    edge.child().and_then(|id| arena.get(id))
}

fn mate(child: Option<&Node>) -> Option<i32> {
    let (wl, _, plies) = child?.terminal_value()?;
    (wl != 0.0).then(|| {
        let distance = plies.round() as i32 / 2 + 1;
        if wl > 0.0 { distance } else { -distance }
    })
}

fn edge_lcb(edge: &Edge, stdevs: f32) -> Option<f32> {
    let stats = edge.stats();
    if stats.visits == 0 {
        return None;
    }
    let n = stats.visits as f32;
    let q = stats.wl_sum / n;
    let variance = (stats.wl_sq_sum / n - q * q).max(0.0);
    Some(q - stdevs * (variance / n).sqrt())
}

fn ranked_edges(arena: &NodeArena, root: NodeId, filter: &[Move], params: &SearchParams) -> Vec<std::sync::Arc<Edge>> {
    let Some(node) = arena.get(root) else { return Vec::new() };
    let edges = node.edges();
    let mut ranked: Vec<_> = edges
        .iter()
        .filter(|edge| filter.is_empty() || filter.contains(&edge.mv()))
        .map(std::sync::Arc::clone)
        .collect();
    ranked.sort_unstable_by(|left, right| {
        let terminal_key = |edge: &Edge| {
            child(arena, edge)
                .filter(|_| edge.completed_visits() > 0)
                .and_then(Node::terminal_value)
                .map(|(wl, _, plies)| (wl.signum() as i8, plies))
                .unwrap_or((0, 0.0))
        };
        let (left_terminal, left_plies) = terminal_key(left);
        let (right_terminal, right_plies) = terminal_key(right);
        right_terminal
            .cmp(&left_terminal)
            .then_with(|| match left_terminal {
                1 => left_plies.total_cmp(&right_plies),
                -1 => right_plies.total_cmp(&left_plies),
                _ => std::cmp::Ordering::Equal,
            })
            .then_with(|| right.completed_visits().cmp(&left.completed_visits()))
            .then_with(|| right.q().total_cmp(&left.q()))
            .then_with(|| right.prior().total_cmp(&left.prior()))
    });
    if params.lcb_stdevs > 0.0 && ranked.len() > 1 {
        let baseline = &ranked[0];
        let min_visits = baseline.completed_visits() as f32 * params.lcb_min_visit_fraction;
        let decisive_terminal = child(arena, baseline)
            .and_then(Node::terminal_value)
            .is_some_and(|(wl, _, _)| wl != 0.0);
        if !decisive_terminal
            && let Some(base_lcb) = edge_lcb(baseline, params.lcb_stdevs)
            && let Some((index, _)) = ranked
                .iter()
                .enumerate()
                .skip(1)
                .filter_map(|(index, edge)| {
                    (edge.completed_visits() as f32 >= min_visits)
                        .then(|| edge_lcb(edge, params.lcb_stdevs).map(|lcb| (index, lcb)))?
                })
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .filter(|(_, lcb)| *lcb >= base_lcb + 0.01)
        {
            ranked.swap(0, index);
        }
    }
    ranked
}

fn best_edge(arena: &NodeArena, root: NodeId, filter: &[Move], params: &SearchParams) -> Option<std::sync::Arc<Edge>> {
    ranked_edges(arena, root, filter, params).into_iter().next()
}

pub(crate) fn best_move_filtered_with_params(
    arena: &NodeArena,
    root: NodeId,
    root_is_black: bool,
    filter: &[Move],
    params: &SearchParams,
) -> Option<Move> {
    best_edge(arena, root, filter, params).map(|edge| orient_move(edge.mv(), root_is_black))
}

pub fn best_move(arena: &NodeArena, root: NodeId, root_is_black: bool) -> Option<Move> {
    best_move_filtered_with_params(arena, root, root_is_black, &[], &SearchParams::default())
}

pub(crate) fn best_mate_with_params(
    arena: &NodeArena,
    root: NodeId,
    filter: &[Move],
    params: &SearchParams,
) -> Option<i32> {
    let node = arena.get(root)?;
    if let Some((wl, _, plies)) = node.terminal_value() {
        let distance = plies.round() as i32 / 2 + 1;
        return (wl != 0.0).then_some(if wl < 0.0 { distance } else { -distance });
    }
    let edge = best_edge(arena, root, filter, params)?;
    mate(child(arena, &edge))
}

fn pv_from_edge(arena: &NodeArena, first: &Edge, root_is_black: bool) -> Vec<Move> {
    let mut pv = vec![orient_move(first.mv(), root_is_black)];
    let Some(mut next) = first.child() else { return pv };
    let mut flip = !root_is_black;
    while let Some(node) = arena.get(next) {
        if node.completed_visits() == 0 || node.expansion_state() != ExpansionState::Expanded {
            break;
        }
        let Some(edge) = best_edge(arena, next, &[], &SearchParams::default()) else {
            break;
        };
        pv.push(orient_move(edge.mv(), flip));
        let Some(child) = edge.child() else { break };
        next = child;
        flip = !flip;
    }
    pv
}

pub(crate) fn principal_variation_with_params(
    arena: &NodeArena,
    root: NodeId,
    root_is_black: bool,
    filter: &[Move],
    params: &SearchParams,
) -> Vec<Move> {
    best_edge(arena, root, filter, params).map_or_else(Vec::new, |edge| pv_from_edge(arena, &edge, root_is_black))
}

pub fn principal_variation(arena: &NodeArena, root: NodeId, root_is_black: bool) -> Vec<Move> {
    principal_variation_with_params(arena, root, root_is_black, &[], &SearchParams::default())
}

pub(crate) fn root_variations(
    arena: &NodeArena,
    root: NodeId,
    root_is_black: bool,
    filter: &[Move],
    max_pv: usize,
    params: &SearchParams,
) -> Vec<RootVariation> {
    let Some(node) = arena.get(root) else { return Vec::new() };
    let default_wl = (-node.q()).clamp(-1.0, 1.0);
    let default_draw = node.draw().clamp(0.0, 1.0);
    ranked_edges(arena, root, filter, params)
        .into_iter()
        .take(max_pv)
        .map(|edge| {
            let visited = edge.completed_visits() > 0;
            let child = child(arena, &edge);
            RootVariation {
                wl: if visited { edge.q() } else { default_wl },
                draw: child.filter(|_| visited).map_or(default_draw, Node::draw),
                mate: visited.then(|| mate(child)).flatten(),
                pv: pv_from_edge(arena, &edge, root_is_black),
            }
        })
        .collect()
}
