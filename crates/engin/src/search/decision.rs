//! 搜后根决策：bestmove、MultiPV、LCB、PV。不参与搜索过程。

use std::ops::Deref;
use std::sync::Arc;
use xiangqi_core::Move;

use super::param::SearchParams;
use super::{Edge, ExpansionState, Node, NodeArena, NodeId};

/// 根节点在既有 completed evidence 上的最终选边规则；不参与 PUCT。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DecisionRule {
    /// 历史默认：按 completed N 最大。
    #[default]
    Auto,
    MaxQ,
    MaxN,
    Lcb,
    Ucb,
    MixNQ,
}

impl DecisionRule {
    pub const fn uci_name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::MaxQ => "MaxQ",
            Self::MaxN => "MaxN",
            Self::Lcb => "Lcb",
            Self::Ucb => "Ucb",
            Self::MixNQ => "MixNQ",
        }
    }

    pub fn parse_uci(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("auto") {
            Some(Self::Auto)
        } else if value.eq_ignore_ascii_case("maxq") {
            Some(Self::MaxQ)
        } else if value.eq_ignore_ascii_case("maxn") {
            Some(Self::MaxN)
        } else if value.eq_ignore_ascii_case("lcb") {
            Some(Self::Lcb)
        } else if value.eq_ignore_ascii_case("ucb") {
            Some(Self::Ucb)
        } else if value.eq_ignore_ascii_case("mixnq") {
            Some(Self::MixNQ)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootEdgeStats {
    pub mv: Move,
    pub completed_visits: u32,
    pub started_visits: u32,
    pub q: f32,
    pub variance: f32,
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

struct EdgeHandle {
    table: Arc<[Edge]>,
    index: usize,
}

impl Deref for EdgeHandle {
    type Target = Edge;

    fn deref(&self) -> &Self::Target {
        &self.table[self.index]
    }
}

struct RankedEdge {
    edge: EdgeHandle,
    visits: u32,
    q: f32,
    prior: f32,
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
            .map(|edge| {
                let stats = edge.stats();
                RootEdgeStats {
                    mv: edge.mv(),
                    completed_visits: stats.visits,
                    started_visits: edge.visits(),
                    q: edge.q(),
                    variance: if stats.visits < 2 {
                        0.0
                    } else {
                        (stats.wl_sq_sum / stats.visits as f32 - edge.q() * edge.q()).max(0.0)
                    },
                    prior: edge.prior(),
                }
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

fn edge_standard_error(edge: &Edge) -> f32 {
    let stats = edge.stats();
    if stats.visits < 2 {
        return 0.0;
    }
    let n = stats.visits as f32;
    let q = stats.wl_sum / n;
    let variance = (stats.wl_sq_sum / n - q * q).max(0.0);
    (variance / n).sqrt()
}

fn rank_by_score(ranked: &mut [RankedEdge], score: impl Fn(&RankedEdge) -> f32) {
    ranked.sort_unstable_by(|left, right| {
        score(right)
            .total_cmp(&score(left))
            .then_with(|| right.visits.cmp(&left.visits))
            .then_with(|| right.q.total_cmp(&left.q))
            .then_with(|| right.prior.total_cmp(&left.prior))
    });
}

fn decision_score(rule: DecisionRule, edge: &RankedEdge, max_visits: u32, params: &SearchParams) -> f32 {
    match rule {
        DecisionRule::Auto | DecisionRule::MaxN => edge.visits as f32,
        DecisionRule::MaxQ => edge.q,
        DecisionRule::Lcb => edge.q - params.decision_lcb_stdevs * edge_standard_error(&edge.edge),
        DecisionRule::Ucb => edge.q + params.decision_ucb_stdevs * edge_standard_error(&edge.edge),
        DecisionRule::MixNQ => edge.q + params.decision_mix_n_weight * edge.visits as f32 / max_visits.max(1) as f32,
    }
}

fn ranked_edges(arena: &NodeArena, root: NodeId, filter: &[Move], params: &SearchParams) -> Vec<EdgeHandle> {
    let Some(node) = arena.get(root) else { return Vec::new() };
    let edges = node.edges();
    let mut ranked: Vec<_> = edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| filter.is_empty() || filter.contains(&edge.mv()))
        .map(|(index, _)| {
            let edge = EdgeHandle {
                table: Arc::clone(&edges),
                index,
            };
            RankedEdge {
                visits: edge.completed_visits(),
                q: edge.q(),
                prior: edge.prior(),
                edge,
            }
        })
        .collect();
    let max_visits = ranked.iter().map(|edge| edge.visits).max().unwrap_or(0);
    rank_by_score(&mut ranked, |edge| {
        decision_score(params.decision_rule, edge, max_visits, params)
    });
    ranked.into_iter().map(|edge| edge.edge).collect()
}

fn best_edge(arena: &NodeArena, root: NodeId, filter: &[Move], params: &SearchParams) -> Option<EdgeHandle> {
    ranked_edges(arena, root, filter, params).into_iter().next()
}

pub fn best_move_with_params(
    arena: &NodeArena,
    root: NodeId,
    root_is_black: bool,
    params: &SearchParams,
) -> Option<Move> {
    best_edge(arena, root, &[], params).map(|edge| orient_move(edge.mv(), root_is_black))
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
    best_move_with_params(arena, root, root_is_black, &SearchParams::default())
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

#[cfg(test)]
mod tests {
    use xiangqi_core::{Move, Square};

    use super::{DecisionRule, NodeArena, SearchParams, best_move_with_params};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    fn complete_samples(node: &super::Node, edge_index: usize, samples: &[f32]) {
        for &sample in samples {
            node.reserve_edge(edge_index).expect("reservation").complete(sample);
        }
    }

    #[test]
    fn explicit_decision_rules_rank_completed_evidence_as_configured() {
        let arena = NodeArena::default();
        let root = arena.allocate();
        let node = arena.get(root).expect("root");
        assert!(node.try_begin_evaluation());
        let first = mv("a0", "a1");
        let second = mv("b0", "b1");
        let third = mv("c0", "c1");
        node.publish_edges(vec![(first, 0.4), (second, 0.3), (third, 0.3)]);
        complete_samples(node, 0, &[0.2; 4]);
        complete_samples(node, 1, &[0.6; 2]);
        complete_samples(node, 2, &[-1.0, 1.0]);

        let select = |rule, mix_weight| {
            best_move_with_params(
                &arena,
                root,
                false,
                &SearchParams {
                    decision_rule: rule,
                    decision_mix_n_weight: mix_weight,
                    decision_lcb_stdevs: 1.0,
                    decision_ucb_stdevs: 1.0,
                    ..SearchParams::default()
                },
            )
        };
        assert_eq!(select(DecisionRule::MaxN, 0.0), Some(first));
        assert_eq!(select(DecisionRule::MaxQ, 0.0), Some(second));
        assert_eq!(select(DecisionRule::Lcb, 0.0), Some(second));
        assert_eq!(select(DecisionRule::Ucb, 0.0), Some(third));
        assert_eq!(select(DecisionRule::MixNQ, 0.5), Some(second));
        assert_eq!(select(DecisionRule::MixNQ, 1.0), Some(first));
    }

    #[test]
    fn auto_is_the_max_n_baseline() {
        let arena = NodeArena::default();
        let root = arena.allocate();
        let node = arena.get(root).expect("root");
        assert!(node.try_begin_evaluation());
        let first = mv("a0", "a1");
        let second = mv("b0", "b1");
        node.publish_edges(vec![(first, 0.5), (second, 0.5)]);
        complete_samples(node, 0, &[0.2, 0.2]);
        complete_samples(node, 1, &[1.0]);

        assert_eq!(
            best_move_with_params(
                &arena,
                root,
                false,
                &SearchParams {
                    ..SearchParams::default()
                },
            ),
            Some(first)
        );
    }
}
