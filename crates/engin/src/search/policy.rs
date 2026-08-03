//! Stream 的选择参数。
//!
//! LC3 Policy 描述 worker/policy 架构，但未公开具体 PUCT 公式。因此在有
//! stream 原生公式前，默认值保持项目批准的 X7 策略
//! （px0 `src/search/classic/search.cc:408-433`）。

use crate::utils::fastmath::fast_log;

/// Stream 自己拥有的最小选择参数集。
///
/// 不包含 root 专用参数、absolute FPU、draw score、contempt 或旧
/// task-worker 参数；stream 固定使用中性和棋分数与 reduction FPU。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParams {
    pub cpuct: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub fpu_reduction: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            cpuct: 1.0,
            cpuct_base: 38_739.0,
            cpuct_factor: 3.894,
            fpu_reduction: 0.220,
        }
    }
}

impl SearchParams {
    pub(crate) fn validate(self) {
        assert!(
            self.cpuct.is_finite() && self.cpuct >= 0.0,
            "stream cpuct must be finite and non-negative"
        );
        assert!(
            self.cpuct_base.is_finite() && self.cpuct_base > 0.0,
            "stream cpuct base must be finite and positive"
        );
        assert!(self.cpuct_factor.is_finite(), "stream cpuct factor must be finite");
        assert!(
            self.fpu_reduction.is_finite() && self.fpu_reduction >= 0.0,
            "stream FPU reduction must be finite and non-negative"
        );
    }
}

/// 项目批准的 PUCT 对齐；等待公开的 LC3 公式。
pub(crate) fn compute_cpuct(params: SearchParams, visits: u32) -> f32 {
    if params.cpuct_factor == 0.0 {
        params.cpuct
    } else {
        params.cpuct + params.cpuct_factor * fast_log((visits as f32 + params.cpuct_base) / params.cpuct_base)
    }
}

// 尚未定义并验证事件/生命周期语义的能力不预留字段：OOO evaluation、MultiPV 与 DAG reuse。

// stream 搜索策略参考 px0 `ComputeCpuct` / `GetFpu` / PUCT edge scoring
// （`src/search/classic/search.cc:408-433`）。`draw_score` 固定为 0（Q 为走子方视角的
// 原始 `wl`）；edge visit 计数包含 in-flight reservation（LC3 node structure）。

use std::sync::Arc;

use xiangqi_core::Move;

use super::{Edge, Node};

/// stream backpropagation 使用的紧凑 WDL 更新。
///
/// - `wl_sum` 对应 px0 `wl_`（走子方 / incoming-edge 视角，非 NN 原始 STM）
/// - `draw_sum` 对应 px0 `d_`
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValueDelta {
    pub visits: u32,
    pub wl_sum: f32,
    pub draw_sum: f32,
    pub m_sum: f32,
}

impl ValueDelta {
    pub fn one(wl: f32, draw: f32) -> Self {
        assert!((-1.0..=1.0).contains(&wl), "WDL wl must be normalized");
        assert!((0.0..=1.0).contains(&draw), "WDL draw must be normalized");
        Self {
            visits: 1,
            wl_sum: wl,
            draw_sum: draw,
            m_sum: 0.0,
        }
    }

    pub fn with_plies_left(wl: f32, draw: f32, plies_left: f32) -> Self {
        assert!(plies_left >= 0.0, "plies-left must be non-negative");
        Self {
            m_sum: plies_left,
            ..Self::one(wl, draw)
        }
    }

    pub fn for_parent(self) -> Self {
        Self {
            visits: self.visits,
            wl_sum: -self.wl_sum,
            draw_sum: self.draw_sum,
            m_sum: self.m_sum,
        }
    }

    pub fn one_ply_up(self) -> Self {
        Self {
            m_sum: self.m_sum + self.visits as f32,
            ..self
        }
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            visits: self.visits + other.visits,
            wl_sum: self.wl_sum + other.wl_sum,
            draw_sum: self.draw_sum + other.draw_sum,
            m_sum: self.m_sum + other.m_sum,
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

/// stream edge 上的 px0 `Node::GetVisitedPolicy`。
pub fn visited_policy(edges: &[Arc<Edge>]) -> f32 {
    edges
        .iter()
        .filter(|edge| edge.completed_visits() > 0)
        .map(|edge| edge.prior())
        .sum()
}

/// `draw_score = 0` 时的 px0 `GetFpu`（`search.cc:408-424`）。
pub fn get_fpu(params: &SearchParams, parent_wl: f32, edges: &[Arc<Edge>]) -> f32 {
    -parent_wl - params.fpu_reduction * visited_policy(edges).sqrt()
}

/// 已访问 edge 返回 Q，未完成访问则返回 FPU。
pub fn edge_utility(edge: &Edge, fpu: f32) -> f32 {
    if edge.completed_visits() == 0 { fpu } else { edge.q() }
}

fn root_filter_allows(is_root: bool, filter: &[Move], mv: Move) -> bool {
    !is_root || filter.is_empty() || filter.contains(&mv)
}

/// 选择 px0 风格 PUCT 最高的 edge。
///
/// `root_move_filter` 对应 px0 `Search::root_move_filter_` 与 UCI `go searchmoves`
/// （`search.cc:53-58,721-724,1668-1739`）；空数组表示不过滤。
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

/// 供已持有 parent node 的 Gather 调用点使用的便利函数。
pub fn select_edge_from_node(
    node: &Node,
    depth: usize,
    params: &SearchParams,
    root_move_filter: &[Move],
) -> Option<usize> {
    select_edge(
        &node.edges(),
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

    use super::{SearchParams, ValueDelta, compute_cpuct, edge_utility, select_edge};
    use crate::search::{NodeKey, NodeRepository};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    #[test]
    fn parent_delta_flips_wl_not_draw() {
        let leaf = ValueDelta::one(0.6, 0.2);
        assert_eq!(leaf.for_parent(), ValueDelta::one(-0.6, 0.2));
    }

    #[test]
    fn defaults_preserve_the_approved_x7_policy() {
        let params = SearchParams::default();
        assert_eq!(params.cpuct, 1.0);
        assert_eq!(params.cpuct_base, 38_739.0);
        assert_eq!(params.cpuct_factor, 3.894);
        assert_eq!(params.fpu_reduction, 0.220);
        assert_eq!(compute_cpuct(params, 0), params.cpuct);
    }

    #[test]
    fn zero_cpuct_factor_keeps_the_initial_value() {
        let params = SearchParams {
            cpuct: 1.25,
            cpuct_factor: 0.0,
            ..SearchParams::default()
        };
        assert_eq!(compute_cpuct(params, 10_000), 1.25);
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
