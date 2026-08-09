//! Stream 的选择参数。
//!
//! LC3 Policy 描述 worker/policy 架构，但未公开具体 PUCT 公式。因此在有
//! stream 原生公式前，PUCT 形状参考 px0。当前默认的常数探索强度由 X7 固定时间
//! 实验选定；FPU 强度采用 LC0 对照值（px0 `src/search/classic/search.cc:408-433`）。

use crate::utils::fastmath::fast_log;

/// Stream 自己拥有的最小选择参数集。
///
/// 不包含 root 专用参数、absolute FPU、draw score、contempt 或旧 task-worker 参数；
/// stream 固定使用中性和棋分数与 reduction FPU。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParams {
    pub cpuct: f32,
    pub cpuct_base: f32,   // 增长何时开始。更小 → 更早、更快变宽；更大 → 更久保持利用 Q。
    pub cpuct_factor: f32, // 增长幅度。更大 → 后期更强地向 PUCT/P 分流；更小 → 后期更容易让已验证的高 Q 分支继续积累 N。
    pub fpu_reduction: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            // 固定时间实验基线：常数 e 足以验证低先验候选，又不会持续打散已有证据。
            // `f32` 的最接近 e；factor 为 0 时 cpuct_base 不参与计算。
            cpuct: 2.718_281_7,
            cpuct_base: 38_739.0,
            cpuct_factor: 0.0,
            fpu_reduction: 0.330,
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

use super::{Edge, Node, NodeRepository};

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

/// MCGS action Q 由该 edge 已观察到的两类 evidence 组成：普通访问读取 shared child
/// 的当前 Q；重复、连将/追击和 rule60 的路径终局只保留为这一次 edge visit 的本地样本。
/// 访问数仍严格属于 edge，不能读取 child N。
pub(crate) fn edge_utility(repository: &NodeRepository, edge: &Edge, fpu: f32) -> f32 {
    let stats = edge.completed_stats();
    if stats.visits == 0 {
        return fpu;
    }
    let propagated = stats.visits.saturating_sub(stats.local_terminal.visits);
    let child_value = if propagated == 0 {
        0.0
    } else {
        let Some(child) = edge.child_key().and_then(|key| repository.get(key)) else {
            return fpu;
        };
        child.q() * propagated as f32
    };
    (stats.local_terminal.wl_sum + child_value) / stats.visits as f32
}

/// 选择 px0 风格 PUCT 最高的 edge。
///
/// `root_move_filter` 对应 px0 `Search::root_move_filter_` 与 UCI `go searchmoves`
/// （`search.cc:53-58,721-724,1668-1739`）；空数组表示不过滤。
pub fn select_edge(
    repository: &NodeRepository,
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
    let children_visits = parent_completed_visits.saturating_sub(1);
    let cpuct = compute_cpuct(*params, parent_completed_visits);
    let u_coeff = cpuct * (children_visits.max(1) as f32).sqrt();
    let fpu = get_fpu(params, parent_wl, edges);
    let mut best: Option<(usize, f32)> = None;
    let filter_root_moves = is_root && !root_move_filter.is_empty();
    for (index, edge) in edges.iter().enumerate() {
        if filter_root_moves && !root_move_filter.contains(&edge.mv()) {
            continue;
        }
        let q = edge_utility(repository, edge, fpu);
        let score = q + u_coeff * edge.prior() / (1 + edge.visits()) as f32;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
}

/// 供已持有 parent node 的 Gather 调用点使用的便利函数。
pub fn select_edge_from_node(
    repository: &NodeRepository,
    node: &Node,
    depth: usize,
    params: &SearchParams,
    root_move_filter: &[Move],
) -> Option<usize> {
    select_edge(
        repository,
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
    fn defaults_use_the_selected_constant_cpuct() {
        let params = SearchParams::default();
        assert_eq!(params.cpuct, 2.718_281_7);
        assert_eq!(params.cpuct_base, 38_739.0);
        assert_eq!(params.cpuct_factor, 0.0);
        assert_eq!(params.fpu_reduction, 0.330);
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
        let key = NodeKey::board(9);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("b2", "b3"), 0.6), (mv("c3", "c4"), 0.4)]);
        let edges = node.edges();
        let params = SearchParams::default();
        assert_eq!(select_edge(&repo, &edges, 0, 0.0, 0, &params, &[]), Some(0));

        let reservation = node.reserve_edge(0).expect("first edge");
        assert_eq!(select_edge(&repo, &edges, 0, 0.0, 0, &params, &[]), Some(1));
        reservation.cancel();
        assert_eq!(edges[0].completed_visits(), 0);
    }

    #[test]
    fn root_move_filter_restricts_selection() {
        let repo = NodeRepository::default();
        let key = NodeKey::board(9);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("b2", "b3"), 0.9), (mv("c3", "c4"), 0.1)]);
        let edges = node.edges();
        let params = SearchParams::default();
        let filter = [mv("c3", "c4")];
        assert_eq!(select_edge(&repo, &edges, 0, 0.0, 0, &params, &filter), Some(1));
    }

    #[test]
    fn visited_edge_utility_reads_shared_child_q() {
        let repo = NodeRepository::default();
        let key = NodeKey::board(3);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("a0", "a1"), 1.0)]);
        let edge = node.edges()[0].clone();
        let child_key = NodeKey::board(4);
        edge.bind_child_key(child_key);
        repo.get_or_insert(child_key).set_graph_value(ValueDelta::one(0.5, 0.0));
        repo.recompute_graph_node(child_key);
        node.reserve_edge(0).expect("res").complete();
        assert!((edge_utility(&repo, &edge, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn path_terminal_is_one_edge_sample_not_a_permanent_child_override() {
        let repo = NodeRepository::default();
        let key = NodeKey::board(30);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("a0", "a1"), 1.0)]);
        let edge = node.edges()[0].clone();
        let child_key = NodeKey::board(31);
        edge.bind_child_key(child_key);
        let child = repo.get_or_insert(child_key);
        child.set_graph_value(ValueDelta::one(-0.2, 0.0));
        repo.recompute_graph_node(child_key);

        node.reserve_edge(0)
            .expect("terminal reservation")
            .complete_path_terminal(ValueDelta::one(0.6, 0.0));
        node.reserve_edge(0).expect("child reservation").complete();

        assert!((edge_utility(&repo, &edge, 0.0) - 0.2).abs() < 1e-6);
    }
}
