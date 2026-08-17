//! Stream 的选择参数。
//!
//! LC3 Policy 描述 worker/policy 架构，但未公开具体 PUCT 公式。本仓 PUCT 形状
//! 历史上参考过 px0；当前默认探索强度由 X7 固定时间实验选定，FPU 采用小网络
//! 基线。这不是 px0 classic search 的等价移植。

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
    /// `Some(scale)` 时 reservation 暂时以 `scale * FPU` 进入 action Q；`None` 是纯
    /// virtual visit。当前实战基线为 `Some(1.0)`。
    pub virtual_mean_fpu_scale: Option<f32>,
    /// 根最终选边的 LCB 半径倍数；0 表示退回既有 N→Q→P 排名。
    pub lcb_stdevs: f32,
    /// LCB 候选至少须达到 N 第一候选的这一 completed-N 比例。
    pub lcb_min_visit_fraction: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            // 所有节点共用同一条曲线；参数由固定节点分流实验选定。
            cpuct: 1.75,
            cpuct_base: 40_000.0,
            cpuct_factor: 4.0,
            // 小网络可能有系统性偏差；降低未知 edge 的首次进入门槛。
            fpu_reduction: 0.200,
            virtual_mean_fpu_scale: None,
            // KataGo 搜索参数的经验起点；只用于根最终 Decision，不参与 PUCT。
            lcb_stdevs: 5.0,
            lcb_min_visit_fraction: 0.15,
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
        assert!(
            self.cpuct_factor.is_finite() && self.cpuct_factor >= 0.0,
            "stream cpuct factor must be finite and non-negative"
        );
        assert!(
            self.fpu_reduction.is_finite() && self.fpu_reduction >= 0.0,
            "stream FPU reduction must be finite and non-negative"
        );
        if let Some(scale) = self.virtual_mean_fpu_scale {
            assert!(
                scale.is_finite() && scale >= 0.0,
                "stream virtual mean FPU scale must be finite and non-negative"
            );
        }
        assert!(
            self.lcb_stdevs.is_finite() && self.lcb_stdevs >= 0.0,
            "stream LCB stdevs must be finite and non-negative"
        );
        assert!(
            self.lcb_min_visit_fraction.is_finite() && (0.0..=1.0).contains(&self.lcb_min_visit_fraction),
            "stream LCB minimum visit fraction must be within [0, 1]"
        );
    }
}

/// cPUCT：常数项加上随访问数缓慢增长的对数项。
///
/// 所有节点使用同一条曲线；根不再有独立初值或独立参数。
/// 默认参数下 `C(0)=1.75`、`C(50k)≈5`。
/// `cpuct_base` 越小，增长越早开始；`cpuct_factor` 控制增长幅度。
/// 形状历史上参考过常见 PUCT 实现；默认参数由 X7 实验选定。
pub(crate) fn compute_cpuct(params: SearchParams, visits: u32) -> f32 {
    params.cpuct + params.cpuct_factor * fast_log((visits as f32 + params.cpuct_base) / params.cpuct_base)
}

/// reduction 越大，未知 edge 的 FPU 越低，越晚获得首次选择。
pub(crate) fn get_fpu(repository: &NodeRepository, params: &SearchParams, parent_q: f32, edges: &[Arc<Edge>]) -> f32 {
    -parent_q - params.fpu_reduction * visited_policy(repository, edges).sqrt()
}

// 未定义并验证的能力不预留字段：OOO evaluation、prefetch、tree-batch gather 等。

// PUCT / FPU / edge scoring 形状历史上参考过常见 MCTS 实现；默认参数与
// `draw_score=0`、in-flight 计入 edge visit 等约定以本仓 stream 为准。

use std::sync::Arc;

use xiangqi_core::Move;

use super::{Edge, NodeRepository};

/// stream backpropagation 使用的紧凑 WDL 更新。
///
/// - `wl_sum`：走子方 / incoming-edge 视角（非 NN 原始 STM）
/// - `draw_sum`：和棋分量
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValueDelta {
    pub visits: u32,
    pub wl_sum: f32,
    /// `wl²` 聚合。仅用于根 LCB 的 value dispersion，不参与 Q / PUCT。
    pub wl_sq_sum: f32,
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
            wl_sq_sum: wl * wl,
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
            wl_sq_sum: self.wl_sq_sum,
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
            wl_sq_sum: self.wl_sq_sum + other.wl_sq_sum,
            draw_sum: self.draw_sum + other.draw_sum,
            m_sum: self.m_sum + other.m_sum,
        }
    }

    /// 同一个 NN / terminal 结果代表多个 logical visit 时的等权展开。
    pub fn repeated(self, visits: u32) -> Self {
        assert_eq!(self.visits, 1, "only a single sample can be repeated");
        Self {
            visits,
            wl_sum: self.wl_sum * visits as f32,
            wl_sq_sum: self.wl_sq_sum * visits as f32,
            draw_sum: self.draw_sum * visits as f32,
            m_sum: self.m_sum * visits as f32,
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

/// 已访问边的 prior 之和，供 FPU 缩放使用。
pub(crate) fn visited_policy(repository: &NodeRepository, edges: &[Arc<Edge>]) -> f32 {
    edges
        .iter()
        .filter(|edge| {
            edge.completed_visits() > 0
                || edge
                    .child_key()
                    .and_then(|key| repository.get(key))
                    .is_some_and(|child| child.completed_visits() > 0)
        })
        .map(|edge| edge.prior())
        .sum()
}

/// MCGS action Q 优先读取共享 child 的当前 Q：即使这条入边尚无本地 N，只要转置 child
/// 已完成过，也已有可复用的 state evidence。FPU 只用于尚无 child Q 的真正未知边。
/// 重复、连将/追击和 rule60 的路径终局则只保留为这一次 edge visit 的本地样本。访问数仍
/// 严格属于 edge，不能读取 child N。
pub(crate) fn edge_utility(repository: &NodeRepository, edge: &Edge, fpu: f32, use_virtual_mean: bool) -> f32 {
    let (stats, started_visits) = edge.selection_snapshot();
    let completed_q = if stats.visits == 0 {
        edge.child_key()
            .and_then(|key| repository.get(key))
            .filter(|child| child.completed_visits() > 0)
            .map_or(fpu, |child| child.q())
    } else {
        let propagated = stats.visits.saturating_sub(stats.local_leaf.visits);
        if propagated == 0 {
            stats.local_leaf.wl_sum / stats.visits as f32
        } else if let Some(child) = edge.child_key().and_then(|key| repository.get(key)) {
            (stats.local_leaf.wl_sum + child.q() * propagated as f32) / stats.visits as f32
        } else {
            fpu
        }
    };
    let in_flight = started_visits.saturating_sub(stats.visits);
    if !use_virtual_mean || in_flight == 0 {
        completed_q
    } else {
        (completed_q * stats.visits as f32 + stats.virtual_wl_sum) / (stats.visits + in_flight) as f32
    }
}

/// 选择 PUCT 最高的 edge。
///
/// completed N 决定 cPUCT 曲线；U 的根号项则使用所有 child 的 started N。这样
/// pending reservation 同时计入父节点已分配预算与该 edge 的分母，等价于 batch 内
/// 临时树继续执行串行 PUCT，而不是只把 virtual visit 压低某一条边的 U。
pub(crate) fn select_edge(
    repository: &NodeRepository,
    edges: &[Arc<Edge>],
    parent_completed_visits: u32,
    parent_q: f32,
    depth: usize,
    params: &SearchParams,
    root_move_filter: &[Move],
) -> Option<(usize, f32)> {
    if edges.is_empty() {
        return None;
    }
    let is_root = depth == 0;
    let children_visits = edges
        .iter()
        .fold(0_u32, |total, edge| total.saturating_add(edge.visits()));
    let cpuct = compute_cpuct(*params, parent_completed_visits);
    let u_coeff = cpuct * (children_visits.max(1) as f32).sqrt();
    let fpu = get_fpu(repository, params, parent_q, edges);
    let mut best: Option<(usize, f32)> = None;
    let filter_root_moves = is_root && !root_move_filter.is_empty();
    for (index, edge) in edges.iter().enumerate() {
        // 这条合法着会闭合 shared-Q 图环，已被 repository 永久排除。它不是
        // history 终局，也不应继续占用 PUCT / reservation。
        if edge.topology_pruned() {
            continue;
        }
        if filter_root_moves && !root_move_filter.contains(&edge.mv()) {
            continue;
        }
        let q = edge_utility(repository, edge, fpu, params.virtual_mean_fpu_scale.is_some());
        let score = q + u_coeff * edge.prior() / (1 + edge.visits()) as f32;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| (index, fpu))
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{Move, Square};

    use super::{SearchParams, ValueDelta, compute_cpuct, edge_utility, select_edge, visited_policy};
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
        assert_eq!(params.cpuct, 1.75);
        assert_eq!(params.cpuct_base, 40_000.0);
        assert_eq!(params.cpuct_factor, 4.0);
        assert_eq!(params.fpu_reduction, 0.200);
        assert_eq!(params.lcb_stdevs, 5.0);
        assert_eq!(params.lcb_min_visit_fraction, 0.15);
        assert_eq!(compute_cpuct(params, 0), params.cpuct);
        // `fast_log` 是热路径近似；默认曲线在 50k 时接近设计目标 5。
        assert!((compute_cpuct(params, 50_000) - 5.0).abs() < 0.05);
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
        assert_eq!(
            select_edge(&repo, &edges, 0, 0.0, 0, &params, &[]).map(|(index, _)| index),
            Some(0)
        );

        let reservation = node.reserve_edge(0).expect("first edge");
        assert_eq!(
            select_edge(&repo, &edges, 0, 0.0, 0, &params, &[]).map(|(index, _)| index),
            Some(1)
        );
        reservation.cancel();
        assert_eq!(edges[0].completed_visits(), 0);
    }

    #[test]
    fn virtual_visits_also_raise_the_parent_puct_numerator() {
        let repo = NodeRepository::default();
        let parent_key = NodeKey::board(51);
        let parent = repo.get_or_insert(parent_key);
        assert!(parent.try_begin_evaluation());
        parent.publish_edges(vec![(mv("b2", "b3"), 0.9), (mv("c3", "c4"), 0.1)]);
        let edges = parent.edges();

        for (edge, key, q) in [
            (&edges[0], NodeKey::board(52), 0.0),
            (&edges[1], NodeKey::board(53), 0.15),
        ] {
            edge.bind_child_key(key);
            repo.get_or_insert(key).set_graph_value(ValueDelta::one(q, 0.0));
            repo.recompute_graph_node(key);
        }
        let reservations: Vec<_> = (0..4).map(|_| parent.reserve_edge(0).expect("virtual visit")).collect();
        let params = SearchParams {
            cpuct: 1.0,
            cpuct_factor: 0.0,
            ..SearchParams::default()
        };

        // sqrt(started children N)=2 makes the high-prior edge win; with the
        // old completed-only numerator (=1), the 0.15-Q edge incorrectly won.
        assert_eq!(
            select_edge(&repo, &edges, 1, 0.0, 0, &params, &[]).map(|(index, _)| index),
            Some(0)
        );
        for reservation in reservations {
            reservation.cancel();
        }
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
        assert_eq!(
            select_edge(&repo, &edges, 0, 0.0, 0, &params, &filter).map(|(index, _)| index),
            Some(1)
        );
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
        assert!((edge_utility(&repo, &edge, 0.0, false) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unvisited_transposition_uses_shared_child_q_not_fpu() {
        let repo = NodeRepository::default();
        let key = NodeKey::board(40);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("a0", "a1"), 1.0)]);
        let edge = node.edges()[0].clone();
        let child_key = NodeKey::board(41);
        edge.bind_child_key(child_key);
        repo.get_or_insert(child_key).set_graph_value(ValueDelta::one(0.5, 0.0));
        repo.recompute_graph_node(child_key);

        assert_eq!(edge.completed_visits(), 0);
        assert!((edge_utility(&repo, &edge, -0.4, false) - 0.5).abs() < 1e-6);
        assert!((visited_policy(&repo, &[edge]) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn virtual_fpu_mean_is_temporary_action_q_only() {
        let repo = NodeRepository::default();
        let key = NodeKey::board(50);
        let node = repo.get_or_insert(key);
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("a0", "a1"), 1.0)]);
        let edge = node.edges()[0].clone();
        let child_key = NodeKey::board(51);
        edge.bind_child_key(child_key);
        repo.get_or_insert(child_key).set_graph_value(ValueDelta::one(0.8, 0.0));
        repo.recompute_graph_node(child_key);
        node.reserve_edge(0).expect("completed evidence").complete();
        assert!((edge_utility(&repo, &edge, -0.3, true) - 0.8).abs() < 1e-6);

        let reservation = node
            .reserve_edge_visits(0, 1, Some(-0.3))
            .expect("virtual mean reservation");
        assert!((edge_utility(&repo, &edge, -0.3, true) - 0.25).abs() < 1e-6);
        reservation.cancel();

        assert!((edge_utility(&repo, &edge, -0.3, true) - 0.8).abs() < 1e-6);
        let stats = edge.stats();
        assert_eq!(stats.visits, 1);
        assert_eq!(stats.virtual_wl_sum, 0.0);
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
            .complete_local_leaf(ValueDelta::one(0.6, 0.0));
        node.reserve_edge(0).expect("child reservation").complete();

        assert!((edge_utility(&repo, &edge, 0.0, false) - 0.2).abs() < 1e-6);
    }
}
