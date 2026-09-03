//! PUCT / FPU / virtual mean。只算该选哪条边与挂多少 μ；不 `reserve`、不 `descend`。
use xiangqi_core::Move;

use super::Edge;
use super::param::SearchParams;
use super::tree::EdgeStats;
use crate::utils::fastmath::fast_log;

/// cPUCT：常数项加上随访问数缓慢增长的对数项。
///
/// 所有节点使用同一条曲线；根不再有独立初值或独立参数。
/// 默认参数下 `C(0)=1.25`、`C(50k)≈4.5`。
/// `cpuct_base` 越小，增长越早开始；`cpuct_factor` 控制增长幅度。
pub fn compute_cpuct(params: SearchParams, visits: u32) -> f32 {
    params.cpuct + params.cpuct_factor * fast_log((visits as f32 + params.cpuct_base) / params.cpuct_base)
}

/// reduction 越大，未知 edge 的 FPU 越低，越晚获得首次选择。
fn get_fpu(params: &SearchParams, parent_q: f32, edges: &[Edge]) -> f32 {
    -parent_q - params.fpu_reduction * visited_policy(edges).sqrt()
}

/// 已访问边的 prior 之和，供 FPU 缩放使用。
fn visited_policy(edges: &[Edge]) -> f32 {
    edges
        .iter()
        .filter(|edge| edge.completed_visits() > 0 || edge.in_flight_visits() > 0)
        .map(|edge| edge.prior())
        .sum()
}

/// 树搜索 action Q：读本 edge 的完成样本；可叠加 in-flight virtual mean。
fn completed_selection_q(stats: EdgeStats) -> f32 {
    stats.wl_sum / stats.visits as f32
}

fn edge_utility(edge: &Edge, fpu: f32, use_virtual_mean: bool) -> f32 {
    let (stats, started_visits) = edge.selection_snapshot();
    let completed_q = if stats.visits == 0 {
        fpu
    } else {
        completed_selection_q(stats)
    };
    let in_flight = started_visits.saturating_sub(stats.visits);
    if !use_virtual_mean || in_flight == 0 {
        completed_q
    } else {
        (completed_q * stats.visits as f32 + stats.virtual_wl_sum) / (stats.visits + in_flight) as f32
    }
}

/// 原始 completed evidence 的均值标准误代理，不把 reservation 当作信息。
fn evidence_standard_error(edge: &Edge) -> f32 {
    let stats = edge.stats();
    if stats.visits < 2 {
        return 0.0;
    }
    let mean = stats.wl_sum / stats.visits as f32;
    let variance = (stats.wl_sq_sum / stats.visits as f32 - mean * mean).max(0.0);
    (variance / stats.visits as f32).sqrt()
}

/// 不带 prior 的均值不确定性 bonus。未访问或仅一个样本时没有方差信息；
/// `SE` 自身随 evidence 增加而衰减，不另设人为截断或停止阈值。
pub fn variance_bonus_from_se(visits: u32, standard_error: f32, params: &SearchParams) -> f32 {
    if visits < 2 {
        0.0
    } else {
        params.variance_bonus_scale * standard_error
    }
}

/// reservation 应写入的 virtual mean；`scale==0` 退化为纯 virtual visit。
fn virtual_mean_for_reservation(params: &SearchParams, fpu: f32) -> Option<f32> {
    if params.virtual_mean_fpu_scale > 0.0 {
        Some(params.virtual_mean_fpu_scale * fpu)
    } else {
        None
    }
}

/// 选择 PUCT 最高的 edge，并给出该边应挂的 virtual mean。
///
/// completed N 决定 cPUCT 曲线；started N（含虚拟损失）进入实际 PUCT。
pub(crate) fn select_edge(
    edges: &[Edge],
    parent_completed_visits: u32,
    parent_q: f32,
    depth: usize,
    params: &SearchParams,
    root_move_filter: &[Move],
) -> Option<(usize, Option<f32>)> {
    if edges.is_empty() {
        return None;
    }
    let is_root = depth == 0;
    let children_visits = edges.iter().map(|edge| edge.visits()).sum::<u32>();
    let cpuct = compute_cpuct(*params, parent_completed_visits);
    let u_coeff = cpuct * (children_visits.max(1) as f32).sqrt();
    let fpu = get_fpu(params, parent_q, edges);
    let mut best: Option<(usize, f32)> = None;
    let filter_root_moves = is_root && !root_move_filter.is_empty();
    for (index, edge) in edges.iter().enumerate() {
        if filter_root_moves && !root_move_filter.contains(&edge.mv()) {
            continue;
        }
        let q = edge_utility(edge, fpu, params.virtual_mean_fpu_scale > 0.0);
        let u = u_coeff * edge.prior() / (1 + edge.visits()) as f32;
        let standard_error = evidence_standard_error(edge);
        let score = q + u + variance_bonus_from_se(edge.completed_visits(), standard_error, params);
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| (index, virtual_mean_for_reservation(params, fpu)))
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{Move, Square};

    use super::{compute_cpuct, edge_utility, select_edge, variance_bonus_from_se, visited_policy};
    use crate::search::NodeArena;
    use crate::search::param::SearchParams;

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }
    #[test]
    fn defaults_use_the_selected_constant_cpuct() {
        let params = SearchParams::default();
        assert_eq!(params.cpuct, 1.25);
        assert_eq!(params.cpuct_base, 40_000.0);
        assert_eq!(params.cpuct_factor, 4.0);
        assert_eq!(params.fpu_reduction, 0.500);
        assert_eq!(params.decision_lcb_stdevs, 1.0);
        assert_eq!(params.decision_ucb_stdevs, 1.0);
        assert_eq!(params.decision_mix_n_weight, 0.25);
        assert_eq!(params.variance_bonus_scale, 0.1);
        assert_eq!(compute_cpuct(params, 0), params.cpuct);
        // `fast_log` 是热路径近似；默认曲线在 50k 时接近设计目标 4.5。
        assert!((compute_cpuct(params, 50_000) - 4.5).abs() < 0.05);
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
    fn variance_bonus_requires_evidence_and_scales_with_standard_error() {
        let params = SearchParams {
            variance_bonus_scale: 0.8,
            ..SearchParams::default()
        };
        assert_eq!(variance_bonus_from_se(0, 1.0, &params), 0.0);
        assert_eq!(variance_bonus_from_se(1, 1.0, &params), 0.0);
        assert!((variance_bonus_from_se(2, 1.0, &params) - 0.8).abs() < 1e-6);
        assert!((variance_bonus_from_se(2, 0.1, &params) - 0.08).abs() < 1e-6);
    }

    #[test]
    fn in_flight_visit_deflects_selection_without_changing_completed_q() {
        let arena = NodeArena::default();
        let node = arena.get(arena.allocate()).expect("node");
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("b2", "b3"), 0.6), (mv("c3", "c4"), 0.4)]);
        let edges = node.edges();
        let params = SearchParams {
            virtual_mean_fpu_scale: 0.0,
            ..SearchParams::default()
        };
        assert_eq!(
            select_edge(&edges, 0, 0.0, 0, &params, &[]).map(|(index, _)| index),
            Some(0)
        );

        let reservation = node.reserve_edge(0).expect("first edge");
        assert_eq!(
            select_edge(&edges, 0, 0.0, 0, &params, &[]).map(|(index, _)| index),
            Some(1)
        );
        reservation.cancel();
        assert_eq!(edges[0].completed_visits(), 0);
    }

    #[test]
    fn root_move_filter_restricts_selection() {
        let arena = NodeArena::default();
        let node = arena.get(arena.allocate()).expect("node");
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("b2", "b3"), 0.9), (mv("c3", "c4"), 0.1)]);
        let edges = node.edges();
        let params = SearchParams::default();
        let filter = [mv("c3", "c4")];
        assert_eq!(
            select_edge(&edges, 0, 0.0, 0, &params, &filter).map(|(index, _)| index),
            Some(1)
        );
    }

    #[test]
    fn visited_edge_utility_reads_edge_q() {
        let arena = NodeArena::default();
        let node = arena.get(arena.allocate()).expect("node");
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("a0", "a1"), 1.0)]);
        let edges = node.edges();
        let edge = &edges[0];
        node.reserve_edge(0).expect("res").complete(0.5);
        assert!((edge_utility(edge, 0.0, false) - 0.5).abs() < 1e-6);
        assert!((visited_policy(std::slice::from_ref(edge)) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn virtual_fpu_mean_is_temporary_action_q_only() {
        let arena = NodeArena::default();
        let node = arena.get(arena.allocate()).expect("node");
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(mv("a0", "a1"), 1.0)]);
        let edges = node.edges();
        let edge = &edges[0];
        node.reserve_edge(0).expect("completed evidence").complete(0.8);
        assert!((edge_utility(edge, -0.3, true) - 0.8).abs() < 1e-6);

        let reservation = node
            .reserve_edge_with_virtual_mean(0, Some(-0.3))
            .expect("virtual mean reservation");
        assert!((edge_utility(edge, -0.3, true) - 0.25).abs() < 1e-6);
        reservation.cancel();

        assert!((edge_utility(edge, -0.3, true) - 0.8).abs() < 1e-6);
        let stats = edge.stats();
        assert_eq!(stats.visits, 1);
        assert_eq!(stats.virtual_wl_sum, 0.0);
    }
}
