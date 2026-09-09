//! Backprop 算法：`complete` reservation + `add_delta`。
//!
//! MCTS 回传实验改这里。worker 循环壳在 `workerpool`。

use std::collections::HashMap;

use super::workerpool::BackpropEvent;
use super::{NodeArena, NodeId};

/// - `visits`：多份 `one()` 样本的合计，不是一次 reservation 携带的 K
/// - `wl_sum`：走子方 / incoming-edge 视角（非 NN 原始 STM）
/// - `draw_sum`：和棋分量
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValueDelta {
    pub visits: u32,
    pub wl_sum: f32,
    pub draw_sum: f32,
    pub m_sum: f32,
}

impl ValueDelta {
    pub fn one(wl: f32, draw: f32, plies_left: f32) -> Self {
        debug_assert!(plies_left >= 0.0, "plies-left must be non-negative");
        debug_assert!((-1.0..=1.0).contains(&wl), "WDL wl must be normalized");
        debug_assert!((0.0..=1.0).contains(&draw), "WDL draw must be normalized");
        Self { visits: 1, wl_sum: wl, draw_sum: draw, m_sum: plies_left }
    }

    pub fn to_parent(self) -> Self {
        Self { wl_sum: -self.wl_sum, m_sum: self.m_sum + self.visits as f32, ..self }
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
        if self.visits == 0 { 0.0 } else { self.wl_sum / self.visits as f32 }
    }
}
type NodeDeltaMap = HashMap<NodeId, ValueDelta>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackpropResult {
    pub(crate) completed_playouts: u32,
    pub(crate) completed_depth: u64,
    pub(crate) max_depth: u64,
}

/// 一条 path 不会重复 node，因此直接完成 edge 与写回 node，不需要聚合表。
pub(crate) fn complete_one<S: super::observer::QueueStamp>(
    event: BackpropEvent<S>,
    arena: &NodeArena,
) -> BackpropResult {
    let BackpropEvent { event, value, .. } = event;
    debug_assert_eq!(event.node_path.len(), event.reservations.len() + 1);
    let depth = event.node_path.len() as u64;
    let mut delta = value;
    let mut reservations = event.reservations.into_iter().rev();
    for (node_index, node_id) in event.node_path.into_iter().enumerate().rev() {
        let node = arena.get(node_id).expect("backprop node lives until job drain");
        if let Some((terminal_wl, terminal_draw, terminal_m)) = node.terminal_value() {
            delta = ValueDelta::one(terminal_wl, terminal_draw, terminal_m);
        }
        node.add_delta(delta);
        if node_index == 0 {
            break;
        }
        let reservation = reservations.next().expect("every non-root backprop edge has a reservation");
        reservation.complete(delta.q());
        delta = delta.to_parent();
    }
    BackpropResult { completed_playouts: 1, completed_depth: depth, max_depth: depth }
}

/// 多条 path 的 node 增量合并后一次写入；edge 仍逐层 complete。
pub(crate) fn complete_batch<S: super::observer::QueueStamp>(
    events: impl IntoIterator<Item = BackpropEvent<S>>,
    arena: &NodeArena,
) -> BackpropResult {
    let mut node_deltas = NodeDeltaMap::default();
    let mut result = BackpropResult::default();

    for event in events {
        let BackpropEvent { event, value, .. } = event;
        debug_assert_eq!(event.node_path.len(), event.reservations.len() + 1);
        let depth = event.node_path.len() as u64;
        let mut delta = value;
        let mut reservations = event.reservations.into_iter().rev();
        for (node_index, node_id) in event.node_path.into_iter().enumerate().rev() {
            let node = arena.get(node_id).expect("backprop node lives until job drain");
            if let Some((terminal_wl, terminal_draw, terminal_m)) = node.terminal_value() {
                delta = ValueDelta::one(terminal_wl, terminal_draw, terminal_m);
            }
            node_deltas.entry(node_id).and_modify(|aggregate| *aggregate = aggregate.merge(delta)).or_insert(delta);
            if node_index == 0 {
                break;
            }
            let reservation = reservations.next().expect("every non-root backprop edge has a reservation");
            reservation.complete(delta.q());
            delta = delta.to_parent();
        }
        result.completed_playouts += 1;
        result.completed_depth += depth;
        result.max_depth = result.max_depth.max(depth);
    }

    for (node_id, delta) in node_deltas {
        arena.get(node_id).expect("backprop node lives until job drain").add_delta(delta);
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, STARTPOS_FEN, Square};

    use super::complete_one;
    use crate::search::NodeArena;
    use crate::search::workerpool::{BackpropEvent, SelectEvent};

    #[test]
    fn backprop_completes_every_reservation_with_alternating_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let arena = NodeArena::default();
        let root_id = arena.allocate();
        let root_node = arena.get(root_id).expect("root node");
        assert!(root_node.try_claim());
        let mv = Move::new(Square::parse("b2").expect("b2"), Square::parse("b3").expect("b3"));
        root_node.publish_edges(vec![(mv, 1.0)]);
        let child_id = arena.child_or_create(&root_node.edges()[0]);
        let child = SelectEvent::<crate::search::NoQueueStamp>::at_root(root_id, Arc::clone(&history))
            .descend(child_id, root_node.reserve_edge(0, None).expect("edge"));

        complete_one(
            BackpropEvent::<crate::search::NoQueueStamp>::without_nn_credit(child.into_event(), 0.4, 0.2, 2.0),
            &arena,
        );

        let edge = &root_node.edges()[0];
        assert_eq!(edge.visits(), 1);
        assert_eq!(edge.completed_visits(), 1);
        assert!((edge.q() - 0.4).abs() < f32::EPSILON);
        assert_eq!(root_node.completed_visits(), 1);
        assert!((root_node.q() + 0.4).abs() < f32::EPSILON);
        assert!((root_node.m() - 3.0).abs() < f32::EPSILON);
        let child_node = arena.get(child_id).expect("child node");
        assert_eq!(child_node.completed_visits(), 1);
        assert!((child_node.q() - 0.4).abs() < f32::EPSILON);
        assert!((child_node.m() - 2.0).abs() < f32::EPSILON);
    }
}
