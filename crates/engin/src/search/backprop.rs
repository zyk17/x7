//! Backprop 算法：`complete` reservation + `add_delta`。
//!
//! MCTS 回传实验改这里。worker 循环壳在 `workerpool`。

use std::collections::HashMap;

use super::workerpool::BackpropEvent;
use super::{Node, NodeArena, NodeId, ValueDelta};

type NodeDeltaMap = HashMap<NodeId, ValueDelta>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackpropResult {
    pub(crate) completed_playouts: u32,
    pub(crate) completed_depth: u64,
    pub(crate) max_depth: u64,
}

/// 路径增量回传：同路径 node 合并 delta，再一次写入；edge 按层 complete。
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
            if let Some((terminal_wl, terminal_draw, terminal_m)) = arena.get(node_id).and_then(Node::terminal_value) {
                delta = ValueDelta::with_plies_left(terminal_wl, terminal_draw, terminal_m);
            }
            node_deltas
                .entry(node_id)
                .and_modify(|aggregate| *aggregate = aggregate.merge(delta))
                .or_insert(delta);
            if node_index == 0 {
                break;
            }
            let Some(reservation) = reservations.next() else {
                debug_assert!(false, "every backprop edge has a reservation");
                break;
            };
            reservation.complete(delta.q());
            delta = delta.for_parent().one_ply_up();
        }
        result.completed_playouts += 1;
        result.completed_depth += depth;
        result.max_depth = result.max_depth.max(depth);
    }

    for (node_id, delta) in node_deltas {
        arena
            .get(node_id)
            .expect("backprop node lives until job drain")
            .add_delta(delta);
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, STARTPOS_FEN, Square};

    use super::complete_batch;
    use crate::search::NodeArena;
    use crate::search::workerpool::{BackpropEvent, GatherEvent};

    #[test]
    fn backprop_completes_every_reservation_with_alternating_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let arena = NodeArena::default();
        let root_id = arena.allocate();
        let root_node = arena.get(root_id).expect("root node");
        assert!(root_node.try_begin_evaluation());
        let mv = Move::new(Square::parse("b2").expect("b2"), Square::parse("b3").expect("b3"));
        root_node.publish_edges(vec![(mv, 1.0)]);
        let child_id = arena.child_or_create(&root_node.edges()[0]);
        let child = GatherEvent::<crate::search::NoQueueStamp>::at_root(root_id, Arc::clone(&history))
            .descend(child_id, root_node.reserve_edge(0).expect("edge"));

        complete_batch(
            [BackpropEvent::<crate::search::NoQueueStamp>::from_gather(
                child.into_event(),
                0.4,
                0.2,
                2.0,
            )],
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
