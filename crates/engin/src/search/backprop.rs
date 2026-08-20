//! Backprop 算法：`complete` reservation + `add_delta`。
//!
//! MCTS 回传实验改这里。worker 循环壳在 `workerpool`。

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use nohash_hasher::NoHashHasher;

use super::workerpool::BackpropEvent;
use super::{NodeKey, NodeRepository, ValueDelta};

type NodeDeltaMap = HashMap<NodeKey, ValueDelta, BuildHasherDefault<NoHashHasher<u64>>>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackpropResult {
    pub(crate) completed_playouts: u32,
    pub(crate) completed_depth: u64,
    pub(crate) max_depth: u64,
}

/// 路径增量回传：同路径 node 合并 delta，再一次写入；edge 按层 complete。
pub(crate) fn complete_batch(
    events: impl IntoIterator<Item = BackpropEvent>,
    repository: &NodeRepository,
) -> BackpropResult {
    let mut node_deltas = NodeDeltaMap::default();
    let mut result = BackpropResult::default();

    for event in events {
        let BackpropEvent { playout, value, .. } = event;
        debug_assert_eq!(playout.node_path.len(), playout.reservations.len() + 1);
        let depth = playout.node_path.len() as u64;
        let mut delta = value;
        let mut reservations = playout.reservations.into_iter().rev();
        for (node_index, node_key) in playout.node_path.into_iter().enumerate().rev() {
            if let Some((terminal_wl, terminal_draw, terminal_m)) =
                repository.get(node_key).and_then(|node| node.terminal_value())
            {
                delta = ValueDelta::with_plies_left(terminal_wl, terminal_draw, terminal_m);
            }
            node_deltas
                .entry(node_key)
                .and_modify(|aggregate| *aggregate = aggregate.merge(delta))
                .or_insert(delta);
            if node_index == 0 {
                break;
            }
            reservations.next().expect("path reservation").complete(delta.q());
            delta = delta.for_parent().one_ply_up();
        }
        result.completed_playouts += 1;
        result.completed_depth += depth;
        result.max_depth = result.max_depth.max(depth);
    }

    for (node_key, delta) in node_deltas {
        repository.get_or_insert(node_key).add_delta(delta);
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, STARTPOS_FEN, Square};

    use super::complete_batch;
    use crate::search::NodeRepository;
    use crate::search::workerpool::{BackpropEvent, PlayoutEvent};

    #[test]
    fn backprop_completes_every_reservation_with_alternating_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let root_key = PlayoutEvent::root(1, Arc::clone(&history)).node_key;
        let root_node = repository.get_or_insert(root_key);
        assert!(root_node.try_begin_evaluation());
        let mv = Move::new(Square::parse("b2").expect("b2"), Square::parse("b3").expect("b3"));
        root_node.publish_edges(vec![(mv, 1.0)]);
        let child = PlayoutEvent::root(1, Arc::clone(&history))
            .descend(root_key.child(mv), root_node.reserve_edge(0).expect("edge"));

        complete_batch([BackpropEvent::evaluation(child, 0.4, 0.2, 2.0)], &repository);

        let edge = &root_node.edges()[0];
        assert_eq!(edge.visits(), 1);
        assert_eq!(edge.completed_visits(), 1);
        assert!((edge.q() - 0.4).abs() < f32::EPSILON);
        assert_eq!(root_node.completed_visits(), 1);
        assert!((root_node.q() + 0.4).abs() < f32::EPSILON);
        assert!((root_node.m() - 3.0).abs() < f32::EPSILON);
        let child_node = repository.get(root_key.child(mv)).expect("child node");
        assert_eq!(child_node.completed_visits(), 1);
        assert!((child_node.q() - 0.4).abs() < f32::EPSILON);
        assert!((child_node.m() - 2.0).abs() < f32::EPSILON);
    }
}
