//! Owned event payloads for LC3-style streaming workers.
//!
//! Reference: LC3 overview, "Workers" and glossary "Variation":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//! <https://lczero.org/dev/lc0/search/lc3/glossary/>

use std::sync::Arc;

use xiangqi_core::{Move, PositionHistory};

use super::{EdgeReservation, NodeKey, ValueDelta};

/// Rejects stale events after `position`, `ucinewgame`, or a replacement `go`.
///
/// LC3 events belong to one streaming search. x7 gives every UCI search a
/// monotonic generation instead of allowing an old event to update a new root.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SearchGeneration(pub u64);

/// A root history plus the moves from that root to a repository node.
///
/// Repetition and rule60 are history-sensitive in Xiangqi, so an event cannot
/// contain only a board hash. The root history is shared immutably; the
/// variation is owned and can be replayed into each worker's local workspace.
#[derive(Clone, Debug)]
pub struct Variation {
    root_history: Arc<PositionHistory>,
    moves: Arc<[Move]>,
}

impl Variation {
    pub fn root(root_history: Arc<PositionHistory>) -> Self {
        Self {
            root_history,
            moves: Arc::from([]),
        }
    }

    pub fn extend(&self, mv: Move) -> Self {
        let mut moves = Vec::with_capacity(self.moves.len() + 1);
        moves.extend_from_slice(&self.moves);
        moves.push(mv);
        Self {
            root_history: Arc::clone(&self.root_history),
            moves: Arc::from(moves),
        }
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        &self.root_history
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// Rebuilds the exact leaf history in a worker-local workspace.
    pub fn replay_history(&self) -> PositionHistory {
        let mut history = self.root_history.as_ref().clone();
        for &mv in self.moves.iter() {
            history.append(mv);
        }
        history
    }
}

/// Work item passed between Gather, Eval, and Backprop workers.
///
/// The event owns all search-specific data. It holds no `&mut` tree/backend
/// references and is therefore safe to send through a bounded worker queue.
#[derive(Debug)]
pub struct NodeEvent {
    pub generation: SearchGeneration,
    pub node_key: NodeKey,
    node_path: Vec<NodeKey>,
    pub variation: Variation,
    pub reservations: Vec<EdgeReservation>,
}

impl NodeEvent {
    pub fn root(generation: SearchGeneration, root_history: Arc<PositionHistory>) -> Self {
        let node_key = NodeKey::root(root_history.last().hash());
        Self {
            generation,
            node_key,
            node_path: vec![node_key],
            variation: Variation::root(root_history),
            reservations: Vec::new(),
        }
    }

    pub fn descend(mut self, child_key: NodeKey, reservation: EdgeReservation) -> Self {
        let variation = self.variation.extend(reservation.mv());
        self.node_key = child_key;
        self.node_path.push(child_key);
        self.variation = variation;
        self.reservations.push(reservation);
        self
    }

    /// Releases every edge-local in-flight visit after a collision, stop, or
    /// failed evaluation. Consuming `self` makes leaving reservations behind
    /// an explicit bug at the caller.
    pub fn cancel(self) {
        for reservation in self.reservations.into_iter().rev() {
            reservation.cancel();
        }
    }

    pub fn node_path(&self) -> &[NodeKey] {
        &self.node_path
    }
}

/// A Gather/Eval outcome routed to Backprop.
///
/// Reference: LC3 Overview, "GatherWorker" and "BackpropWorker":
/// <https://lczero.org/dev/lc0/search/lc3/overview/>. A collision must reach
/// Backprop so the same terminal stage releases its edge-local in-flight visit.
#[derive(Debug)]
pub struct BackpropEvent {
    node: NodeEvent,
    outcome: BackpropOutcome,
}

#[derive(Debug)]
enum BackpropOutcome {
    Evaluation { value: f32, draw: f32 },
    Collision,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackpropResult {
    pub completed_playouts: u32,
    pub collisions: u32,
}

impl BackpropEvent {
    pub(crate) fn evaluation(node: NodeEvent, value: f32, draw: f32) -> Self {
        Self {
            node,
            outcome: BackpropOutcome::Evaluation { value, draw },
        }
    }

    pub(crate) fn collision(node: NodeEvent) -> Self {
        Self {
            node,
            outcome: BackpropOutcome::Collision,
        }
    }

    /// Completes the path bottom-up. The leaf value is from the leaf side to
    /// move; each parent edge therefore receives the sign-flipped update.
    pub(crate) fn complete(self, repository: &super::NodeRepository) -> BackpropResult {
        let Self { node, outcome } = self;
        let BackpropOutcome::Evaluation { value, draw } = outcome else {
            node.cancel();
            return BackpropResult {
                collisions: 1,
                ..BackpropResult::default()
            };
        };
        debug_assert_eq!(node.node_path.len(), node.reservations.len() + 1);
        let mut delta = ValueDelta::one(value, draw);
        let mut reservations = node.reservations.into_iter().rev();
        for (node_index, node_key) in node.node_path.into_iter().enumerate().rev() {
            repository.get_or_insert(node_key).add_value(delta);
            if node_index == 0 {
                break;
            }
            delta = delta.for_parent();
            reservations.next().expect("path reservation").complete(delta.q());
        }
        BackpropResult {
            completed_playouts: 1,
            ..BackpropResult::default()
        }
    }

    pub fn cancel(self) {
        self.node.cancel();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, STARTPOS_FEN};

    use super::{BackpropEvent, NodeEvent, SearchGeneration};
    use crate::search::stream::{NodeKey, NodeRepository};

    #[test]
    fn variation_keeps_root_history_and_owns_its_path() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            SearchGeneration(7),
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        let child_key = root.node_key.child(mv);
        let root_history_len = root.variation.root_history().len();
        let next = root.descend(child_key, crate::search::stream::EdgeReservation::test_only(mv));

        assert_eq!(next.variation.moves(), &[mv]);
        assert_eq!(next.variation.root_history().len(), root_history_len);
        assert_eq!(next.generation, SearchGeneration(7));
        let mut expected = xiangqi_core::PositionHistory::from_positions(state.positions());
        expected.append(mv);
        assert_eq!(next.variation.replay_history().last().hash(), expected.last().hash());
    }

    #[test]
    fn backprop_completes_every_reservation_with_alternating_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            SearchGeneration(1),
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let repository = NodeRepository::default();
        let root_node = repository.get_or_insert(root.node_key);
        assert!(root_node.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        root_node.publish_edges(vec![(mv, 1.0)]);
        let child = root.descend(NodeKey::root(42), root_node.reserve_edge(0).expect("edge"));

        BackpropEvent::evaluation(child, 0.4, 0.2).complete(&repository);

        let edge = &root_node.edges()[0];
        assert_eq!(edge.visits(), 1);
        assert_eq!(edge.completed_visits(), 1);
        assert!((edge.q() + 0.4).abs() < f32::EPSILON);
        assert_eq!(root_node.completed_visits(), 1);
        assert!((root_node.q() + 0.4).abs() < f32::EPSILON);
        let child_node = repository.get(NodeKey::root(42)).expect("child node");
        assert_eq!(child_node.completed_visits(), 1);
        assert!((child_node.q() - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn collision_backprop_releases_reservation_without_updating_nodes() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            SearchGeneration(1),
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let repository = NodeRepository::default();
        let root_node = repository.get_or_insert(root.node_key);
        assert!(root_node.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        root_node.publish_edges(vec![(mv, 1.0)]);
        let child = root.descend(NodeKey::root(42), root_node.reserve_edge(0).expect("edge"));

        let result = BackpropEvent::collision(child).complete(&repository);

        let edge = &root_node.edges()[0];
        assert_eq!(result.completed_playouts, 0);
        assert_eq!(result.collisions, 1);
        assert_eq!(edge.visits(), 0);
        assert_eq!(edge.completed_visits(), 0);
        assert_eq!(root_node.completed_visits(), 0);
        assert!(repository.get(NodeKey::root(42)).is_none());
    }
}
