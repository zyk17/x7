//! Owned event payloads for LC3-style streaming workers.
//!
//! Reference: LC3 overview, "Workers" and glossary "Variation":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//! <https://lczero.org/dev/lc0/search/lc3/glossary/>

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::time::Instant;

use nohash_hasher::NoHashHasher;
use xiangqi_core::{Move, PositionHistory};

use super::{EdgeReservation, NodeKey, ValueDelta};

type NodeDeltaMap = HashMap<NodeKey, ValueDelta, BuildHasherDefault<NoHashHasher<u64>>>;

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
/// move path is owned and grown in place as Gather descends.
#[derive(Clone, Debug)]
pub struct Variation {
    root_history: Arc<PositionHistory>,
    moves: Vec<Move>,
}

impl Variation {
    pub fn root(root_history: Arc<PositionHistory>) -> Self {
        Self {
            root_history,
            moves: Vec::new(),
        }
    }

    pub fn push(&mut self, mv: Move) {
        self.moves.push(mv);
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
    queued_at: Option<Instant>,
}

impl NodeEvent {
    pub fn root(generation: SearchGeneration, root_history: Arc<PositionHistory>) -> Self {
        Self::at_root(generation, NodeKey::root(root_history.last().hash()), root_history)
    }

    pub fn at_root(generation: SearchGeneration, root_key: NodeKey, root_history: Arc<PositionHistory>) -> Self {
        Self {
            generation,
            node_key: root_key,
            node_path: vec![root_key],
            variation: Variation::root(root_history),
            reservations: Vec::new(),
            queued_at: None,
        }
    }

    pub fn descend(mut self, child_key: NodeKey, reservation: EdgeReservation) -> Self {
        self.variation.push(reservation.mv());
        self.node_key = child_key;
        self.node_path.push(child_key);
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

    /// Marks one worker-queue handoff for optional benchmark telemetry.
    ///
    /// Reference: LC3 overview, "Stats Collection". Normal UCI searches keep
    /// this unset, so timing collection is outside their hot path.
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    pub(crate) fn take_queue_wait(&mut self) -> Option<std::time::Duration> {
        self.queued_at.take().map(|queued_at| queued_at.elapsed())
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
    queued_at: Option<Instant>,
}

#[derive(Debug)]
enum BackpropOutcome {
    Evaluation { wl: f32, draw: f32, plies_left: f32 },
    Collision,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackpropResult {
    pub completed_playouts: u32,
    pub collisions: u32,
    pub collision_depths: Vec<usize>,
    /// Sum of completed leaf depths, with the root at depth one. This matches
    /// px0 classic's `cum_depth_` accounting (`search.cc:2157-2167`).
    pub completed_depth: u64,
    pub max_depth: u64,
}

impl BackpropEvent {
    pub(crate) fn evaluation(node: NodeEvent, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self {
            node,
            outcome: BackpropOutcome::Evaluation { wl, draw, plies_left },
            queued_at: None,
        }
    }

    pub(crate) fn collision(node: NodeEvent) -> Self {
        Self {
            node,
            outcome: BackpropOutcome::Collision,
            queued_at: None,
        }
    }

    /// Completes events through the aggregate path used by Backprop workers.
    /// Edge reservations remain per-event because each owns exactly one
    /// in-flight visit.
    ///
    /// LC3 Overview, "BackpropWorker":
    /// <https://lczero.org/dev/lc0/search/lc3/overview/>.
    pub(crate) fn complete_batch(
        events: impl IntoIterator<Item = Self>,
        repository: &super::NodeRepository,
    ) -> BackpropResult {
        // `NodeKey` is already mixed; same identity hasher as the repository.
        let mut node_deltas = NodeDeltaMap::default();
        let mut result = BackpropResult::default();

        for event in events {
            let Self { node, outcome, .. } = event;
            let BackpropOutcome::Evaluation { wl, draw, plies_left } = outcome else {
                result.collision_depths.push(node.variation.moves().len());
                node.cancel();
                result.collisions += 1;
                continue;
            };
            debug_assert_eq!(node.node_path.len(), node.reservations.len() + 1);
            let depth = node.node_path.len() as u64;
            let mut delta = ValueDelta::with_plies_left(wl, draw, plies_left);
            let mut reservations = node.reservations.into_iter().rev();
            for (node_index, node_key) in node.node_path.into_iter().enumerate().rev() {
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
                // px0 `EdgeAndNode::GetQ` reads child.wl (mover view). Complete
                // the edge with the child delta, then flip for the parent node
                // (`search.cc:2175-2257`: finalize(v); v = -v).
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

    pub fn cancel(self) {
        self.node.cancel();
    }

    pub(crate) fn mark_queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    /// Returns the optional delay since the Backprop queue handoff.
    ///
    /// Reference: LC3 overview, "Stats Collection".
    pub(crate) fn take_queue_wait(&mut self) -> Option<std::time::Duration> {
        self.queued_at.take().map(|queued_at| queued_at.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, STARTPOS_FEN};

    use super::{BackpropEvent, NodeEvent, SearchGeneration};
    use crate::search::{NodeKey, NodeRepository};

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
        let next = root.descend(child_key, crate::search::EdgeReservation::test_only(mv));

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

        BackpropEvent::complete_batch([BackpropEvent::evaluation(child, 0.4, 0.2, 2.0)], &repository);

        let edge = &root_node.edges()[0];
        assert_eq!(edge.visits(), 1);
        assert_eq!(edge.completed_visits(), 1);
        // Mover-perspective leaf value is stored on the child and edge; parent
        // receives the flipped backup (px0 search.cc:2129,2175-2257).
        assert!((edge.q() - 0.4).abs() < f32::EPSILON);
        assert_eq!(root_node.completed_visits(), 1);
        assert!((root_node.q() + 0.4).abs() < f32::EPSILON);
        assert!((root_node.m() - 3.0).abs() < f32::EPSILON);
        let child_node = repository.get(NodeKey::root(42)).expect("child node");
        assert_eq!(child_node.completed_visits(), 1);
        assert!((child_node.q() - 0.4).abs() < f32::EPSILON);
        assert!((child_node.m() - 2.0).abs() < f32::EPSILON);
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

        let result = BackpropEvent::complete_batch([BackpropEvent::collision(child)], &repository);

        let edge = &root_node.edges()[0];
        assert_eq!(result.completed_playouts, 0);
        assert_eq!(result.collisions, 1);
        assert_eq!(edge.visits(), 0);
        assert_eq!(edge.completed_visits(), 0);
        assert_eq!(root_node.completed_visits(), 0);
        assert!(repository.get(NodeKey::root(42)).is_none());
    }

    #[test]
    fn batch_backprop_merges_node_updates_and_completes_each_edge_visit() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root_history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let first = NodeEvent::root(SearchGeneration(1), Arc::clone(&root_history));
        let root_node = repository.get_or_insert(first.node_key);
        assert!(root_node.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        root_node.publish_edges(vec![(mv, 1.0)]);
        let child_key = NodeKey::root(42);
        let first = first.descend(child_key, root_node.reserve_edge(0).expect("first edge"));
        let second = NodeEvent::root(SearchGeneration(1), root_history)
            .descend(child_key, root_node.reserve_edge(0).expect("second edge"));

        let result = BackpropEvent::complete_batch(
            [
                BackpropEvent::evaluation(first, 0.4, 0.2, 0.0),
                BackpropEvent::evaluation(second, 0.2, 0.4, 0.0),
            ],
            &repository,
        );

        let edge = &root_node.edges()[0];
        let child_node = repository.get(child_key).expect("child node");
        assert_eq!(result.completed_playouts, 2);
        assert_eq!(result.collisions, 0);
        assert_eq!(result.completed_depth, 4);
        assert_eq!(result.max_depth, 2);
        assert_eq!(edge.visits(), 2);
        assert_eq!(edge.completed_visits(), 2);
        assert!((edge.q() - 0.3).abs() < f32::EPSILON);
        assert_eq!(root_node.completed_visits(), 2);
        assert!((root_node.q() + 0.3).abs() < f32::EPSILON);
        assert_eq!(child_node.completed_visits(), 2);
        assert!((child_node.q() - 0.3).abs() < f32::EPSILON);
    }
}
