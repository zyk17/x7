//! Sharded node repository and edge-local reservations for stream search.
//!
//! Reference: LC3 overview, "Node repository" and "Node structure":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! x7 deliberately uses tree keys in the first version: a child key combines
//! its parent key and move, so transpositions are not merged into a DAG yet.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use xiangqi_core::Move;

use super::ValueDelta;

/// Repository identity. It is a tree key, not a position-only transposition
/// key: equal positions reached by different paths remain different nodes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeKey(u64);

impl NodeKey {
    pub const fn root(position_hash: u64) -> Self {
        Self(position_hash)
    }

    /// Equivalent operation to LC3's documented `HashConcatenate(parentHash,
    /// Move)`, using the existing px0 `hashcat` mixing primitive.
    pub const fn child(self, mv: Move) -> Self {
        Self(xiangqi_core::hashcat::hash_cat(self.0, mv.raw() as u64))
    }
}

/// The immutable expansion lifecycle of a repository node.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ExpansionState {
    #[default]
    Unexpanded = 0,
    Evaluating = 1,
    Expanded = 2,
    Terminal = 3,
}

impl ExpansionState {
    fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Unexpanded,
            1 => Self::Evaluating,
            2 => Self::Expanded,
            3 => Self::Terminal,
            _ => unreachable!("invalid stream expansion state"),
        }
    }
}

/// A child edge. In-flight visits live on the incoming edge, never in the
/// child node's completed visit count, matching the LC3 node invariant.
#[derive(Debug)]
pub struct StreamEdge {
    mv: Move,
    prior_bits: AtomicU32,
    started: AtomicU32,
    /// Protected aggregate: Q and completed N must be observed together.
    completed: Mutex<CompletedStats>,
}

#[derive(Debug, Default)]
struct CompletedStats {
    visits: u32,
    value_sum: f32,
}

#[derive(Debug, Default)]
struct NodeStats {
    visits: u32,
    value_sum: f32,
    draw_sum: f32,
}

impl StreamEdge {
    pub fn mv(&self) -> Move {
        self.mv
    }

    pub fn prior(&self) -> f32 {
        f32::from_bits(self.prior_bits.load(Ordering::Acquire))
    }

    /// LC3 edge N includes in-flight visits. Completed N is separately needed
    /// to form a stable Q while a GPU evaluation is outstanding.
    pub fn visits(&self) -> u32 {
        self.started.load(Ordering::Acquire)
    }

    pub fn completed_visits(&self) -> u32 {
        self.completed.lock().visits
    }

    pub fn q(&self) -> f32 {
        let completed = self.completed.lock();
        if completed.visits == 0 {
            return 0.0;
        }
        completed.value_sum / completed.visits as f32
    }

    fn new(mv: Move, prior: f32) -> Self {
        assert!((0.0..=1.0).contains(&prior), "policy prior must be normalized");
        Self {
            mv,
            prior_bits: AtomicU32::new(prior.to_bits()),
            started: AtomicU32::new(0),
            completed: Mutex::new(CompletedStats::default()),
        }
    }

    fn reserve(self: &Arc<Self>) -> EdgeReservation {
        self.started.fetch_add(1, Ordering::AcqRel);
        EdgeReservation { edge: Arc::clone(self) }
    }

    fn cancel(&self) {
        loop {
            let started = self.started.load(Ordering::Acquire);
            assert!(started > self.completed_visits(), "stream edge reservation underflow");
            if self
                .started
                .compare_exchange_weak(started, started - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    fn complete(&self, value: f32) {
        let mut completed = self.completed.lock();
        assert!(
            self.started.load(Ordering::Acquire) > completed.visits,
            "stream edge completion without reservation"
        );
        completed.visits += 1;
        completed.value_sum += value;
    }
}

/// One pending visit. It must be consumed exactly once by `complete` or
/// `cancel`; this is the stream replacement for classic's node virtual-loss
/// cleanup.
#[derive(Debug)]
pub struct EdgeReservation {
    edge: Arc<StreamEdge>,
}

impl EdgeReservation {
    pub fn mv(&self) -> Move {
        self.edge.mv()
    }

    pub fn complete(self, value: f32) {
        self.edge.complete(value);
    }

    pub fn cancel(self) {
        self.edge.cancel();
    }

    #[cfg(test)]
    pub(crate) fn test_only(mv: Move) -> Self {
        Arc::new(StreamEdge::new(mv, 1.0)).reserve()
    }
}

/// A repository value. Expansion publishes its edge vector once; per-edge
/// statistics can then progress independently without a whole-tree lock.
#[derive(Debug, Default)]
pub struct StreamNode {
    expansion: AtomicU8,
    edges: RwLock<Arc<[Arc<StreamEdge>]>>,
    /// LC3 nodes retain their completed aggregate. In-flight visits remain
    /// edge-local and are deliberately excluded from this value.
    stats: Mutex<NodeStats>,
    terminal: Mutex<Option<(f32, f32)>>,
}

impl StreamNode {
    pub fn completed_visits(&self) -> u32 {
        self.stats.lock().visits
    }

    pub fn q(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 {
            0.0
        } else {
            stats.value_sum / stats.visits as f32
        }
    }

    pub fn draw(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 {
            0.0
        } else {
            stats.draw_sum / stats.visits as f32
        }
    }

    pub(crate) fn add_value(&self, delta: ValueDelta) {
        let mut stats = self.stats.lock();
        stats.visits += delta.visits;
        stats.value_sum += delta.value_sum;
        stats.draw_sum += delta.draw_sum;
    }

    pub fn expansion_state(&self) -> ExpansionState {
        ExpansionState::from_raw(self.expansion.load(Ordering::Acquire))
    }

    /// Exactly one Eval worker claims an unexpanded node. Other workers report
    /// a collision and backprop/cancel their reservation instead of evaluating
    /// the same position again.
    pub fn try_begin_evaluation(&self) -> bool {
        self.expansion
            .compare_exchange(
                ExpansionState::Unexpanded as u8,
                ExpansionState::Evaluating as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn publish_edges(&self, edges: Vec<(Move, f32)>) {
        assert_eq!(
            self.expansion_state(),
            ExpansionState::Evaluating,
            "node must be evaluating"
        );
        let edges: Arc<[Arc<StreamEdge>]> = edges
            .into_iter()
            .map(|(mv, prior)| Arc::new(StreamEdge::new(mv, prior)))
            .collect();
        *self.edges.write() = edges;
        self.expansion.store(ExpansionState::Expanded as u8, Ordering::Release);
    }

    /// Restores a node after Eval failed before publishing terminal data or
    /// policy. This prevents future Gather events from treating a failed NN
    /// request as a permanent collision.
    pub fn abort_evaluation(&self) {
        self.expansion
            .compare_exchange(
                ExpansionState::Evaluating as u8,
                ExpansionState::Unexpanded as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .expect("only evaluating stream nodes can abort evaluation");
    }

    pub fn mark_terminal(&self, value: f32, draw: f32) {
        assert!(
            self.expansion_state() == ExpansionState::Evaluating,
            "node must be evaluating"
        );
        *self.terminal.lock() = Some((value, draw));
        self.expansion.store(ExpansionState::Terminal as u8, Ordering::Release);
    }

    pub fn terminal_value(&self) -> Option<(f32, f32)> {
        *self.terminal.lock()
    }

    pub fn edges(&self) -> Arc<[Arc<StreamEdge>]> {
        Arc::clone(&self.edges.read())
    }

    pub fn reserve_edge(&self, edge_index: usize) -> Option<EdgeReservation> {
        self.edges().get(edge_index).map(StreamEdge::reserve)
    }
}

#[derive(Debug)]
struct RepositoryShard {
    nodes: RwLock<HashMap<NodeKey, Arc<StreamNode>>>,
}

/// Sharded key-value repository. A shard lock only protects map lookup and
/// insertion; node statistics live behind the node/edge objects themselves.
#[derive(Debug)]
pub struct NodeRepository {
    shards: Box<[RepositoryShard]>,
}

impl NodeRepository {
    pub fn new(shard_count: usize) -> Self {
        assert!(
            shard_count.is_power_of_two(),
            "stream shard count must be a power of two"
        );
        assert!(shard_count > 0, "stream shard count must be non-zero");
        Self {
            shards: (0..shard_count)
                .map(|_| RepositoryShard {
                    nodes: RwLock::new(HashMap::new()),
                })
                .collect(),
        }
    }

    pub fn get_or_insert(&self, key: NodeKey) -> Arc<StreamNode> {
        let shard = &self.shards[key.0 as usize & (self.shards.len() - 1)];
        if let Some(node) = shard.nodes.read().get(&key) {
            return Arc::clone(node);
        }
        let mut nodes = shard.nodes.write();
        Arc::clone(nodes.entry(key).or_insert_with(|| Arc::new(StreamNode::default())))
    }

    pub fn get(&self, key: NodeKey) -> Option<Arc<StreamNode>> {
        let shard = &self.shards[key.0 as usize & (self.shards.len() - 1)];
        shard.nodes.read().get(&key).cloned()
    }
}

impl Default for NodeRepository {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use xiangqi_core::{Move, Square};

    use super::{ExpansionState, NodeKey, NodeRepository};

    fn b2_b3() -> Move {
        Move::new(Square::parse("b2").expect("b2"), Square::parse("b3").expect("b3"))
    }

    #[test]
    fn tree_key_distinguishes_different_parents() {
        let mv = b2_b3();
        assert_ne!(NodeKey::root(1).child(mv), NodeKey::root(2).child(mv));
        assert_eq!(NodeKey::root(1).child(mv), NodeKey::root(1).child(mv));
    }

    #[test]
    fn one_worker_claims_evaluation_and_edge_reservation_balances() {
        let repository = Arc::new(NodeRepository::default());
        let key = NodeKey::root(123);
        let mut workers = Vec::new();
        for _ in 0..8 {
            let repository = Arc::clone(&repository);
            workers.push(thread::spawn(move || {
                repository.get_or_insert(key).try_begin_evaluation()
            }));
        }
        assert_eq!(
            workers
                .into_iter()
                .filter_map(|worker| worker.join().ok())
                .filter(|&won| won)
                .count(),
            1
        );

        let node = repository.get(key).expect("node");
        assert_eq!(node.expansion_state(), ExpansionState::Evaluating);
        node.publish_edges(vec![(b2_b3(), 1.0)]);
        let edge = node.reserve_edge(0).expect("edge");
        assert_eq!(node.edges()[0].visits(), 1);
        edge.complete(0.5);
        assert_eq!(node.edges()[0].completed_visits(), 1);
        assert!((node.edges()[0].q() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn cancelled_reservation_does_not_leave_virtual_visit() {
        let node = NodeRepository::default().get_or_insert(NodeKey::root(321));
        assert!(node.try_begin_evaluation());
        node.publish_edges(vec![(b2_b3(), 1.0)]);
        node.reserve_edge(0).expect("edge").cancel();
        assert_eq!(node.edges()[0].visits(), 0);
        assert_eq!(node.edges()[0].completed_visits(), 0);
    }

    #[test]
    fn failed_evaluation_returns_node_to_claimable_state() {
        let node = NodeRepository::default().get_or_insert(NodeKey::root(456));
        assert!(node.try_begin_evaluation());
        node.abort_evaluation();
        assert_eq!(node.expansion_state(), ExpansionState::Unexpanded);
        assert!(node.try_begin_evaluation());
    }
}
