//! Sharded node repository and edge-local reservations for stream search.
//!
//! Reference: LC3 overview, "Node repository" and "Node structure":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! x7 deliberately uses tree keys in the first version: a child key combines
//! its parent key and move, so transpositions are not merged into a DAG yet.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use nohash_hasher::{IsEnabled, NoHashHasher};
use parking_lot::{Mutex, RwLock};
use xiangqi_core::Move;

use super::ValueDelta;

/// Repository identity. It is a tree key, not a position-only transposition
/// key: equal positions reached by different paths remain different nodes.
///
/// The `u64` is already mixed by `hash_cat`. The shard map uses
/// `nohash_hasher::NoHashHasher` so this value is used as the bucket index
/// directly (no second hash pass).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodeKey(u64);

impl Hash for NodeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0);
    }
}

// Asserts `Hash` only calls `write_u64` once — required by `NoHashHasher`.
impl IsEnabled for NodeKey {}

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
pub struct Edge {
    mv: Move,
    /// Policy prior as IEEE-754 `f32` bit pattern (`f32::to_bits` /
    /// `from_bits`). Stored in `AtomicU32` because std has no `AtomicF32`.
    prior_bits: AtomicU32,
    started: AtomicU32,
    /// Protected aggregate: Q and completed N must be observed together.
    completed: Mutex<CompletedStats>,
}

/// Completed edge aggregate (no draw on edges).
/// `wl_sum` is mover-perspective totals; Q = wl_sum / visits.
#[derive(Debug, Default)]
struct CompletedStats {
    visits: u32,
    wl_sum: f32,
}

/// Completed node WDL aggregate (`wl_sum` / `draw_sum` ↔ px0 `wl_` / `d_`).
#[derive(Debug, Default)]
struct NodeStats {
    visits: u32,
    wl_sum: f32,
    draw_sum: f32,
    m_sum: f32,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    lower: f32,
    upper: f32,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            lower: -1.0,
            upper: 1.0,
        }
    }
}

impl Edge {
    fn new(mv: Move, prior: f32) -> Self {
        assert!((0.0..=1.0).contains(&prior), "policy prior must be normalized");
        Self {
            mv,
            prior_bits: AtomicU32::new(prior.to_bits()),
            started: AtomicU32::new(0),
            completed: Mutex::new(CompletedStats::default()),
        }
    }

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
        completed.wl_sum / completed.visits as f32
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

    fn complete(&self, wl: f32) {
        let mut completed = self.completed.lock();
        assert!(
            self.started.load(Ordering::Acquire) > completed.visits,
            "stream edge completion without reservation"
        );
        completed.visits += 1;
        completed.wl_sum += wl;
    }
}

/// One pending visit. It must be consumed exactly once by `complete` or
/// `cancel`; this is the stream reservation cleanup for pending visits.
/// cleanup.
#[derive(Debug)]
pub struct EdgeReservation {
    edge: Arc<Edge>,
}

impl EdgeReservation {
    pub fn mv(&self) -> Move {
        self.edge.mv()
    }

    pub fn complete(self, wl: f32) {
        self.edge.complete(wl);
    }

    pub fn cancel(self) {
        self.edge.cancel();
    }

    #[cfg(test)]
    pub(crate) fn test_only(mv: Move) -> Self {
        Arc::new(Edge::new(mv, 1.0)).reserve()
    }
}

/// A repository value. Expansion publishes its edge vector once; per-edge
/// statistics can then progress independently without a whole-tree lock.
#[derive(Debug, Default)]
pub struct Node {
    /// Lifecycle: Unexpanded → Evaluating → Expanded|Terminal (`ExpansionState` as u8 for CAS).
    expansion: AtomicU8,
    edges: RwLock<Arc<[Arc<Edge>]>>,
    /// LC3 nodes retain their completed aggregate. In-flight visits remain
    /// edge-local and are deliberately excluded from this value.
    stats: Mutex<NodeStats>,
    /// Terminal WDL + plies: `(wl, draw≡d, plies_left≡m)`.
    /// `m` is stored in plies (half-moves), same as px0 `MakeTerminal(plies_left)`;
    /// UCI “moves left” is a separate full-move conversion.
    terminal: Mutex<Option<(f32, f32, f32)>>,
    /// Exact interval in the incoming-edge perspective. This is the stream
    /// equivalent of px0 `lower_bound_` / `upper_bound_` (`node.h:191-204`).
    bounds: Mutex<Bounds>,
}

impl Node {
    pub fn completed_visits(&self) -> u32 {
        self.stats.lock().visits
    }

    pub fn q(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 {
            0.0
        } else {
            stats.wl_sum / stats.visits as f32
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

    pub fn m(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 {
            0.0
        } else {
            stats.m_sum / stats.visits as f32
        }
    }

    pub(crate) fn add_delta(&self, delta: ValueDelta) {
        let mut stats = self.stats.lock();
        stats.visits += delta.visits;
        stats.wl_sum += delta.wl_sum;
        stats.draw_sum += delta.draw_sum;
        stats.m_sum += delta.m_sum;
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
        // px0 `Node::SortEdges` after policy init (`node.cc:291-297`).
        let mut edges = edges;
        edges.sort_unstable_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(std::cmp::Ordering::Equal));
        let edges: Arc<[Arc<Edge>]> = edges
            .into_iter()
            .map(|(mv, prior)| Arc::new(Edge::new(mv, prior)))
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

    pub fn mark_terminal(&self, wl: f32, draw: f32, plies_left: f32) {
        assert_eq!(
            self.expansion_state(),
            ExpansionState::Evaluating,
            "node must be evaluating"
        );
        *self.terminal.lock() = Some((wl, draw, plies_left));
        *self.bounds.lock() = Bounds { lower: wl, upper: wl };
        self.expansion.store(ExpansionState::Terminal as u8, Ordering::Release);
    }

    fn bounds(&self) -> Bounds {
        *self.bounds.lock()
    }

    fn update_bounds(&self, bounds: Bounds) {
        let mut current = self.bounds.lock();
        current.lower = current.lower.max(bounds.lower);
        current.upper = current.upper.min(bounds.upper);
        debug_assert!(current.lower <= current.upper + f32::EPSILON);
    }

    fn mark_proven_terminal(&self, wl: f32, draw: f32, plies_left: f32) -> bool {
        let mut terminal = self.terminal.lock();
        if self.expansion_state() != ExpansionState::Expanded {
            return false;
        }
        *terminal = Some((wl, draw, plies_left));
        *self.bounds.lock() = Bounds { lower: wl, upper: wl };
        self.expansion.store(ExpansionState::Terminal as u8, Ordering::Release);
        true
    }

    pub fn terminal_wl(&self) -> Option<(f32, f32)> {
        (*self.terminal.lock()).map(|(wl, draw, _)| (wl, draw))
    }

    pub fn terminal_value(&self) -> Option<(f32, f32, f32)> {
        *self.terminal.lock()
    }

    pub fn terminal_plies_left(&self) -> Option<f32> {
        (*self.terminal.lock()).map(|(_, _, plies)| plies)
    }

    pub fn edges(&self) -> Arc<[Arc<Edge>]> {
        Arc::clone(&self.edges.read())
    }

    pub fn reserve_edge(&self, edge_index: usize) -> Option<EdgeReservation> {
        self.edges().get(edge_index).map(Edge::reserve)
    }
}

#[derive(Debug)]
struct RepositoryShard {
    /// `NoHashHasher`: `NodeKey` is already `hash_cat`'d; do not hash again.
    nodes: RwLock<HashMap<NodeKey, Arc<Node>, BuildHasherDefault<NoHashHasher<u64>>>>,
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
                    nodes: RwLock::new(HashMap::default()),
                })
                .collect(),
        }
    }

    pub fn get_or_insert(&self, key: NodeKey) -> Arc<Node> {
        let shard = &self.shards[key.0 as usize & (self.shards.len() - 1)];
        if let Some(node) = shard.nodes.read().get(&key) {
            return Arc::clone(node);
        }
        let mut nodes = shard.nodes.write();
        Arc::clone(nodes.entry(key).or_insert_with(|| Arc::new(Node::default())))
    }

    pub fn get(&self, key: NodeKey) -> Option<Arc<Node>> {
        let shard = &self.shards[key.0 as usize & (self.shards.len() - 1)];
        shard.nodes.read().get(&key).cloned()
    }

    /// Propagates exact terminal bounds along one owned variation. A parent is
    /// made terminal only when its child intervals prove one exact outcome;
    /// an unknown sibling keeps the parent interval open. This is the stream
    /// counterpart of px0 `MaybeSetBounds` (`search.cc:2229-2289`).
    pub(crate) fn propagate_proven_bounds(&self, node_path: &[NodeKey], root: NodeKey) {
        for &parent_key in node_path.iter().rev().skip(1) {
            if parent_key == root {
                break;
            }
            let Some(parent) = self.get(parent_key) else {
                continue;
            };
            if parent.expansion_state() == ExpansionState::Terminal {
                continue;
            }
            let mut child_lower = -1.0_f32;
            let mut child_upper = -1.0_f32;
            let mut children = Vec::new();
            for edge in parent.edges().iter() {
                let child = self.get(parent_key.child(edge.mv()));
                let bounds = child.as_ref().map_or_else(Bounds::default, |node| node.bounds());
                child_lower = child_lower.max(bounds.lower);
                child_upper = child_upper.max(bounds.upper);
                children.push((bounds, child.and_then(|node| node.terminal_plies_left())));
            }
            // Children are measured in the parent side-to-move perspective;
            // this node is measured from its incoming edge, hence negation.
            parent.update_bounds(Bounds {
                lower: -child_upper,
                upper: -child_lower,
            });
            let bounds = parent.bounds();
            if (bounds.upper - bounds.lower).abs() > f32::EPSILON {
                continue;
            }
            let wl = bounds.lower;
            let plies_left = if wl < 0.0 {
                // The side to move can force a win: choose the shortest win.
                children
                    .iter()
                    .filter(|(child, _)| child.lower > 0.0)
                    .filter_map(|(_, m)| *m)
                    .reduce(f32::min)
                    .unwrap_or(0.0)
                    + 1.0
            } else if wl > 0.0 {
                // Every move loses: choose the longest loss.
                children.iter().filter_map(|(_, m)| *m).reduce(f32::max).unwrap_or(0.0) + 1.0
            } else {
                children.iter().filter_map(|(_, m)| *m).reduce(f32::min).unwrap_or(0.0) + 1.0
            };
            parent.mark_proven_terminal(wl, if wl == 0.0 { 1.0 } else { 0.0 }, plies_left);
        }
    }

    fn remove(&self, key: NodeKey) -> Option<Arc<Node>> {
        let shard = &self.shards[key.0 as usize & (self.shards.len() - 1)];
        shard.nodes.write().remove(&key)
    }

    /// Removes one detached tree subtree and returns its node count.
    ///
    /// Reference: LC3 overview, "Node repository". LC3 does not define a
    /// tree-reuse GC policy; x7's tree-only policy follows the sibling release
    /// shape of px0 `Node::ReleaseChildrenExceptOne` (`node.cc:417-445`).
    /// The caller must first drain all events and reservations.
    pub(crate) fn remove_subtree(&self, root: NodeKey) -> usize {
        let mut pending = vec![root];
        let mut removed = 0;
        while let Some(key) = pending.pop() {
            let Some(node) = self.remove(key) else {
                continue;
            };
            removed += 1;
            pending.extend(node.edges().iter().map(|edge| key.child(edge.mv())));
        }
        removed
    }

    /// Checks the edge-local reservation invariant below `root`.
    pub(crate) fn subtree_is_settled(&self, root: NodeKey) -> bool {
        let mut pending = vec![root];
        let mut seen: HashSet<NodeKey, BuildHasherDefault<NoHashHasher<u64>>> = HashSet::default();
        while let Some(key) = pending.pop() {
            if !seen.insert(key) {
                continue;
            }
            let Some(node) = self.get(key) else {
                continue;
            };
            let edges = node.edges();
            if edges.iter().any(|edge| edge.visits() != edge.completed_visits()) {
                return false;
            }
            pending.extend(edges.iter().map(|edge| key.child(edge.mv())));
        }
        true
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.nodes.read().len()).sum()
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
    fn sticky_proof_needs_all_losing_replies_and_keeps_the_longest_loss() {
        let repository = NodeRepository::default();
        let root = NodeKey::root(1);
        let parent_move = b2_b3();
        let parent_key = root.child(parent_move);
        let first_move = Move::new(Square::parse("c3").expect("from"), Square::parse("c4").expect("to"));
        let second_move = Move::new(Square::parse("d3").expect("from"), Square::parse("d4").expect("to"));
        let root_node = repository.get_or_insert(root);
        assert!(root_node.try_begin_evaluation());
        root_node.publish_edges(vec![(parent_move, 1.0)]);
        let parent = repository.get_or_insert(parent_key);
        assert!(parent.try_begin_evaluation());
        parent.publish_edges(vec![(first_move, 0.5), (second_move, 0.5)]);

        let first = repository.get_or_insert(parent_key.child(first_move));
        assert!(first.try_begin_evaluation());
        first.mark_terminal(-1.0, 0.0, 2.0);
        repository.propagate_proven_bounds(&[root, parent_key, parent_key.child(first_move)], root);
        assert_eq!(parent.expansion_state(), ExpansionState::Expanded);

        let second = repository.get_or_insert(parent_key.child(second_move));
        assert!(second.try_begin_evaluation());
        second.mark_terminal(-1.0, 0.0, 6.0);
        repository.propagate_proven_bounds(&[root, parent_key, parent_key.child(second_move)], root);
        assert_eq!(parent.expansion_state(), ExpansionState::Terminal);
        assert_eq!(parent.terminal_wl(), Some((1.0, 0.0)));
        assert_eq!(parent.terminal_plies_left(), Some(7.0));
        assert_ne!(root_node.expansion_state(), ExpansionState::Terminal);
    }

    #[test]
    fn sticky_proof_keeps_the_shortest_forced_win() {
        let repository = NodeRepository::default();
        let root = NodeKey::root(2);
        let parent_move = b2_b3();
        let parent_key = root.child(parent_move);
        let first_move = Move::new(Square::parse("c3").expect("from"), Square::parse("c4").expect("to"));
        let second_move = Move::new(Square::parse("d3").expect("from"), Square::parse("d4").expect("to"));
        let root_node = repository.get_or_insert(root);
        assert!(root_node.try_begin_evaluation());
        root_node.publish_edges(vec![(parent_move, 1.0)]);
        let parent = repository.get_or_insert(parent_key);
        assert!(parent.try_begin_evaluation());
        parent.publish_edges(vec![(first_move, 0.5), (second_move, 0.5)]);
        for (mv, m) in [(first_move, 2.0), (second_move, 6.0)] {
            let child = repository.get_or_insert(parent_key.child(mv));
            assert!(child.try_begin_evaluation());
            child.mark_terminal(1.0, 0.0, m);
            repository.propagate_proven_bounds(&[root, parent_key, parent_key.child(mv)], root);
        }
        assert_eq!(parent.expansion_state(), ExpansionState::Terminal);
        assert_eq!(parent.terminal_wl(), Some((-1.0, 0.0)));
        assert_eq!(parent.terminal_plies_left(), Some(3.0));
    }

    #[test]
    fn cancelled_reservation_does_not_leave_virtual_visit() {
        let root = NodeRepository::default().get_or_insert(NodeKey::root(321));
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(b2_b3(), 1.0)]);
        root.reserve_edge(0).expect("edge").cancel();
        assert_eq!(root.edges()[0].visits(), 0);
        assert_eq!(root.edges()[0].completed_visits(), 0);
    }

    #[test]
    fn failed_evaluation_returns_node_to_claimable_state() {
        let root = NodeRepository::default().get_or_insert(NodeKey::root(456));
        assert!(root.try_begin_evaluation());
        root.abort_evaluation();
        assert_eq!(root.expansion_state(), ExpansionState::Unexpanded);
        assert!(root.try_begin_evaluation());
    }
}
