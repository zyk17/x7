//! stream 搜索的分片 node repository 与 edge-local reservation。
//!
//! 参考：LC3 Overview 的 “Node repository” 与 “Node structure”：
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! x7 首版刻意使用 tree key：child key 由 parent key 和走法组成，因此暂不把换位合并为
//! DAG。

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use nohash_hasher::{IsEnabled, NoHashHasher};
use parking_lot::{Mutex, RwLock};
use xiangqi_core::Move;

use super::ValueDelta;

/// repository 的标识。它是 tree key 而非仅按局面划分的 transposition key：不同路径
/// 到达的相同局面仍是不同 node。
///
/// `u64` 已由 `hash_cat` 混合。分片 map 使用 `nohash_hasher::NoHashHasher`，直接
/// 以此值为 bucket index，不再进行第二次 hash。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodeKey(u64);

impl Hash for NodeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0);
    }
}

// 断言 `Hash` 只调用一次 `write_u64`，这是 `NoHashHasher` 的要求。
impl IsEnabled for NodeKey {}

impl NodeKey {
    pub const fn root(position_hash: u64) -> Self {
        Self(position_hash)
    }

    /// 等价于 LC3 文档的 `HashConcatenate(parentHash, Move)`，使用既有 px0
    /// `hashcat` 混合原语。
    pub const fn child(self, mv: Move) -> Self {
        Self(xiangqi_core::hashcat::hash_cat(self.0, mv.raw() as u64))
    }
}

/// repository node 不可逆的展开生命周期。
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

/// child edge。in-flight visit 保存在入边，绝不计入 child node 的 completed visit，
/// 对齐 LC3 的 node 不变量。
#[derive(Debug)]
pub struct Edge {
    mv: Move,
    /// IEEE-754 `f32` 位模式的 policy prior（`f32::to_bits` / `from_bits`）。
    /// std 没有 `AtomicF32`，故保存为 `AtomicU32`。
    prior_bits: AtomicU32,
    started: AtomicU32,
    /// 受保护的聚合值：必须同时读取 Q 与 completed N。
    completed: Mutex<CompletedStats>,
}

/// 已完成 edge 聚合值（edge 不保存 draw）。`wl_sum` 是走子方视角总和；
/// Q = wl_sum / visits。
#[derive(Debug, Default)]
struct CompletedStats {
    visits: u32,
    wl_sum: f32,
}

/// 已完成 node WDL 聚合值（`wl_sum` / `draw_sum` 对应 px0 `wl_` / `d_`）。
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

    /// LC3 edge N 包含 in-flight visit。GPU 评估未返回时，另存 completed N 才能形成
    /// 稳定的 Q。
    pub fn visits(&self) -> u32 {
        self.started.load(Ordering::Acquire)
    }

    pub fn completed_visits(&self) -> u32 {
        self.completed.lock().visits
    }

    /// 此 edge 上待完成的 reservation。
    ///
    /// 由 `started - completed` 推导而非单独存储，所以 `complete` / `cancel`
    /// 会自动释放它。参考：LC3 Overview 的 "Node structure"。KataGo 将类似临时
    /// 计数放在 child node；x7 是 tree，入边是唯一且更简单的 owner。
    pub fn in_flight_visits(&self) -> u32 {
        let completed = self.completed.lock().visits;
        self.started.load(Ordering::Acquire).saturating_sub(completed)
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

/// 一次待完成访问。它必须恰好被 `complete` 或 `cancel` 消费一次，确保
/// stream 的 reservation 不泄漏。
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

/// repository 的 node 值。展开时只发布一次 edge vector；之后各 edge 统计可独立推进，
/// 不需要整棵 tree 锁。
#[derive(Debug, Default)]
pub struct Node {
    /// 生命周期：Unexpanded → Evaluating → Expanded|Terminal（`ExpansionState` 以 u8
    /// 供 CAS 使用）。
    expansion: AtomicU8,
    edges: RwLock<Arc<[Arc<Edge>]>>,
    /// LC3 node 保留其 completed 聚合值。in-flight visit 保持在 edge-local，刻意不计入
    /// 此值。
    stats: Mutex<NodeStats>,
    /// 终局 WDL 与 plies：`(wl, draw≡d, plies_left≡m)`。`m` 以 ply（半回合）保存，
    /// 与 px0 `MakeTerminal(plies_left)` 一致；UCI “moves left” 另行换算为完整回合。
    terminal: Mutex<Option<(f32, f32, f32)>>,
    /// incoming-edge 视角的精确区间。相当于 stream 版本的 px0 `lower_bound_` /
    /// `upper_bound_`（`node.h:191-204`）。
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

    /// 恰好一个 Eval worker 取得未展开 node。其他 worker 报告 collision，并
    /// backprop/cancel 它们的 reservation，不重复评估同一局面。
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
        // px0 在 policy 初始化后调用 `Node::SortEdges`（`node.cc:291-297`）。
        let mut edges = edges;
        edges.sort_unstable_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(std::cmp::Ordering::Equal));
        let edges: Arc<[Arc<Edge>]> = edges
            .into_iter()
            .map(|(mv, prior)| Arc::new(Edge::new(mv, prior)))
            .collect();
        *self.edges.write() = edges;
        self.expansion.store(ExpansionState::Expanded as u8, Ordering::Release);
    }

    /// Eval 在发布终局数据或 policy 前失败后恢复 node，避免后续 Gather event 将失败的
    /// NN 请求当作永久 collision。
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
    /// `NoHashHasher`：`NodeKey` 已经 `hash_cat`，不得再次 hash。
    nodes: RwLock<HashMap<NodeKey, Arc<Node>, BuildHasherDefault<NoHashHasher<u64>>>>,
}

/// 分片 key-value repository。分片锁只保护 map 查找和插入；node 统计存放在各自的
/// node/edge 对象之后。
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

    /// 沿一条 owned variation 传播精确终局边界。只有 child 区间证明唯一结果时才将
    /// parent 标记为终局；未知 sibling 会保持 parent 区间开放。对应 stream 版本的
    /// px0 `MaybeSetBounds`（`search.cc:2229-2289`）。
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
            // child 按 parent side-to-move 视角度量；本 node 按 incoming edge 视角度量，
            // 因此取反。
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
                // 当前行棋方可强制胜：选最短胜利。
                children
                    .iter()
                    .filter(|(child, _)| child.lower > 0.0)
                    .filter_map(|(_, m)| *m)
                    .reduce(f32::min)
                    .unwrap_or(0.0)
                    + 1.0
            } else if wl > 0.0 {
                // 每步都输：选最长败局。
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

    /// 删除一棵已脱离的 tree subtree，并返回 node 数。
    ///
    /// 参考：LC3 Overview 的 “Node repository”。LC3 未定义 tree-reuse GC 策略；x7
    /// 的 tree-only 策略遵循 px0 `Node::ReleaseChildrenExceptOne`（`node.cc:417-445`）
    /// 的 sibling 释放方式。调用方必须先 drain 所有 event 和 reservation。
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

    /// 检查 `root` 以下的 edge-local reservation 不变量。
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
