//! stream 搜索的分片 node repository 与 edge-local reservation。
//!
//! 参考：LC3 Overview 的 “Node repository” 与 “Node structure”：
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! x7 首版刻意使用 tree key：child key 由 parent key 和走法组成，因此暂不把换位合并为
//! DAG。

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use nohash_hasher::{IsEnabled, NoHashHasher};
use parking_lot::{Mutex, RwLock};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;

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
        // `complete` 也持有此锁。必须在同一临界区内检查并减少 `started`，否则
        // complete 可在检查与 CAS 之间增加 completed，破坏 started >= completed。
        let completed = self.completed.lock();
        let started = self.started.load(Ordering::Acquire);
        assert!(started > completed.visits, "stream edge reservation underflow");
        self.started.fetch_sub(1, Ordering::AcqRel);
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

    /// 恰好一个 Eval worker 取得未展开 node。其他 worker 报告 collision 并直接
    /// cancel 自己的 reservation，不重复评估同一局面。
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
        let shard = &self.shards[self.shard_index(key)];
        if let Some(node) = shard.nodes.read().get(&key) {
            return Arc::clone(node);
        }
        let mut nodes = shard.nodes.write();
        Arc::clone(nodes.entry(key).or_insert_with(|| Arc::new(Node::default())))
    }

    pub fn get(&self, key: NodeKey) -> Option<Arc<Node>> {
        let shard = &self.shards[self.shard_index(key)];
        shard.nodes.read().get(&key).cloned()
    }

    fn shard_index(&self, key: NodeKey) -> usize {
        key.0 as usize & (self.shards.len() - 1)
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

    /// 批量删除已脱离的 tree subtree，并返回实际删除的 node 数。
    ///
    /// 参考：LC3 Overview 的 “Node repository”。LC3 未定义 tree-reuse GC 策略；x7
    /// 的 tree-only 策略遵循 px0 `Node::ReleaseChildrenExceptOne`（`node.cc:417-445`）
    /// 的 sibling 释放方式。调用方必须先 drain 所有 event 和 reservation。
    ///
    /// 先在只读阶段收集全部 key，再按 repository shard 一次性写入删除。跨回合路径
    /// 已经 drain，故不需要让每个 node 的删除都单独获取一次 shard 写锁。
    pub(crate) fn remove_subtrees(&self, roots: impl IntoIterator<Item = NodeKey>) -> usize {
        let mut pending: Vec<_> = roots.into_iter().collect();
        let mut keys_by_shard: Vec<Vec<NodeKey>> = (0..self.shards.len()).map(|_| Vec::new()).collect();
        while let Some(key) = pending.pop() {
            let Some(node) = self.get(key) else {
                continue;
            };
            keys_by_shard[self.shard_index(key)].push(key);
            pending.extend(node.edges().iter().map(|edge| key.child(edge.mv())));
        }

        let mut removed = 0;
        for (shard, keys) in self.shards.iter().zip(keys_by_shard) {
            if keys.is_empty() {
                continue;
            }
            let mut nodes = shard.nodes.write();
            for key in keys {
                removed += usize::from(nodes.remove(&key).is_some());
            }
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

// tree reuse 保留已走主线，并回收不可达 sibling subtree。
// 参考 px0 `NodeTree::MakeMove` / `ResetToPosition`（`src/search/classic/node.cc:465-520`）。

/// 已走着替换当前 root 时回收的 node 数。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcStats {
    pub removed_nodes: usize,
}

/// 两次已完成 stream 搜索之间保留的 tree 状态。
#[derive(Debug)]
pub struct Tree {
    repository: Arc<NodeRepository>,
    /// 保留的已走主线：从最早 root 到当前 root。
    root_keys: Vec<NodeKey>,
    /// 与 `root_keys` 对齐的完整历史；末项就是当前 root。快照使 UCI 能精确悔棋和
    /// 复用，不必根据 hash 重建局面。
    root_histories: Vec<Arc<PositionHistory>>,
}

impl Tree {
    pub fn new(root_history: Arc<PositionHistory>) -> Self {
        let root = NodeKey::root(root_history.last().hash());
        Self {
            repository: Arc::new(NodeRepository::default()),
            root_keys: vec![root],
            root_histories: vec![root_history],
        }
    }

    pub fn repository(&self) -> &Arc<NodeRepository> {
        &self.repository
    }

    pub fn root_key(&self) -> NodeKey {
        *self.root_keys.last().expect("stream tree always has a root")
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        self.root_histories
            .last()
            .expect("stream tree always has a root history")
    }

    /// 在当前 root 以下的 event 都完成或取消后，推进到一个合法 child。
    /// 旧 root 留在已走主线，仅回收 sibling subtree。
    pub fn advance(&mut self, mv: Move) -> Result<GcStats, EnginError> {
        let old_root = self.root_key();
        if !self.repository.subtree_is_settled(old_root) {
            return Err(EnginError::PortIncomplete(
                "stream tree advance requires settled reservations",
            ));
        }
        self.advance_settled(mv)
    }

    /// Engine 已停止并 drain worker 后使用的推进路径。
    ///
    /// `SearchSession::set_position` 已保证 reservation 全部归还；保留 debug
    /// 断言以防生命周期边界被破坏。参考 LC3 Overview 的 "Node repository" 与
    /// px0 `NodeTree::MakeMove`（`src/search/classic/node.cc:465-483`）。
    fn advance_after_drain(&mut self, mv: Move) -> Result<GcStats, EnginError> {
        debug_assert!(self.repository.subtree_is_settled(self.root_key()));
        self.advance_settled(mv)
    }

    fn advance_settled(&mut self, mv: Move) -> Result<GcStats, EnginError> {
        let old_root = self.root_key();
        if !self.root_history().last().board().is_legal_move(mv) {
            return Err(EnginError::PortIncomplete("stream tree advance requires a legal move"));
        }

        let mut siblings = Vec::new();
        if let Some(root) = self.repository.get(old_root) {
            siblings.extend(
                root.edges()
                    .iter()
                    .filter(|edge| edge.mv() != mv)
                    .map(|edge| old_root.child(edge.mv())),
            );
        }
        let stats = GcStats {
            removed_nodes: self.repository.remove_subtrees(siblings),
        };

        let new_root = old_root.child(mv);
        self.repository.get_or_insert(new_root);
        let mut history = self.root_history().as_ref().clone();
        history.append(mv);
        self.root_keys.push(new_root);
        self.root_histories.push(Arc::new(history));
        Ok(stats)
    }

    /// Returns to the immediately previous retained root. It does not reclaim
    /// the future child; a later different `advance` will prune it as a sibling.
    pub fn rewind_one(&mut self) -> Result<bool, EnginError> {
        if self.root_keys.len() == 1 {
            return Ok(false);
        }
        if !self.repository.subtree_is_settled(self.root_key()) {
            return Err(EnginError::PortIncomplete(
                "stream tree rewind requires settled reservations",
            ));
        }
        self.root_keys.pop();
        self.root_histories.pop();
        Ok(true)
    }

    /// Repositions this reusable tree for a complete UCI position history.
    ///
    /// A retained ancestor is restored directly; a continuation is advanced
    /// move-by-move so each played edge prunes its siblings. An unrelated
    /// history starts a fresh repository. This is the tree-only counterpart of
    /// px0 `NodeTree::ResetToPosition` (`src/search/classic/node.cc:484-520`).
    pub fn reset_to_history(&mut self, target: Arc<PositionHistory>) -> Result<GcStats, EnginError> {
        if !self.repository.subtree_is_settled(self.root_key()) {
            return Err(EnginError::PortIncomplete(
                "stream tree reset requires settled reservations",
            ));
        }

        self.reset_to_history_settled(target, false)
    }

    /// Engine 的 `position` 路径在调用前已经 abort 并 drain 当前 job，因此无需在
    /// release 构建重复遍历整个 repository 检查 reservation。
    pub(crate) fn reset_to_history_after_drain(&mut self, target: Arc<PositionHistory>) -> Result<GcStats, EnginError> {
        debug_assert!(self.repository.subtree_is_settled(self.root_key()));
        self.reset_to_history_settled(target, true)
    }

    fn reset_to_history_settled(
        &mut self,
        target: Arc<PositionHistory>,
        after_drain: bool,
    ) -> Result<GcStats, EnginError> {
        if let Some(index) = self.root_histories.iter().position(|history| history == &target) {
            self.root_keys.truncate(index + 1);
            self.root_histories.truncate(index + 1);
            return Ok(GcStats::default());
        }

        if target.len() > self.root_history().len()
            && target.positions()[..self.root_history().len()] == *self.root_history().positions()
        {
            let mut stats = GcStats::default();
            while self.root_history().len() < target.len() {
                let next = target.get(self.root_history().len());
                let mv = self
                    .root_history()
                    .last()
                    .board()
                    .generate_legal_moves()
                    .into_iter()
                    .find(|mv| {
                        let mut candidate = self.root_history().as_ref().clone();
                        candidate.append(*mv);
                        candidate.last().board() == next.board()
                    })
                    .ok_or(EnginError::PortIncomplete(
                        "stream tree reset could not derive legal move",
                    ))?;
                let advanced = if after_drain {
                    self.advance_after_drain(mv)?
                } else {
                    self.advance(mv)?
                };
                stats.removed_nodes += advanced.removed_nodes;
            }
            return Ok(stats);
        }

        let root = NodeKey::root(target.last().hash());
        self.repository = Arc::new(NodeRepository::default());
        self.root_keys = vec![root];
        self.root_histories = vec![target];
        Ok(GcStats::default())
    }
}

#[cfg(test)]
mod tree_tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{NodeKey, Tree};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    fn tree() -> Tree {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        Tree::new(Arc::new(PositionHistory::from_positions(state.positions())))
    }

    #[test]
    fn advance_keeps_old_root_and_prunes_sibling_subtree() {
        let mut tree = tree();
        let old_root = tree.root_key();
        let keep = mv("a0", "a1");
        let drop = mv("a0", "a2");
        let root = tree.repository().get_or_insert(old_root);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(keep, 0.5), (drop, 0.5)]);
        let kept_child = old_root.child(keep);
        let dropped_child = old_root.child(drop);
        let dropped_grandchild = dropped_child.child(mv("a9", "a8"));
        tree.repository().get_or_insert(kept_child);
        let dropped = tree.repository().get_or_insert(dropped_child);
        assert!(dropped.try_begin_evaluation());
        dropped.publish_edges(vec![(mv("a9", "a8"), 1.0)]);
        tree.repository().get_or_insert(dropped_grandchild);
        assert_eq!(tree.repository().len(), 4);

        let stats = tree.advance(keep).expect("advance");
        assert_eq!(stats.removed_nodes, 2);
        assert_eq!(tree.repository().len(), 2);
        assert_eq!(tree.root_key(), kept_child);
        assert_eq!(tree.root_history().len(), 2);
        assert!(tree.repository().get(old_root).is_some());
        assert!(tree.repository().get(kept_child).is_some());
        assert!(tree.repository().get(dropped_child).is_none());
        assert!(tree.repository().get(dropped_grandchild).is_none());
    }

    #[test]
    fn rewind_keeps_played_child_for_future_reuse() {
        let mut tree = tree();
        let old_root = tree.root_key();
        let played = mv("a0", "a1");
        tree.advance(played).expect("advance");
        let played_root = tree.root_key();
        assert!(tree.rewind_one().expect("rewind"));
        assert_eq!(tree.root_key(), old_root);
        assert_eq!(tree.root_history().len(), 1);
        assert!(tree.repository().get(played_root).is_some());
        assert!(!tree.rewind_one().expect("root cannot rewind"));
    }

    #[test]
    fn advance_rejects_an_in_flight_reservation() {
        let mut tree = tree();
        let root_key = tree.root_key();
        let played = mv("a0", "a1");
        let root = tree.repository().get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(played, 1.0)]);
        let reservation = root.reserve_edge(0).expect("reservation");
        assert!(tree.advance(played).is_err());
        reservation.cancel();
        assert!(tree.advance(played).is_ok());
    }

    #[test]
    fn reset_to_history_reuses_retained_ancestor_and_continuation() {
        let mut tree = tree();
        let game = GameState::from_fen_moves(STARTPOS_FEN, &["a0a1", "a9a8"]).expect("legal line");
        let first = game.moves[0];
        let second = game.moves[1];
        tree.advance(first).expect("first advance");
        let first_history = tree.root_history().clone();
        tree.advance(second).expect("second advance");

        tree.reset_to_history(first_history.clone())
            .expect("rewind through reset");
        assert_eq!(tree.root_history(), &first_history);
        assert!(tree.repository().get(tree.root_key().child(second)).is_some());

        let target = Arc::new(PositionHistory::from_positions(game.positions()));
        tree.reset_to_history(target).expect("replay continuation");
        assert_eq!(tree.root_history().len(), 3);
    }

    #[test]
    fn reset_to_unrelated_history_starts_fresh_repository() {
        let mut tree = tree();
        tree.advance(mv("a0", "a1")).expect("advance");
        let unrelated = GameState::from_fen_moves(STARTPOS_FEN, &["b0b1"]).expect("other legal line");
        let target = Arc::new(PositionHistory::from_positions(unrelated.positions()));

        tree.reset_to_history(target.clone()).expect("fresh tree");
        assert_eq!(tree.root_history(), &target);
        assert_eq!(tree.repository().len(), 0);
        assert_eq!(tree.root_key(), NodeKey::root(target.last().hash()));
    }
}

#[cfg(test)]
mod repository_tests {
    use std::sync::{Arc, Barrier};
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
    fn cancelled_reservation_restores_started_visit_count() {
        let root = NodeRepository::default().get_or_insert(NodeKey::root(321));
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(b2_b3(), 1.0)]);
        root.reserve_edge(0).expect("edge").cancel();
        assert_eq!(root.edges()[0].visits(), 0);
        assert_eq!(root.edges()[0].completed_visits(), 0);
    }

    #[test]
    // LC3 Overview 的 owned event 约束：每个 reservation 只能完成或取消一次。
    fn concurrent_complete_and_cancel_keep_reservation_balanced() {
        for _ in 0..128 {
            let root = NodeRepository::default().get_or_insert(NodeKey::root(654));
            assert!(root.try_begin_evaluation());
            root.publish_edges(vec![(b2_b3(), 1.0)]);
            let completed = root.reserve_edge(0).expect("completed reservation");
            let cancelled = root.reserve_edge(0).expect("cancelled reservation");
            let barrier = Arc::new(Barrier::new(3));

            thread::scope(|scope| {
                let complete_barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    complete_barrier.wait();
                    completed.complete(0.5);
                });
                let cancel_barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    cancel_barrier.wait();
                    cancelled.cancel();
                });
                barrier.wait();
            });

            assert_eq!(root.edges()[0].visits(), 1);
            assert_eq!(root.edges()[0].completed_visits(), 1);
        }
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
