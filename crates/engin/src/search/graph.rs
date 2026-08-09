//! stream 搜索的分片 node repository 与 edge-local reservation。
//!
//! 参考：LC3 Overview 的 “Node repository” 与 “Node structure”：
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! MCGS 使用 board key：相同棋盘（含行棋方）共享 node；走法统计仍留在 parent edge。

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use nohash_hasher::{IsEnabled, NoHashHasher};
use parking_lot::{Mutex, RwLock};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;

use super::ValueDelta;

/// repository 的标识。MCGS 只以当前棋盘（含行棋方）划分 node；历史规则仍由 event
/// 的 variation 在 path-local 层面裁决。
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
    /// MCGS 的共享 state identity：只由当前棋盘（含行棋方）决定。
    pub const fn board(board_hash: u64) -> Self {
        Self(board_hash)
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
            _ => unreachable!("invalid expansion state"),
        }
    }
}

/// child edge。in-flight visit 保存在入边，绝不计入 child node 的 completed visit，
/// 对齐 LC3 的 node 不变量。
#[derive(Debug)]
pub struct Edge {
    mv: Move,
    /// MCGS 中 move 指向的共享 board node。首次沿此 edge 下降时绑定；同一 parent
    /// board 与 move 必然导向同一 child board。KataGo GraphSearch 的 action/state 分离。
    child_key: OnceLock<NodeKey>,
    /// IEEE-754 `f32` 位模式的 policy prior（`f32::to_bits` / `from_bits`）。
    /// std 没有 `AtomicF32`，故保存为 `AtomicU32`。
    prior_bits: AtomicU32,
    started: AtomicU32,
    /// 已完成访问和其中 variation-local 终局的实际样本。
    ///
    /// 同一 board edge 可由不同 history 到达。重复、连将/追击与 rule60 的裁决不能
    /// first-writer-wins；每次命中的路径终局只计入自己的这一次 edge visit。
    /// 参考 KataGo `docs/GraphSearch.md` 的 edge-local action statistic。
    ///
    /// 受保护的聚合值：必须同时读取 completed N 与 local terminal 样本。
    /// 与 `started` 协同维护 `started >= completed`。`complete` 与 `cancel` 必须在
    /// 同一把锁下先检查、再修改，不能拆成独立原子计数。
    completed: Mutex<EdgeStats>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EdgeStats {
    pub visits: u32,
    pub local_terminal: ValueDelta,
}

impl Edge {
    fn new(mv: Move, prior: f32) -> Self {
        assert!((0.0..=1.0).contains(&prior), "policy prior must be normalized");
        Self {
            mv,
            child_key: OnceLock::new(),
            prior_bits: AtomicU32::new(prior.to_bits()),
            started: AtomicU32::new(0),
            completed: Mutex::new(EdgeStats::default()),
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

    pub(crate) fn completed_stats(&self) -> EdgeStats {
        *self.completed.lock()
    }

    /// 此 edge 上待完成的 reservation。
    ///
    /// 由 `started - completed` 推导而非单独存储，所以 `complete` / `cancel`
    /// 会自动释放它。参考：LC3 Overview 的 "Node structure"。KataGo 将类似临时
    /// 计数放在 child node；x7 将它放在入边，owner 更明确也更简单。
    pub fn in_flight_visits(&self) -> u32 {
        let completed = self.completed.lock().visits;
        self.started.load(Ordering::Acquire).saturating_sub(completed)
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

    pub fn child_key(&self) -> Option<NodeKey> {
        self.child_key.get().copied()
    }

    pub fn bind_child_key(&self, key: NodeKey) {
        if let Err(existing) = self.child_key.set(key) {
            assert_eq!(existing, key, "one edge must keep one child board");
        }
    }

    fn complete(&self, local_terminal: Option<ValueDelta>) {
        let mut completed = self.completed.lock();
        assert!(
            self.started.load(Ordering::Acquire) > completed.visits,
            "stream edge completion without reservation"
        );
        completed.visits += 1;
        if let Some(value) = local_terminal {
            assert_eq!(value.visits, 1, "one path terminal completes one edge visit");
            completed.local_terminal = completed.local_terminal.merge(value);
        }
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

    pub fn complete(self) {
        self.edge.complete(None);
    }

    pub(crate) fn complete_path_terminal(self, value: ValueDelta) {
        self.edge.complete(Some(value));
    }

    pub fn cancel(self) {
        self.edge.cancel();
    }

    #[cfg(test)]
    pub(crate) fn test_only(mv: Move) -> Self {
        Arc::new(Edge::new(mv, 1.0)).reserve()
    }
}

/// 已完成 node WDL 聚合值（`wl_sum` / `draw_sum` 对应 px0 `wl_` / `d_`）。
#[derive(Debug, Default)]
struct NodeStats {
    visits: u32,
    wl_sum: f32,
    draw_sum: f32,
    m_sum: f32,
}

/// repository 的 node 值。展开时只发布一次 edge vector；之后各 edge 统计可独立推进，
/// 不需要整张图锁。
#[derive(Debug, Default)]
pub struct Node {
    /// 生命周期：Unexpanded → Evaluating → Expanded|Terminal（`ExpansionState` 以 u8
    /// 供 CAS 使用）。
    expansion: AtomicU8,
    edges: RwLock<Arc<[Arc<Edge>]>>,
    /// LC3 node 保留其 completed 聚合值。in-flight visit 保持在 edge-local，刻意不计入
    /// 此值。
    stats: Mutex<NodeStats>,
    /// 首次 NN / 非共享终局的原始预测 U。图回传从它和 edge action N 幂等重算。
    graph_value: Mutex<Option<ValueDelta>>,
    /// 终局 WDL 与 plies：`(wl, draw≡d, plies_left≡m)`。`m` 以 ply（半回合）保存，
    /// UCI “moves left” 另行换算为完整回合。
    terminal: Mutex<Option<(f32, f32, f32)>>,
}

impl Node {
    pub(crate) fn set_graph_value(&self, value: ValueDelta) {
        assert_eq!(value.visits, 1, "graph node base value has weight one");
        let mut graph_value = self.graph_value.lock();
        if let Some(existing) = *graph_value {
            assert_eq!(existing, value, "shared node keeps its first evaluation");
        } else {
            *graph_value = Some(value);
        }
    }
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
        self.set_graph_value(ValueDelta::with_plies_left(wl, draw, plies_left));
        self.expansion.store(ExpansionState::Terminal as u8, Ordering::Release);
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

type ShardNodes = RwLock<HashMap<NodeKey, Arc<Node>, BuildHasherDefault<NoHashHasher<u64>>>>;

/// 分片 key-value repository。分片锁只保护 map 查找和插入；node 统计存放在各自的
/// node/edge 对象之后。
#[derive(Debug)]
pub struct NodeRepository {
    /// `NoHashHasher`：`NodeKey` 已经 `hash_cat`，不得再次 hash。
    shards: Box<[ShardNodes]>,
}

impl NodeRepository {
    /// KataGo `GraphSearch.md` 的 idempotent 更新：只读各 edge 的局部 N 与 child
    /// shared Q，从本 node 的首次预测 U 重算统计；不向其他 parent 广播。
    pub(crate) fn recompute_graph_node(&self, key: NodeKey) {
        let Some(node) = self.get(key) else { return };
        let base = *node.graph_value.lock();
        let Some(base) = base else { return };
        let edges = node.edges();
        let mut total = base;
        for edge in edges.iter() {
            let edge_stats = edge.completed_stats();
            let visits = edge_stats.visits;
            if visits == 0 {
                continue;
            }
            total.visits += visits;
            let local = edge_stats.local_terminal;
            total.wl_sum -= local.wl_sum;
            total.draw_sum += local.draw_sum;
            total.m_sum += local.m_sum + local.visits as f32;

            let propagated = visits.saturating_sub(local.visits);
            if propagated == 0 {
                continue;
            }
            let Some(child) = edge.child_key().and_then(|child| self.get(child)) else {
                continue;
            };
            let weight = propagated as f32;
            total.wl_sum -= child.q() * weight;
            total.draw_sum += child.draw() * weight;
            total.m_sum += (child.m() + 1.0) * weight;
        }
        let mut stats = node.stats.lock();
        stats.visits = total.visits;
        stats.wl_sum = total.wl_sum;
        stats.draw_sum = total.draw_sum;
        stats.m_sum = total.m_sum;
    }
    pub fn new(shard_count: usize) -> Self {
        assert!(
            shard_count.is_power_of_two(),
            "stream shard count must be a power of two"
        );
        assert!(shard_count > 0, "stream shard count must be non-zero");
        Self {
            shards: (0..shard_count).map(|_| RwLock::new(HashMap::default())).collect(),
        }
    }

    pub fn get_or_insert(&self, key: NodeKey) -> Arc<Node> {
        let shard = &self.shards[self.shard_index(key)];
        if let Some(node) = shard.read().get(&key) {
            return Arc::clone(node);
        }
        let mut nodes = shard.write();
        Arc::clone(nodes.entry(key).or_insert_with(|| Arc::new(Node::default())))
    }

    pub fn get(&self, key: NodeKey) -> Option<Arc<Node>> {
        let shard = &self.shards[self.shard_index(key)];
        shard.read().get(&key).cloned()
    }

    fn shard_index(&self, key: NodeKey) -> usize {
        key.0 as usize & (self.shards.len() - 1)
    }

    /// 只保留所有 retained root 可达的 graph node。图中 child 可有多个 parent，不能按
    /// sibling subtree 删除；必须先从根遍历再按 shard 批量回收。
    pub(crate) fn retain_reachable(&self, roots: impl IntoIterator<Item = NodeKey>) -> usize {
        let mut pending: Vec<_> = roots.into_iter().collect();
        let mut reachable: HashSet<NodeKey, BuildHasherDefault<NoHashHasher<u64>>> = HashSet::default();
        while let Some(key) = pending.pop() {
            if !reachable.insert(key) {
                continue;
            }
            let Some(node) = self.get(key) else {
                continue;
            };
            pending.extend(node.edges().iter().filter_map(|edge| edge.child_key()));
        }

        let mut removed = 0;
        for shard in self.shards.iter() {
            let mut nodes = shard.write();
            let before = nodes.len();
            nodes.retain(|key, _| reachable.contains(key));
            removed += before - nodes.len();
        }
        removed
    }

    /// 检查 `root` 以下的 edge-local reservation 不变量。
    pub(crate) fn graph_is_settled(&self, root: NodeKey) -> bool {
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
            pending.extend(edges.iter().filter_map(|edge| edge.child_key()));
        }
        true
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.read().len()).sum()
    }
}

impl Default for NodeRepository {
    fn default() -> Self {
        Self::new(64)
    }
}

// 跨回合复用保留已走根，并回收不再从任一 retained root 可达的 node。

/// 已走着替换当前 root 时回收的 node 数。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcStats {
    pub removed_nodes: usize,
}

/// 两次已完成 stream 搜索之间保留的 graph 状态。
#[derive(Debug)]
pub struct SearchGraph {
    repository: Arc<NodeRepository>,
    /// 保留的已走主线：从最早 root 到当前 root。
    root_keys: Vec<NodeKey>,
    /// 与 `root_keys` 对齐的完整历史；末项就是当前 root。快照使 UCI 能精确悔棋和
    /// 复用，不必根据 hash 重建局面。
    root_histories: Vec<Arc<PositionHistory>>,
}

impl SearchGraph {
    pub fn new(root_history: Arc<PositionHistory>) -> Self {
        let root = NodeKey::board(root_history.last().board().hash());
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
        *self.root_keys.last().expect("search graph always has a root")
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        self.root_histories
            .last()
            .expect("search graph always has a root history")
    }

    /// 在当前 root 以下的 event 都完成或取消后，推进到一个合法 child。
    /// 旧 root 留作悔棋根；随后按所有 retained root 的可达性回收。
    pub fn advance(&mut self, mv: Move) -> Result<GcStats, EnginError> {
        let old_root = self.root_key();
        if !self.repository.graph_is_settled(old_root) {
            return Err(EnginError::PortIncomplete(
                "search graph advance requires settled reservations",
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
        debug_assert!(self.repository.graph_is_settled(self.root_key()));
        self.advance_settled(mv)
    }

    fn advance_settled(&mut self, mv: Move) -> Result<GcStats, EnginError> {
        if !self.root_history().last().board().is_legal_move(mv) {
            return Err(EnginError::PortIncomplete("search graph advance requires a legal move"));
        }

        let mut history = self.root_history().as_ref().clone();
        history.append(mv);
        let new_root = NodeKey::board(history.last().board().hash());
        self.repository.get_or_insert(new_root);
        self.root_keys.push(new_root);
        self.root_histories.push(Arc::new(history));
        Ok(GcStats {
            removed_nodes: self.repository.retain_reachable(self.root_keys.iter().copied()),
        })
    }

    /// 返回前一个 retained root，并回收不再能从任一保留 root 到达的 future node。
    pub fn rewind_one(&mut self) -> Result<bool, EnginError> {
        if self.root_keys.len() == 1 {
            return Ok(false);
        }
        if !self.repository.graph_is_settled(self.root_key()) {
            return Err(EnginError::PortIncomplete(
                "search graph rewind requires settled reservations",
            ));
        }
        self.root_keys.pop();
        self.root_histories.pop();
        self.repository.retain_reachable(self.root_keys.iter().copied());
        Ok(true)
    }

    /// 将可复用图定位到完整 UCI history。已保留前缀复用其 root；无关 history
    /// 创建新 repository。参考 px0 `NodeTree::ResetToPosition`
    /// （`src/search/classic/node.cc:484-520`）。
    pub fn reset_to_history(&mut self, target: Arc<PositionHistory>) -> Result<GcStats, EnginError> {
        if !self.repository.graph_is_settled(self.root_key()) {
            return Err(EnginError::PortIncomplete(
                "search graph reset requires settled reservations",
            ));
        }

        self.reset_to_history_settled(target, false)
    }

    /// Engine 的 `position` 路径在调用前已经 abort 并 drain 当前 job，因此无需在
    /// release 构建重复遍历整个 repository 检查 reservation。
    pub(crate) fn reset_to_history_after_drain(&mut self, target: Arc<PositionHistory>) -> Result<GcStats, EnginError> {
        debug_assert!(self.repository.graph_is_settled(self.root_key()));
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
            return Ok(GcStats {
                removed_nodes: self.repository.retain_reachable(self.root_keys.iter().copied()),
            });
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
                        "search graph reset could not derive legal move",
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

        let root = NodeKey::board(target.last().board().hash());
        self.repository = Arc::new(NodeRepository::default());
        self.root_keys = vec![root];
        self.root_histories = vec![target];
        Ok(GcStats::default())
    }
}

#[cfg(test)]
mod graph_tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{NodeKey, SearchGraph};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    fn graph() -> SearchGraph {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        SearchGraph::new(Arc::new(PositionHistory::from_positions(state.positions())))
    }

    #[test]
    fn advance_keeps_old_root_and_all_nodes_reachable_from_undo_root() {
        let mut tree = graph();
        let old_root = tree.root_key();
        let keep = mv("a0", "a1");
        let drop = mv("a0", "a2");
        let root = tree.repository().get_or_insert(old_root);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(keep, 0.5), (drop, 0.5)]);
        let mut kept_history = tree.root_history().as_ref().clone();
        kept_history.append(keep);
        let kept_child = NodeKey::board(kept_history.last().board().hash());
        let mut dropped_history = tree.root_history().as_ref().clone();
        dropped_history.append(drop);
        let dropped_child = NodeKey::board(dropped_history.last().board().hash());
        let dropped_grandchild = NodeKey::board(0xdead_beef);
        root.edges()[0].bind_child_key(kept_child);
        root.edges()[1].bind_child_key(dropped_child);
        tree.repository().get_or_insert(kept_child);
        let dropped = tree.repository().get_or_insert(dropped_child);
        assert!(dropped.try_begin_evaluation());
        dropped.publish_edges(vec![(mv("a9", "a8"), 1.0)]);
        dropped.edges()[0].bind_child_key(dropped_grandchild);
        tree.repository().get_or_insert(dropped_grandchild);
        assert_eq!(tree.repository().len(), 4);

        let stats = tree.advance(keep).expect("advance");
        assert_eq!(stats.removed_nodes, 0);
        assert_eq!(tree.repository().len(), 4);
        assert_eq!(tree.root_key(), kept_child);
        assert_eq!(tree.root_history().len(), 2);
        assert!(tree.repository().get(old_root).is_some());
        assert!(tree.repository().get(kept_child).is_some());
        assert!(tree.repository().get(dropped_child).is_some());
        assert!(tree.repository().get(dropped_grandchild).is_some());
    }

    #[test]
    fn rewind_reclaims_unretained_future_root() {
        let mut tree = graph();
        let old_root = tree.root_key();
        let played = mv("a0", "a1");
        tree.advance(played).expect("advance");
        let played_root = tree.root_key();
        assert!(tree.rewind_one().expect("rewind"));
        assert_eq!(tree.root_key(), old_root);
        assert_eq!(tree.root_history().len(), 1);
        assert!(tree.repository().get(played_root).is_none());
        assert!(!tree.rewind_one().expect("root cannot rewind"));
    }

    #[test]
    fn advance_rejects_an_in_flight_reservation() {
        let mut tree = graph();
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
        let mut tree = graph();
        let game = GameState::from_fen_moves(STARTPOS_FEN, &["a0a1", "a9a8"]).expect("legal line");
        let first = game.moves[0];
        let second = game.moves[1];
        tree.advance(first).expect("first advance");
        let first_history = tree.root_history().clone();
        tree.advance(second).expect("second advance");

        tree.reset_to_history(first_history.clone())
            .expect("rewind through reset");
        assert_eq!(tree.root_history(), &first_history);
        let target = Arc::new(PositionHistory::from_positions(game.positions()));
        tree.reset_to_history(target).expect("replay continuation");
        assert_eq!(tree.root_history().len(), 3);
    }

    #[test]
    fn reset_to_unrelated_history_starts_fresh_repository() {
        let mut tree = graph();
        tree.advance(mv("a0", "a1")).expect("advance");
        let unrelated = GameState::from_fen_moves(STARTPOS_FEN, &["b0b1"]).expect("other legal line");
        let target = Arc::new(PositionHistory::from_positions(unrelated.positions()));

        tree.reset_to_history(target.clone()).expect("fresh tree");
        assert_eq!(tree.root_history(), &target);
        assert_eq!(tree.repository().len(), 0);
        assert_eq!(tree.root_key(), NodeKey::board(target.last().board().hash()));
    }
}

#[cfg(test)]
mod repository_tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use xiangqi_core::{Move, Square};

    use super::{ExpansionState, NodeKey, NodeRepository};
    use crate::search::ValueDelta;

    fn b2_b3() -> Move {
        Move::new(Square::parse("b2").expect("b2"), Square::parse("b3").expect("b3"))
    }

    #[test]
    fn graph_recompute_keeps_action_visits_local_to_its_parent() {
        let repo = NodeRepository::default();
        let a = NodeKey::board(1);
        let b = NodeKey::board(2);
        let c = NodeKey::board(3);
        let d = NodeKey::board(4);
        let ab = b2_b3();
        let ac = Move::new(Square::parse("c2").unwrap(), Square::parse("c3").unwrap());
        let bd = Move::new(Square::parse("d2").unwrap(), Square::parse("d3").unwrap());
        let cd = Move::new(Square::parse("e2").unwrap(), Square::parse("e3").unwrap());
        for (key, edges, value) in [
            (a, vec![(ab, 0.5), (ac, 0.5)], 0.0),
            (b, vec![(bd, 1.0)], 0.0),
            (c, vec![(cd, 1.0)], 0.0),
            (d, vec![], 0.8),
        ] {
            let node = repo.get_or_insert(key);
            assert!(node.try_begin_evaluation());
            node.publish_edges(edges);
            node.set_graph_value(ValueDelta::one(value, 0.0));
        }
        repo.get(a).unwrap().edges()[0].bind_child_key(b);
        repo.get(a).unwrap().edges()[1].bind_child_key(c);
        repo.get(b).unwrap().edges()[0].bind_child_key(d);
        repo.get(c).unwrap().edges()[0].bind_child_key(d);
        for key in [a, b, c, d] {
            repo.recompute_graph_node(key);
        }
        repo.get(b).unwrap().reserve_edge(0).unwrap().complete();
        repo.recompute_graph_node(b);
        repo.get(a).unwrap().reserve_edge(0).unwrap().complete();
        repo.recompute_graph_node(a);
        assert_eq!(repo.get(c).unwrap().edges()[0].completed_visits(), 0);
        assert_eq!(repo.get(c).unwrap().completed_visits(), 1);
        assert_eq!(repo.get(b).unwrap().completed_visits(), 2);
        assert!((repo.get(b).unwrap().q() + 0.4).abs() < 1e-6);
    }

    #[test]
    fn one_worker_claims_evaluation_and_edge_reservation_balances() {
        let repository = Arc::new(NodeRepository::default());
        let key = NodeKey::board(123);
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
        edge.complete();
        assert_eq!(node.edges()[0].completed_visits(), 1);
    }

    #[test]
    fn cancelled_reservation_restores_started_visit_count() {
        let root = NodeRepository::default().get_or_insert(NodeKey::board(321));
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
            let root = NodeRepository::default().get_or_insert(NodeKey::board(654));
            assert!(root.try_begin_evaluation());
            root.publish_edges(vec![(b2_b3(), 1.0)]);
            let completed = root.reserve_edge(0).expect("completed reservation");
            let cancelled = root.reserve_edge(0).expect("cancelled reservation");
            let barrier = Arc::new(Barrier::new(3));

            thread::scope(|scope| {
                let complete_barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    complete_barrier.wait();
                    completed.complete();
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
        let root = NodeRepository::default().get_or_insert(NodeKey::board(456));
        assert!(root.try_begin_evaluation());
        root.abort_evaluation();
        assert_eq!(root.expansion_state(), ExpansionState::Unexpanded);
        assert!(root.try_begin_evaluation());
    }
}
