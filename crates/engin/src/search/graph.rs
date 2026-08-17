//! stream 搜索的分片 node repository 与 edge-local reservation。
//!
//! 普通 MCGS 使用 board key：相同棋盘（含行棋方）共享 node；真实重复后进入的
//! ContinuationTree 改用带规则 history 的 key。走法统计始终留在 parent edge。
//! repository 角色可参考 LC3 Overview 的 “Node repository”；统计语义见 `MCGS.md`。
//! <https://lczero.org/dev/lc0/search/lc3/overview/>

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use nohash_hasher::{IsEnabled, NoHashHasher};
use parking_lot::{Mutex, RwLock};
use xiangqi_core::{Move, PositionHistory, hashcat::hash_cat};

use crate::EnginError;

use super::ValueDelta;

/// repository 的标识。普通 MCGS node 只以当前棋盘（含行棋方）划分；ContinuationTree
/// node 额外纳入规则 history。历史终局仍由 event variation 的私有 history 裁决。
///
/// `u64` 已由 `hash_cat` 混合。分片 map 使用 `nohash_hasher::NoHashHasher`，直接
/// 以此值为 bucket index，不再进行第二次 hash。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum NodeKey {
    /// 普通 MCGS state。相同棋盘（含行棋方）共享 node 与后续统计。
    GraphNode { board_hash: u64 },
    /// 首次重复后的规则敏感 state。相同棋盘但不同历史不得共享 N/Q。
    TreeNode { board_hash: u64, rule_context_hash: u64 },
}

impl Default for NodeKey {
    fn default() -> Self {
        Self::GraphNode { board_hash: 0 }
    }
}

impl Hash for NodeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.storage_hash());
    }
}

// 断言 `Hash` 只调用一次 `write_u64`，这是 `NoHashHasher` 的要求。
impl IsEnabled for NodeKey {}

impl NodeKey {
    /// MCGS 的共享 state identity：只由当前棋盘（含行棋方）决定。
    pub const fn board(board_hash: u64) -> Self {
        Self::GraphNode { board_hash }
    }

    /// 图闭环处的纯树形续搜 node。相同棋盘但规则 history 不同，不共享 N/Q。
    pub fn continuation(board_hash: u64, context: u64) -> Self {
        Self::TreeNode {
            board_hash,
            rule_context_hash: context,
        }
    }

    /// 当前 history 应使用的 repository identity。
    ///
    /// 普通 MCGS 只按 board 共享。真实重复进入 TreeNode；零化着清空重复规则
    /// 上下文，Tree 内该 edge 的 child 随即回到普通 GraphNode。这样本回合已展开的
    /// 零化后子图可在实际走子后的下一回合直接复用，而 Tree 先前的统计不会迁移。
    pub fn for_history(history: &PositionHistory) -> Self {
        let board_hash = history.last().board().hash();
        if history.did_repeat_since_last_zeroing_move() {
            Self::continuation(board_hash, history.rule_context_hash())
        } else {
            Self::board(board_hash)
        }
    }

    pub const fn is_continuation(self) -> bool {
        matches!(self, Self::TreeNode { .. })
    }

    fn storage_hash(self) -> u64 {
        match self {
            Self::GraphNode { board_hash } => board_hash,
            Self::TreeNode {
                board_hash,
                rule_context_hash,
            } => hash_cat(hash_cat(board_hash, rule_context_hash), 1),
        }
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
    /// 已绑定的 child state。普通 Graph edge 与 Tree 的零化 edge 都可永久绑定；
    /// Graph → Tree 入口因 child context 取决于本次 variation，刻意保持未绑定。
    child_key: OnceLock<NodeKey>,
    /// 若接入 child 会闭合 shared-Q 图环，此合法着不参与本 graph 的 PUCT。它不是
    /// 规则裁决；真实 variation 重复由 ContinuationTree 继续搜索。
    topology_pruned: OnceLock<()>,
    /// policy prior 在 node 发布后不再变化；普通不可变 `f32` 即可安全地被所有
    /// Gather worker 读取，不需要为 PUCT 热路径使用原子加载。
    prior: f32,
    started: AtomicU32,
    /// 已完成访问和其中 variation-local 终局的实际样本。
    ///
    /// 同一 board edge 可由不同 history 到达。重复、连将/追击与 rule60 的裁决不能
    /// first-writer-wins；每次命中的路径终局只计入自己的这一次 edge visit。
    /// 参考 KataGo `docs/GraphSearch.md` 的 edge-local action statistic。
    ///
    /// 受保护的聚合值：必须同时读取 completed N 与 local leaf 样本。
    /// 与 `started` 协同维护 `started >= completed`。`complete` 与 `cancel` 必须在
    /// 同一把锁下先检查、再修改，不能拆成独立原子计数。
    completed: Mutex<EdgeStats>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EdgeStats {
    /// 已完成的 logical visit。它决定 PUCT 的 N 与 action Q 的权重。
    pub visits: u32,
    /// 实际完成的 leaf event 数。一个 batch leaf 可代表多个 logical visit；因此它
    /// 不能由 `visits` 推导。根 LCB 用它避免把同一次 NN 结果当成独立证据。
    pub observations: u32,
    /// 每个物理 leaf 的 logical 权重平方和。LCB 用它计算加权 observation 的有效
    /// 样本量：`visits² / Σ(weight²)`。
    pub observation_weight_sq_sum: u64,
    pub local_leaf: ValueDelta,
    /// 尚未完成的 virtual mean 之和；in-flight 数直接由 `started - visits` 推导。
    pub virtual_wl_sum: f32,
}

impl Edge {
    fn new(mv: Move, prior: f32) -> Self {
        assert!((0.0..=1.0).contains(&prior), "policy prior must be normalized");
        Self {
            mv,
            child_key: OnceLock::new(),
            topology_pruned: OnceLock::new(),
            prior,
            started: AtomicU32::new(0),
            completed: Mutex::new(EdgeStats::default()),
        }
    }

    pub fn mv(&self) -> Move {
        self.mv
    }

    pub fn prior(&self) -> f32 {
        self.prior
    }

    /// LC3 edge N 包含 in-flight visit。GPU 评估未返回时，另存 completed N 才能形成
    /// 稳定的 Q。
    pub fn visits(&self) -> u32 {
        self.started.load(Ordering::Acquire)
    }

    pub fn completed_visits(&self) -> u32 {
        self.completed.lock().visits
    }

    pub(crate) fn stats(&self) -> EdgeStats {
        *self.completed.lock()
    }

    /// 供 selection 同时读取已完成统计与 started N。所有更新二者的路径都先持有
    /// `completed`，因此这个快照不会把另一份 reservation 的 virtual mean 与旧 N 混用。
    pub(crate) fn selection_snapshot(&self) -> (EdgeStats, u32) {
        let stats = *self.completed.lock();
        (stats, self.started.load(Ordering::Acquire))
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

    pub(crate) fn reserve(self: &Arc<Self>, visits: u32, virtual_mean: Option<f32>) -> EdgeReservation {
        assert!(visits > 0, "edge reservation must have positive weight");
        let virtual_wl_sum = virtual_mean.unwrap_or(0.0) * visits as f32;
        let mut completed = self.completed.lock();
        self.started.fetch_add(visits, Ordering::AcqRel);
        completed.virtual_wl_sum += virtual_wl_sum;
        EdgeReservation {
            edge: Arc::clone(self),
            visits,
            virtual_wl_sum,
        }
    }

    fn cancel(&self, visits: u32, virtual_wl_sum: f32) {
        // `complete` 也持有此锁。必须在同一临界区内检查并减少 `started`，否则
        // complete 可在检查与 CAS 之间增加 completed，破坏 started >= completed。
        let mut completed = self.completed.lock();
        let started = self.started.load(Ordering::Acquire);
        assert!(
            started.saturating_sub(completed.visits) >= visits,
            "stream edge reservation underflow"
        );
        completed.virtual_wl_sum -= virtual_wl_sum;
        let started = self.started.fetch_sub(visits, Ordering::AcqRel) - visits;
        if started == completed.visits {
            completed.virtual_wl_sum = 0.0;
        }
    }

    pub fn child_key(&self) -> Option<NodeKey> {
        self.child_key.get().copied()
    }

    pub(crate) fn topology_pruned(&self) -> bool {
        self.topology_pruned.get().is_some()
    }

    pub fn bind_child_key(&self, key: NodeKey) {
        assert!(!self.topology_pruned(), "topology-pruned edge cannot bind a child");
        if let Err(existing) = self.child_key.set(key) {
            assert_eq!(existing, key, "one edge must keep one child board");
        }
    }

    fn complete(&self, visits: u32, virtual_wl_sum: f32, local_leaf: Option<ValueDelta>) {
        let mut completed = self.completed.lock();
        assert!(
            self.started.load(Ordering::Acquire).saturating_sub(completed.visits) >= visits,
            "stream edge completion without reservation"
        );
        completed.virtual_wl_sum -= virtual_wl_sum;
        completed.visits += visits;
        if self.started.load(Ordering::Acquire) == completed.visits {
            completed.virtual_wl_sum = 0.0;
        }
        // 一个 BackpropEvent 对应一个真正到达的 leaf；即使它把一个未展开叶子的
        // NN 结果展开为 K 个 logical visit，也仍只有一份独立 Evidence。
        completed.observations += 1;
        let weight = u64::from(visits);
        completed.observation_weight_sq_sum = completed
            .observation_weight_sq_sum
            .saturating_add(weight.saturating_mul(weight));
        if let Some(value) = local_leaf {
            assert_eq!(value.visits, visits, "local leaf must match reservation weight");
            completed.local_leaf = completed.local_leaf.merge(value);
        }
    }
}

/// 一次待完成访问。它必须恰好被 `complete` 或 `cancel` 消费一次，确保
/// stream 的 reservation 不泄漏。
#[derive(Debug)]
pub struct EdgeReservation {
    edge: Arc<Edge>,
    visits: u32,
    virtual_wl_sum: f32,
}

impl EdgeReservation {
    pub fn mv(&self) -> Move {
        self.edge.mv()
    }

    pub fn complete(self) {
        self.edge.complete(self.visits, self.virtual_wl_sum, None);
    }

    /// 将已保留的 visit 预算交给多个后继 event。它不改动 edge 的 started N；各份
    /// reservation 之后各自 complete/cancel，合计恰好消费原来的 reservation。
    pub(crate) fn split(self, weights: &[u32]) -> Vec<Self> {
        assert_eq!(
            weights.iter().sum::<u32>(),
            self.visits,
            "reservation split must preserve visits"
        );
        weights
            .iter()
            .map(|&visits| Self {
                edge: Arc::clone(&self.edge),
                visits,
                virtual_wl_sum: self.virtual_wl_sum * visits as f32 / self.visits as f32,
            })
            .collect()
    }

    pub(crate) fn merge(parts: Vec<Self>) -> Self {
        assert!(!parts.is_empty(), "reservation merge requires a part");
        let edge = Arc::clone(&parts[0].edge);
        let visits = parts
            .iter()
            .map(|part| {
                assert!(Arc::ptr_eq(&edge, &part.edge), "only one edge can be merged");
                part.visits
            })
            .sum();
        let virtual_wl_sum = parts.iter().map(|part| part.virtual_wl_sum).sum();
        Self {
            edge,
            visits,
            virtual_wl_sum,
        }
    }

    pub(crate) fn visits(&self) -> u32 {
        self.visits
    }

    pub(crate) fn complete_local_leaf(self, value: ValueDelta) {
        self.edge.complete(self.visits, self.virtual_wl_sum, Some(value));
    }

    pub fn cancel(self) {
        self.edge.cancel(self.visits, self.virtual_wl_sum);
    }

    #[cfg(test)]
    pub(crate) fn test_only(mv: Move) -> Self {
        Arc::new(Edge::new(mv, 1.0)).reserve(1, None)
    }
}

/// 已完成 node WDL 聚合值（`wl_sum` / `draw_sum` 对应 px0 `wl_` / `d_`）。
#[derive(Debug, Default)]
struct NodeStats {
    visits: u32,
    wl_sum: f32,
    wl_sq_sum: f32,
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
    /// 节点首次 Prediction 也要带 multivisit 权重，保证后续 shared-Q 重算的 base N
    /// 与入边 completed N 一致。
    pub(crate) fn set_graph_value(&self, value: ValueDelta) {
        assert!(value.visits > 0, "graph node base value must have positive weight");
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
        self.value_snapshot().0
    }

    /// 在同一把统计锁下读取 Q、WDL/M 与二阶矩，避免 parent 图回算混合 child
    /// 不同完成时刻的快照。
    pub(crate) fn value_moments_snapshot(&self) -> (f32, f32, f32, f32) {
        let stats = self.stats.lock();
        if stats.visits == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let visits = stats.visits as f32;
        (
            stats.wl_sum / visits,
            stats.draw_sum / visits,
            stats.m_sum / visits,
            stats.wl_sq_sum / visits,
        )
    }

    pub fn draw(&self) -> f32 {
        self.value_snapshot().1
    }

    pub fn m(&self) -> f32 {
        self.value_snapshot().2
    }

    /// 在同一把统计锁下读取 WDL/M，避免 parent 重算混合 child 的两个不同版本。
    pub(crate) fn value_snapshot(&self) -> (f32, f32, f32) {
        let (wl, draw, m, _) = self.value_moments_snapshot();
        (wl, draw, m)
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
        edges.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
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

    pub(crate) fn set_terminal_value_weighted(&self, wl: f32, draw: f32, plies_left: f32, visits: u32) {
        assert_eq!(
            self.expansion_state(),
            ExpansionState::Evaluating,
            "node must be evaluating"
        );
        *self.terminal.lock() = Some((wl, draw, plies_left));
        self.set_graph_value(ValueDelta::with_plies_left(wl, draw, plies_left).repeated(visits));
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
        self.reserve_edge_visits(edge_index, 1, None)
    }

    /// Gather 的一组 logical visit 已在此 edge 上预占；只在分裂/终局合并时使用。
    pub(crate) fn reserve_edge_visits(
        &self,
        edge_index: usize,
        visits: u32,
        virtual_mean: Option<f32>,
    ) -> Option<EdgeReservation> {
        self.edges()
            .get(edge_index)
            .map(|edge| edge.reserve(visits, virtual_mean))
    }
}

type ShardNodes = RwLock<HashMap<NodeKey, Arc<Node>, BuildHasherDefault<NoHashHasher<u64>>>>;

/// 分片 key-value repository。分片锁只保护 map 查找和插入；node 统计存放在各自的
/// node/edge 对象之后。
#[derive(Debug)]
pub struct NodeRepository {
    /// `NoHashHasher`：`NodeKey` 已经 `hash_cat`，不得再次 hash。
    shards: Box<[ShardNodes]>,
    /// 只保护 node 创建、edge 绑定和后台 mark/sweep 的拓扑一致性；N/Q 不经过它。
    topology_lock: RwLock<()>,
    /// 仅序列化“检查新边是否会闭环 + 绑定新边”这一极少发生的操作。正常 PUCT、
    /// reservation 与统计更新均不经过此锁。
    ///
    /// KataGo `cpp/search/search.cpp:1426-1445` 在单条 playout 的 graph path 上截断
    /// cycle；这里的 shared-Q 重算要求更强：不允许 repository 的已绑定边形成 Q
    /// 依赖环，故在第一次连接 edge 时做一次 DFS。
    link_lock: Mutex<()>,
}

/// 新 edge 指向已有 board 时的连接结果。
#[derive(Clone, Copy, Debug)]
pub(crate) enum ChildLink {
    Bound,
    TopologyPruned,
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
            let edge_stats = edge.stats();
            let visits = edge_stats.visits;
            if visits == 0 {
                continue;
            }
            total.visits += visits;
            let local = edge_stats.local_leaf;
            total.wl_sum -= local.wl_sum;
            total.wl_sq_sum += local.wl_sq_sum;
            total.draw_sum += local.draw_sum;
            total.m_sum += local.m_sum + local.visits as f32;

            let propagated = visits.saturating_sub(local.visits);
            if propagated == 0 {
                continue;
            }
            let Some(child) = edge.child_key().and_then(|child| self.get(child)) else {
                continue;
            };
            let (child_q, child_draw, child_m, child_q_sq) = child.value_moments_snapshot();
            let weight = propagated as f32;
            total.wl_sum -= child_q * weight;
            total.wl_sq_sum += child_q_sq * weight;
            total.draw_sum += child_draw * weight;
            total.m_sum += (child_m + 1.0) * weight;
        }
        let mut stats = node.stats.lock();
        stats.visits = total.visits;
        stats.wl_sum = total.wl_sum;
        stats.wl_sq_sum = total.wl_sq_sum;
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
            topology_lock: RwLock::new(()),
            link_lock: Mutex::new(()),
        }
    }

    pub fn get_or_insert(&self, key: NodeKey) -> Arc<Node> {
        let _topology = self.topology_lock.read();
        self.get_or_insert_unlocked(key)
    }

    fn get_or_insert_unlocked(&self, key: NodeKey) -> Arc<Node> {
        let shard = &self.shards[self.shard_index(key)];
        if let Some(node) = shard.read().get(&key) {
            return Arc::clone(node);
        }
        let mut nodes = shard.write();
        Arc::clone(nodes.entry(key).or_insert_with(|| Arc::new(Node::default())))
    }

    pub fn get(&self, key: NodeKey) -> Option<Arc<Node>> {
        let _topology = self.topology_lock.read();
        self.get_unlocked(key)
    }

    fn get_unlocked(&self, key: NodeKey) -> Option<Arc<Node>> {
        let shard = &self.shards[self.shard_index(key)];
        shard.read().get(&key).cloned()
    }

    /// 原子地检查并绑定一条此前未绑定的 graph edge。
    ///
    /// 若 `child` 已能沿已绑定 edge 到达 `parent`，再连接 `parent -> child` 会让
    /// `recompute_graph_node` 的 shared Q 产生循环依赖。此时永久过滤这条 edge；它
    /// 不是棋规终局，调用方取消本次 reservation 后继续选择其他 edge。
    pub(crate) fn bind_child_or_cut_cycle(&self, parent: NodeKey, edge: &Edge, child: NodeKey) -> ChildLink {
        let _topology = self.topology_lock.read();
        let _link = self.link_lock.lock();
        if let Some(existing) = edge.child_key() {
            assert_eq!(existing, child, "one edge must keep one child board");
            return ChildLink::Bound;
        }
        if edge.topology_pruned() {
            return ChildLink::TopologyPruned;
        }
        // 新 child 尚未进入 repository 时不可能沿既有图边回到 parent，直接绑定。
        // 这避免绝大多数首次 expansion 的无意义 DFS。
        if self.get_unlocked(child).is_none() {
            self.get_or_insert_unlocked(child);
            edge.bind_child_key(child);
            return ChildLink::Bound;
        }
        if self.reaches_unlocked(child, parent) {
            edge.topology_pruned
                .set(())
                .expect("cycle-cut edge is initialized once under link lock");
            return ChildLink::TopologyPruned;
        }
        edge.bind_child_key(child);
        ChildLink::Bound
    }

    /// 新边检查专用 DFS。它只读取已绑定 edge，且只在首次连接 edge 时调用。
    fn reaches_unlocked(&self, from: NodeKey, target: NodeKey) -> bool {
        let mut pending = vec![from];
        let mut seen: HashSet<NodeKey, BuildHasherDefault<NoHashHasher<u64>>> = HashSet::default();
        while let Some(key) = pending.pop() {
            if !seen.insert(key) {
                continue;
            }
            if key == target {
                return true;
            }
            let Some(node) = self.get_unlocked(key) else {
                continue;
            };
            pending.extend(node.edges().iter().filter_map(|edge| edge.child_key()));
        }
        false
    }

    /// 以当前 root 为唯一保留根回收不可达图。
    ///
    /// GC 持有 topology 写锁，node 创建与 edge 绑定会短暂等待，因而不会删掉刚被
    /// transposition 接回的 node；统计更新不受此锁影响。参考 LC3 Overview 的
    /// "Node repository"；LC3 未公开 GC 细节，mark/sweep 是当前单图语义的直接实现。
    pub(crate) fn retain_from_root(&self, root: NodeKey) -> usize {
        let _topology = self.topology_lock.write();
        let _link = self.link_lock.lock();
        let mut pending = vec![root];
        let mut reachable: HashSet<NodeKey, BuildHasherDefault<NoHashHasher<u64>>> = HashSet::default();
        while let Some(key) = pending.pop() {
            if !reachable.insert(key) {
                continue;
            }
            let Some(node) = self.get_unlocked(key) else {
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

    fn shard_index(&self, key: NodeKey) -> usize {
        key.storage_hash() as usize & (self.shards.len() - 1)
    }

    /// 检查整个 repository 是否没有 edge-local reservation。
    ///
    /// Graph → ContinuationTree 的入口不能绑定 contextual child；只从 graph root
    /// 遍历会漏掉这类 Tree node。因此 reset / advance 的安全边界必须检查全部 node。
    pub(crate) fn is_settled(&self) -> bool {
        self.shards.iter().all(|shard| {
            shard
                .read()
                .values()
                .all(|node| node.edges().iter().all(|edge| edge.visits() == edge.completed_visits()))
        })
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.read().len()).sum()
    }

    /// 无关 position 换图后由后台线程逐 shard 释放整张旧图。
    ///
    /// 该 repository 已与 Engine 脱离且不再被 worker 使用；逐 shard `yield` 只降低释放
    /// 峰值，不改变任何活跃图的内容。
    pub(crate) fn release_incrementally(self) {
        for shard in self.shards {
            drop(shard);
            std::thread::yield_now();
        }
    }
}

impl Default for NodeRepository {
    fn default() -> Self {
        Self::new(64)
    }
}

// 跨回合复用保留已走根，并回收不再从任一 retained root 可达的 node。

/// 两次已完成 stream 搜索之间保留的 graph 状态。
#[derive(Debug)]
pub struct SearchGraph {
    repository: Arc<NodeRepository>,
    root_key: NodeKey,
    /// 当前 root 的完整历史。悔棋由下一条 UCI `position ... moves` 重建，不保留旧
    /// root 的所有 sibling 图，否则 GC 永远无法回收它们。
    root_history: Arc<PositionHistory>,
    gc_pending: bool,
}

impl SearchGraph {
    pub fn new(root_history: Arc<PositionHistory>) -> Self {
        let root = NodeKey::for_history(root_history.as_ref());
        Self {
            repository: Arc::new(NodeRepository::default()),
            root_key: root,
            root_history,
            gc_pending: false,
        }
    }

    pub fn repository(&self) -> &Arc<NodeRepository> {
        &self.repository
    }

    pub fn root_key(&self) -> NodeKey {
        self.root_key
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        &self.root_history
    }

    /// 在当前 root 以下的 event 都完成或取消后，推进到一个合法 child。
    ///
    /// 旧 root 留作悔棋；它仍可到达原图的所有已绑定分支，所以同步 DFS 回收不会删掉
    /// 正常对局中的 node，只会阻塞下一回合。这条路径只更新 root/history。
    pub fn advance(&mut self, mv: Move) -> Result<(), EnginError> {
        if !self.repository.is_settled() {
            return Err(EnginError::PortIncomplete(
                "search graph advance requires settled reservations",
            ));
        }
        self.advance_settled(mv)
    }

    /// Engine 已停止并 drain worker 后使用的推进路径。
    ///
    /// Engine 的 `set_position` 已保证 reservation 全部归还；此处不再重复全图扫描。
    fn advance_after_drain(&mut self, mv: Move) -> Result<(), EnginError> {
        self.advance_settled(mv)
    }

    fn advance_settled(&mut self, mv: Move) -> Result<(), EnginError> {
        if !self.root_history().last().board().is_legal_move(mv) {
            return Err(EnginError::PortIncomplete("search graph advance requires a legal move"));
        }

        let mut history = self.root_history().as_ref().clone();
        history.append(mv);
        let new_root = NodeKey::for_history(&history);
        self.repository.get_or_insert(new_root);
        self.root_key = new_root;
        self.root_history = Arc::new(history);
        self.gc_pending = true;
        Ok(())
    }

    /// 取出一次 root 推进请求的后台 mark/sweep 根。
    pub(crate) fn take_pending_gc_root(&mut self) -> Option<NodeKey> {
        self.gc_pending.then(|| {
            self.gc_pending = false;
            self.root_key
        })
    }

    /// 将可复用图定位到完整 UCI history。已保留前缀复用其 root；无关 history
    /// 创建新 repository。
    pub fn reset_to_history(
        &mut self,
        target: Arc<PositionHistory>,
    ) -> Result<Option<Arc<NodeRepository>>, EnginError> {
        if !self.repository.is_settled() {
            return Err(EnginError::PortIncomplete(
                "search graph reset requires settled reservations",
            ));
        }

        self.reset_to_history_settled(target, false)
    }

    /// Engine 的 `position` 路径在调用前已经 abort 并 drain 当前 job，因此无需重复
    /// 遍历整个 repository 检查 reservation。
    pub(crate) fn reset_to_history_after_drain(
        &mut self,
        target: Arc<PositionHistory>,
    ) -> Result<Option<Arc<NodeRepository>>, EnginError> {
        self.reset_to_history_settled(target, true)
    }

    fn reset_to_history_settled(
        &mut self,
        target: Arc<PositionHistory>,
        after_drain: bool,
    ) -> Result<Option<Arc<NodeRepository>>, EnginError> {
        if self.root_history == target {
            return Ok(None);
        }

        if target.len() > self.root_history().len()
            && target.positions()[..self.root_history().len()] == *self.root_history().positions()
        {
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
                if after_drain {
                    self.advance_after_drain(mv)?
                } else {
                    self.advance(mv)?
                };
            }
            return Ok(None);
        }

        let root = NodeKey::for_history(target.as_ref());
        let retired = std::mem::replace(&mut self.repository, Arc::new(NodeRepository::default()));
        self.root_key = root;
        self.root_history = target;
        self.gc_pending = false;
        // 已脱离 Engine 的旧 repository 不需要对共享图做 DFS；交给后台释放即可。
        Ok(Some(retired))
    }
}

#[cfg(test)]
mod graph_tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{NodeKey, SearchGraph};
    use crate::search::Variation;

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    fn graph() -> SearchGraph {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        SearchGraph::new(Arc::new(PositionHistory::from_positions(state.positions())))
    }

    #[test]
    fn graph_and_tree_keys_keep_their_identity_separate() {
        let graph = NodeKey::board(42);
        let first_tree = NodeKey::continuation(42, 1);
        let second_tree = NodeKey::continuation(42, 2);

        assert!(matches!(graph, NodeKey::GraphNode { .. }));
        assert!(matches!(first_tree, NodeKey::TreeNode { .. }));
        assert_ne!(graph, first_tree);
        assert_ne!(first_tree, second_tree);
    }

    fn repeated_history() -> Arc<PositionHistory> {
        let (board, _) = xiangqi_core::ChessBoard::from_fen("3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = PositionHistory::default();
        history.reset(board, 2, 30);
        for text in ["d9e9", "d2e2", "e9d9", "e2d2"] {
            let mv = history.last().board().parse_move(text).expect(text);
            history.append(mv);
        }
        assert!(history.did_repeat_since_last_zeroing_move());
        Arc::new(history)
    }

    #[test]
    fn advancing_and_collecting_keeps_only_the_new_root_graph() {
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

        tree.advance(keep).expect("advance");
        let gc_root = tree.take_pending_gc_root().expect("advanced root schedules GC");
        assert_eq!(tree.repository().retain_from_root(gc_root), 3);
        assert_eq!(tree.repository().len(), 1);
        assert_eq!(tree.root_key(), kept_child);
        assert_eq!(tree.root_history().len(), 2);
        assert!(tree.repository().get(old_root).is_none());
        assert!(tree.repository().get(kept_child).is_some());
        assert!(tree.repository().get(dropped_child).is_none());
        assert!(tree.repository().get(dropped_grandchild).is_none());
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
    fn repeated_history_keeps_continuation_tree_across_turns() {
        let history = repeated_history();
        let mut graph = SearchGraph::new(Arc::clone(&history));
        let root = NodeKey::for_history(history.as_ref());
        assert!(root.is_continuation());
        assert_eq!(graph.root_key(), root);

        let mv = history.last().board().parse_move("d9e9").expect("legal move");
        let mut variation = Variation::root(Arc::clone(&history));
        let expected_child = variation.child_key_for_history(mv);
        let root_node = graph.repository().get_or_insert(root);
        assert!(root_node.try_begin_evaluation());
        root_node.publish_edges(vec![(mv, 1.0)]);
        root_node.edges()[0].bind_child_key(expected_child);
        let child_node = graph.repository().get_or_insert(expected_child);
        child_node.set_graph_value(crate::search::ValueDelta::one(0.6, 0.0));
        graph.repository().recompute_graph_node(expected_child);

        graph.advance(mv).expect("advance contextual root");
        let gc_root = graph.take_pending_gc_root().expect("advance schedules GC");
        assert_eq!(graph.repository().retain_from_root(gc_root), 1);

        assert!(graph.root_key().is_continuation());
        assert_eq!(graph.root_key(), expected_child);
        let retained = graph
            .repository()
            .get(expected_child)
            .expect("retained contextual root");
        assert_eq!(retained.completed_visits(), 1);
        assert!((retained.q() - 0.6).abs() < f32::EPSILON);
        assert_eq!(NodeKey::for_history(graph.root_history().as_ref()), expected_child);
    }

    #[test]
    fn reset_to_old_history_starts_a_fresh_graph_then_replays_continuation() {
        let mut tree = graph();
        let game = GameState::from_fen_moves(STARTPOS_FEN, &["a0a1", "a9a8"]).expect("legal line");
        let first = game.moves[0];
        let second = game.moves[1];
        tree.advance(first).expect("first advance");
        let first_history = tree.root_history().clone();
        tree.advance(second).expect("second advance");

        assert!(
            tree.reset_to_history(first_history.clone())
                .expect("reset old history")
                .is_some()
        );
        assert_eq!(tree.root_history(), &first_history);
        let target = Arc::new(PositionHistory::from_positions(game.positions()));
        assert!(tree.reset_to_history(target).expect("replay continuation").is_none());
        assert_eq!(tree.root_history().len(), 3);
    }

    #[test]
    fn reset_to_unrelated_history_starts_fresh_repository() {
        let mut tree = graph();
        tree.advance(mv("a0", "a1")).expect("advance");
        let unrelated = GameState::from_fen_moves(STARTPOS_FEN, &["b0b1"]).expect("other legal line");
        let target = Arc::new(PositionHistory::from_positions(unrelated.positions()));

        let retired = tree
            .reset_to_history(target.clone())
            .expect("fresh tree")
            .expect("unrelated history retires old repository");
        assert_eq!(retired.len(), 1);
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

    use super::{ChildLink, EdgeReservation, ExpansionState, Node, NodeKey, NodeRepository};
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
    fn new_edge_that_would_close_a_graph_cycle_is_permanently_pruned() {
        // A -> B -> D 与 A -> C -> D 合并后，D -> B 会让 B/D 的 Q 互相依赖。
        // 这不是棋规重复；X7 直接将该 edge 排除在 shared 图之外。
        let repo = NodeRepository::default();
        let a = NodeKey::board(10);
        let b = NodeKey::board(11);
        let c = NodeKey::board(12);
        let d = NodeKey::board(13);
        let moves = [
            b2_b3(),
            Move::new(Square::parse("c2").unwrap(), Square::parse("c3").unwrap()),
            Move::new(Square::parse("d2").unwrap(), Square::parse("d3").unwrap()),
            Move::new(Square::parse("e2").unwrap(), Square::parse("e3").unwrap()),
            Move::new(Square::parse("f2").unwrap(), Square::parse("f3").unwrap()),
        ];
        for (key, edges, value) in [
            (a, vec![(moves[0], 0.5), (moves[1], 0.5)], 0.1),
            (b, vec![(moves[2], 1.0)], 0.25),
            (c, vec![(moves[3], 1.0)], 0.2),
            (d, vec![(moves[4], 1.0)], 0.4),
        ] {
            let node = repo.get_or_insert(key);
            assert!(node.try_begin_evaluation());
            node.set_graph_value(ValueDelta::one(value, 0.0));
            node.publish_edges(edges);
        }
        assert!(matches!(
            repo.bind_child_or_cut_cycle(a, &repo.get(a).unwrap().edges()[0], b),
            ChildLink::Bound
        ));
        assert!(matches!(
            repo.bind_child_or_cut_cycle(a, &repo.get(a).unwrap().edges()[1], c),
            ChildLink::Bound
        ));
        assert!(matches!(
            repo.bind_child_or_cut_cycle(b, &repo.get(b).unwrap().edges()[0], d),
            ChildLink::Bound
        ));
        assert!(matches!(
            repo.bind_child_or_cut_cycle(c, &repo.get(c).unwrap().edges()[0], d),
            ChildLink::Bound
        ));

        let d_node = repo.get(d).expect("D node");
        let d_edges = d_node.edges();
        let d_edge = &d_edges[0];
        let ChildLink::TopologyPruned = repo.bind_child_or_cut_cycle(d, d_edge, b) else {
            panic!("D -> B must be cut")
        };
        assert!(d_edge.child_key().is_none());
        assert!(d_edge.topology_pruned());
        assert!(matches!(
            repo.bind_child_or_cut_cycle(d, d_edge, b),
            ChildLink::TopologyPruned
        ));
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
    fn split_virtual_mean_is_fully_removed_on_completion() {
        let root = NodeRepository::default().get_or_insert(NodeKey::board(322));
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(b2_b3(), 1.0)]);
        let parts = root
            .reserve_edge_visits(0, 4, Some(-0.25))
            .expect("virtual mean reservation")
            .split(&[1, 3]);
        let reservation = EdgeReservation::merge(parts);
        let edge = &root.edges()[0];
        assert!((edge.stats().virtual_wl_sum + 1.0).abs() < f32::EPSILON);

        reservation.complete();

        let stats = edge.stats();
        assert_eq!(stats.visits, 4);
        assert_eq!(stats.virtual_wl_sum, 0.0);
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
    fn split_reservation_preserves_the_parent_visit_budget() {
        let node = Arc::new(Node::default());
        assert!(node.try_begin_evaluation());
        node.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        node.publish_edges(vec![(b2_b3(), 1.0)]);

        let parts = node
            .reserve_edge_visits(0, 4, None)
            .expect("weighted reservation")
            .split(&[1, 3]);
        assert_eq!(node.edges()[0].visits(), 4);
        parts.into_iter().for_each(|part| part.complete());
        assert_eq!(node.edges()[0].completed_visits(), 4);
        assert_eq!(node.edges()[0].in_flight_visits(), 0);
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
