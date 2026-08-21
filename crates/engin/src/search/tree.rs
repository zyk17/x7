//! 搜索树数据面：`NodeKey` / `Edge` / `Node` / `NodeRepository` / reservation。
//!
//! 只定义结构与原子操作，不调度流水线。`child = hash_cat(parent, move)`，不合并换位。

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use nohash_hasher::{IsEnabled, NoHashHasher};
use parking_lot::{Mutex, RwLock};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;

/// stream backpropagation 使用的紧凑 WDL 更新。
///
/// - `visits`：多份 `one()` 样本的合计，不是一次 reservation 携带的 K
/// - `wl_sum`：走子方 / incoming-edge 视角（非 NN 原始 STM）
/// - `draw_sum`：和棋分量
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValueDelta {
    pub visits: u32,
    pub wl_sum: f32,
    /// `wl²` 聚合。仅用于根 LCB 的 value dispersion，不参与 Q / PUCT。
    pub wl_sq_sum: f32,
    pub draw_sum: f32,
    pub m_sum: f32,
}

impl ValueDelta {
    pub fn one(wl: f32, draw: f32) -> Self {
        // 归一化由 Eval/终局入口保证；此处只在 debug 抓回归，避免 backprop 热路径开销。
        debug_assert!((-1.0..=1.0).contains(&wl), "WDL wl must be normalized");
        debug_assert!((0.0..=1.0).contains(&draw), "WDL draw must be normalized");
        Self {
            visits: 1,
            wl_sum: wl,
            wl_sq_sum: wl * wl,
            draw_sum: draw,
            m_sum: 0.0,
        }
    }

    pub fn with_plies_left(wl: f32, draw: f32, plies_left: f32) -> Self {
        debug_assert!(plies_left >= 0.0, "plies-left must be non-negative");
        Self {
            m_sum: plies_left,
            ..Self::one(wl, draw)
        }
    }

    pub fn for_parent(self) -> Self {
        Self {
            wl_sum: -self.wl_sum,
            ..self
        }
    }

    pub fn one_ply_up(self) -> Self {
        Self {
            m_sum: self.m_sum + self.visits as f32,
            ..self
        }
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            visits: self.visits + other.visits,
            wl_sum: self.wl_sum + other.wl_sum,
            wl_sq_sum: self.wl_sq_sum + other.wl_sq_sum,
            draw_sum: self.draw_sum + other.draw_sum,
            m_sum: self.m_sum + other.m_sum,
        }
    }

    pub fn q(self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.wl_sum / self.visits as f32
        }
    }
}

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

    pub const fn child(self, mv: Move) -> Self {
        Self(xiangqi_core::hashcat::hash_cat(self.0, mv.raw() as u64))
    }
}

/// child edge。in-flight visit 保存在入边，绝不计入 child node 的 completed visit。
#[derive(Debug)]
pub struct Edge {
    mv: Move,
    prior: f32,
    started: AtomicU32,
    /// 已完成 N/Q 与尚未完成的 virtual mean；选边要一起读。
    stats: Mutex<EdgeStats>,
}

/// edge 聚合。`wl_sum` 是走子方视角；`wl_sq_sum` 仅供根 LCB。
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EdgeStats {
    pub visits: u32,
    pub wl_sum: f32,
    pub wl_sq_sum: f32,
    pub virtual_wl_sum: f32,
}

impl Edge {
    fn new(mv: Move, prior: f32) -> Self {
        assert!((0.0..=1.0).contains(&prior), "policy prior must be normalized");
        Self {
            mv,
            prior,
            started: AtomicU32::new(0),
            stats: Mutex::new(EdgeStats::default()),
        }
    }

    pub fn mv(&self) -> Move {
        self.mv
    }

    pub fn prior(&self) -> f32 {
        self.prior
    }

    /// edge N 包含 in-flight visit。
    pub fn visits(&self) -> u32 {
        self.started.load(Ordering::Acquire)
    }

    pub fn completed_visits(&self) -> u32 {
        self.stats.lock().visits
    }

    pub fn in_flight_visits(&self) -> u32 {
        let completed = self.stats.lock().visits;
        self.started.load(Ordering::Acquire).saturating_sub(completed)
    }

    pub(crate) fn stats(&self) -> EdgeStats {
        *self.stats.lock()
    }

    pub(crate) fn selection_snapshot(&self) -> (EdgeStats, u32) {
        let stats = *self.stats.lock();
        (stats, self.started.load(Ordering::Acquire))
    }

    pub fn q(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 {
            return 0.0;
        }
        stats.wl_sum / stats.visits as f32
    }

    fn reserve(self: &Arc<Self>, virtual_mean: Option<f32>) -> EdgeReservation {
        let virtual_wl_sum = virtual_mean.unwrap_or(0.0);
        let mut stats = self.stats.lock();
        self.started.fetch_add(1, Ordering::AcqRel);
        stats.virtual_wl_sum += virtual_wl_sum;
        EdgeReservation {
            edge: Arc::clone(self),
            virtual_wl_sum,
        }
    }

    fn cancel(&self, virtual_wl_sum: f32) {
        let mut stats = self.stats.lock();
        let started = self.started.fetch_sub(1, Ordering::AcqRel);
        assert!(started > stats.visits, "stream edge reservation underflow");
        stats.virtual_wl_sum -= virtual_wl_sum;
        if started - 1 == stats.visits {
            stats.virtual_wl_sum = 0.0;
        }
    }

    fn complete(&self, virtual_wl_sum: f32, wl: f32) {
        let mut stats = self.stats.lock();
        let started = self.started.load(Ordering::Acquire);
        assert!(started > stats.visits, "stream edge completion without reservation");
        stats.virtual_wl_sum -= virtual_wl_sum;
        stats.visits += 1;
        stats.wl_sum += wl;
        stats.wl_sq_sum += wl * wl;
        if started == stats.visits {
            stats.virtual_wl_sum = 0.0;
        }
    }
}

/// 一次待完成访问。它必须恰好被 `complete` 或 `cancel` 消费一次。
#[derive(Debug)]
pub struct EdgeReservation {
    edge: Arc<Edge>,
    virtual_wl_sum: f32,
}

impl EdgeReservation {
    pub fn mv(&self) -> Move {
        self.edge.mv()
    }

    pub fn complete(self, wl: f32) {
        self.edge.complete(self.virtual_wl_sum, wl);
    }

    pub fn cancel(self) {
        self.edge.cancel(self.virtual_wl_sum);
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
    /// 精确证明只靠 `Terminal` 状态 + 本字段；不另存半开 bounds。
    terminal: Mutex<Option<(f32, f32, f32)>>,
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
        stats.wl_sq_sum += delta.wl_sq_sum;
        stats.draw_sum += delta.draw_sum;
        stats.m_sum += delta.m_sum;
    }

    /// 在同一把统计锁下读取 Q、WDL/M 与二阶矩。
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

    pub(crate) fn value_snapshot(&self) -> (f32, f32, f32) {
        let (wl, draw, m, _) = self.value_moments_snapshot();
        (wl, draw, m)
    }

    pub fn expansion_state(&self) -> ExpansionState {
        ExpansionState::from_raw(self.expansion.load(Ordering::Acquire))
    }

    /// `Unexpanded → Evaluating`：至多一个 Gather claim 成功，该叶子交给 Eval。
    /// 其余撞上 `Evaluating` 的路径由 Gather `park_collision`（保留 reservation / μ），
    /// 等该叶子 backprop complete 后再 cancel，不立刻取消、也不重复 Eval。
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
        self.expansion.store(ExpansionState::Terminal as u8, Ordering::Release);
    }

    fn mark_proven_terminal(&self, wl: f32, draw: f32, plies_left: f32) -> bool {
        let mut terminal = self.terminal.lock();
        if self.expansion_state() != ExpansionState::Expanded {
            return false;
        }
        *terminal = Some((wl, draw, plies_left));
        self.expansion.store(ExpansionState::Terminal as u8, Ordering::Release);
        true
    }

    /// 父 STM 强制胜 → 钉成 incoming `wl=-1`；已是强制胜则只缩短 plies。
    fn apply_forced_win(&self, plies_left: f32) {
        match self.expansion_state() {
            ExpansionState::Expanded => {
                self.mark_proven_terminal(-1.0, 0.0, plies_left);
            }
            ExpansionState::Terminal => self.shorten_terminal_plies(plies_left),
            _ => {}
        }
    }

    fn shorten_terminal_plies(&self, plies_left: f32) {
        let mut terminal = self.terminal.lock();
        let Some((wl, draw, old_plies)) = *terminal else {
            return;
        };
        if wl < 0.0 && plies_left + f32::EPSILON < old_plies {
            *terminal = Some((wl, draw, plies_left));
        }
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
        self.reserve_edge_with_virtual_mean(edge_index, None)
    }

    pub(crate) fn reserve_edge_with_virtual_mean(
        &self,
        edge_index: usize,
        virtual_mean: Option<f32>,
    ) -> Option<EdgeReservation> {
        self.edges().get(edge_index).map(|edge| edge.reserve(virtual_mean))
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

    /// 沿 path 向上传播强制终局（不存半开 bounds）。每层扫父的全部边：
    /// 任一儿子对父 STM 必胜 → 立刻钉父；必败/必和要全部儿子都 Terminal。
    /// `root` 不钉死。
    pub(crate) fn propagate_proven_terminals(&self, node_path: &[NodeKey], root: NodeKey) {
        for &parent_key in node_path.iter().rev().skip(1) {
            if parent_key == root {
                break;
            }
            let Some(parent) = self.get(parent_key) else {
                continue;
            };
            let edges = parent.edges();
            if edges.is_empty() {
                continue;
            }

            let mut all_terminal = true;
            let mut best_for_stm = f32::NEG_INFINITY;
            let mut min_win_plies: Option<f32> = None;
            let mut min_plies = f32::INFINITY;
            let mut max_plies = f32::NEG_INFINITY;
            for edge in edges.iter() {
                let Some((wl, _, plies)) = self
                    .get(parent_key.child(edge.mv()))
                    .filter(|child| child.expansion_state() == ExpansionState::Terminal)
                    .and_then(|child| child.terminal_value())
                else {
                    all_terminal = false;
                    continue;
                };
                best_for_stm = best_for_stm.max(wl);
                min_plies = min_plies.min(plies);
                max_plies = max_plies.max(plies);
                if wl > 0.0 {
                    min_win_plies = Some(min_win_plies.map_or(plies, |best| best.min(plies)));
                }
            }

            if let Some(plies) = min_win_plies {
                parent.apply_forced_win(plies + 1.0);
                continue;
            }
            // 无必胜着：必败用最长 plies，必和用最短；Expanded 由 mark_proven_terminal 把关。
            if !all_terminal {
                continue;
            }
            let wl = -best_for_stm;
            debug_assert!(wl >= 0.0, "forced-win branch should have continued");
            let plies_left = if wl > 0.0 { max_plies } else { min_plies } + 1.0;
            parent.mark_proven_terminal(wl, if wl == 0.0 { 1.0 } else { 0.0 }, plies_left);
        }
    }

    /// 批量删除已脱离的 tree subtree，并返回实际删除的 node 数。
    ///
    /// 跨回合由 reaper 异步调用：root 已推进后，待删 sibling 与下一手搜索 key 空间不相交；
    /// 按 shard 写锁删除，可与活跃搜索重叠，不挡 `go`。
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

    /// 无关 position 换图后由后台线程逐 shard 释放整张旧图。
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

/// 两次已完成 stream 搜索之间保留的 tree 状态。
#[derive(Debug)]
pub struct SearchTree {
    repository: Arc<NodeRepository>,
    /// 保留的已走主线：从最早 root 到当前 root。
    root_keys: Vec<NodeKey>,
    /// 与 `root_keys` 对齐的完整历史；末项就是当前 root。快照使 UCI 能精确悔棋和
    /// 复用，不必根据 hash 重建局面。
    root_histories: Vec<Arc<PositionHistory>>,
    /// `advance` 后待后台删除的 sibling 子树根；由 `take_pending_gc_roots` 交给 reaper。
    pending_gc_roots: Vec<NodeKey>,
}

impl SearchTree {
    pub fn new(root_history: Arc<PositionHistory>) -> Self {
        let root = NodeKey::root(root_history.last().hash());
        Self {
            repository: Arc::new(NodeRepository::default()),
            root_keys: vec![root],
            root_histories: vec![root_history],
            pending_gc_roots: Vec::new(),
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
    /// 旧 root 留在已走主线；sibling 子树只挂到 `pending_gc_roots`，不在此同步删除。
    pub fn advance(&mut self, mv: Move) -> Result<(), EnginError> {
        let old_root = self.root_key();
        if !self.repository.subtree_is_settled(old_root) {
            return Err(EnginError::PortIncomplete(
                "stream tree advance requires settled reservations",
            ));
        }
        self.advance_settled(mv)
    }

    /// Engine 已停止并 drain worker 后使用的推进路径。
    fn advance_after_drain(&mut self, mv: Move) -> Result<(), EnginError> {
        debug_assert!(self.repository.subtree_is_settled(self.root_key()));
        self.advance_settled(mv)
    }

    fn advance_settled(&mut self, mv: Move) -> Result<(), EnginError> {
        let old_root = self.root_key();
        if !self.root_history().last().board().is_legal_move(mv) {
            return Err(EnginError::PortIncomplete("stream tree advance requires a legal move"));
        }

        if let Some(root) = self.repository.get(old_root) {
            self.pending_gc_roots.extend(
                root.edges()
                    .iter()
                    .filter(|edge| edge.mv() != mv)
                    .map(|edge| old_root.child(edge.mv())),
            );
        }

        let new_root = old_root.child(mv);
        self.repository.get_or_insert(new_root);
        let mut history = self.root_history().as_ref().clone();
        history.append(mv);
        self.root_keys.push(new_root);
        self.root_histories.push(Arc::new(history));
        Ok(())
    }

    /// 取出 `advance` 挂起的 sibling 子树根，交给 Engine reaper 异步 `remove_subtrees`。
    pub(crate) fn take_pending_gc_roots(&mut self) -> Vec<NodeKey> {
        std::mem::take(&mut self.pending_gc_roots)
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
        self.pending_gc_roots.clear();
        Ok(true)
    }

    /// 将可复用树定位到完整 UCI history。已保留前缀复用；无关 history 换新
    /// repository，并把旧图交给调用方后台释放。
    pub fn reset_to_history(
        &mut self,
        target: Arc<PositionHistory>,
    ) -> Result<Option<Arc<NodeRepository>>, EnginError> {
        if !self.repository.subtree_is_settled(self.root_key()) {
            return Err(EnginError::PortIncomplete(
                "stream tree reset requires settled reservations",
            ));
        }
        self.reset_to_history_settled(target, false)
    }

    /// Engine 的 `position` 路径在调用前已经 abort 并 drain 当前 job。
    pub(crate) fn reset_to_history_after_drain(
        &mut self,
        target: Arc<PositionHistory>,
    ) -> Result<Option<Arc<NodeRepository>>, EnginError> {
        debug_assert!(self.repository.subtree_is_settled(self.root_key()));
        self.reset_to_history_settled(target, true)
    }

    fn reset_to_history_settled(
        &mut self,
        target: Arc<PositionHistory>,
        after_drain: bool,
    ) -> Result<Option<Arc<NodeRepository>>, EnginError> {
        if let Some(index) = self.root_histories.iter().position(|history| history == &target) {
            self.root_keys.truncate(index + 1);
            self.root_histories.truncate(index + 1);
            // 悔棋后这些 sibling 仍可能从保留根走到；丢掉 pending，避免误删。
            self.pending_gc_roots.clear();
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
                        "stream tree reset could not derive legal move",
                    ))?;
                if after_drain {
                    self.advance_after_drain(mv)?;
                } else {
                    self.advance(mv)?;
                }
            }
            return Ok(None);
        }

        let root = NodeKey::root(target.last().hash());
        let retired = std::mem::replace(&mut self.repository, Arc::new(NodeRepository::default()));
        self.root_keys = vec![root];
        self.root_histories = vec![target];
        self.pending_gc_roots.clear();
        Ok(Some(retired))
    }
}

#[cfg(test)]
mod tree_tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{NodeKey, SearchTree};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    fn tree() -> SearchTree {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        SearchTree::new(Arc::new(PositionHistory::from_positions(state.positions())))
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

        tree.advance(keep).expect("advance");
        assert_eq!(tree.repository().len(), 4);
        let pending = tree.take_pending_gc_roots();
        assert_eq!(pending, vec![dropped_child]);
        assert_eq!(tree.repository().remove_subtrees(pending), 2);
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
        assert!(tree.reset_to_history(target).expect("replay continuation").is_none());
        assert_eq!(tree.root_history().len(), 3);
    }

    #[test]
    fn reset_to_unrelated_history_starts_fresh_repository() {
        let mut tree = tree();
        tree.advance(mv("a0", "a1")).expect("advance");
        let unrelated = GameState::from_fen_moves(STARTPOS_FEN, &["b0b1"]).expect("other legal line");
        let target = Arc::new(PositionHistory::from_positions(unrelated.positions()));

        let retired = tree.reset_to_history(target.clone()).expect("fresh tree");
        assert!(retired.is_some());
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

    use super::{ExpansionState, NodeKey, NodeRepository, ValueDelta};

    fn b2_b3() -> Move {
        Move::new(Square::parse("b2").expect("b2"), Square::parse("b3").expect("b3"))
    }

    #[test]
    fn parent_delta_flips_wl_not_draw() {
        let leaf = ValueDelta::one(0.6, 0.2);
        assert_eq!(leaf.for_parent(), ValueDelta::one(-0.6, 0.2));
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
        repository.propagate_proven_terminals(&[root, parent_key, parent_key.child(first_move)], root);
        assert_eq!(parent.expansion_state(), ExpansionState::Expanded);

        let second = repository.get_or_insert(parent_key.child(second_move));
        assert!(second.try_begin_evaluation());
        second.mark_terminal(-1.0, 0.0, 6.0);
        repository.propagate_proven_terminals(&[root, parent_key, parent_key.child(second_move)], root);
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

        // 先钉较长必胜：父应立刻成为 Terminal（不必等兄弟）。
        let long = repository.get_or_insert(parent_key.child(second_move));
        assert!(long.try_begin_evaluation());
        long.mark_terminal(1.0, 0.0, 6.0);
        repository.propagate_proven_terminals(&[root, parent_key, parent_key.child(second_move)], root);
        assert_eq!(parent.expansion_state(), ExpansionState::Terminal);
        assert_eq!(parent.terminal_wl(), Some((-1.0, 0.0)));
        assert_eq!(parent.terminal_plies_left(), Some(7.0));

        // 再钉更短必胜：plies 缩短。
        let short = repository.get_or_insert(parent_key.child(first_move));
        assert!(short.try_begin_evaluation());
        short.mark_terminal(1.0, 0.0, 2.0);
        repository.propagate_proven_terminals(&[root, parent_key, parent_key.child(first_move)], root);
        assert_eq!(parent.terminal_plies_left(), Some(3.0));
    }

    #[test]
    fn sticky_proof_one_winning_reply_marks_parent_while_sibling_open() {
        let repository = NodeRepository::default();
        let root = NodeKey::root(3);
        let parent_move = b2_b3();
        let parent_key = root.child(parent_move);
        let win_move = Move::new(Square::parse("c3").expect("from"), Square::parse("c4").expect("to"));
        let open_move = Move::new(Square::parse("d3").expect("from"), Square::parse("d4").expect("to"));
        let root_node = repository.get_or_insert(root);
        assert!(root_node.try_begin_evaluation());
        root_node.publish_edges(vec![(parent_move, 1.0)]);
        let parent = repository.get_or_insert(parent_key);
        assert!(parent.try_begin_evaluation());
        parent.publish_edges(vec![(win_move, 0.5), (open_move, 0.5)]);

        let winner = repository.get_or_insert(parent_key.child(win_move));
        assert!(winner.try_begin_evaluation());
        winner.mark_terminal(1.0, 0.0, 4.0);
        repository.propagate_proven_terminals(&[root, parent_key, parent_key.child(win_move)], root);

        assert_eq!(parent.expansion_state(), ExpansionState::Terminal);
        assert_eq!(parent.terminal_wl(), Some((-1.0, 0.0)));
        assert_eq!(parent.terminal_plies_left(), Some(5.0));
        // 兄弟仍未展开，不能挡「一胜即钉」。
        assert!(repository.get(parent_key.child(open_move)).is_none());
        assert_ne!(root_node.expansion_state(), ExpansionState::Terminal);
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
