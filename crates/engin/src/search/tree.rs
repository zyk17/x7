//! 搜索树数据面：`NodeId` / `Edge` / `Node` / `NodeArena` / reservation。
//!
//! 只定义结构与原子操作，不调度流水线。child 由 edge 一次性绑定 arena slot，不合并换位。

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};

use parking_lot::{Mutex, RwLock};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;
use crate::search::backprop::ValueDelta;

/// 路径树 node 的稳定地址。它只用于 arena 寻址，不携带棋盘或路径 hash。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct NodeId(u64);

impl NodeId {
    const SLOT_BITS: u32 = 10;
    const SLOT_MASK: u64 = (1 << Self::SLOT_BITS) - 1;

    const fn new(page: u32, slot: u16) -> Self {
        Self(((page as u64) << Self::SLOT_BITS) | slot as u64)
    }

    const fn page(self) -> usize {
        (self.0 >> Self::SLOT_BITS) as usize
    }

    const fn slot(self) -> usize {
        (self.0 & Self::SLOT_MASK) as usize
    }
}

/// child edge。in-flight visit 保存在入边，绝不计入 child node 的 completed visit。
#[derive(Debug)]
pub struct Edge {
    mv: Move,
    prior: f32,
    started: AtomicU32,
    child: OnceLock<NodeId>,
    /// 已完成 N/Q 与尚未完成的 virtual mean；选边要一起读。
    stats: Mutex<EdgeStats>,
}

/// edge 聚合。completed 原始矩是 action-Q、方差与 SE 的唯一真相。
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EdgeStats {
    pub visits: u32,
    pub wl_sum: f32,
    pub wl_sq_sum: f32,
    pub virtual_wl_sum: f32,
}

impl EdgeStats {
    pub(crate) fn q(self) -> f32 {
        if self.visits == 0 { 0.0 } else { self.wl_sum / self.visits as f32 }
    }

    pub(crate) fn variance(self) -> f32 {
        if self.visits < 2 { 0.0 } else { (self.wl_sq_sum / self.visits as f32 - self.q() * self.q()).max(0.0) }
    }

    /// completed evidence 的样本均值标准误；reservation 不属于证据。
    pub(crate) fn standard_error(self) -> f32 {
        if self.visits < 2 { 0.0 } else { (self.variance() / self.visits as f32).sqrt() }
    }
}

impl Edge {
    fn new(mv: Move, prior: f32) -> Self {
        debug_assert!((0.0..=1.0).contains(&prior), "policy prior must be normalized");
        Self { mv, prior, started: AtomicU32::new(0), child: OnceLock::new(), stats: Mutex::new(EdgeStats::default()) }
    }

    pub fn mv(&self) -> Move {
        self.mv
    }

    pub fn prior(&self) -> f32 {
        self.prior
    }

    pub fn child(&self) -> Option<NodeId> {
        self.child.get().copied()
    }

    fn install_child(&self, child: NodeId) -> Result<(), NodeId> {
        self.child.set(child)
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
        self.stats.lock().q()
    }

    fn reserve(&self, virtual_mean: f32) -> f32 {
        let virtual_wl_sum = virtual_mean;
        let mut stats = self.stats.lock();
        self.started.fetch_add(1, Ordering::AcqRel);
        stats.virtual_wl_sum += virtual_wl_sum;
        virtual_wl_sum
    }

    fn cancel(&self, virtual_wl_sum: f32) {
        let mut stats = self.stats.lock();
        let started = self.started.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(started > stats.visits, "stream edge reservation underflow");
        stats.virtual_wl_sum -= virtual_wl_sum;
        if started - 1 == stats.visits {
            stats.virtual_wl_sum = 0.0;
        }
    }

    fn complete(&self, virtual_wl_sum: f32, wl: f32) {
        let mut stats = self.stats.lock();
        let started = self.started.load(Ordering::Acquire);
        debug_assert!(started > stats.visits, "stream edge completion without reservation");
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
    edges: Arc<[Edge]>,
    edge_index: usize,
    virtual_wl_sum: f32,
}

impl EdgeReservation {
    pub fn mv(&self) -> Move {
        self.edges[self.edge_index].mv()
    }

    pub fn complete(self, wl: f32) {
        self.edges[self.edge_index].complete(self.virtual_wl_sum, wl);
    }

    pub fn cancel(self) {
        self.edges[self.edge_index].cancel(self.virtual_wl_sum);
    }
}

/// node 不可逆的展开生命周期。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ExpansionState {
    #[default]
    Unexpanded = 0,
    Claimed = 1,
    Expanded = 2,
    Terminal = 3,
}

impl ExpansionState {
    fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Unexpanded,
            1 => Self::Claimed,
            2 => Self::Expanded,
            3 => Self::Terminal,
            _ => unreachable!("invalid stream expansion state"),
        }
    }
}

/// 已完成 node 的 WDL 聚合值。
#[derive(Debug, Default)]
struct NodeStats {
    visits: u32,
    wl_sum: f32,
    draw_sum: f32,
    m_sum: f32,
}

/// arena 的 node 值。展开时只发布一次 edge vector；之后各 edge 统计可独立推进，
/// 不需要整棵 tree 锁。
#[derive(Debug, Default)]
pub struct Node {
    /// 生命周期：Unexpanded → Claimed → Expanded|Terminal
    expansion: AtomicU8,
    edges: OnceLock<Arc<[Edge]>>,
    ///node 保留其 completed 聚合值。in-flight visit 保持在 edge-local
    stats: Mutex<NodeStats>,
    /// `(wl, draw≡d, plies_left≡m)`。`m` 以 ply（半回合）保存，
    /// UCI “moves left” 另行换算为完整回合。
    /// 精确证明只靠 `Terminal` 状态 + 本字段；
    terminal: Mutex<Option<(f32, f32, f32)>>,
}

impl Node {
    pub fn completed_visits(&self) -> u32 {
        self.stats.lock().visits
    }

    pub fn q(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 { 0.0 } else { stats.wl_sum / stats.visits as f32 }
    }

    pub fn draw(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 { 0.0 } else { stats.draw_sum / stats.visits as f32 }
    }

    pub fn m(&self) -> f32 {
        let stats = self.stats.lock();
        if stats.visits == 0 { 0.0 } else { stats.m_sum / stats.visits as f32 }
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

    /// `Unexpanded → Claimed`：至多一个 Select claim 成功，该叶子交给 Expand。
    /// 其余撞上 `Claimed` 的路径由 Select `park_collision`（保留 reservation / μ），
    /// 等该叶子 backprop complete 后再 cancel，不立刻取消、也不重复 Eval。
    pub fn try_claim(&self) -> bool {
        self.expansion
            .compare_exchange(
                ExpansionState::Unexpanded as u8,
                ExpansionState::Claimed as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn publish_edges(&self, edges: impl IntoIterator<Item = (Move, f32)>) {
        debug_assert_eq!(self.expansion_state(), ExpansionState::Claimed, "node must be claimed");
        let mut edges: smallvec::SmallVec<[(Move, f32); 64]> = edges.into_iter().collect();
        edges.sort_unstable_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(std::cmp::Ordering::Equal));
        let edges: Arc<[Edge]> = edges.into_iter().map(|(mv, prior)| Edge::new(mv, prior)).collect();
        let published = self.edges.set(edges).is_ok();
        debug_assert!(published, "stream node publishes edges once");
        self.expansion.store(ExpansionState::Expanded as u8, Ordering::Release);
    }

    /// Expand 或 Eval 在发布终局数据或 policy 前失败后恢复 node，避免后续 Select event 将失败的
    /// 请求当作永久 collision。
    pub fn abort_claim(&self) {
        let aborted = self
            .expansion
            .compare_exchange(
                ExpansionState::Claimed as u8,
                ExpansionState::Unexpanded as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        debug_assert!(aborted, "only claimed stream nodes can abort their claim");
    }

    pub fn mark_terminal(&self, wl: f32, draw: f32, plies_left: f32) {
        debug_assert_eq!(self.expansion_state(), ExpansionState::Claimed, "node must be claimed");
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
    /// 返回 terminal result 是否实际改变，以决定是否继续向祖先传播。
    fn apply_forced_win(&self, plies_left: f32) -> bool {
        match self.expansion_state() {
            ExpansionState::Expanded => self.mark_proven_terminal(-1.0, 0.0, plies_left),
            ExpansionState::Terminal => self.shorten_terminal_plies(plies_left),
            _ => false,
        }
    }

    fn shorten_terminal_plies(&self, plies_left: f32) -> bool {
        let mut terminal = self.terminal.lock();
        let Some((wl, draw, old_plies)) = *terminal else {
            return false;
        };
        if wl < 0.0 && plies_left + f32::EPSILON < old_plies {
            *terminal = Some((wl, draw, plies_left));
            true
        } else {
            false
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

    pub fn edges(&self) -> Arc<[Edge]> {
        self.edges.get().cloned().unwrap_or_default()
    }

    pub(crate) fn reserve_edge(&self, edge_index: usize, virtual_mean: f32) -> Option<EdgeReservation> {
        let edges = self.edges();
        let edge = edges.get(edge_index)?;
        let virtual_wl_sum = edge.reserve(virtual_mean);
        Some(EdgeReservation { edges, edge_index, virtual_wl_sum })
    }
}

const NODES_PER_PAGE: usize = 1 << NodeId::SLOT_BITS;

struct NodeSlot {
    initialized: AtomicBool,
    value: UnsafeCell<MaybeUninit<Node>>,
}

// `Node` 的可变部分自身由原子量和锁保护。slot 的初始化、回收与复用只发生在
// arena 的 allocator 锁下；GC 只处理已不可达且已 settled 的 subtree。
unsafe impl Sync for NodeSlot {}

impl NodeSlot {
    fn empty() -> Self {
        Self { initialized: AtomicBool::new(false), value: UnsafeCell::new(MaybeUninit::uninit()) }
    }
}

struct NodePage {
    slots: Box<[NodeSlot]>,
}

impl NodePage {
    fn new() -> Self {
        Self { slots: std::iter::repeat_with(NodeSlot::empty).take(NODES_PER_PAGE).collect() }
    }
}

#[derive(Default)]
struct ArenaAllocator {
    free: Vec<NodeId>,
    next_page: u32,
    next_slot: u16,
}

/// append-only page arena。page 永不搬迁，`NodeId` 只在 stop/drain 后的 GC 确认
/// 无任何旧 event 可达时才会被复用。
pub struct NodeArena {
    pages: RwLock<Vec<Arc<NodePage>>>,
    allocator: Mutex<ArenaAllocator>,
}

impl NodeArena {
    pub fn new() -> Self {
        Self { pages: RwLock::new(Vec::new()), allocator: Mutex::new(ArenaAllocator::default()) }
    }

    fn page(&self, id: NodeId) -> Option<Arc<NodePage>> {
        self.pages.read().get(id.page()).cloned()
    }

    pub fn allocate(&self) -> NodeId {
        let (id, page) = {
            let mut allocator = self.allocator.lock();
            if let Some(id) = allocator.free.pop() {
                let page = self.page(id).expect("reusable node page exists");
                (id, page)
            } else {
                if allocator.next_slot as usize == NODES_PER_PAGE {
                    allocator.next_page += 1;
                    allocator.next_slot = 0;
                }
                let page_index = allocator.next_page;
                let slot = allocator.next_slot;
                allocator.next_slot += 1;
                let id = NodeId::new(page_index, slot);
                let page = {
                    let mut pages = self.pages.write();
                    while pages.len() <= page_index as usize {
                        pages.push(Arc::new(NodePage::new()));
                    }
                    Arc::clone(&pages[page_index as usize])
                };
                (id, page)
            }
        };
        let slot = &page.slots[id.slot()];
        assert!(!slot.initialized.load(Ordering::Acquire), "arena slot is free");
        // SAFETY: allocator gives each live slot to exactly one initializer; page storage is stable.
        unsafe { (*slot.value.get()).write(Node::default()) };
        slot.initialized.store(true, Ordering::Release);
        id
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        let page = self.page(id)?;
        let slot = page.slots.get(id.slot())?;
        if !slot.initialized.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: initialized slots are never moved. GC only frees unreachable slots after the
        // owning search job has drained, so no caller can retain a reference to a freed slot.
        Some(unsafe { (&*slot.value.get()).assume_init_ref() })
    }

    pub fn child_or_create(&self, edge: &Edge) -> NodeId {
        if let Some(child) = edge.child() {
            return child;
        }
        let candidate = self.allocate();
        match edge.install_child(candidate) {
            Ok(()) => candidate,
            Err(_) => {
                self.free(candidate);
                edge.child().expect("racing edge installs a child")
            }
        }
    }

    /// 沿 path 向上传播强制终局（不存半开 bounds）。每层扫父的全部边：
    /// 任一儿子对父 STM 必胜 → 立刻钉父；必败/必和要全部儿子都 Terminal。
    /// `root` 不钉死。
    pub(crate) fn propagate_proven_terminals(&self, node_path: &[NodeId], root: NodeId) {
        for &parent_id in node_path.iter().rev().skip(1) {
            if parent_id == root {
                break;
            }
            let parent = self.get(parent_id).expect("event path nodes live until job drain");
            let edges = parent.edges();
            debug_assert!(!edges.is_empty(), "event parent has a selected edge");
            let mut all_terminal = true;
            let mut best_for_stm = f32::NEG_INFINITY;
            let mut min_win_plies: Option<f32> = None;
            let mut max_draw_plies = f32::NEG_INFINITY;
            let mut max_plies = f32::NEG_INFINITY;
            for edge in edges.iter() {
                let Some(child_id) = edge.child() else {
                    all_terminal = false;
                    continue;
                };
                let child = self.get(child_id).expect("terminal propagation child lives");
                if child.expansion_state() != ExpansionState::Terminal {
                    all_terminal = false;
                    continue;
                }
                let (wl, _, plies) = child.terminal_value().expect("terminal child has exact value");
                best_for_stm = best_for_stm.max(wl);
                max_plies = max_plies.max(plies);
                if wl > 0.0 {
                    min_win_plies = Some(min_win_plies.map_or(plies, |best| best.min(plies)));
                } else if wl == 0.0 {
                    max_draw_plies = max_draw_plies.max(plies);
                }
            }

            let changed = if let Some(plies) = min_win_plies {
                parent.apply_forced_win(plies + 1.0)
            } else {
                // 无必胜着：必败用最长败着、必和用最长和棋；Expanded 由 mark_proven_terminal 把关。
                if !all_terminal {
                    false
                } else {
                    let wl = -best_for_stm;
                    debug_assert!(wl >= 0.0, "forced-win branch should have continued");
                    let plies_left = if wl > 0.0 { max_plies } else { max_draw_plies } + 1.0;
                    parent.mark_proven_terminal(wl, if wl == 0.0 { 1.0 } else { 0.0 }, plies_left)
                }
            };
            if !changed {
                break;
            }
        }
    }

    fn free(&self, id: NodeId) {
        let page = self.page(id).expect("reusable node page exists");
        let slot = &page.slots[id.slot()];
        assert!(slot.initialized.swap(false, Ordering::AcqRel), "reusable node lives");
        // SAFETY: GC reaches only settled, unreachable nodes and removes each slot once.
        unsafe { std::ptr::drop_in_place((*slot.value.get()).as_mut_ptr()) };
        self.allocator.lock().free.push(id);
    }

    /// 销毁已断开的 node，并一次归还所有 slot，避免 reaper 按 node 反复争用 allocator lock。
    fn free_many(&self, ids: Vec<NodeId>) -> usize {
        for &id in &ids {
            let page = self.page(id).expect("reaper node page exists");
            let slot = &page.slots[id.slot()];
            assert!(slot.initialized.swap(false, Ordering::AcqRel), "reaper node lives");
            // SAFETY: callers reach only settled, unreachable nodes and remove each slot once.
            unsafe { std::ptr::drop_in_place((*slot.value.get()).as_mut_ptr()) };
        }
        let removed = ids.len();
        if removed != 0 {
            self.allocator.lock().free.extend(ids);
        }
        removed
    }

    /// 批量删除已脱离的 tree subtree，并返回实际删除的 node 数。
    ///
    /// 跨回合由 reaper 异步调用：root 已推进后，待删 sibling 与新 root 不相交；
    /// 只在旧 job drain 后入队，可与下一手搜索重叠，不挡 `go`。
    pub(crate) fn remove_subtrees(&self, roots: impl IntoIterator<Item = NodeId>) -> usize {
        let mut pending: Vec<_> = roots.into_iter().collect();
        let mut removed = Vec::new();
        while let Some(id) = pending.pop() {
            let node = self.get(id).expect("reaper subtree node lives");
            pending.extend(node.edges().iter().filter_map(|edge| edge.child()));
            removed.push(id);
        }
        self.free_many(removed)
    }

    /// 回收已与当前 root 脱钩、但其某个 child 被保留的祖先 slot。
    pub(crate) fn remove_nodes(&self, nodes: impl IntoIterator<Item = NodeId>) {
        self.free_many(nodes.into_iter().collect());
    }

    /// 检查 `root` 以下的 edge-local reservation 不变量。
    pub(crate) fn subtree_is_settled(&self, root: NodeId) -> bool {
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            let node = self.get(id).expect("tree node lives while checking reservations");
            let edges = node.edges();
            if edges.iter().any(|edge| edge.visits() != edge.completed_visits()) {
                return false;
            }
            pending.extend(edges.iter().filter_map(|edge| edge.child()));
        }
        true
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pages
            .read()
            .iter()
            .flat_map(|page| page.slots.iter())
            .filter(|slot| slot.initialized.load(Ordering::Acquire))
            .count()
    }
}

impl Default for NodeArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NodeArena {
    fn drop(&mut self) {
        for page in self.pages.get_mut().iter() {
            for slot in page.slots.iter() {
                if slot.initialized.swap(false, Ordering::AcqRel) {
                    // SAFETY: arena 销毁时已不存在外部引用；每个 live slot 恰好 drop 一次。
                    unsafe { std::ptr::drop_in_place((*slot.value.get()).as_mut_ptr()) };
                }
            }
        }
    }
}

/// 两次已完成 stream 搜索之间保留的 tree 状态。只支持向前复用；悔棋或无关
/// `position` 直接换一棵新 tree。
pub struct SearchTree {
    arena: Arc<NodeArena>,
    root: NodeId,
    root_history: Arc<PositionHistory>,
    /// `advance` 后待后台删除的 sibling 子树根；由 `take_pending_gc_roots` 交给 reaper。
    pending_gc_roots: Vec<NodeId>,
    /// 已推进的旧 root 不能 DFS 删除（chosen child 仍存活），单独回收其 slot。
    pending_gc_nodes: Vec<NodeId>,
}

impl SearchTree {
    pub fn new(root_history: Arc<PositionHistory>) -> Self {
        let arena = Arc::new(NodeArena::default());
        Self { root: arena.allocate(), arena, root_history, pending_gc_roots: Vec::new(), pending_gc_nodes: Vec::new() }
    }

    pub fn arena(&self) -> &Arc<NodeArena> {
        &self.arena
    }

    pub fn root_id(&self) -> NodeId {
        self.root
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        &self.root_history
    }

    /// 在当前 root 以下的 event 都完成或取消后，推进到一个合法 child。
    /// 旧 root 留在已走主线；sibling 子树只挂到 `pending_gc_roots`，不在此同步删除。
    pub(crate) fn advance(&mut self, mv: Move) -> Result<(), EnginError> {
        debug_assert!(self.arena.subtree_is_settled(self.root_id()));
        let old_root = self.root_id();
        if !self.root_history().last().board().is_legal_move(mv) {
            return Err(EnginError::Internal("stream tree advance requires a legal move"));
        }

        let root = self.arena.get(old_root).ok_or(EnginError::Internal("stream tree root is unavailable"))?;
        let edges = root.edges();
        let chosen = edges
            .iter()
            .find(|edge| edge.mv() == mv)
            .and_then(|edge| edge.child())
            .ok_or(EnginError::Internal("stream tree cannot reuse an unexpanded child"))?;
        self.pending_gc_roots.extend(edges.iter().filter(|edge| edge.mv() != mv).filter_map(|edge| edge.child()));
        self.pending_gc_nodes.push(old_root);
        let mut history = self.root_history().as_ref().clone();
        history.append(mv);
        self.root = chosen;
        self.root_history = Arc::new(history);
        Ok(())
    }

    /// 取出 `advance` 挂起的 sibling 子树根，交给 Engine reaper 异步 `remove_subtrees`。
    pub(crate) fn take_pending_gc(&mut self) -> (Vec<NodeId>, Vec<NodeId>) {
        (std::mem::take(&mut self.pending_gc_roots), std::mem::take(&mut self.pending_gc_nodes))
    }

    /// Engine 的 `position` 路径在调用前已经 abort 并 drain 当前 job。
    pub(crate) fn reset_to_history_after_drain(
        &mut self,
        target: Arc<PositionHistory>,
    ) -> Result<Option<Arc<NodeArena>>, EnginError> {
        debug_assert!(self.arena.subtree_is_settled(self.root_id()));
        if target.len() >= self.root_history().len()
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
                    .ok_or(EnginError::Internal("stream tree reset could not derive legal move"))?;
                if self.advance(mv).is_err() {
                    return Ok(Some(self.replace_with_fresh(target)));
                }
            }
            return Ok(None);
        }
        Ok(Some(self.replace_with_fresh(target)))
    }

    fn replace_with_fresh(&mut self, target: Arc<PositionHistory>) -> Arc<NodeArena> {
        let arena = Arc::new(NodeArena::default());
        let retired = std::mem::replace(&mut self.arena, arena);
        self.root = self.arena.allocate();
        self.root_history = target;
        self.pending_gc_roots.clear();
        self.pending_gc_nodes.clear();
        retired
    }
}
#[cfg(test)]
mod arena_tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN, Square};

    use super::{ExpansionState, NodeArena, NodeId, SearchTree};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    fn parent_with_two_children() -> (NodeArena, NodeId, NodeId) {
        let arena = NodeArena::default();
        let root = arena.allocate();
        let root_node = arena.get(root).expect("root");
        assert!(root_node.try_claim());
        root_node.publish_edges([(mv("a0", "a1"), 1.0)]);
        let parent = arena.child_or_create(&root_node.edges()[0]);
        let parent_node = arena.get(parent).expect("parent");
        assert!(parent_node.try_claim());
        parent_node.publish_edges([(mv("b0", "b1"), 0.5), (mv("c0", "c1"), 0.5)]);
        (arena, root, parent)
    }

    fn mark_child(arena: &NodeArena, parent: NodeId, edge_index: usize, wl: f32, plies: f32) -> NodeId {
        let parent_node = arena.get(parent).expect("parent");
        let child = arena.child_or_create(&parent_node.edges()[edge_index]);
        let child_node = arena.get(child).expect("child");
        assert!(child_node.try_claim());
        child_node.mark_terminal(wl, if wl == 0.0 { 1.0 } else { 0.0 }, plies);
        child
    }

    #[test]
    fn edge_binds_one_child_and_advance_reclaims_siblings() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = SearchTree::new(history);
        let root_id = tree.root_id();
        let root = tree.arena().get(root_id).expect("root");
        let moves = tree.root_history().last().board().generate_legal_moves();
        root.try_claim();
        root.publish_edges(vec![(moves[0], 0.6), (moves[1], 0.4)]);

        let edges = root.edges();
        let kept = tree.arena().child_or_create(&edges[0]);
        assert_eq!(kept, tree.arena().child_or_create(&edges[0]));
        let dropped = tree.arena().child_or_create(&edges[1]);
        assert_eq!(tree.arena().len(), 3);

        tree.advance(moves[0]).expect("settled advance");
        assert_eq!(tree.root_id(), kept);
        let (roots, nodes) = tree.take_pending_gc();
        assert_eq!(roots, vec![dropped]);
        assert_eq!(nodes, vec![root_id]);
        assert_eq!(tree.arena().remove_subtrees(roots), 1);
        tree.arena().remove_nodes(nodes);
        assert!(tree.arena().get(root_id).is_none());
        assert!(tree.arena().get(kept).is_some());
        assert!(tree.arena().get(dropped).is_none());
    }

    #[test]
    fn fresh_arena_drops_pending_gc_ids_from_the_retired_arena() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let initial = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = SearchTree::new(Arc::clone(&initial));
        let root = tree.arena().get(tree.root_id()).expect("root");
        let mv = tree.root_history().last().board().generate_legal_moves()[0];
        root.try_claim();
        root.publish_edges(vec![(mv, 1.0)]);
        tree.arena().child_or_create(&root.edges()[0]);
        tree.advance(mv).expect("advance");

        let retired = tree.reset_to_history_after_drain(initial).expect("reset").expect("fresh arena");
        assert!(retired.get(tree.root_id()).is_some());
        assert!(tree.arena().get(tree.root_id()).is_some());
        assert_eq!(tree.take_pending_gc(), (Vec::new(), Vec::new()));
    }

    #[test]
    fn terminal_win_marks_parent_without_waiting_for_siblings_or_root() {
        let (arena, root, parent) = parent_with_two_children();
        let winning_child = mark_child(&arena, parent, 0, 1.0, 6.0);

        arena.propagate_proven_terminals(&[root, parent, winning_child], root);

        let parent = arena.get(parent).expect("parent");
        assert_eq!(parent.expansion_state(), ExpansionState::Terminal);
        assert_eq!(parent.terminal_value(), Some((-1.0, 0.0, 7.0)));
        assert_eq!(arena.get(root).expect("root").expansion_state(), ExpansionState::Expanded);
    }

    #[test]
    fn all_terminal_children_choose_the_longest_draw_or_loss() {
        let (arena, root, parent) = parent_with_two_children();
        let loss = mark_child(&arena, parent, 0, -1.0, 2.0);
        arena.propagate_proven_terminals(&[root, parent, loss], root);
        assert_eq!(arena.get(parent).expect("parent").expansion_state(), ExpansionState::Expanded);

        let draw = mark_child(&arena, parent, 1, 0.0, 8.0);
        arena.propagate_proven_terminals(&[root, parent, draw], root);
        assert_eq!(arena.get(parent).expect("parent").terminal_value(), Some((0.0, 1.0, 9.0)));

        let (arena, root, parent) = parent_with_two_children();
        let short_loss = mark_child(&arena, parent, 0, -1.0, 2.0);
        arena.propagate_proven_terminals(&[root, parent, short_loss], root);
        let long_loss = mark_child(&arena, parent, 1, -1.0, 6.0);
        arena.propagate_proven_terminals(&[root, parent, long_loss], root);
        assert_eq!(arena.get(parent).expect("parent").terminal_value(), Some((1.0, 0.0, 7.0)));
    }

    #[test]
    fn shorter_proven_win_updates_an_already_terminal_parent() {
        let (arena, root, parent) = parent_with_two_children();
        let long_win = mark_child(&arena, parent, 0, 1.0, 6.0);
        arena.propagate_proven_terminals(&[root, parent, long_win], root);
        let short_win = mark_child(&arena, parent, 1, 1.0, 2.0);
        arena.propagate_proven_terminals(&[root, parent, short_win], root);

        assert_eq!(arena.get(parent).expect("parent").terminal_value(), Some((-1.0, 0.0, 3.0)));
    }
}
