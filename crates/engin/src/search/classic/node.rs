//! px0 `src/search/classic/node.h:84-260`、`node.cc:161-373,465-543`。

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use parking_lot::RwLock;

use xiangqi_core::{GameResult, Move, MoveList, Position, PositionHistory};

const EMPTY_CHILD: usize = usize::MAX;
const RESERVED_CHILD: usize = usize::MAX - 1;

/// px0 `Node::Terminal` (`node.h:132`)。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Terminal {
    #[default]
    NonTerminal, // 游戏未结束
    EndOfGame, // 游戏正常结束
    Tablebase, // 命中库
    TwoFold,   // 两次重复
}

/// px0 `Edge` (`node.h:85-112`)。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Edge {
    pub mv: Move,
    p: u16, // value of Move policy prior returned from the neural net (but can be changed by adding Dirichlet noise).Must be in [0,1].
}

impl Edge {
    pub const fn new(mv: Move) -> Self {
        Self { mv, p: 0 }
    }

    /// px0 `Edge::SetP` (`node.cc:161-168`)。
    pub fn set_p(&mut self, p: f32) {
        debug_assert!((0.0..=1.0).contains(&p));
        const ROUNDINGS: i32 = (1 << 11) - (3 << 28);
        let bits = p.to_bits() as i32;
        let tmp = bits.wrapping_add(ROUNDINGS);
        self.p = if tmp < 0 { 0 } else { (tmp >> 12) as u16 };
    }

    /// px0 `Edge::GetP` (`node.cc:170-176`)。
    pub fn get_p(self) -> f32 {
        let tmp = ((self.p as u32) << 12) | (3 << 28);
        f32::from_bits(tmp)
    }
}

/// px0 `EdgeAndNode` (`src/search/classic/node.h:356-410`)。
/// 保存一对边和节点, 提供代理函数, 简化访问他们
///
/// C++ 版本把一条 edge 与其可选 child node 组合为只读代理。Rust 的树使用
/// arena 索引保存 child，因此此类型只在读取统计时短暂借用二者。
#[derive(Clone, Copy, Debug)]
pub struct EdgeAndNode<'a> {
    edge: &'a Edge,
    node: Option<&'a Node>,
}

impl<'a> EdgeAndNode<'a> {
    pub const fn new(edge: &'a Edge, node: Option<&'a Node>) -> Self {
        Self { edge, node }
    }

    /// px0 `EdgeAndNode::GetQ` (`node.h:375-377`)。
    pub fn q(self, default_q: f32, draw_score: f32) -> f32 {
        self.node
            .filter(|node| node.n() > 0)
            .map(|node| node.q(draw_score))
            .unwrap_or(default_q)
    }

    /// px0 `EdgeAndNode::GetNStarted` (`node.h:387-390`)。
    pub fn n_started(self) -> u32 {
        self.node.map_or(0, Node::n_started)
    }

    /// px0 `EdgeAndNode::GetP` (`node.h:400-401`)。
    pub fn p(self) -> f32 {
        self.edge.get_p()
    }

    /// px0 `EdgeAndNode::GetMove` (`node.h:402-404`)。
    pub const fn mv(self) -> Move {
        self.edge.mv
    }

    /// px0 `EdgeAndNode::GetU` (`node.h:406-410`)。
    /// Returns U = numerator * p / N.
    /// Passed numerator is expected to be equal to (cpuct * sqrt(N[parent])).
    pub fn u(self, numerator: f32) -> f32 {
        numerator * self.p() / (1 + self.n_started()) as f32
    }

    pub const fn child(self) -> Option<&'a Node> {
        self.node
    }
}

/// px0 `Node` 统计与树结构（单线程：children 与 edges 平行索引）。
#[derive(Debug)]
pub struct Node {
    parent: Option<usize>,
    edge_index: u16,
    edges: Vec<Edge>,
    children: Vec<AtomicUsize>,
    terminal: Terminal,
    lower_bound: GameResult,
    upper_bound: GameResult,
    // win-loss, 视角: 刚刚落子的那个人（也就是带我们来到这个新局面的那个人）的视角。
    // 因为在 MCTS 搜索树中，当我们反向传播（Backpropagation）更新父节点的胜率时，我们是用当前节点的评估值去更新父节点。
    // 如果使用“刚刚落子的人”的视角，当前节点的胜负分（比如 +1 表示我赢了）可以直接加到父节点上，
    // 不需要频繁地在每一步乘以 -1 去翻转正负号，从而极大地减少了代码出错（把正负号搞反）的概率。
    wl: f32,
    d: f32,
    m: f32,
    // How many completed visits this node had.
    n: AtomicU32,
    // (AKA virtual loss.) How many threads currently process this node (started
    // but not finished). This value is added to n during selection which node
    // to pick in MCTS, and also when selecting the best move.
    n_in_flight: AtomicU32,
    // px0 `solid_children_` (`node.h:253-255,329-330`). In C++ this changes
    // `child_` from a sibling list into a contiguous Node array. The Rust arena
    // already gives every child a stable allocation, so the equivalent state is
    // that every edge owns an allocated child slot.
    solid_children: bool,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            parent: None,
            edge_index: 0,
            edges: Vec::new(),
            children: Vec::new(),
            terminal: Terminal::NonTerminal,
            lower_bound: GameResult::BlackWon,
            upper_bound: GameResult::WhiteWon,
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            n: AtomicU32::new(0),
            n_in_flight: AtomicU32::new(0),
            solid_children: false,
        }
    }
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent,
            edge_index: self.edge_index,
            edges: self.edges.clone(),
            children: self
                .children
                .iter()
                .map(|child| AtomicUsize::new(child.load(Ordering::Acquire)))
                .collect(),
            terminal: self.terminal,
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            wl: self.wl,
            d: self.d,
            m: self.m,
            n: AtomicU32::new(self.n()),
            n_in_flight: AtomicU32::new(self.n_in_flight()),
            solid_children: self.solid_children,
        }
    }
}

impl Node {
    pub fn new(parent: Option<usize>, edge_index: u16) -> Self {
        Self {
            parent,
            edge_index,
            ..Self::default()
        }
    }

    pub const fn parent(&self) -> Option<usize> {
        self.parent
    }

    pub const fn edge_index(&self) -> u16 {
        self.edge_index
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn n(&self) -> u32 {
        self.n.load(Ordering::Acquire)
    }

    pub fn n_in_flight(&self) -> u32 {
        self.n_in_flight.load(Ordering::Acquire)
    }

    pub fn n_started(&self) -> u32 {
        self.n().saturating_add(self.n_in_flight())
    }

    pub fn children_visits(&self) -> u32 {
        let n = self.n();
        if n > 0 {
            n - 1
        } else {
            0
        }
    }

    pub const fn wl(&self) -> f32 {
        self.wl
    }

    pub const fn d(&self) -> f32 {
        self.d
    }

    pub const fn m(&self) -> f32 {
        self.m
    }

    /// px0 `Node::GetLowerBound` / `GetUpperBound`
    /// (`src/search/classic/node.h:153-156`).
    pub const fn lower_bound(&self) -> GameResult {
        self.lower_bound
    }

    pub const fn upper_bound(&self) -> GameResult {
        self.upper_bound
    }

    pub fn q(&self, draw_score: f32) -> f32 {
        self.wl + draw_score * self.d
    }

    pub const fn is_terminal(&self) -> bool {
        !matches!(self.terminal, Terminal::NonTerminal)
    }

    /// px0 `Node::IsTwoFoldTerminal` (`src/search/classic/node.h:147-149`)。
    pub const fn is_twofold_terminal(&self) -> bool {
        matches!(self.terminal, Terminal::TwoFold)
    }

    /// px0 `EdgeAndNode::IsTbTerminal` (`src/search/classic/node.h:394-398`).
    pub const fn is_tablebase_terminal(&self) -> bool {
        matches!(self.terminal, Terminal::Tablebase)
    }

    pub fn edge(&self, index: usize) -> &Edge {
        &self.edges[index]
    }

    pub fn edge_mut(&mut self, index: usize) -> &mut Edge {
        &mut self.edges[index]
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// px0 `Node::SortEdges` (`src/search/classic/node.cc:291-298`).
    ///
    /// px0 only sorts immediately after policy initialization, before child
    /// nodes exist, so the parallel child slots are all empty and need no
    /// reordering.
    pub fn sort_edges(&mut self) {
        assert!(self
            .children
            .iter()
            .all(|child| child.load(Ordering::Acquire) == EMPTY_CHILD));
        self.edges.sort_unstable_by_key(|edge| std::cmp::Reverse(edge.p));
    }

    pub fn child(&self, index: usize) -> Option<usize> {
        let child = self.children.get(index)?;
        #[cfg(debug_assertions)]
        let mut spins = 0usize;
        loop {
            let index = child.load(Ordering::Acquire);
            if index == RESERVED_CHILD {
                #[cfg(debug_assertions)]
                {
                    spins += 1;
                    assert!(spins < 10_000_000, "child reservation was never published");
                }
                std::hint::spin_loop();
                continue;
            }
            return (index != EMPTY_CHILD).then_some(index);
        }
    }

    /// Reserves one px0 `GetOrSpawnNode` child slot. The winner must publish
    /// a stable arena index with `publish_child`; every other caller waits in
    /// `child()` and observes that same index. Reference:
    /// `src/search/classic/node.h:468-525`.
    fn try_reserve_child(&self, index: usize) -> bool {
        self.children[index]
            .compare_exchange(EMPTY_CHILD, RESERVED_CHILD, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn publish_child(&self, edge_idx: usize, child_idx: usize) {
        debug_assert_ne!(child_idx, EMPTY_CHILD);
        debug_assert_ne!(child_idx, RESERVED_CHILD);
        self.children[edge_idx].store(child_idx, Ordering::Release);
    }

    /// px0 `Node::MakeSolid` state predicate (`node.cc:245-288`).
    pub const fn has_solid_children(&self) -> bool {
        self.solid_children
    }

    /// px0 `Node::CreateEdges` (`node.cc:205-210`)。
    pub fn create_edges(&mut self, moves: &MoveList) {
        assert!(self.edges.is_empty());
        assert!(self
            .children
            .iter()
            .all(|child| child.load(Ordering::Acquire) == EMPTY_CHILD));
        assert!(moves.len() <= u8::MAX as usize, "px0 Node::num_edges_ is uint8_t");
        self.edges = moves.iter().copied().map(Edge::new).collect();
        self.children = (0..moves.len()).map(|_| AtomicUsize::new(EMPTY_CHILD)).collect();
    }

    /// px0 `Node::CreateSingleChildNode` (`node.cc:196-203`)。
    pub fn create_single_child_node(&mut self, mv: Move) -> usize {
        assert!(self.edges.is_empty());
        self.edges = vec![Edge::new(mv)];
        self.children = vec![AtomicUsize::new(EMPTY_CHILD)];
        0
    }

    /// px0 `Node::TryStartScoreUpdate` (`node.cc:348-352`)。
    pub fn try_start_score_update(&self) -> bool {
        loop {
            let completed = self.n.load(Ordering::Acquire);
            let in_flight = self.n_in_flight.load(Ordering::Acquire);
            if completed == 0 && in_flight > 0 {
                return false;
            }
            if self
                .n_in_flight
                .compare_exchange_weak(in_flight, in_flight + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// px0 `Node::CancelScoreUpdate` (`node.cc:354`)。
    pub fn cancel_score_update(&self, multivisit: u32) {
        self.n_in_flight.fetch_sub(multivisit, Ordering::AcqRel);
    }

    /// px0 `Node::IncrementNInFlight` (`node.cc:346`)。
    pub fn increment_n_in_flight(&self, count: u32) {
        self.n_in_flight.fetch_add(count, Ordering::AcqRel);
    }

    /// px0 `Node::CopyPolicy` (`node.cc:378-384`)。
    pub fn copy_policy(&self, max_needed: usize, out: &mut [f32]) {
        let count = max_needed.min(self.edges.len());
        for (idx, policy) in out.iter_mut().enumerate().take(count) {
            *policy = self.edges[idx].get_p();
        }
    }

    /// px0 `Node::FinalizeScoreUpdate` (`node.cc:356-366`)。
    pub fn finalize_score_update(&mut self, v: f32, d: f32, m: f32, multivisit: u32) {
        let n = self.n.load(Ordering::Acquire);
        self.wl += multivisit as f32 * (v - self.wl) / (n + multivisit) as f32;
        self.d += multivisit as f32 * (d - self.d) / (n + multivisit) as f32;
        self.m += multivisit as f32 * (m - self.m) / (n + multivisit) as f32;
        self.n.store(n + multivisit, Ordering::Release);
        self.n_in_flight.fetch_sub(multivisit, Ordering::AcqRel);
    }

    /// px0 `Node::AdjustForTerminal` (`src/search/classic/node.cc:368-373`).
    ///
    /// A sticky terminal discovered below this non-terminal node changes the
    /// already accumulated average by `multivisit / n`, rather than adding a
    /// new visit. `MaybeSetBounds` supplies the exact delta to apply.
    pub fn adjust_for_terminal(&mut self, v: f32, d: f32, m: f32, multivisit: u32) {
        let n = self.n();
        debug_assert!(n >= multivisit);
        let n = n as f32;
        self.wl += multivisit as f32 * v / n;
        self.d += multivisit as f32 * d / n;
        self.m += multivisit as f32 * m / n;
    }

    /// px0 `Node::RevertTerminalVisits` (`src/search/classic/node.cc:375-392`)。
    ///
    /// Tree reuse can move the root inside a previously assumed two-fold
    /// repetition. In that case those inherited terminal visits are no longer
    /// valid and must be removed before the node is extended again.
    pub fn revert_terminal_visits(&mut self, v: f32, d: f32, m: f32, multivisit: u32) {
        let new_n = self.n() as i64 - multivisit as i64;
        if new_n <= 0 {
            self.wl = 0.0;
            self.d = 1.0;
            self.m = 0.0;
            self.n.store(0, Ordering::Release);
            return;
        }
        let new_n = new_n as f32;
        self.wl -= multivisit as f32 * (v - self.wl) / new_n;
        self.d -= multivisit as f32 * (d - self.d) / new_n;
        self.m -= multivisit as f32 * (m - self.m) / new_n;
        self.n.store(new_n as u32, Ordering::Release);
    }

    /// px0 `Node::MakeTerminal` (`node.cc:300-317`)。
    fn make_terminal(&mut self, result: GameResult, plies_left: f32, terminal: Terminal) {
        if !matches!(terminal, Terminal::TwoFold) {
            self.set_bounds(result, result);
        }
        self.terminal = terminal;
        self.m = plies_left;
        match result {
            GameResult::Draw => {
                self.wl = 0.0;
                self.d = 1.0;
            }
            GameResult::WhiteWon => {
                self.wl = 1.0;
                self.d = 0.0;
            }
            GameResult::BlackWon => {
                self.wl = -1.0;
                self.d = 0.0;
            }
            GameResult::Undecided => {}
        }
    }

    /// px0 `Node::MakeNotTerminal` (`node.cc:319-341`)。
    fn make_not_terminal(&mut self, child_stats: &[(u32, f32, f32)]) {
        self.terminal = Terminal::NonTerminal;
        let mut n = 0;
        if !self.edges.is_empty() {
            n = 1;
            self.wl = 0.0;
            self.d = 0.0;
            for &(child_n, child_wl, child_d) in child_stats {
                if child_n > 0 {
                    n += child_n;
                    self.wl += -child_wl * child_n as f32;
                    self.d += child_d * child_n as f32;
                }
            }
            self.wl /= n as f32;
            self.d /= n as f32;
        }
        self.n.store(n, Ordering::Release);
    }

    pub fn set_bounds(&mut self, lower: GameResult, upper: GameResult) {
        self.lower_bound = lower;
        self.upper_bound = upper;
    }

    pub fn visited_policy(&self, arena: &NodeArena) -> f32 {
        let mut sum = 0.0;
        for (idx, child) in self.children.iter().enumerate() {
            if (child.load(Ordering::Acquire) != EMPTY_CHILD)
                .then(|| child.load(Ordering::Acquire))
                .and_then(|child_idx| arena.get(child_idx))
                .is_some_and(|node| node.n() > 0)
            {
                sum += self.edges[idx].get_p();
            }
        }
        sum
    }

    fn release_children_except_one(&mut self, keep_edge: Option<usize>) {
        for (idx, child) in self.children.iter().enumerate() {
            if Some(idx) != keep_edge {
                child.store(EMPTY_CHILD, Ordering::Release);
            }
        }
        // px0 `ReleaseChildrenExceptOne(nullptr)` resets both `num_edges_`
        // and `edges_` (`node.cc:445-448`). This is required before
        // `CreateSingleChildNode()` handles a line absent from the old tree.
        if keep_edge.is_none() {
            self.children.clear();
            self.edges.clear();
        }
        self.solid_children = false;
    }

    fn reset_in_place(&mut self, parent: Option<usize>, edge_index: u16) {
        let sibling_edges = std::mem::take(&mut self.edges);
        let _ = sibling_edges;
        *self = Self::new(parent, edge_index);
    }
}

/// px0 `Node` 的稳定分配存储（`src/search/classic/node.h:282-300`）。
///
/// px0 用 `unique_ptr<Node>` 保存 child；Rust 保留 arena 索引作为 parent/child
/// 链接，但每个 Node 使用 Box 单独分配，避免 arena 扩容时改变节点地址。
#[derive(Debug, Default)]
#[allow(clippy::vec_box)] // px0 child nodes retain stable heap addresses across arena growth.
pub struct NodeArena {
    /// `Box<Node>` keeps each node address stable while the vector grows.
    /// The lock protects vector metadata and allocation only; node-level
    /// ownership remains the px0 task-split/in-flight responsibility.
    /// Reference: `src/search/classic/node.h:282-300,468-525`.
    nodes: RwLock<Vec<Box<Node>>>,
}

impl NodeArena {
    /// px0 creates child ownership while its caller owns the tree phase
    /// (`src/search/classic/node.cc:196-210,465-520`). The safe Rust path
    /// requires an exclusive arena borrow; scoped task workers remain gated.
    pub fn alloc(&mut self, node: Node) -> usize {
        let nodes = self.nodes.get_mut();
        let idx = nodes.len();
        nodes.push(Box::new(node));
        idx
    }

    pub fn get(&self, idx: usize) -> Option<&Node> {
        let node = self
            .nodes
            .read()
            .get(idx)
            .map(|node| Box::as_ref(node) as *const Node)?;
        // SAFETY: every entry is a Box and NodeArena never removes or moves a
        // Node allocation until the complete tree is dropped after workers
        // have joined. The RwLock only protects the Vec metadata.
        Some(unsafe { &*node })
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Node> {
        self.nodes.get_mut().get_mut(idx).map(Box::as_mut)
    }

    pub fn spawn_child(&mut self, parent_idx: usize, edge_idx: usize) -> usize {
        if !self
            .get(parent_idx)
            .expect("valid parent node")
            .try_reserve_child(edge_idx)
        {
            return self
                .get(parent_idx)
                .expect("valid parent node")
                .child(edge_idx)
                .expect("reserved child slot must be published");
        }
        let child_idx = self.alloc(Node::new(Some(parent_idx), edge_idx as u16));
        self.get(parent_idx)
            .expect("valid parent node")
            .publish_child(edge_idx, child_idx);
        child_idx
    }
}

impl Clone for NodeArena {
    fn clone(&self) -> Self {
        let nodes = self
            .nodes
            .read()
            .iter()
            .map(|node| Box::new((**node).clone()))
            .collect();
        Self {
            nodes: RwLock::new(nodes),
        }
    }
}

/// px0 `NodeTree` (`node.h` + `node.cc:465-543`)。
#[derive(Clone, Debug, Default)]
pub struct NodeTree {
    arena: NodeArena,
    gamebegin: Option<usize>,    // Root node of a game tree.
    current_head: Option<usize>, // A node which to start search from.
    history: PositionHistory,
}

impl NodeTree {
    pub fn current_head(&self) -> usize {
        self.current_head.expect("NodeTree has no current head")
    }

    pub fn history(&self) -> &PositionHistory {
        &self.history
    }

    pub fn arena(&self) -> &NodeArena {
        &self.arena
    }

    pub fn arena_mut(&mut self) -> &mut NodeArena {
        &mut self.arena
    }

    pub fn node(&self, idx: usize) -> &Node {
        self.arena.get(idx).expect("valid node index")
    }

    pub fn node_mut(&mut self, idx: usize) -> &mut Node {
        self.arena.get_mut(idx).expect("valid node index")
    }

    /// px0 `EdgeAndNode(Edge*, Node*)` (`node.h:358-410`) 的 arena 适配。
    pub fn edge_and_node(&self, node_idx: usize, edge_idx: usize) -> EdgeAndNode<'_> {
        let node = self.node(node_idx);
        EdgeAndNode::new(
            node.edge(edge_idx),
            node.child(edge_idx).and_then(|idx| self.arena.get(idx)),
        )
    }

    /// px0 `Node::MakeTerminal` (`node.cc:300-317`) 与 `GetOwnEdge`
    /// (`node.h:244-248`) 的 arena 适配。
    pub fn make_terminal(&mut self, node_idx: usize, result: GameResult, plies_left: f32, terminal: Terminal) {
        let parent = self.node(node_idx).parent;
        let edge_index = self.node(node_idx).edge_index as usize;
        self.node_mut(node_idx).make_terminal(result, plies_left, terminal);
        if result == GameResult::BlackWon {
            if let Some(parent) = parent {
                self.node_mut(parent).edge_mut(edge_index).set_p(0.0);
            }
        }
    }

    /// px0 `Node::MakeNotTerminal` (`node.cc:319-341`) 的 arena 适配。
    pub fn make_not_terminal(&mut self, node_idx: usize) {
        let child_indices: Vec<usize> = self
            .node(node_idx)
            .children
            .iter()
            .filter_map(|child| {
                let index = child.load(Ordering::Acquire);
                (index != EMPTY_CHILD).then_some(index)
            })
            .collect();
        let child_stats = child_indices
            .into_iter()
            .map(|child| {
                let node = self.node(child);
                (node.n(), node.wl, node.d)
            })
            .collect::<Vec<_>>();
        self.node_mut(node_idx).make_not_terminal(&child_stats);
    }

    /// px0 `Node::MakeSolid` (`src/search/classic/node.cc:245-288`) adapted
    /// to indexed stable Rust allocations. px0 reallocates the sibling chain
    /// into a `Node[]`; this arena instead fills every missing edge slot with a
    /// stable boxed child. The externally observable condition is identical:
    /// a solid node has one child node for every edge and may not be solidified
    /// while an immediate leaf/terminal child is in flight.
    pub fn make_solid(&mut self, node_idx: usize) -> bool {
        let node = self.node(node_idx);
        if node.has_solid_children() || node.num_edges() == 0 || node.is_terminal() {
            return false;
        }

        let mut total_in_flight = 0u32;
        for child_idx in node.children.iter().filter_map(|child| {
            let index = child.load(Ordering::Acquire);
            (index != EMPTY_CHILD).then_some(index)
        }) {
            let child = self.node(child_idx);
            if (child.n() <= 1 || child.is_terminal()) && child.n_in_flight() > 0 {
                return false;
            }
            total_in_flight += child.n_in_flight();
        }
        if total_in_flight != node.n_in_flight() {
            return false;
        }

        let missing_edges = self
            .node(node_idx)
            .children
            .iter()
            .enumerate()
            .filter_map(|(edge_idx, child)| (child.load(Ordering::Acquire) == EMPTY_CHILD).then_some(edge_idx))
            .collect::<Vec<_>>();
        for edge_idx in missing_edges {
            self.arena.spawn_child(node_idx, edge_idx);
        }
        self.node_mut(node_idx).solid_children = true;
        true
    }

    /// px0 `NodeTree::MakeMove` (`node.cc:465-481`)。
    pub fn make_move(&mut self, mv: Move) {
        let head = self.current_head();
        let mut kept_edge = None;
        for edge_idx in 0..self.node(head).num_edges() {
            if self.node(head).edge(edge_idx).mv != mv {
                continue;
            }
            let child_idx = match self.node(head).child(edge_idx) {
                Some(idx) => idx,
                None => self.arena.spawn_child(head, edge_idx),
            };
            if self.node(child_idx).is_terminal() {
                self.make_not_terminal(child_idx);
            }
            kept_edge = Some(edge_idx);
            self.current_head = Some(child_idx);
            break;
        }
        self.node_mut(head).release_children_except_one(kept_edge);
        if kept_edge.is_none() {
            let head = self.current_head();
            self.node_mut(head).create_single_child_node(mv);
            let child_idx = self.arena.spawn_child(head, 0);
            self.current_head = Some(child_idx);
        }
        self.history.append(mv);
    }

    /// px0 `NodeTree::TrimTreeAtHead` (`node.cc:483-491`)。
    pub fn trim_tree_at_head(&mut self) {
        let head = self.current_head();
        let parent = self.node(head).parent;
        let edge_index = self.node(head).edge_index;
        self.node_mut(head).release_children_except_one(None);
        self.node_mut(head).reset_in_place(parent, edge_index);
    }

    /// px0 `NodeTree::ResetToPosition` (`node.cc:493-520`)。
    pub fn reset_to_position(&mut self, startpos: &Position, moves: &[Move]) -> bool {
        if self.gamebegin.is_some() && self.history.starting() != startpos {
            self.deallocate_tree();
        }
        if self.gamebegin.is_none() {
            let root = self.arena.alloc(Node::new(None, 0));
            self.gamebegin = Some(root);
        }
        self.history.reset_position(startpos.clone());
        let old_head = self.current_head;
        self.current_head = self.gamebegin;
        let mut seen_old_head = old_head == self.current_head;
        for &mv in moves {
            self.make_move(mv);
            if old_head == self.current_head {
                seen_old_head = true;
            }
        }
        if !seen_old_head {
            self.trim_tree_at_head();
        }
        seen_old_head
    }

    fn deallocate_tree(&mut self) {
        self.arena = NodeArena::default();
        self.gamebegin = None;
        self.current_head = None;
        self.history = PositionHistory::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_prior_roundtrip() {
        for p in [0.0, 0.001, 0.25, 0.5, 0.999, 1.0] {
            let mut edge = Edge::new(Move::NULL);
            edge.set_p(p);
            let decoded = edge.get_p();
            assert!((decoded - p).abs() < 0.02, "p={p} decoded={decoded}");
        }
    }

    #[test]
    fn edge_and_node_matches_px0_q_u_proxy() {
        let a0 = xiangqi_core::Square::parse("a0").expect("a0");
        let a1 = xiangqi_core::Square::parse("a1").expect("a1");
        let mut arena = NodeArena::default();
        let root = arena.alloc(Node::default());
        arena
            .get_mut(root)
            .expect("root")
            .create_single_child_node(Move::new(a0, a1));
        arena.get_mut(root).expect("root").edge_mut(0).set_p(0.5);

        let unvisited = EdgeAndNode::new(
            arena.get(root).expect("root").edge(0),
            arena.get(root).expect("root").child(0).and_then(|idx| arena.get(idx)),
        );
        assert!((unvisited.q(0.25, 0.0) - 0.25).abs() < f32::EPSILON);
        assert!((unvisited.u(4.0) - 2.0).abs() < 0.02);

        let child = arena.spawn_child(root, 0);
        let child_node = arena.get_mut(child).expect("child");
        assert!(child_node.try_start_score_update());
        child_node.finalize_score_update(0.75, 0.0, 0.0, 1);
        let visited = EdgeAndNode::new(
            arena.get(root).expect("root").edge(0),
            arena.get(root).expect("root").child(0).and_then(|idx| arena.get(idx)),
        );
        assert!((visited.q(0.25, 0.0) - 0.75).abs() < f32::EPSILON);
        assert_eq!(visited.n_started(), 1);
        assert!((visited.u(4.0) - 1.0).abs() < 0.02);
    }

    #[test]
    fn finalize_score_update_matches_running_average() {
        let mut node = Node::default();
        assert!(node.try_start_score_update());
        node.finalize_score_update(0.5, 0.1, 2.0, 1);
        assert_eq!(node.n(), 1);
        assert!((node.wl() - 0.5).abs() < 1e-6);
        assert!(node.try_start_score_update());
        node.finalize_score_update(0.0, 0.0, 0.0, 1);
        assert_eq!(node.n(), 2);
        assert!((node.wl() - 0.25).abs() < 1e-6);
        assert_eq!(node.n_in_flight(), 0);
    }

    /// px0 `Node::TryStartScoreUpdate` rejects a second in-flight visit for
    /// an unexpanded node (`src/search/classic/node.cc:348-352`). Rust keeps
    /// that rule atomic so gathering tasks cannot both win the first extend.
    #[test]
    fn first_score_update_has_one_concurrent_winner() {
        let node = std::sync::Arc::new(Node::default());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let node = std::sync::Arc::clone(&node);
            let barrier = std::sync::Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                node.try_start_score_update()
            }));
        }
        assert_eq!(
            workers
                .into_iter()
                .filter_map(|worker| worker.join().ok())
                .filter(|started| *started)
                .count(),
            1
        );
        assert_eq!(node.n_in_flight(), 1);
    }

    /// px0 `GetOrSpawnNode` has one child-slot winner even when several
    /// gathering tasks reach the same edge (`src/search/classic/node.h:468-525`).
    #[test]
    fn child_slot_has_one_concurrent_reservation_winner() {
        let mut node = Node::default();
        node.create_single_child_node(Move::new(
            xiangqi_core::Square::parse("a0").expect("a0"),
            xiangqi_core::Square::parse("a1").expect("a1"),
        ));
        let node = std::sync::Arc::new(node);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let node = std::sync::Arc::clone(&node);
            let barrier = std::sync::Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                node.try_reserve_child(0)
            }));
        }
        assert_eq!(
            workers
                .into_iter()
                .filter_map(|worker| worker.join().ok())
                .filter(|reserved| *reserved)
                .count(),
            1
        );
        node.publish_child(0, 42);
        assert_eq!(node.child(0), Some(42));
    }

    #[test]
    fn revert_terminal_visits_matches_px0_zero_reset() {
        let mut node = Node::default();
        assert!(node.try_start_score_update());
        node.finalize_score_update(0.5, 0.25, 4.0, 1);

        node.revert_terminal_visits(0.5, 0.25, 4.0, 1);

        assert_eq!(node.n(), 0);
        assert!((node.wl() - 0.0).abs() < f32::EPSILON);
        assert!((node.d() - 1.0).abs() < f32::EPSILON);
        assert!((node.m() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn revert_terminal_visits_restores_prior_average() {
        let mut node = Node::default();
        assert!(node.try_start_score_update());
        node.finalize_score_update(0.2, 0.4, 3.0, 1);
        assert!(node.try_start_score_update());
        node.finalize_score_update(1.0, 0.0, 1.0, 1);

        node.revert_terminal_visits(1.0, 0.0, 1.0, 1);

        assert_eq!(node.n(), 1);
        assert!((node.wl() - 0.2).abs() < 1e-6);
        assert!((node.d() - 0.4).abs() < 1e-6);
        assert!((node.m() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn sort_edges_orders_policy_before_children_exist() {
        let a0 = xiangqi_core::Square::parse("a0").expect("a0");
        let a1 = xiangqi_core::Square::parse("a1").expect("a1");
        let a2 = xiangqi_core::Square::parse("a2").expect("a2");
        let mut node = Node::default();
        node.create_edges(&vec![Move::new(a0, a1), Move::new(a0, a2)]);
        node.edge_mut(0).set_p(0.1);
        node.edge_mut(1).set_p(0.9);

        node.sort_edges();

        assert!(node.edge(0).get_p() > node.edge(1).get_p());
        assert_eq!(node.edge(0).mv, Move::new(a0, a2));
    }

    /// px0 `Node::MakeSolid` makes every edge addressable as a child only when
    /// no immediate leaf/terminal child is in flight (`node.cc:245-288`).
    #[test]
    fn make_solid_fills_all_child_slots_after_safety_check() {
        let a0 = xiangqi_core::Square::parse("a0").expect("a0");
        let a1 = xiangqi_core::Square::parse("a1").expect("a1");
        let a2 = xiangqi_core::Square::parse("a2").expect("a2");
        let startpos = Position::from_fen(xiangqi_core::STARTPOS_FEN).expect("startpos");
        let mut tree = NodeTree::default();
        tree.reset_to_position(&startpos, &[]);
        let root = tree.current_head();
        tree.node_mut(root)
            .create_edges(&vec![Move::new(a0, a1), Move::new(a0, a2)]);

        assert!(tree.make_solid(root));
        assert!(tree.node(root).has_solid_children());
        for edge_idx in 0..tree.node(root).num_edges() {
            let child = tree.node(root).child(edge_idx).expect("solid child");
            assert_eq!(tree.node(child).parent(), Some(root));
            assert_eq!(tree.node(child).edge_index(), edge_idx as u16);
        }
        assert!(!tree.make_solid(root));
    }

    #[test]
    fn make_solid_rejects_immediate_leaf_in_flight() {
        let a0 = xiangqi_core::Square::parse("a0").expect("a0");
        let a1 = xiangqi_core::Square::parse("a1").expect("a1");
        let startpos = Position::from_fen(xiangqi_core::STARTPOS_FEN).expect("startpos");
        let mut tree = NodeTree::default();
        tree.reset_to_position(&startpos, &[]);
        let root = tree.current_head();
        tree.node_mut(root).create_edges(&vec![Move::new(a0, a1)]);
        let child = tree.arena_mut().spawn_child(root, 0);
        assert!(tree.node_mut(root).try_start_score_update());
        assert!(tree.node_mut(child).try_start_score_update());

        assert!(!tree.make_solid(root));
        assert!(!tree.node(root).has_solid_children());
    }

    #[test]
    fn reset_to_new_line_clears_unmatched_edges() {
        use xiangqi_core::{GameState, Position, STARTPOS_FEN};

        let startpos = Position::from_fen(STARTPOS_FEN).expect("startpos");
        let first = startpos.board().parse_move("a0a1").expect("first move");
        let new_line = startpos.board().parse_move("h2h4").expect("new line");

        let mut tree = NodeTree::default();
        tree.reset_to_position(&startpos, &[]);
        let root = tree.current_head();
        tree.node_mut(root).create_edges(&vec![first]);

        let state = GameState::new(startpos, vec![new_line]);
        tree.reset_to_position(&state.startpos, &state.moves);
        assert_eq!(tree.history().len(), 2);
    }
}
