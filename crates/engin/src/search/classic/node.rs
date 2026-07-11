//! px0 `src/search/classic/node.h:84-260`、`node.cc:161-373,465-543`。

use xiangqi_core::{GameResult, Move, MoveList, Position, PositionHistory};

/// px0 `Node::Terminal` (`node.h:132`)。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Terminal {
    #[default]
    NonTerminal,
    EndOfGame,
    Tablebase,
    TwoFold,
}

/// px0 `Edge` (`node.h:85-112`)。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Edge {
    pub mv: Move,
    p: u16,
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

/// px0 `Node` 统计与树结构（单线程：children 与 edges 平行索引）。
#[derive(Clone, Debug)]
pub struct Node {
    parent: Option<usize>,
    edge_index: u16,
    edges: Vec<Edge>,
    children: Vec<Option<usize>>,
    terminal: Terminal,
    lower_bound: GameResult,
    upper_bound: GameResult,
    wl: f32,
    d: f32,
    m: f32,
    n: u32,
    n_in_flight: u32,
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
            n: 0,
            n_in_flight: 0,
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

    pub const fn n(&self) -> u32 {
        self.n
    }

    pub const fn n_in_flight(&self) -> u32 {
        self.n_in_flight
    }

    pub fn n_started(&self) -> u32 {
        self.n + self.n_in_flight
    }

    pub fn children_visits(&self) -> u32 {
        if self.n > 0 {
            self.n - 1
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

    pub fn q(&self, draw_score: f32) -> f32 {
        self.wl + draw_score * self.d
    }

    pub const fn is_terminal(&self) -> bool {
        !matches!(self.terminal, Terminal::NonTerminal)
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

    pub fn child(&self, index: usize) -> Option<usize> {
        self.children.get(index).copied().flatten()
    }

    /// px0 `Node::CreateEdges` (`node.cc:205-210`)。
    pub fn create_edges(&mut self, moves: &MoveList) {
        assert!(self.edges.is_empty());
        assert!(self.children.iter().all(|child| child.is_none()));
        self.edges = moves.iter().copied().map(Edge::new).collect();
        self.children = vec![None; moves.len()];
    }

    /// px0 `Node::CreateSingleChildNode` (`node.cc:196-203`)。
    pub fn create_single_child_node(&mut self, mv: Move) -> usize {
        assert!(self.edges.is_empty());
        self.edges = vec![Edge::new(mv)];
        self.children = vec![None];
        0
    }

    /// px0 `Node::TryStartScoreUpdate` (`node.cc:348-352`)。
    pub fn try_start_score_update(&mut self) -> bool {
        if self.n == 0 && self.n_in_flight > 0 {
            return false;
        }
        self.n_in_flight += 1;
        true
    }

    /// px0 `Node::CancelScoreUpdate` (`node.cc:354`)。
    pub fn cancel_score_update(&mut self, multivisit: u32) {
        self.n_in_flight -= multivisit;
    }

    /// px0 `Node::FinalizeScoreUpdate` (`node.cc:356-366`)。
    pub fn finalize_score_update(&mut self, v: f32, d: f32, m: f32, multivisit: u32) {
        self.wl += multivisit as f32 * (v - self.wl) / (self.n + multivisit) as f32;
        self.d += multivisit as f32 * (d - self.d) / (self.n + multivisit) as f32;
        self.m += multivisit as f32 * (m - self.m) / (self.n + multivisit) as f32;
        self.n += multivisit;
        self.n_in_flight -= multivisit;
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
        self.n = 0;
        if !self.edges.is_empty() {
            self.n = 1;
            self.wl = 0.0;
            self.d = 0.0;
            for &(child_n, child_wl, child_d) in child_stats {
                if child_n > 0 {
                    self.n += child_n;
                    self.wl += -child_wl * child_n as f32;
                    self.d += child_d * child_n as f32;
                }
            }
            self.wl /= self.n as f32;
            self.d /= self.n as f32;
        }
    }

    pub fn set_bounds(&mut self, lower: GameResult, upper: GameResult) {
        self.lower_bound = lower;
        self.upper_bound = upper;
    }

    pub fn visited_policy(&self, arena: &NodeArena) -> f32 {
        let mut sum = 0.0;
        for (idx, child) in self.children.iter().enumerate() {
            if child
                .and_then(|child_idx| arena.get(child_idx))
                .is_some_and(|node| node.n > 0)
            {
                sum += self.edges[idx].get_p();
            }
        }
        sum
    }

    fn release_children_except_one(&mut self, keep_edge: Option<usize>) {
        for (idx, child) in self.children.iter_mut().enumerate() {
            if Some(idx) != keep_edge {
                *child = None;
            }
        }
        // px0 `ReleaseChildrenExceptOne(nullptr)` resets both `num_edges_`
        // and `edges_` (`node.cc:445-448`). This is required before
        // `CreateSingleChildNode()` handles a line absent from the old tree.
        if keep_edge.is_none() {
            self.children.clear();
            self.edges.clear();
        }
    }

    fn reset_in_place(&mut self, parent: Option<usize>, edge_index: u16) {
        let sibling_edges = std::mem::take(&mut self.edges);
        let _ = sibling_edges;
        *self = Self::new(parent, edge_index);
    }
}

/// 节点存储区。
#[derive(Clone, Debug, Default)]
pub struct NodeArena {
    nodes: Vec<Node>,
}

impl NodeArena {
    pub fn alloc(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub fn get(&self, idx: usize) -> Option<&Node> {
        self.nodes.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Node> {
        self.nodes.get_mut(idx)
    }

    pub fn spawn_child(&mut self, parent_idx: usize, edge_idx: usize) -> usize {
        let child_idx = self.alloc(Node::new(Some(parent_idx), edge_idx as u16));
        self.nodes[parent_idx].children[edge_idx] = Some(child_idx);
        child_idx
    }
}

/// px0 `NodeTree` (`node.h` + `node.cc:465-543`)。
#[derive(Clone, Debug, Default)]
pub struct NodeTree {
    arena: NodeArena,
    gamebegin: Option<usize>,
    current_head: Option<usize>,
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
    fn make_not_terminal(&mut self, node_idx: usize) {
        let child_indices: Vec<usize> = self.node(node_idx).children.iter().copied().flatten().collect();
        let child_stats = child_indices
            .into_iter()
            .map(|child| {
                let node = self.node(child);
                (node.n, node.wl, node.d)
            })
            .collect::<Vec<_>>();
        self.node_mut(node_idx).make_not_terminal(&child_stats);
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
