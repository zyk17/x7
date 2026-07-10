use std::collections::{HashSet, VecDeque};

use super::node::{MctsNode, MctsNodeId, TerminalKind};
use xiangqi_core::types::Move;

/// 延后 GC：死节点数或占比超阈值才做物理压缩。
const GC_MIN_DEAD_NODES: usize = 128;

/// 面向 MCTS 的简易树容器。
#[derive(Debug, Default)]
pub struct MctsTree {
    nodes: Vec<MctsNode>,
    pub(crate) gamebegin_id: Option<MctsNodeId>,
    gamebegin_key: u64,
}

impl MctsTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            gamebegin_id: None,
            gamebegin_key: 0,
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.gamebegin_id = None;
        self.gamebegin_key = 0;
    }

    /// `Vec` 槽位数（含已断开但仍占位的节点）。
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 从 `gamebegin` 仍可达的活跃节点数；无 gamebegin 时等同 `len()`。
    pub fn reachable_len(&self) -> usize {
        self.collect_reachable().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn gamebegin_id(&self) -> Option<MctsNodeId> {
        self.gamebegin_id
    }

    pub(crate) fn gamebegin_start_key(&self) -> u64 {
        self.gamebegin_key
    }

    pub fn add_node(&mut self, node: MctsNode) -> MctsNodeId {
        let id = MctsNodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn get(&self, id: MctsNodeId) -> Option<&MctsNode> {
        self.nodes.get(id.0)
    }

    pub fn get_mut(&mut self, id: MctsNodeId) -> Option<&mut MctsNode> {
        self.nodes.get_mut(id.0)
    }

    pub fn set_gamebegin(&mut self, id: MctsNodeId, start_key: u64) {
        self.gamebegin_id = Some(id);
        self.gamebegin_key = start_key;
    }

    /// px0 `NodeTree::MakeMove`：断兄弟引用，不在热路径做全量压缩。
    pub fn make_move(&mut self, head_id: MctsNodeId, mv: Move, child_state_key: u64) -> Option<MctsNodeId> {
        let edge_idx = self.get(head_id)?.children.iter().position(|edge| edge.mv == mv)?;
        let child_id = {
            let head = self.get_mut(head_id)?;
            release_siblings_except(head, edge_idx);
            head.children[edge_idx].child
        };
        let new_head = if let Some(child_id) = child_id {
            if let Some(child) = self.get_mut(child_id) {
                if child.is_terminal() {
                    child.make_not_terminal();
                }
            }
            child_id
        } else {
            self.create_single_child_node(head_id, edge_idx, child_state_key)?
        };
        Some(new_head)
    }

    /// px0 `ResetToPosition`：从 gamebegin 重放全路径。
    pub fn reset_to_position(
        &mut self,
        game_start_key: u64,
        moves: &[Move],
        position_keys: &[u64],
        old_head: Option<MctsNodeId>,
    ) -> (MctsNodeId, bool) {
        if self.gamebegin_id.is_none() || self.gamebegin_key != game_start_key {
            self.clear();
        }
        let gamebegin = self.gamebegin_id.expect("gamebegin must exist after init");
        let mut current = gamebegin;
        let mut seen_old = old_head == Some(current);
        for (idx, &mv) in moves.iter().enumerate() {
            let child_key = position_keys.get(idx).copied().unwrap_or(0);
            if let Some(next) = self.make_move(current, mv, child_key) {
                current = next;
                if old_head == Some(current) {
                    seen_old = true;
                }
            } else {
                seen_old = false;
                break;
            }
        }
        if !seen_old {
            self.trim_tree_at_head(current);
        }
        (self.compact_if_bloated(current), seen_old)
    }

    /// 沿 path 推进根节点，每步裁剪兄弟子树引用（不拷贝节点）。
    pub fn advance_root(&mut self, root_id: MctsNodeId, path: &[Move], position_keys: &[u64]) -> Option<MctsNodeId> {
        let mut node_id = root_id;
        for (idx, mv) in path.iter().enumerate() {
            let child_key = position_keys.get(idx).copied().unwrap_or(0);
            node_id = self.make_move(node_id, *mv, child_key)?;
        }
        Some(node_id)
    }

    /// px0 `TrimTreeAtHead`：回退到祖先局面时清空当前根子树统计。
    pub fn trim_tree_at_head(&mut self, root_id: MctsNodeId) {
        if let Some(node) = self.get_mut(root_id) {
            for edge in &mut node.children {
                edge.child = None;
                edge.visits = 0;
                edge.in_flight = 0;
                edge.wl = 0.0;
                edge.d = 0.0;
                edge.m = 0.0;
            }
            node.visits = 0;
            node.in_flight = 0;
            node.wl = 0.0;
            node.d = 0.0;
            node.m = 0.0;
            node.terminal_kind = TerminalKind::NonTerminal;
            node.terminal_value = None;
        }
    }

    /// 在 `prepare_root` / `reset_to_position` 等边界做延后 GC，热路径 `make_move` 不调用。
    pub(crate) fn compact_if_bloated(&mut self, head_id: MctsNodeId) -> MctsNodeId {
        let reachable = self.collect_reachable().len();
        let dead = self.nodes.len().saturating_sub(reachable);
        if dead == 0 {
            return head_id;
        }
        if dead < GC_MIN_DEAD_NODES && dead <= reachable {
            return head_id;
        }
        self.compact_reachable(head_id)
    }

    pub(crate) fn for_each_reachable<F>(&self, mut f: F)
    where
        F: FnMut(MctsNodeId),
    {
        for idx in self.collect_reachable() {
            f(MctsNodeId(idx));
        }
    }

    fn create_single_child_node(
        &mut self,
        head_id: MctsNodeId,
        edge_idx: usize,
        state_key: u64,
    ) -> Option<MctsNodeId> {
        let child_id = self.add_node(MctsNode {
            state_key,
            visits: 0,
            in_flight: 0,
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            expanded: false,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        });
        let head = self.get_mut(head_id)?;
        head.children[edge_idx].child = Some(child_id);
        Some(child_id)
    }

    fn collect_reachable(&self) -> HashSet<usize> {
        let mut reachable = HashSet::<usize>::new();
        let Some(gb) = self.gamebegin_id else {
            return (0..self.nodes.len()).collect();
        };
        if self.nodes.get(gb.0).is_none() {
            return reachable;
        }
        let mut queue = VecDeque::from([gb.0]);
        while let Some(idx) = queue.pop_front() {
            if !reachable.insert(idx) {
                continue;
            }
            let Some(node) = self.nodes.get(idx) else {
                continue;
            };
            for edge in &node.children {
                if let Some(child) = edge.child {
                    queue.push_back(child.0);
                }
            }
        }
        reachable
    }

    pub(crate) fn compact_reachable(&mut self, head_id: MctsNodeId) -> MctsNodeId {
        let roots = match self.gamebegin_id {
            Some(gb) => vec![gb],
            None => vec![head_id],
        };
        let remap = self.compact_unreachable(&roots);
        MctsNodeId(remap.get(&head_id.0).copied().unwrap_or(head_id.0))
    }

    /// 从 `keep_roots` 可达的节点子集压缩 `nodes`，丢弃被 `child=None` 切断的孤儿子树。
    fn compact_unreachable(&mut self, keep_roots: &[MctsNodeId]) -> std::collections::HashMap<usize, usize> {
        let mut reachable = HashSet::<usize>::new();
        let mut queue = VecDeque::new();
        for root in keep_roots {
            if self.nodes.get(root.0).is_some() {
                queue.push_back(root.0);
            }
        }
        while let Some(idx) = queue.pop_front() {
            if !reachable.insert(idx) {
                continue;
            }
            let Some(node) = self.nodes.get(idx) else {
                continue;
            };
            for edge in &node.children {
                if let Some(child) = edge.child {
                    queue.push_back(child.0);
                }
            }
        }

        if reachable.len() == self.nodes.len() {
            return (0..self.nodes.len()).map(|i| (i, i)).collect();
        }

        let mut old_to_new = std::collections::HashMap::with_capacity(reachable.len());
        let mut new_nodes = Vec::with_capacity(reachable.len());
        for (old_idx, node) in self.nodes.iter().enumerate() {
            if reachable.contains(&old_idx) {
                old_to_new.insert(old_idx, new_nodes.len());
                new_nodes.push(node.clone());
            }
        }
        for node in &mut new_nodes {
            for edge in &mut node.children {
                if let Some(child) = edge.child {
                    edge.child = old_to_new.get(&child.0).copied().map(MctsNodeId);
                }
            }
        }
        if let Some(gb) = self.gamebegin_id {
            if let Some(&new_idx) = old_to_new.get(&gb.0) {
                self.gamebegin_id = Some(MctsNodeId(new_idx));
            } else {
                self.gamebegin_id = None;
            }
        }
        self.nodes = new_nodes;
        old_to_new
    }
}

fn release_siblings_except(node: &mut MctsNode, keep_idx: usize) {
    for (idx, edge) in node.children.iter_mut().enumerate() {
        if idx != keep_idx {
            edge.child = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mcts::EdgeStats;
    use super::*;
    use xiangqi_core::types::{Move, Square};

    #[test]
    fn make_move_disconnects_siblings_without_immediate_compact() {
        let mut tree = MctsTree::new();
        let root = tree.add_node(MctsNode::default());
        tree.set_gamebegin(root, 0);
        tree.get_mut(root).unwrap().children = vec![
            EdgeStats {
                mv: Move::make(Square::SQ_A0, Square::SQ_A1),
                child: Some(tree.add_node(MctsNode {
                    state_key: 1,
                    ..Default::default()
                })),
                ..Default::default()
            },
            EdgeStats {
                mv: Move::make(Square::SQ_B0, Square::SQ_B1),
                child: Some(tree.add_node(MctsNode {
                    state_key: 2,
                    ..Default::default()
                })),
                ..Default::default()
            },
        ];

        let kept = tree
            .make_move(root, Move::make(Square::SQ_A0, Square::SQ_A1), 1)
            .expect("move");

        assert_eq!(tree.len(), 3, "dead slots remain until deferred gc");
        assert_eq!(tree.reachable_len(), 2);
        assert!(tree.get(kept).is_some());
        assert!(tree.get(MctsNodeId(2)).is_some(), "orphan slot still present");
    }

    #[test]
    fn compact_if_bloated_reclaims_unreachable_nodes() {
        let mut tree = MctsTree::new();
        let root = tree.add_node(MctsNode::default());
        tree.set_gamebegin(root, 0);
        tree.get_mut(root).unwrap().children = vec![
            EdgeStats {
                mv: Move::make(Square::SQ_A0, Square::SQ_A1),
                child: Some(tree.add_node(MctsNode {
                    state_key: 1,
                    ..Default::default()
                })),
                ..Default::default()
            },
            EdgeStats {
                mv: Move::make(Square::SQ_B0, Square::SQ_B1),
                child: Some(tree.add_node(MctsNode {
                    state_key: 2,
                    ..Default::default()
                })),
                ..Default::default()
            },
        ];
        let kept = tree
            .make_move(root, Move::make(Square::SQ_A0, Square::SQ_A1), 1)
            .expect("move");

        let kept = tree.compact_reachable(kept);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.reachable_len(), 2);
        assert!(tree.get(kept).is_some());
        assert!(tree.get(MctsNodeId(2)).is_none());
    }
}
