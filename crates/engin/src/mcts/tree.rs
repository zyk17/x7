use super::{MctsNode, MctsNodeId};
use std::collections::HashMap;

/// 面向 MCTS 的简易树容器。
#[derive(Debug, Default)]
pub struct MctsTree {
    nodes: Vec<MctsNode>,
}

impl MctsTree {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
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

    pub fn copy_subtree(&self, root_id: MctsNodeId) -> (MctsTree, MctsNodeId) {
        let mut new_tree = MctsTree::new();
        let mut remap = HashMap::<usize, MctsNodeId>::new();
        let new_root = copy_subtree_rec(self, root_id, &mut new_tree, &mut remap);
        (new_tree, new_root)
    }
}

fn copy_subtree_rec(
    src: &MctsTree,
    old_id: MctsNodeId,
    dst: &mut MctsTree,
    remap: &mut HashMap<usize, MctsNodeId>,
) -> MctsNodeId {
    if let Some(&existing) = remap.get(&old_id.0) {
        return existing;
    }
    let old_node = src.get(old_id).expect("subtree node must exist");
    let new_id = dst.add_node(MctsNode {
        state_key: old_node.state_key,
        visits: old_node.visits,
        value_sum: old_node.value_sum,
        expanded: old_node.expanded,
        terminal_value: old_node.terminal_value,
        children: old_node.children.clone(),
    });
    remap.insert(old_id.0, new_id);
    let child_ids = old_node
        .children
        .iter()
        .map(|edge| edge.child)
        .collect::<Vec<_>>();
    for (idx, child) in child_ids.into_iter().enumerate() {
        if let Some(child_id) = child {
            let mapped_child = copy_subtree_rec(src, child_id, dst, remap);
            dst.get_mut(new_id).expect("new subtree node").children[idx].child = Some(mapped_child);
        }
    }
    new_id
}
