use super::{MctsNode, MctsNodeId};

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
}
