use xiangqi_core::types::Move;

/// 树中节点句柄。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MctsNodeId(pub usize);

/// 单条边的累计统计。
#[derive(Clone, Debug)]
pub struct EdgeStats {
    pub mv: Move,
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
    pub child: Option<MctsNodeId>,
}

impl Default for EdgeStats {
    fn default() -> Self {
        Self {
            mv: Move::none(),
            prior: 0.0,
            visits: 0,
            value_sum: 0.0,
            child: None,
        }
    }
}

impl EdgeStats {
    #[inline]
    pub fn mean_q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

/// MCTS 节点最小记录。
#[derive(Clone, Debug, Default)]
pub struct MctsNode {
    pub state_key: u64,
    pub visits: u32,
    pub value_sum: f32,
    pub expanded: bool,
    pub terminal_value: Option<f32>,
    pub children: Vec<EdgeStats>,
}

impl MctsNode {
    #[inline]
    pub fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}
