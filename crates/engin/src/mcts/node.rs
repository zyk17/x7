use xiangqi_core::types::Move;

/// 终局类型，对齐 lc0 classic `Node::Terminal`。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerminalKind {
    #[default]
    NonTerminal,
    Generic,
    TwoFold,
}

/// 树中节点句柄。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MctsNodeId(pub usize);

/// lc0 classic：`wl/d/m` 增量均值 + `n_in_flight` 在边上跟踪虚拟损失。
#[derive(Clone, Debug)]
pub struct EdgeStats {
    pub mv: Move,
    pub prior: f32,
    pub visits: u32,
    pub in_flight: u32,
    pub wl: f32,
    pub d: f32,
    pub m: f32,
    pub child: Option<MctsNodeId>,
}

impl Default for EdgeStats {
    fn default() -> Self {
        Self {
            mv: Move::none(),
            prior: 0.0,
            visits: 0,
            in_flight: 0,
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            child: None,
        }
    }
}

impl EdgeStats {
    #[inline]
    pub fn n_started(&self) -> u32 {
        self.visits.saturating_add(self.in_flight)
    }

    /// lc0 classic `GetQ(draw_score=0)`；`q = wl + draw_score * d`。
    #[inline]
    pub fn mean_q(&self) -> f32 {
        self.wl
    }

    #[inline]
    pub fn mean_q_with_draw(&self, draw_score: f32) -> f32 {
        self.wl + draw_score * self.d
    }

    /// lc0 classic `Node::TryStartScoreUpdate`（边级 virtual loss）。
    #[inline]
    pub fn try_start_score_update(&mut self) -> bool {
        if self.visits == 0 && self.in_flight > 0 {
            return false;
        }
        self.in_flight = self.in_flight.saturating_add(1);
        true
    }

    #[inline]
    pub fn get_m(&self, parent_m: f32) -> f32 {
        if self.visits > 0 {
            self.m
        } else {
            parent_m
        }
    }

    /// lc0 classic `RevertTerminalVisits`（边级 twofold 深度修正）。
    #[inline]
    pub fn revert_terminal_visits(&mut self, wl: f32, d: f32, m: f32, multivisit: u32) {
        let n_new = self.visits.saturating_sub(multivisit);
        if n_new == 0 {
            self.visits = 0;
            self.in_flight = 0;
            self.wl = 0.0;
            self.d = 0.0;
            self.m = 0.0;
            return;
        }
        self.wl -= multivisit as f32 * wl / self.visits as f32;
        self.d -= multivisit as f32 * d / self.visits as f32;
        self.m -= multivisit as f32 * m / self.visits as f32;
        self.visits = n_new;
    }

    #[inline]
    pub fn finalize_score_update(&mut self, v: f32, d: f32, m: f32, multivisit: u32) {
        finalize_wdl_stats(
            &mut self.visits,
            &mut self.in_flight,
            &mut self.wl,
            &mut self.d,
            &mut self.m,
            v,
            d,
            m,
            multivisit,
        );
    }
}

/// MCTS 节点：统计语义对齐 lc0 classic `Node`。
#[derive(Clone, Debug)]
pub struct MctsNode {
    pub state_key: u64,
    pub visits: u32,
    pub in_flight: u32,
    pub wl: f32,
    pub d: f32,
    pub m: f32,
    pub expanded: bool,
    pub terminal_kind: TerminalKind,
    pub terminal_value: Option<f32>,
    pub children: Vec<EdgeStats>,
}

impl Default for MctsNode {
    fn default() -> Self {
        Self {
            state_key: 0,
            visits: 0,
            in_flight: 0,
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            expanded: false,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        }
    }
}

impl MctsNode {
    #[inline]
    pub fn mean_value(&self) -> f32 {
        self.wl
    }

    #[inline]
    pub fn mean_value_with_draw(&self, draw_score: f32) -> f32 {
        self.wl + draw_score * self.d
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.terminal_value.is_some()
    }

    #[inline]
    pub fn is_twofold_terminal(&self) -> bool {
        self.terminal_kind == TerminalKind::TwoFold
    }

    /// lc0 classic `Node::TryStartScoreUpdate`。
    #[inline]
    pub fn try_start_score_update(&mut self) -> bool {
        if self.visits == 0 && self.in_flight > 0 {
            return false;
        }
        self.in_flight = self.in_flight.saturating_add(1);
        true
    }

    /// lc0 classic `Node::GetNStarted()`。
    #[inline]
    pub fn n_started(&self) -> u32 {
        self.visits.saturating_add(self.in_flight)
    }

    /// lc0 `Node::IncrementNInFlight`：`PickNodesToExtend` 批量 virtual loss。
    #[inline]
    pub fn increment_n_in_flight(&mut self, multivisit: u32) {
        self.in_flight = self.in_flight.saturating_add(multivisit);
    }

    /// lc0 classic `GetChildrenVisits()`。
    #[inline]
    pub fn children_visits(&self) -> u32 {
        if self.visits > 0 {
            self.visits - 1
        } else {
            0
        }
    }

    #[inline]
    pub fn finalize_score_update(&mut self, v: f32, d: f32, m: f32, multivisit: u32) {
        finalize_wdl_stats(
            &mut self.visits,
            &mut self.in_flight,
            &mut self.wl,
            &mut self.d,
            &mut self.m,
            v,
            d,
            m,
            multivisit,
        );
    }

    /// lc0 classic `MakeNotTerminal()`：复用子树作新根时从子节点重算统计。
    pub fn make_not_terminal(&mut self) {
        self.terminal_kind = TerminalKind::NonTerminal;
        self.terminal_value = None;
        self.visits = 0;
        if self.children.is_empty() {
            self.wl = 0.0;
            self.d = 0.0;
            self.m = 0.0;
            return;
        }
        self.visits = 1;
        let mut wl_sum = 0.0f32;
        let mut d_sum = 0.0f32;
        for edge in &self.children {
            if edge.visits > 0 {
                self.visits = self.visits.saturating_add(edge.visits);
                wl_sum += -edge.wl * edge.visits as f32;
                d_sum += edge.d * edge.visits as f32;
            }
        }
        if self.visits > 0 {
            self.wl = wl_sum / self.visits as f32;
            self.d = d_sum / self.visits as f32;
        } else {
            self.wl = 0.0;
            self.d = 0.0;
        }
        self.m = 0.0;
    }

    /// lc0 classic `RevertTerminalVisits`（twofold 深度修正用）。
    pub fn revert_terminal_visits(&mut self, wl: f32, d: f32, m: f32, multivisit: u32) {
        let n_new = self.visits.saturating_sub(multivisit);
        if n_new == 0 {
            self.visits = 0;
            self.in_flight = 0;
            self.wl = 0.0;
            self.d = 0.0;
            self.m = 0.0;
            self.terminal_kind = TerminalKind::NonTerminal;
            self.terminal_value = None;
            return;
        }
        self.wl -= multivisit as f32 * wl / self.visits as f32;
        self.d -= multivisit as f32 * d / self.visits as f32;
        self.m -= multivisit as f32 * m / self.visits as f32;
        self.visits = n_new;
        self.terminal_kind = TerminalKind::NonTerminal;
        self.terminal_value = None;
    }
}

/// lc0 classic `Node::FinalizeScoreUpdate`。
pub(crate) fn finalize_wdl_stats(
    visits: &mut u32,
    in_flight: &mut u32,
    wl: &mut f32,
    d: &mut f32,
    m: &mut f32,
    v: f32,
    dv: f32,
    mv: f32,
    multivisit: u32,
) {
    if multivisit == 0 {
        return;
    }
    let n = *visits;
    let denom = (n + multivisit) as f32;
    *wl += multivisit as f32 * (v - *wl) / denom;
    *d += multivisit as f32 * (dv - *d) / denom;
    *m += multivisit as f32 * (mv - *m) / denom;
    *visits = n.saturating_add(multivisit);
    *in_flight = in_flight.saturating_sub(multivisit);
}

/// lc0 classic `Node::CancelScoreUpdate`。
pub(crate) fn cancel_score_update(in_flight: &mut u32, multivisit: u32) {
    *in_flight = in_flight.saturating_sub(multivisit);
}

/// lc0 classic 终局赋值。
pub(crate) fn terminal_wdl(value: f32) -> (f32, f32, f32) {
    if value.abs() < f32::EPSILON {
        (0.0, 1.0, 0.0)
    } else if value > 0.0 {
        (1.0, 0.0, 0.0)
    } else {
        (-1.0, 0.0, 0.0)
    }
}
