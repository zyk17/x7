//! stream 流水线 worker 之间传递的 owned event。
//!
//! 术语可参考 LC3 Overview 的 “Workers” 与 Glossary 的 “Variation”；event 必须
//! owned，并携带完整 root history / variation / generation / reservation。
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//! <https://lczero.org/dev/lc0/search/lc3/glossary/>

use std::sync::Arc;
#[cfg(feature = "benchmark")]
use std::time::Instant;

use xiangqi_core::{Move, Position, PositionHistory};

use super::{EdgeReservation, NodeKey, ValueDelta};

/// root history 加上从 root 到 repository node 的走法。
///
/// 中国象棋重复局面和 rule60 依赖历史，因此 event 不能只保存棋盘 hash。Search 在
/// 创建时已将完整 UCI history 裁成规则/NN 所需的最小窗口，并以不可变方式共享；
/// Gather 下行时在本 event 内扩展其 owned 走法路径。
#[derive(Clone, Debug)]
pub struct Variation {
    root_history: Arc<PositionHistory>,
    moves: Vec<Move>,
    /// 首次需要规则或 NN 编码时才重放 root history；之后随 `push` 增量追加。
    /// 同一个 Gather event 不会在每层 Expanded node 重放整条 variation。
    history: Option<PositionHistory>,
    /// Gather 当前所在局面。MCGS 必须从真实棋盘取得 child board key；保留这份轻量
    /// snapshot 避免每下降一层都从 root 重放整条 variation。
    /// 完整 history 仍只在 Eval 的规则裁决与 NN 编码时重建。
    position: Position,
}

impl Variation {
    pub fn root(root_history: Arc<PositionHistory>) -> Self {
        Self {
            position: root_history.last().clone(),
            root_history,
            moves: Vec::new(),
            history: None,
        }
    }

    pub fn push(&mut self, mv: Move) {
        self.position = Position::after(&self.position, mv);
        self.moves.push(mv);
        if let Some(history) = self.history.as_mut() {
            history.append(mv);
        }
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        &self.root_history
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// 返回 worker 私有的准确规则与 NN history。root window 已由 Search 共享；首次
    /// 读取只复制这一份窗口，之后由 `push` 增量维护。
    pub(crate) fn history(&mut self) -> &PositionHistory {
        if self.history.is_none() {
            let mut history = self.root_history.as_ref().clone();
            for &mv in &self.moves {
                history.append(mv);
            }
            self.history = Some(history);
        }
        self.history.as_ref().expect("variation history is initialized")
    }

    /// MCGS child identity 只使用走子后的棋盘，不混入 repetition/rule60/history。
    /// `Position::after`（px0 `position.cc:31-60`）只复制当前 position，不重放整个
    /// variation。
    pub fn child_board_key(&self, mv: Move) -> NodeKey {
        NodeKey::board(Position::after(&self.position, mv).board().hash())
    }

    /// 走后节点的真实 identity。重复上下文仍有效时返回 TreeNode；零化着清空
    /// 上下文后返回 GraphNode，使本次搜索已展开的普通子图可跨回合直接复用。
    pub(crate) fn child_key_for_history(&mut self, mv: Move) -> NodeKey {
        let child = Position::after(&self.position, mv);
        let mut history = self.history().clone();
        history.append(mv);
        debug_assert_eq!(child.board().hash(), history.last().board().hash());
        NodeKey::for_history(&history)
    }
}

/// Gather、Eval 与 Backprop worker 间传递的工作项。
///
/// event 拥有全部搜索专用数据，不持有 graph/backend 的 `&mut` 引用，因此可安全地通过
/// 有界 worker 队列发送。
#[derive(Debug)]
pub struct NodeEvent {
    /// 拒绝 `position` / `ucinewgame` / 替换 `go` 之后残留的旧 event。
    /// 每次 UCI 搜索单调递增，旧 event 不得更新新的 root。
    pub generation: u64,
    pub node_key: NodeKey,
    node_path: Vec<NodeKey>,
    /// 每次 Graph → Tree 的 `(入边 reservation 索引, Tree 根 key)`。一条 variation
    /// 可以在零化后回到 Graph，再进入新的 Tree，因此这里不是单值。
    tree_entries: Vec<(usize, NodeKey)>,
    pub variation: Variation,
    pub reservations: Vec<EdgeReservation>,
    #[cfg(feature = "benchmark")]
    queued_at: Option<Instant>,
}

impl NodeEvent {
    pub fn root(generation: u64, root_history: Arc<PositionHistory>) -> Self {
        Self::at_root(generation, NodeKey::for_history(root_history.as_ref()), root_history)
    }

    pub fn at_root(generation: u64, root_key: NodeKey, root_history: Arc<PositionHistory>) -> Self {
        Self {
            generation,
            node_key: root_key,
            node_path: vec![root_key],
            tree_entries: Vec::new(),
            variation: Variation::root(root_history),
            reservations: Vec::new(),
            #[cfg(feature = "benchmark")]
            queued_at: None,
        }
    }

    pub fn descend(mut self, child_key: NodeKey, reservation: EdgeReservation) -> Self {
        self.variation.push(reservation.mv());
        self.node_key = child_key;
        self.node_path.push(child_key);
        self.reservations.push(reservation);
        self
    }

    /// 若走后会形成棋规认可的第一次重复，返回对应 TreeNode key。
    ///
    /// 不能只在全部 history 中查相同 board hash：相差奇数 ply 的 canonical board
    /// 可能相同，却不是同一行棋方的重复。先按 parity 过滤，再由
    /// `PositionHistory::append` 作为规则真相确认 `repetitions`。
    pub(crate) fn repeated_child_key(&mut self, mv: Move) -> Option<NodeKey> {
        let board = Position::after(&self.variation.position, mv).board().hash();
        let history = self.variation.history();
        let child_parity = history.len() & 1;
        if !history
            .positions()
            .iter()
            .enumerate()
            .any(|(index, position)| index & 1 == child_parity && position.board().hash() == board)
        {
            return None;
        }

        let mut child_history = history.clone();
        child_history.append(mv);
        (child_history.last().repetitions() > 0).then(|| NodeKey::for_history(&child_history))
    }

    /// 进入 ContinuationTree。入口 edge 不绑定 child，因为同一个 Graph edge 在不同
    /// variation 下可通向不同的规则上下文；Tree root 仍按其 key 在 repository 复用。
    pub fn descend_continuation(mut self, child_key: NodeKey, reservation: EdgeReservation) -> Self {
        assert!(child_key.is_continuation(), "continuation entry must target a TreeNode");
        self.tree_entries.push((self.reservations.len(), child_key));
        self.variation.push(reservation.mv());
        self.node_key = child_key;
        self.node_path.push(child_key);
        self.reservations.push(reservation);
        self
    }

    /// 当前 leaf 是路径终局但不是共享 node 时，在回传前丢弃它；实际入边仍保留在
    /// `reservations` 中。这样与环边一样，只重算真正可共享的 parent node。
    pub(crate) fn discard_leaf_node(mut self) -> Self {
        assert_eq!(
            self.node_path.len(),
            self.reservations.len() + 1,
            "normal path must end at one shared leaf"
        );
        self.node_path.pop();
        self
    }

    /// collision、停止或评估失败后释放全部 edge-local in-flight visit。消费 `self`
    /// 让调用方遗漏 reservation 成为显式错误。
    pub fn cancel(self) {
        for reservation in self.reservations.into_iter().rev() {
            reservation.cancel();
        }
    }

    pub fn node_path(&self) -> &[NodeKey] {
        &self.node_path
    }

    /// 标记一次 worker 队列交接，仅编入 benchmark 二进制。
    #[cfg(feature = "benchmark")]
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    #[cfg(feature = "benchmark")]
    pub(crate) fn take_queue_wait(&mut self) -> Option<std::time::Duration> {
        self.queued_at.take().map(|queued_at| queued_at.elapsed())
    }
}

/// 由 Gather/Eval 路由给 Backprop 的结果。
///
/// 参考：LC3 Overview 的 “GatherWorker” 和 “BackpropWorker”：
/// <https://lczero.org/dev/lc0/search/lc3/overview/>。
#[derive(Debug)]
pub struct BackpropEvent {
    node: NodeEvent,
    /// 此 event 的 leaf 只是 variation-local 的结果，没有可重算的 shared leaf node。
    /// 它在完成最后一条 reservation 时成为一个 edge-local 样本，绝不能永久覆盖该
    /// board edge 的其他历史。参见 KataGo `docs/GraphSearch.md` 的 edge-local N。
    local_leaf: Option<ValueDelta>,
    #[cfg(feature = "benchmark")]
    queued_at: Option<Instant>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackpropResult {
    pub completed_playouts: u32,
    /// 已完成叶子深度之和，root 记为深度一。语义参考自 `cum_depth_` 的计数方式
    /// （`search.cc:2157-2167`）。
    pub completed_depth: u64,
    pub max_depth: u64,
}

impl BackpropEvent {
    pub(crate) fn evaluation(node: NodeEvent) -> Self {
        Self {
            node,
            local_leaf: None,
            #[cfg(feature = "benchmark")]
            queued_at: None,
        }
    }

    pub(crate) fn local_leaf(node: NodeEvent, value: ValueDelta) -> Self {
        Self {
            node,
            local_leaf: Some(value),
            #[cfg(feature = "benchmark")]
            queued_at: None,
        }
    }

    /// 经 Backprop worker 完成一批独立 event。
    ///
    /// LC3 Overview, "BackpropWorker":
    /// <https://lczero.org/dev/lc0/search/lc3/overview/>.
    pub(crate) fn complete_batch(
        events: impl IntoIterator<Item = Self>,
        repository: &super::NodeRepository,
    ) -> BackpropResult {
        let mut result = BackpropResult::default();

        for event in events {
            let Self { node, local_leaf, .. } = event;
            let expected_nodes = node.reservations.len() + usize::from(local_leaf.is_none());
            debug_assert_eq!(
                node.node_path.len(),
                expected_nodes,
                "shared-leaf path has one more node; local leaf path has no shared leaf"
            );
            let depth = node.reservations.len() as u64 + 1;
            let tree_entries = node.tree_entries;
            let mut reservations: Vec<_> = node.reservations.into_iter().map(Some).collect();
            let mut local_leaf = local_leaf;
            for index in (0..reservations.len()).rev() {
                if tree_entries.iter().any(|(entry, _)| *entry == index) {
                    continue;
                }
                let reservation = reservations[index].take().expect("one reservation");
                if let Some(value) = local_leaf.take() {
                    reservation.complete_local_leaf(value);
                } else {
                    reservation.complete();
                }
            }
            // 从最内层 Tree 开始采样。若零化后又发生第一次重复，内层 Tree 的结果
            // 先进入其 Graph parent，外层入口随后读到已更新的局部树根。
            for (index, root) in tree_entries.into_iter().rev() {
                let entry = reservations[index].take().expect("continuation entry reservation");
                if let Some(value) = local_leaf.take() {
                    // ContinuationTree 根本身就是 path terminal（例如重复正好触发
                    // rule60）时，它已被 `discard_leaf_node` 移出 node_path，不能读
                    // 未展开 node 的默认值；直接把本次规则裁决写入 entry edge。
                    entry.complete_local_leaf(value);
                } else {
                    for node_key in node.node_path[index + 1..].iter().rev() {
                        repository.recompute_graph_node(*node_key);
                        repository.prove_terminal(*node_key);
                    }
                    let child = repository.get(root).expect("continuation root");
                    let (wl, draw, plies_left) = child.value_snapshot();
                    entry.complete_local_leaf(ValueDelta::with_plies_left(wl, draw, plies_left));
                }
            }
            // KataGo GraphSearch 的 idempotent 回传：只重算本 variation 上已实际经过的
            // node。共享 child 不向所有 parent 广播，未走到的 parent 以后被访问时再重算。
            for (index, node_key) in node.node_path.into_iter().enumerate().rev() {
                repository.recompute_graph_node(node_key);
                // 根保留为普通 node；UCI 通过其已证明 terminal child 输出正确的 mate
                // 距离。其余节点可把已绑定 terminal 沿当前 backprop path 继续证明上去。
                if index != 0 {
                    repository.prove_terminal(node_key);
                }
            }
            result.completed_playouts += 1;
            result.completed_depth += depth;
            result.max_depth = result.max_depth.max(depth);
        }

        result
    }

    pub fn cancel(self) {
        self.node.cancel();
    }

    #[cfg(feature = "benchmark")]
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    /// 返回进入 Backprop 队列后的可选等待时间。
    ///
    /// 参考：LC3 Overview 的 “Stats Collection”。
    #[cfg(feature = "benchmark")]
    pub(crate) fn take_queue_wait(&mut self) -> Option<std::time::Duration> {
        self.queued_at.take().map(|queued_at| queued_at.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, Position, STARTPOS_FEN};

    use super::{BackpropEvent, NodeEvent, Variation};
    use crate::search::{ExpansionState, NodeKey, NodeRepository, ValueDelta};

    #[test]
    fn variation_keeps_root_history_and_owns_its_path() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut root = NodeEvent::root(
            7,
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        // 先物化 cached history，再下行；`push` 必须同步追加而不是重新回放。
        assert_eq!(root.variation.history().len(), state.positions().len());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        let child_key = root.variation.child_board_key(mv);
        let root_history_len = root.variation.root_history().len();
        let mut next = root.descend(child_key, crate::search::EdgeReservation::test_only(mv));

        assert_eq!(next.variation.moves(), &[mv]);
        assert_eq!(next.variation.root_history().len(), root_history_len);
        assert_eq!(next.generation, 7);
        let mut expected = xiangqi_core::PositionHistory::from_positions(state.positions());
        expected.append(mv);
        assert_eq!(next.variation.history().last().hash(), expected.last().hash());
    }

    #[test]
    fn first_path_repetition_enters_a_contextual_continuation_root() {
        let (board, _) = xiangqi_core::ChessBoard::from_fen("3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = xiangqi_core::PositionHistory::default();
        history.reset(board, 2, 30);
        for text in ["d9e9", "d2e2", "e9d9", "e2d2"] {
            let mv = history.last().board().parse_move(text).expect(text);
            history.append(mv);
        }
        let history = Arc::new(history);
        let mut event = NodeEvent::root(8, Arc::clone(&history));
        let mv = event
            .variation
            .position
            .board()
            .parse_move("d9e9")
            .expect("repeat move");
        let child = event.repeated_child_key(mv).expect("first repetition");

        let mut event = event.descend_continuation(child, crate::search::EdgeReservation::test_only(mv));
        assert!(event.node_key.is_continuation());
        assert_eq!(event.variation.history().last().repetitions(), 1);
    }

    #[test]
    fn opposite_ply_board_match_does_not_enter_the_continuation_tree() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let parent = state.current_position();
        let mv = parent.board().parse_move("b2b3").expect("legal move");
        let child = Position::after(&parent, mv);
        // `PositionHistory` 的奇偶位置代表不同的行棋方。这里故意放入一个只在
        // 相反 parity 出现的相同 board，模拟旧的全量 hash 扫描误判。
        let history = xiangqi_core::PositionHistory::from_positions(vec![child, parent.clone(), parent]);
        let mut event = NodeEvent::root(3, Arc::new(history));
        let child_hash = Position::after(&event.variation.position, mv).board().hash();

        assert!(
            event
                .variation
                .history()
                .positions()
                .iter()
                .any(|position| position.board().hash() == child_hash)
        );
        assert!(event.repeated_child_key(mv).is_none());
    }

    #[test]
    fn zeroing_move_leaves_the_continuation_tree() {
        let (board, _) = xiangqi_core::ChessBoard::from_fen("3k5/9/9/r3R4/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = xiangqi_core::PositionHistory::default();
        history.reset(board, 2, 30);
        for text in ["d9e9", "d2e2", "e9d9", "e2d2"] {
            let mv = history.last().board().parse_move(text).expect(text);
            history.append(mv);
        }
        let mut variation = Variation::root(Arc::new(history));

        let repeat = variation.position.board().parse_move("d9e9").expect("repeat move");
        assert!(variation.child_key_for_history(repeat).is_continuation());
        variation.push(repeat);
        let reply = variation.position.board().parse_move("d2e2").expect("reply");
        variation.push(reply);
        let zeroing = variation
            .position
            .board()
            .generate_legal_moves()
            .into_iter()
            .find(|mv| xiangqi_core::Position::after(&variation.position, *mv).rule60_ply() == 0)
            .expect("legal capture");

        assert!(matches!(
            variation.child_key_for_history(zeroing),
            NodeKey::GraphNode { .. }
        ));
    }

    #[test]
    fn real_history_can_enter_tree_then_zero_then_enter_tree_again() {
        // 黑将、红车的四步往返先形成第一次重复。黑车吃掉 e6 红车后，重复
        // 上下文清零而回到 Graph；再用同样的四步往返，必须重新进入 Tree。
        // 这是实战中 Graph → Tree → Graph → Tree 的最小规则形状，不依赖 NN。
        let (board, _) = xiangqi_core::ChessBoard::from_fen("3k5/9/9/r3R4/9/9/9/3R5/9/5K3 b - - 2 30").expect("fen");
        let mut history = xiangqi_core::PositionHistory::default();
        history.reset(board, 2, 30);
        for text in ["d9e9", "d2e2", "e9d9", "e2d2"] {
            let mv = history.last().board().parse_move(text).expect(text);
            history.append(mv);
        }
        assert!(history.did_repeat_since_last_zeroing_move());
        assert!(NodeKey::for_history(&history).is_continuation());

        let capture = history.last().board().parse_move("a6e6").expect("capture");
        history.append(capture);
        assert!(!history.did_repeat_since_last_zeroing_move());
        assert!(matches!(NodeKey::for_history(&history), NodeKey::GraphNode { .. }));

        for text in ["d2e2", "d9e9", "e2d2", "e9d9"] {
            let mv = history.last().board().parse_move(text).expect(text);
            history.append(mv);
        }
        assert_eq!(history.last().repetitions(), 1);
        assert!(NodeKey::for_history(&history).is_continuation());
    }

    #[test]
    fn backprop_completes_every_reservation_with_alternating_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            1,
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let repository = NodeRepository::default();
        let root_node = repository.get_or_insert(root.node_key);
        assert!(root_node.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        root_node.publish_edges(vec![(mv, 1.0)]);
        root_node.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        let child_key = NodeKey::board(42);
        root_node.edges()[0].bind_child_key(child_key);
        let child_node = repository.get_or_insert(child_key);
        child_node.set_graph_value(crate::search::ValueDelta::with_plies_left(0.4, 0.2, 2.0));
        let child = root.descend(child_key, root_node.reserve_edge(0).expect("edge"));

        BackpropEvent::complete_batch([BackpropEvent::evaluation(child)], &repository);

        let edge = &root_node.edges()[0];
        assert_eq!(edge.visits(), 1);
        assert_eq!(edge.completed_visits(), 1);
        // child Q 是走子方视角；parent 的 shared Q 幂等重算后取反。
        assert_eq!(root_node.completed_visits(), 2);
        assert!((root_node.q() + 0.2).abs() < f32::EPSILON);
        assert!((root_node.m() - 1.5).abs() < f32::EPSILON);
        let child_node = repository.get(child_key).expect("child node");
        assert_eq!(child_node.completed_visits(), 1);
        assert!((child_node.q() - 0.4).abs() < f32::EPSILON);
        assert!((child_node.m() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn backprop_proves_a_terminal_grandchild_immediately() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let root = NodeEvent::root(18, history);
        let root_node = repository.get_or_insert(root.node_key);
        let first = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        let reply = Move::new(
            xiangqi_core::Square::parse("b7").expect("b7"),
            xiangqi_core::Square::parse("b6").expect("b6"),
        );
        let parent_key = NodeKey::board(18_001);
        let terminal_key = NodeKey::board(18_002);
        assert!(root_node.try_begin_evaluation());
        root_node.set_graph_value(ValueDelta::one(0.0, 0.0));
        root_node.publish_edges(vec![(first, 1.0)]);
        root_node.edges()[0].bind_child_key(parent_key);

        let parent = repository.get_or_insert(parent_key);
        assert!(parent.try_begin_evaluation());
        parent.set_graph_value(ValueDelta::one(0.0, 0.0));
        parent.publish_edges(vec![(reply, 1.0)]);
        parent.edges()[0].bind_child_key(terminal_key);

        let terminal = repository.get_or_insert(terminal_key);
        assert!(terminal.try_begin_evaluation());
        terminal.mark_terminal(1.0, 0.0, 0.0);

        let event = root
            .descend(parent_key, root_node.reserve_edge(0).expect("root edge"))
            .descend(terminal_key, parent.reserve_edge(0).expect("parent edge"));
        BackpropEvent::complete_batch([BackpropEvent::evaluation(event)], &repository);

        assert_eq!(parent.expansion_state(), ExpansionState::Terminal);
        assert_eq!(parent.terminal_value(), Some((-1.0, 0.0, 1.0)));
        assert_eq!(root_node.edges()[0].completed_visits(), 1);
    }

    #[test]
    fn continuation_entry_samples_the_terminal_proof_from_this_event() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let root = NodeEvent::root(19, history);
        let root_node = repository.get_or_insert(root.node_key);
        let first = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        let reply = Move::new(
            xiangqi_core::Square::parse("b7").expect("b7"),
            xiangqi_core::Square::parse("b6").expect("b6"),
        );
        let continuation_key = NodeKey::continuation(19_001, 19_002);
        let terminal_key = NodeKey::continuation(19_003, 19_004);
        assert!(root_node.try_begin_evaluation());
        root_node.set_graph_value(ValueDelta::one(0.0, 0.0));
        root_node.publish_edges(vec![(first, 1.0)]);

        let continuation = repository.get_or_insert(continuation_key);
        assert!(continuation.try_begin_evaluation());
        continuation.set_graph_value(ValueDelta::one(0.0, 0.0));
        continuation.publish_edges(vec![(reply, 1.0)]);
        continuation.edges()[0].bind_child_key(terminal_key);

        let terminal = repository.get_or_insert(terminal_key);
        assert!(terminal.try_begin_evaluation());
        terminal.mark_terminal(1.0, 0.0, 0.0);

        let event = root
            .descend_continuation(continuation_key, root_node.reserve_edge(0).expect("entry edge"))
            .descend(terminal_key, continuation.reserve_edge(0).expect("tree edge"));
        BackpropEvent::complete_batch([BackpropEvent::evaluation(event)], &repository);

        assert_eq!(continuation.terminal_value(), Some((-1.0, 0.0, 1.0)));
        let entry = root_node.edges()[0].stats();
        assert_eq!(entry.local_leaf.visits, 1);
        assert!((entry.local_leaf.wl_sum + 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn batch_backprop_merges_node_updates_and_completes_each_edge_visit() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root_history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let first = NodeEvent::root(1, Arc::clone(&root_history));
        let root_node = repository.get_or_insert(first.node_key);
        assert!(root_node.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        root_node.publish_edges(vec![(mv, 1.0)]);
        root_node.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        let child_key = NodeKey::board(42);
        root_node.edges()[0].bind_child_key(child_key);
        repository
            .get_or_insert(child_key)
            .set_graph_value(crate::search::ValueDelta::one(0.3, 0.3));
        let first = first.descend(child_key, root_node.reserve_edge(0).expect("first edge"));
        let second =
            NodeEvent::root(1, root_history).descend(child_key, root_node.reserve_edge(0).expect("second edge"));

        let result = BackpropEvent::complete_batch(
            [BackpropEvent::evaluation(first), BackpropEvent::evaluation(second)],
            &repository,
        );

        let edge = &root_node.edges()[0];
        let child_node = repository.get(child_key).expect("child node");
        assert_eq!(result.completed_playouts, 2);
        assert_eq!(result.completed_depth, 4);
        assert_eq!(result.max_depth, 2);
        assert_eq!(edge.visits(), 2);
        assert_eq!(edge.completed_visits(), 2);
        assert_eq!(root_node.completed_visits(), 3);
        assert!((root_node.q() + 0.2).abs() < f32::EPSILON);
        assert_eq!(child_node.completed_visits(), 1);
        assert!((child_node.q() - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn continuation_tree_backprop_keeps_its_value_local_to_the_entry_edge() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            2,
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let repository = NodeRepository::default();
        let node = repository.get_or_insert(root.node_key);
        assert!(node.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        node.publish_edges(vec![(mv, 1.0)]);
        node.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        let continuation_key = NodeKey::continuation(42, 99);
        let continuation = repository.get_or_insert(continuation_key);
        continuation.set_graph_value(crate::search::ValueDelta::one(0.6, 0.0));
        let event = root.descend_continuation(continuation_key, node.reserve_edge(0).expect("entry edge"));
        BackpropEvent::complete_batch([BackpropEvent::evaluation(event)], &repository);

        let edge = node.edges()[0].clone();
        assert_eq!(edge.completed_visits(), 1);
        assert!(edge.child_key().is_none());
        assert_eq!(node.completed_visits(), 2);
        assert!((node.q() + 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn continuation_tree_updates_inside_then_samples_its_root_at_the_shard_entry() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            12,
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let repository = NodeRepository::default();
        let shard = repository.get_or_insert(root.node_key);
        assert!(shard.try_begin_evaluation());
        let entry_move = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        shard.publish_edges(vec![(entry_move, 1.0)]);
        shard.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));

        let continuation_key = NodeKey::continuation(42, 99);
        let continuation = repository.get_or_insert(continuation_key);
        assert!(continuation.try_begin_evaluation());
        let inside_move = Move::new(
            xiangqi_core::Square::parse("c2").expect("c2"),
            xiangqi_core::Square::parse("c3").expect("c3"),
        );
        continuation.publish_edges(vec![(inside_move, 1.0)]);
        continuation.set_graph_value(crate::search::ValueDelta::with_plies_left(0.4, 0.0, 2.0));

        let leaf_key = NodeKey::continuation(43, 100);
        let leaf = repository.get_or_insert(leaf_key);
        leaf.set_graph_value(crate::search::ValueDelta::with_plies_left(0.8, 0.0, 3.0));
        continuation.edges()[0].bind_child_key(leaf_key);

        let event = root
            .descend_continuation(continuation_key, shard.reserve_edge(0).expect("entry edge"))
            .descend(leaf_key, continuation.reserve_edge(0).expect("inside edge"));
        BackpropEvent::complete_batch([BackpropEvent::evaluation(event)], &repository);

        // 树内：leaf 的 +0.8 先按换手传给 continuation；base +0.4 与该一次
        // child visit 平均后为 -0.2。进入树的普通 Shard edge 再只采样这个
        // contextual root 的当前值，不绑定它为 shared child。
        assert_eq!(leaf.completed_visits(), 1);
        assert!((leaf.q() - 0.8).abs() < f32::EPSILON);
        assert_eq!(continuation.edges()[0].completed_visits(), 1);
        assert_eq!(continuation.completed_visits(), 2);
        assert!((continuation.q() + 0.2).abs() < f32::EPSILON);
        assert_eq!(shard.edges()[0].completed_visits(), 1);
        assert!(shard.edges()[0].child_key().is_none());
        assert!((shard.edges()[0].stats().local_leaf.q() + 0.2).abs() < f32::EPSILON);
        assert_eq!(shard.completed_visits(), 2);
        assert!((shard.q() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn backprop_handles_a_second_tree_after_a_zeroing_graph_segment() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            13,
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let repository = NodeRepository::default();
        let root_node = repository.get_or_insert(root.node_key);
        let first_tree_key = NodeKey::continuation(42, 1);
        let graph_key = NodeKey::board(43);
        let second_tree_key = NodeKey::continuation(44, 2);
        let first_tree = repository.get_or_insert(first_tree_key);
        let graph = repository.get_or_insert(graph_key);
        let second_tree = repository.get_or_insert(second_tree_key);
        let first = Move::new(
            xiangqi_core::Square::parse("b2").expect("from"),
            xiangqi_core::Square::parse("b3").expect("to"),
        );
        let second = Move::new(
            xiangqi_core::Square::parse("b9").expect("from"),
            xiangqi_core::Square::parse("b8").expect("to"),
        );
        let third = Move::new(
            xiangqi_core::Square::parse("c3").expect("from"),
            xiangqi_core::Square::parse("c4").expect("to"),
        );
        for (node, mv, value) in [
            (root_node.as_ref(), first, 0.0),
            (first_tree.as_ref(), second, 0.2),
            (graph.as_ref(), third, 0.4),
        ] {
            assert!(node.try_begin_evaluation());
            node.publish_edges(vec![(mv, 1.0)]);
            node.set_graph_value(ValueDelta::one(value, 0.0));
        }
        second_tree.set_graph_value(ValueDelta::one(0.6, 0.0));
        first_tree.edges()[0].bind_child_key(graph_key);

        let event = root
            .descend_continuation(first_tree_key, root_node.reserve_edge(0).expect("first entry"))
            .descend(graph_key, first_tree.reserve_edge(0).expect("zeroing edge"))
            .descend_continuation(second_tree_key, graph.reserve_edge(0).expect("second entry"));
        BackpropEvent::complete_batch([BackpropEvent::evaluation(event)], &repository);

        assert_eq!(root_node.edges()[0].completed_visits(), 1);
        assert_eq!(first_tree.edges()[0].completed_visits(), 1);
        assert_eq!(graph.edges()[0].completed_visits(), 1);
    }

    #[test]
    fn path_terminal_at_continuation_root_uses_its_local_rule_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            9,
            Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions())),
        );
        let repository = NodeRepository::default();
        let node = repository.get_or_insert(root.node_key);
        assert!(node.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        node.publish_edges(vec![(mv, 1.0)]);
        node.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        let continuation = NodeKey::continuation(42, 100);
        repository.get_or_insert(continuation);
        let event = root
            .descend_continuation(continuation, node.reserve_edge(0).expect("entry edge"))
            .discard_leaf_node();

        BackpropEvent::complete_batch(
            [BackpropEvent::local_leaf(
                event,
                crate::search::ValueDelta::one(0.8, 0.0),
            )],
            &repository,
        );

        assert_eq!(node.edges()[0].completed_visits(), 1);
        assert!((node.edges()[0].stats().local_leaf.q() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn path_terminal_samples_do_not_make_a_shared_edge_first_writer_wins() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let root_key = NodeEvent::root(4, Arc::clone(&history)).node_key;
        let root = repository.get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        root.publish_edges(vec![(mv, 1.0)]);
        root.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        let child_key = NodeKey::board(42);
        root.edges()[0].bind_child_key(child_key);

        for value in [0.6, -0.2] {
            let event =
                NodeEvent::root(4, Arc::clone(&history)).descend(child_key, root.reserve_edge(0).expect("edge"));
            BackpropEvent::complete_batch(
                [BackpropEvent::local_leaf(
                    event.discard_leaf_node(),
                    crate::search::ValueDelta::one(value, 0.0),
                )],
                &repository,
            );
        }

        let stats = root.edges()[0].stats();
        assert_eq!(stats.local_leaf.visits, 2);
        assert!((stats.local_leaf.q() - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn path_terminal_discards_its_unshared_leaf_before_backprop() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root_history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let root_event = NodeEvent::root(3, root_history);
        let repository = NodeRepository::default();
        let root = repository.get_or_insert(root_event.node_key);
        assert!(root.try_begin_evaluation());
        let mv = Move::new(
            xiangqi_core::Square::parse("b2").expect("b2"),
            xiangqi_core::Square::parse("b3").expect("b3"),
        );
        root.publish_edges(vec![(mv, 1.0)]);
        root.set_graph_value(crate::search::ValueDelta::one(0.0, 0.0));
        let child_key = root_event.variation.child_board_key(mv);
        root.edges()[0].bind_child_key(child_key);
        let child = repository.get_or_insert(child_key);
        assert!(child.try_begin_evaluation());

        let event = root_event.descend(child_key, root.reserve_edge(0).expect("root edge"));
        child.abort_evaluation();
        BackpropEvent::complete_batch(
            [BackpropEvent::local_leaf(
                event.discard_leaf_node(),
                crate::search::ValueDelta::one(0.0, 1.0),
            )],
            &repository,
        );

        assert_eq!(child.expansion_state(), crate::search::ExpansionState::Unexpanded);
        assert_eq!(root.edges()[0].completed_visits(), 1);
        assert_eq!(root.completed_visits(), 2);
    }
}
