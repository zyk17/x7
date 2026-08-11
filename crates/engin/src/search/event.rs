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

/// 拒绝 `position`、`ucinewgame` 或替换 `go` 之后残留的旧 event。
///
/// 一个 event 只属于一次流式搜索。x7 为每次 UCI 搜索分配单调递增 generation，
/// 不让旧 event 更新新的 root。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SearchGeneration(pub u64);

/// root history 加上从 root 到 repository node 的走法。
///
/// 中国象棋重复局面和 rule60 依赖历史，因此 event 不能只保存棋盘 hash。root history
/// 以不可变方式共享；Gather 下行时在本 event 内扩展其 owned 走法路径。
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

    /// 返回 worker 私有的准确 history。首次读取才从共享 root 重放一次，之后由
    /// `push` 维护，供重复规则、RuleJudge 与 NN 编码共用。
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

    pub(crate) fn continuation_child_key(&mut self, mv: Move) -> NodeKey {
        let child = Position::after(&self.position, mv);
        let mut history = self.history().clone();
        history.append(mv);
        NodeKey::continuation(child.board().hash(), history.rule_context_hash())
    }
}

/// Gather、Eval 与 Backprop worker 间传递的工作项。
///
/// event 拥有全部搜索专用数据，不持有 graph/backend 的 `&mut` 引用，因此可安全地通过
/// 有界 worker 队列发送。
#[derive(Debug)]
pub struct NodeEvent {
    pub generation: SearchGeneration,
    pub node_key: NodeKey,
    node_path: Vec<NodeKey>,
    /// `(入边 reservation 索引, ContinuationTree 根 key)`。正常 path 为 `None`。
    continuation: Option<(usize, NodeKey)>,
    pub variation: Variation,
    pub reservations: Vec<EdgeReservation>,
    #[cfg(feature = "benchmark")]
    queued_at: Option<Instant>,
}

impl NodeEvent {
    pub fn root(generation: SearchGeneration, root_history: Arc<PositionHistory>) -> Self {
        Self::at_root(
            generation,
            NodeKey::board(root_history.last().board().hash()),
            root_history,
        )
    }

    pub fn at_root(generation: SearchGeneration, root_key: NodeKey, root_history: Arc<PositionHistory>) -> Self {
        Self {
            generation,
            node_key: root_key,
            node_path: vec![root_key],
            continuation: None,
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

    pub(crate) fn repeats_in_history(&mut self, mv: Move) -> bool {
        let board = Position::after(&self.variation.position, mv).board().hash();
        self.variation
            .history()
            .positions()
            .iter()
            .any(|position| position.board().hash() == board)
    }

    /// 进入 ContinuationTree。`child_key` 必须是本 variation 第一次重复后的局面：
    /// 这条入边不绑定 shared graph，树根和其后代只按完整规则上下文保存。
    pub fn descend_continuation(mut self, child_key: NodeKey, reservation: EdgeReservation) -> Self {
        assert!(self.continuation.is_none());
        self.continuation = Some((self.reservations.len(), child_key));
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

    /// 经 Backprop worker 使用的聚合路径完成 event。edge reservation 保持为每 event
    /// 一份，因为每份恰好拥有一个 in-flight visit。
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
            let continuation = node.continuation;
            let mut reservations: Vec<_> = node.reservations.into_iter().map(Some).collect();
            let mut local_leaf = local_leaf;
            for index in (0..reservations.len()).rev() {
                if continuation.is_some_and(|(entry, _)| entry == index) {
                    continue;
                }
                let reservation = reservations[index].take().expect("one reservation");
                if let Some(value) = local_leaf.take() {
                    reservation.complete_local_leaf(value);
                } else {
                    reservation.complete();
                }
            }
            if let Some((index, root)) = continuation {
                let entry = reservations[index].take().expect("continuation entry reservation");
                if let Some(value) = local_leaf.take() {
                    // ContinuationTree 根本身就是 path terminal（例如重复正好触发
                    // rule60）时，它已被 `discard_leaf_node` 移出 node_path，不能读
                    // 未展开 node 的默认值；直接把本次规则裁决写入 entry edge。
                    entry.complete_local_leaf(value);
                } else {
                    for node_key in node.node_path[index + 1..].iter().rev() {
                        repository.recompute_graph_node(*node_key);
                    }
                    let child = repository.get(root).expect("continuation root");
                    let (wl, draw, plies_left) = child.value_snapshot();
                    entry.complete_local_leaf(ValueDelta::with_plies_left(wl, draw, plies_left));
                }
            }
            // KataGo GraphSearch 的 idempotent 回传：只重算本 variation 上已实际经过的
            // node。共享 child 不向所有 parent 广播，未走到的 parent 以后被访问时再重算。
            for node_key in node.node_path.into_iter().rev() {
                repository.recompute_graph_node(node_key);
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

    use xiangqi_core::{GameState, Move, STARTPOS_FEN};

    use super::{BackpropEvent, NodeEvent, SearchGeneration};
    use crate::search::{NodeKey, NodeRepository};

    #[test]
    fn variation_keeps_root_history_and_owns_its_path() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut root = NodeEvent::root(
            SearchGeneration(7),
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
        assert_eq!(next.generation, SearchGeneration(7));
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
        let mut event = NodeEvent::root(SearchGeneration(8), Arc::clone(&history));
        let mv = event
            .variation
            .position
            .board()
            .parse_move("d9e9")
            .expect("repeat move");
        assert!(event.repeats_in_history(mv));
        let child = event.variation.continuation_child_key(mv);

        let mut event = event.descend_continuation(child, crate::search::EdgeReservation::test_only(mv));
        assert!(event.node_key.is_continuation());
        assert_eq!(event.variation.history().last().repetitions(), 1);
    }

    #[test]
    fn backprop_completes_every_reservation_with_alternating_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            SearchGeneration(1),
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
    fn batch_backprop_merges_node_updates_and_completes_each_edge_visit() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root_history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let first = NodeEvent::root(SearchGeneration(1), Arc::clone(&root_history));
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
        let second = NodeEvent::root(SearchGeneration(1), root_history)
            .descend(child_key, root_node.reserve_edge(0).expect("second edge"));

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
            SearchGeneration(2),
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
    fn path_terminal_at_continuation_root_uses_its_local_rule_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root = NodeEvent::root(
            SearchGeneration(9),
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
        assert!((node.edges()[0].completed_stats().local_leaf.q() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn path_terminal_samples_do_not_make_a_shared_edge_first_writer_wins() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let repository = NodeRepository::default();
        let root_key = NodeEvent::root(SearchGeneration(4), Arc::clone(&history)).node_key;
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
            let event = NodeEvent::root(SearchGeneration(4), Arc::clone(&history))
                .descend(child_key, root.reserve_edge(0).expect("edge"));
            BackpropEvent::complete_batch(
                [BackpropEvent::local_leaf(
                    event.discard_leaf_node(),
                    crate::search::ValueDelta::one(value, 0.0),
                )],
                &repository,
            );
        }

        let stats = root.edges()[0].completed_stats();
        assert_eq!(stats.local_leaf.visits, 2);
        assert!((stats.local_leaf.q() - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn path_terminal_discards_its_unshared_leaf_before_backprop() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let root_history = Arc::new(xiangqi_core::PositionHistory::from_positions(state.positions()));
        let root_event = NodeEvent::root(SearchGeneration(3), root_history);
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
