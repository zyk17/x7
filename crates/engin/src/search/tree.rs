//! 按 px0 主线保留规则复用 stream tree。
//!
//! LC3 Overview 描述 repository，但未描述 tree reuse。此 tree-only 策略对照 px0
//! `NodeTree::MakeMove` / `ResetToPosition`（`src/search/classic/node.cc:465-520`）：
//! 保留已走主线，释放 sibling subtree；不是 DAG/TT 策略。

use std::sync::Arc;

use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;

use super::{NodeKey, NodeRepository};

/// 已走着替换当前 root 时回收的 node 数。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcStats {
    pub removed_nodes: usize,
}

/// 两次已完成 stream 搜索之间保留的 tree 状态。
#[derive(Debug)]
pub struct Tree {
    repository: Arc<NodeRepository>,
    /// Retained played line, from its oldest root to the current root.
    root_keys: Vec<NodeKey>,
    /// Full histories aligned with `root_keys`; their final entry is the
    /// current root history. Snapshots make UCI rewind/reuse exact without
    /// reconstructing positions from hashes.
    root_histories: Vec<Arc<PositionHistory>>,
}

impl Tree {
    pub fn new(root_history: Arc<PositionHistory>) -> Self {
        let root = NodeKey::root(root_history.last().hash());
        Self {
            repository: Arc::new(NodeRepository::default()),
            root_keys: vec![root],
            root_histories: vec![root_history],
        }
    }

    pub fn repository(&self) -> &Arc<NodeRepository> {
        &self.repository
    }

    pub fn root_key(&self) -> NodeKey {
        *self.root_keys.last().expect("stream tree always has a root")
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        self.root_histories
            .last()
            .expect("stream tree always has a root history")
    }

    /// Advances to a legal child after all events below the current root have
    /// completed or cancelled. The old root stays on the retained played line;
    /// only its sibling subtrees are reclaimed.
    pub fn advance(&mut self, mv: Move) -> Result<GcStats, EnginError> {
        let old_root = self.root_key();
        if !self.repository.subtree_is_settled(old_root) {
            return Err(EnginError::PortIncomplete(
                "stream tree advance requires settled reservations",
            ));
        }
        if !self.root_history().last().board().is_legal_move(mv) {
            return Err(EnginError::PortIncomplete("stream tree advance requires a legal move"));
        }

        let mut stats = GcStats::default();
        if let Some(root) = self.repository.get(old_root) {
            for edge in root.edges().iter() {
                if edge.mv() != mv {
                    stats.removed_nodes += self.repository.remove_subtree(old_root.child(edge.mv()));
                }
            }
        }

        let new_root = old_root.child(mv);
        self.repository.get_or_insert(new_root);
        let mut history = self.root_history().as_ref().clone();
        history.append(mv);
        self.root_keys.push(new_root);
        self.root_histories.push(Arc::new(history));
        Ok(stats)
    }

    /// Returns to the immediately previous retained root. It does not reclaim
    /// the future child; a later different `advance` will prune it as a sibling.
    pub fn rewind_one(&mut self) -> Result<bool, EnginError> {
        if self.root_keys.len() == 1 {
            return Ok(false);
        }
        if !self.repository.subtree_is_settled(self.root_key()) {
            return Err(EnginError::PortIncomplete(
                "stream tree rewind requires settled reservations",
            ));
        }
        self.root_keys.pop();
        self.root_histories.pop();
        Ok(true)
    }

    /// Repositions this reusable tree for a complete UCI position history.
    ///
    /// A retained ancestor is restored directly; a continuation is advanced
    /// move-by-move so each played edge prunes its siblings. An unrelated
    /// history starts a fresh repository. This is the tree-only counterpart of
    /// px0 `NodeTree::ResetToPosition` (`src/search/classic/node.cc:484-520`).
    pub fn reset_to_history(&mut self, target: Arc<PositionHistory>) -> Result<GcStats, EnginError> {
        if !self.repository.subtree_is_settled(self.root_key()) {
            return Err(EnginError::PortIncomplete(
                "stream tree reset requires settled reservations",
            ));
        }

        if let Some(index) = self.root_histories.iter().position(|history| history == &target) {
            self.root_keys.truncate(index + 1);
            self.root_histories.truncate(index + 1);
            return Ok(GcStats::default());
        }

        if target.len() > self.root_history().len()
            && target.positions()[..self.root_history().len()] == *self.root_history().positions()
        {
            let mut stats = GcStats::default();
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
                    .ok_or(EnginError::PortIncomplete(
                        "stream tree reset could not derive legal move",
                    ))?;
                let advanced = self.advance(mv)?;
                stats.removed_nodes += advanced.removed_nodes;
            }
            return Ok(stats);
        }

        let root = NodeKey::root(target.last().hash());
        self.repository = Arc::new(NodeRepository::default());
        self.root_keys = vec![root];
        self.root_histories = vec![target];
        Ok(GcStats::default())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, Square, STARTPOS_FEN};

    use super::{NodeKey, Tree};

    fn mv(from: &str, to: &str) -> Move {
        Move::new(Square::parse(from).expect("from"), Square::parse(to).expect("to"))
    }

    fn tree() -> Tree {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        Tree::new(Arc::new(PositionHistory::from_positions(state.positions())))
    }

    #[test]
    fn advance_keeps_old_root_and_prunes_sibling_subtree() {
        let mut tree = tree();
        let old_root = tree.root_key();
        let keep = mv("a0", "a1");
        let drop = mv("a0", "a2");
        let root = tree.repository().get_or_insert(old_root);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(keep, 0.5), (drop, 0.5)]);
        let kept_child = old_root.child(keep);
        let dropped_child = old_root.child(drop);
        let dropped_grandchild = dropped_child.child(mv("a9", "a8"));
        tree.repository().get_or_insert(kept_child);
        let dropped = tree.repository().get_or_insert(dropped_child);
        assert!(dropped.try_begin_evaluation());
        dropped.publish_edges(vec![(mv("a9", "a8"), 1.0)]);
        tree.repository().get_or_insert(dropped_grandchild);
        assert_eq!(tree.repository().len(), 4);

        let stats = tree.advance(keep).expect("advance");
        assert_eq!(stats.removed_nodes, 2);
        assert_eq!(tree.repository().len(), 2);
        assert_eq!(tree.root_key(), kept_child);
        assert_eq!(tree.root_history().len(), 2);
        assert!(tree.repository().get(old_root).is_some());
        assert!(tree.repository().get(kept_child).is_some());
        assert!(tree.repository().get(dropped_child).is_none());
        assert!(tree.repository().get(dropped_grandchild).is_none());
    }

    #[test]
    fn rewind_keeps_played_child_for_future_reuse() {
        let mut tree = tree();
        let old_root = tree.root_key();
        let played = mv("a0", "a1");
        tree.advance(played).expect("advance");
        let played_root = tree.root_key();
        assert!(tree.rewind_one().expect("rewind"));
        assert_eq!(tree.root_key(), old_root);
        assert_eq!(tree.root_history().len(), 1);
        assert!(tree.repository().get(played_root).is_some());
        assert!(!tree.rewind_one().expect("root cannot rewind"));
    }

    #[test]
    fn advance_rejects_an_in_flight_reservation() {
        let mut tree = tree();
        let root_key = tree.root_key();
        let played = mv("a0", "a1");
        let root = tree.repository().get_or_insert(root_key);
        assert!(root.try_begin_evaluation());
        root.publish_edges(vec![(played, 1.0)]);
        let reservation = root.reserve_edge(0).expect("reservation");
        assert!(tree.advance(played).is_err());
        reservation.cancel();
        assert!(tree.advance(played).is_ok());
    }

    #[test]
    fn reset_to_history_reuses_retained_ancestor_and_continuation() {
        let mut tree = tree();
        let game = GameState::from_fen_moves(STARTPOS_FEN, &["a0a1", "a9a8"]).expect("legal line");
        let first = game.moves[0];
        let second = game.moves[1];
        tree.advance(first).expect("first advance");
        let first_history = tree.root_history().clone();
        tree.advance(second).expect("second advance");

        tree.reset_to_history(first_history.clone())
            .expect("rewind through reset");
        assert_eq!(tree.root_history(), &first_history);
        assert!(tree.repository().get(tree.root_key().child(second)).is_some());

        let target = Arc::new(PositionHistory::from_positions(game.positions()));
        tree.reset_to_history(target).expect("replay continuation");
        assert_eq!(tree.root_history().len(), 3);
    }

    #[test]
    fn reset_to_unrelated_history_starts_fresh_repository() {
        let mut tree = tree();
        tree.advance(mv("a0", "a1")).expect("advance");
        let unrelated = GameState::from_fen_moves(STARTPOS_FEN, &["b0b1"]).expect("other legal line");
        let target = Arc::new(PositionHistory::from_positions(unrelated.positions()));

        tree.reset_to_history(target.clone()).expect("fresh tree");
        assert_eq!(tree.root_history(), &target);
        assert_eq!(tree.repository().len(), 0);
        assert_eq!(tree.root_key(), NodeKey::root(target.last().hash()));
    }
}
