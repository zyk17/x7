//! Deterministic Gather -> Eval -> Backprop baseline for stream search.
//!
//! Reference: LC3 overview, "Workers":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>
//!
//! This is intentionally single-threaded. It is the semantic baseline for the
//! later bounded queues: every transition already uses an owned `NodeEvent`, so
//! queue workers will not need access to a mutable DFS tree.

use std::sync::Arc;

use xiangqi_core::{GameResult, PositionHistory};

use crate::neural::backend::Backend;
use crate::EnginError;

use super::{
    select_edge, terminal_value_for_side_to_move, BackpropEvent, ExpansionState, NodeEvent, NodeKey, NodeRepository,
    SearchGeneration,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamStats {
    pub completed_playouts: u64,
    pub collisions: u64,
    pub network_batches: u64,
    pub network_evaluations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamOutcome {
    Completed,
    Collision,
}

/// First runnable stream search. It owns a repository for one root and only
/// shares the immutable backend/history through `Arc`.
pub struct StreamSearch {
    backend: Arc<dyn Backend>,
    repository: NodeRepository,
    generation: SearchGeneration,
    root_history: Arc<PositionHistory>,
    root_key: NodeKey,
    cpuct: f32,
    stats: StreamStats,
}

impl StreamSearch {
    pub fn new(
        backend: Arc<dyn Backend>,
        generation: SearchGeneration,
        root_history: Arc<PositionHistory>,
        cpuct: f32,
    ) -> Self {
        assert!(
            cpuct.is_finite() && cpuct >= 0.0,
            "cpuct must be finite and non-negative"
        );
        let root_key = NodeKey::root(root_history.last().hash());
        Self {
            backend,
            repository: NodeRepository::default(),
            generation,
            root_history,
            root_key,
            cpuct,
            stats: StreamStats::default(),
        }
    }

    pub fn root_key(&self) -> NodeKey {
        self.root_key
    }

    pub fn repository(&self) -> &NodeRepository {
        &self.repository
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    pub fn run_playouts(&mut self, count: u64) -> Result<StreamStats, EnginError> {
        for _ in 0..count {
            self.run_playout()?;
        }
        Ok(self.stats)
    }

    pub fn run_playout(&mut self) -> Result<StreamOutcome, EnginError> {
        let mut event = NodeEvent::root(self.generation, Arc::clone(&self.root_history));
        loop {
            let node = self.repository.get_or_insert(event.node_key);
            match node.expansion_state() {
                ExpansionState::Unexpanded => {
                    if !node.try_begin_evaluation() {
                        continue;
                    }
                    return self.evaluate_and_backprop(node.as_ref(), event);
                }
                ExpansionState::Evaluating => {
                    event.cancel();
                    self.stats.collisions += 1;
                    return Ok(StreamOutcome::Collision);
                }
                ExpansionState::Terminal => {
                    let (value, draw) = node.terminal_value().expect("terminal stream value");
                    BackpropEvent {
                        node: event,
                        value,
                        draw,
                    }
                    .complete(&self.repository);
                    self.stats.completed_playouts += 1;
                    return Ok(StreamOutcome::Completed);
                }
                ExpansionState::Expanded => {
                    let edges = node.edges();
                    let edge_index = select_edge(&edges, node.completed_visits(), self.cpuct)
                        .expect("expanded stream node must have an edge");
                    let reservation = node.reserve_edge(edge_index).expect("selected stream edge");
                    let child_key = event.node_key.child(reservation.mv());
                    event = event.descend(child_key, reservation);
                }
            }
        }
    }

    fn evaluate_and_backprop(
        &mut self,
        node: &super::StreamNode,
        event: NodeEvent,
    ) -> Result<StreamOutcome, EnginError> {
        let history = event.variation.replay_history();
        match history.compute_game_result() {
            GameResult::Undecided => {
                let legal_moves = history.last().board().generate_legal_moves();
                let eval = self.backend.evaluate(&history, &legal_moves);
                if eval.policies.len() != legal_moves.len() {
                    event.cancel();
                    node.abort_evaluation();
                    return Err(EnginError::PortIncomplete("stream backend policy length"));
                }
                node.publish_edges(legal_moves.into_iter().zip(eval.policies.iter().copied()).collect());
                BackpropEvent {
                    node: event,
                    value: eval.wl,
                    draw: eval.d,
                }
                .complete(&self.repository);
            }
            result => {
                let (value, draw) = terminal_value_for_side_to_move(result, history.last().is_black_to_move());
                node.mark_terminal(value, draw);
                BackpropEvent {
                    node: event,
                    value,
                    draw,
                }
                .complete(&self.repository);
            }
        }
        self.stats.completed_playouts += 1;
        Ok(StreamOutcome::Completed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use super::{StreamOutcome, StreamSearch};
    use crate::neural::backend::UniformBackend;
    use crate::search::stream::{ExpansionState, SearchGeneration};

    fn startpos_history() -> Arc<PositionHistory> {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        Arc::new(PositionHistory::from_positions(state.positions()))
    }

    #[test]
    fn fixed_playouts_leave_no_in_flight_edge_visits() {
        let backend = Arc::new(UniformBackend::default());
        let mut search = StreamSearch::new(backend, SearchGeneration(1), startpos_history(), 1.0);
        let stats = search.run_playouts(32).expect("playouts");
        assert_eq!(stats.completed_playouts, 32);
        assert_eq!(stats.collisions, 0);

        let root = search.repository().get(search.root_key()).expect("root");
        assert_eq!(root.completed_visits(), 32);
        assert_eq!(root.expansion_state(), ExpansionState::Expanded);
        for edge in root.edges().iter() {
            assert_eq!(edge.visits(), edge.completed_visits());
        }
    }

    #[test]
    fn known_terminal_root_completes_without_creating_edges() {
        let state = GameState::from_fen_moves("4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", &[] as &[&str])
            .expect("checkmate fen");
        let backend = Arc::new(UniformBackend::default());
        let mut search = StreamSearch::new(
            backend,
            SearchGeneration(2),
            Arc::new(PositionHistory::from_positions(state.positions())),
            1.0,
        );
        assert_eq!(
            search.run_playout().expect("terminal playout"),
            StreamOutcome::Completed
        );
        let root = search.repository().get(search.root_key()).expect("root");
        assert_eq!(root.expansion_state(), ExpansionState::Terminal);
        assert!(root.edges().is_empty());
        assert_eq!(root.completed_visits(), 1);
    }
}
