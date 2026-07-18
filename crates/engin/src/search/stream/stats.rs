//! Read-only stream root statistics for diagnostics and future watchdog output.
//!
//! Reference: LC3 overview, "Stats Collection":
//! <https://lczero.org/dev/lc0/search/lc3/overview/>.
//! The LC3 policy documentation still leaves final move choice TBD, so this
//! module intentionally exposes raw root edges without inventing a bestmove or
//! PV selection rule.

use xiangqi_core::Move;

use super::{NodeKey, NodeRepository};

/// Stable statistics for one root edge. `started_visits` includes in-flight
/// work; `completed_visits` and `q` only include completed backpropagation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamRootEdgeStats {
    pub mv: Move,
    pub completed_visits: u32,
    pub started_visits: u32,
    pub q: f32,
    pub prior: f32,
}

/// A consistent one-node repository snapshot. It deliberately does not claim
/// a globally atomic tree view: LC3 avoids locking more than one node at once.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamRootStats {
    pub completed_visits: u32,
    pub q: f32,
    pub draw: f32,
    pub edges: Vec<StreamRootEdgeStats>,
}

/// Reads only the root repository value. Callers may format or sort the copy,
/// but must not derive a formal bestmove until `SearchPolicy` defines that
/// rule.
pub fn root_stats(repository: &NodeRepository, root_key: NodeKey) -> Option<StreamRootStats> {
    let root = repository.get(root_key)?;
    Some(StreamRootStats {
        completed_visits: root.completed_visits(),
        q: root.q(),
        draw: root.draw(),
        edges: root
            .edges()
            .iter()
            .map(|edge| StreamRootEdgeStats {
                mv: edge.mv(),
                completed_visits: edge.completed_visits(),
                started_visits: edge.visits(),
                q: edge.q(),
                prior: edge.prior(),
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use super::root_stats;
    use crate::neural::backend::UniformBackend;
    use crate::search::stream::{SearchGeneration, StreamSearch};

    #[test]
    fn root_snapshot_reports_completed_and_in_flight_visits_separately() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut search = StreamSearch::new(Arc::new(UniformBackend::default()), SearchGeneration(31), history, 1.0);
        search.run_playouts(16).expect("playouts");

        let stats = root_stats(search.repository(), search.root_key()).expect("root snapshot");
        assert_eq!(stats.completed_visits, 16);
        assert!(stats.edges.iter().any(|edge| edge.completed_visits > 0));
        assert!(stats
            .edges
            .iter()
            .all(|edge| edge.started_visits == edge.completed_visits));
    }
}
