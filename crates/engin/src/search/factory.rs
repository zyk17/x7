//! Search construction boundary corresponding to px0 `SearchFactory`.

use std::sync::Arc;

use crate::neural::backend::Backend;

use super::SearchBase;

/// Produces one independent search implementation for an Engine-owned backend.
///
/// px0 reference: `src/search/search.h` `SearchFactory` and
/// `src/engine.cc:137-167`.
pub trait SearchFactory: Send + Sync {
    fn create(&self, backend: Arc<dyn Backend>) -> Box<dyn SearchBase>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, STARTPOS_FEN};

    use crate::neural::backend::UniformBackend;
    use crate::GoParams;

    use super::SearchFactory;

    #[test]
    fn classic_and_stream_factories_create_independent_searches() {
        let backend: Arc<dyn crate::neural::backend::Backend> = Arc::new(UniformBackend::default());
        let mut classic = crate::search::classic::Factory.create(Arc::clone(&backend));
        let mut stream = crate::search::stream::Factory.create(backend);
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        classic.set_position(&state).expect("classic position");
        stream.set_position(&state).expect("stream position");
        stream
            .start_search(&GoParams {
                nodes: Some(1),
                ..GoParams::default()
            })
            .expect("stream start");
        stream.wait_search().expect("stream wait");
        assert!(stream.best_move().is_some());
    }
}
