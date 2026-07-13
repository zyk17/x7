//! px0 `src/engine.cc:137-220` 的 P3 引擎接线。

use xiangqi_core::{GameState, STARTPOS_FEN};

use crate::search::classic::{backend::Backend, search::ClassicSearch, UniformBackend};
use crate::search::SearchBase;
use crate::uci_loop::{EngineController, GoParams, StringUciResponder};
use crate::EnginError;

/// px0 `Engine` 的 P3 子集：搜索 + UCI controller。
pub struct ClassicEngine {
    search: ClassicSearch,
    position: Option<GameState>,
}

impl ClassicEngine {
    pub fn with_backend(backend: Box<dyn Backend>) -> Self {
        Self {
            search: ClassicSearch::new(backend),
            position: None,
        }
    }

    pub fn uniform() -> Self {
        Self::with_backend(Box::new(UniformBackend::default()))
    }

    pub fn search(&self) -> &ClassicSearch {
        &self.search
    }
}

impl EngineController for ClassicEngine {
    fn register_uci_responder(&mut self, _responder: &mut dyn StringUciResponder) {}

    fn unregister_uci_responder(&mut self, _responder: &mut dyn StringUciResponder) {}

    fn ensure_ready(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn new_game(&mut self) -> Result<(), EnginError> {
        self.search.new_game()?;
        self.set_position(STARTPOS_FEN, &[])
    }

    fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        let state = GameState::from_fen_moves(fen, moves)?;
        self.search.set_position(&state)?;
        self.position = Some(state);
        Ok(())
    }

    fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        if self.position.is_none() {
            self.new_game()?;
        }
        self.search.start_search(params)?;
        for output in std::mem::take(&mut self.search.outputs) {
            responder.output_thinking_info(&[output.info]);
            responder.output_best_move(&output.bestmove);
        }
        Ok(())
    }

    fn ponder_hit(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn wait(&mut self) -> Result<(), EnginError> {
        self.search.wait_search()
    }

    fn stop(&mut self, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        self.search.stop_search()?;
        for output in std::mem::take(&mut self.search.outputs) {
            responder.output_thinking_info(&[output.info]);
            responder.output_best_move(&output.bestmove);
        }
        Ok(())
    }
}
