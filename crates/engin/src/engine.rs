//! px0 `src/engine.cc:137-220` 的 P3 引擎接线。

use xiangqi_core::{GameState, STARTPOS_FEN};

use crate::neural::backend::{Backend, UniformBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::classic::ClassicSearch;
use crate::search::SearchBase;
use crate::uci_loop::UciOptions;
use crate::uci_loop::{EngineController, GoParams, StringUciResponder};
use crate::EnginError;

/// px0 `Engine` 的 P3 子集：搜索 + UCI controller。
pub struct ClassicEngine {
    search: ClassicSearch,
    position: Option<GameState>,
    unavailable_reason: Option<String>,
    uci_weights_file: Option<String>,
    loaded_weights_file: Option<String>,
}

impl ClassicEngine {
    pub fn with_backend(backend: Box<dyn Backend>) -> Self {
        Self {
            search: ClassicSearch::new(backend),
            position: None,
            unavailable_reason: None,
            uci_weights_file: None,
            loaded_weights_file: None,
        }
    }

    /// px0 `Engine::UpdateBackendConfig` (`src/engine.cc:153-167`) creates a
    /// real backend before search. P4 has no UCI weights configuration yet;
    /// callers can still create the validated ONNX backend directly.
    pub fn from_onnx_file(path: impl AsRef<std::path::Path>) -> Result<Self, EnginError> {
        Ok(Self::with_backend(Box::new(OnnxBackend::from_file(path)?)))
    }

    /// px0 configures a real backend before search (`src/engine.cc:153-167`).
    /// Main UCI must not silently search with the UniformBackend test stub.
    pub fn unavailable() -> Self {
        Self {
            search: ClassicSearch::new(Box::new(UniformBackend::default())),
            position: None,
            unavailable_reason: Some("WeightsFile is not configured".into()),
            uci_weights_file: None,
            loaded_weights_file: None,
        }
    }

    pub fn uniform() -> Self {
        Self::with_backend(Box::new(UniformBackend::default()))
    }

    pub fn search(&self) -> &ClassicSearch {
        &self.search
    }

    /// px0 `Engine::UpdateBackendConfig` (`src/engine.cc:153-167`) restricted
    /// to this port's single formal ONNX backend. px0's backend registry and
    /// protobuf weight discovery are intentionally not reproduced here.
    fn update_backend_config(&mut self) -> Result<(), EnginError> {
        let Some(path) = self.uci_weights_file.as_deref() else {
            self.loaded_weights_file = None;
            self.unavailable_reason = Some("WeightsFile is not configured".into());
            return Ok(());
        };
        if path.is_empty() {
            self.loaded_weights_file = None;
            self.unavailable_reason = Some("WeightsFile is not configured".into());
            return Ok(());
        }
        if self.loaded_weights_file.as_deref() == Some(path) {
            return Ok(());
        }
        self.search.abort_search()?;
        match OnnxBackend::from_file(path) {
            Ok(backend) => {
                self.search.set_backend(Box::new(backend))?;
                self.loaded_weights_file = Some(path.to_string());
                self.unavailable_reason = None;
                Ok(())
            }
            Err(error) => {
                self.loaded_weights_file = None;
                self.unavailable_reason = Some(format!("cannot load WeightsFile {path}: {error}"));
                Ok(())
            }
        }
    }
}

impl EngineController for ClassicEngine {
    fn register_uci_responder(&mut self, _responder: &mut dyn StringUciResponder) {}

    fn unregister_uci_responder(&mut self, _responder: &mut dyn StringUciResponder) {}

    fn set_uci_options(&mut self, options: &UciOptions) -> Result<(), EnginError> {
        self.uci_weights_file = Some(options.weights_file.clone());
        Ok(())
    }

    fn ensure_ready(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn new_game(&mut self) -> Result<(), EnginError> {
        self.search.new_game()?;
        self.set_position(STARTPOS_FEN, &[])
    }

    fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        // px0 `Engine::SetPosition` first calls `EnsureSearchStopped()`
        // (`src/engine.cc:187-197`).  The old worker owns mutable tree state;
        // replacing it before all worker threads joined is a UCI race.
        self.search.abort_search()?;
        // px0 `Engine::SetPosition` updates backend configuration after
        // stopping search and before making the new `GameState`
        // (`src/engine.cc:187-197`). Retrying here also handles a file that
        // appeared after an earlier failed `setoption`.
        if self.uci_weights_file.is_some() {
            self.update_backend_config()?;
        }
        let state = GameState::from_fen_moves(fen, moves)?;
        self.search.set_position(&state)?;
        self.position = Some(state);
        Ok(())
    }

    fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        if let Some(reason) = &self.unavailable_reason {
            responder.send_raw_response(&format!("info string cannot search: {reason}"));
            return Ok(());
        }
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
