//! px0 `src/engine.cc:137-250` 的 UCI Engine lifecycle.

use std::sync::Arc;

use xiangqi_core::{GameState, STARTPOS_FEN};

use crate::callbacks::SearchResponder;
use crate::neural::backend::{Backend, CachingBackend, UniformBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::SearchSession;
use crate::uci_loop::{GoParams, StringUciResponder};
use crate::EnginError;
use crate::Options;

/// px0 `Engine` 的 P3 子集：搜索与 UCI 调度。
pub struct Engine {
    // ONNX initialization can fail while the UCI process is already alive;
    // `None` means no usable stream session. It is deliberately not a
    // UniformBackend fallback.
    search: Option<SearchSession>,
    options: Options,
    position: Option<GameState>,
    manages_weights_file: bool,
    backend_error: Option<String>,
    loaded_weights_file: Option<String>,
    responder: Option<Arc<dyn SearchResponder>>,
}

impl Default for Engine {
    /// Rust adapter for the formal px0 `Engine::Engine` constructor
    /// (`src/engine.cc:137-167`); it adds no separate initialization path.
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// px0 `Engine::Engine` + `Engine::UpdateBackendConfig`
    /// (`src/engine.cc:137-167`), with ONNX initialization deferred until
    /// `SetPosition`. This is the formal UCI constructor.
    pub fn new() -> Self {
        Self {
            search: None,
            options: Options::default(),
            position: None,
            manages_weights_file: true,
            backend_error: None,
            loaded_weights_file: None,
            responder: None,
        }
    }

    pub fn with_backend(backend: Box<dyn Backend>) -> Self {
        let backend: Arc<dyn Backend> = Arc::from(backend);
        let mut search = SearchSession::new(Arc::clone(&backend));
        search.set_responder(None);
        Self {
            search: Some(search),
            options: Options::default(),
            position: None,
            manages_weights_file: false,
            backend_error: None,
            loaded_weights_file: None,
            responder: None,
        }
    }

    /// Explicit backend construction for tests and direct callers. Formal UCI
    /// startup uses `new` and receives its model through `WeightsFile`.
    pub fn from_onnx_file(path: impl AsRef<std::path::Path>) -> Result<Self, EnginError> {
        Ok(Self::with_backend(Box::new(CachingBackend::new(Box::new(
            OnnxBackend::from_file(path)?,
        )))))
    }

    /// Deterministic test-only constructor. It never participates in the
    /// formal UCI `WeightsFile` lifecycle.
    pub fn uniform() -> Self {
        Self::with_backend(Box::new(UniformBackend::default()))
    }

    pub fn has_search(&self) -> bool {
        self.search.is_some()
    }

    /// Installs the structured search output callback for library users.
    /// UCI uses the same API with its queue-backed text adapter.
    pub fn set_search_responder(&mut self, responder: Option<Arc<dyn SearchResponder>>) {
        self.responder = responder;
        if let Some(search) = self.search.as_mut() {
            search.set_responder(self.responder.clone());
        }
    }

    /// px0 `Engine::UpdateBackendConfig` (`src/engine.cc:153-167`), restricted
    /// to this port's single formal ONNX backend. A failed backend result
    /// removes the old search, so a changed/broken `WeightsFile` cannot keep
    /// searching with stale weights.
    fn update_backend_config(&mut self) -> Result<(), EnginError> {
        if !self.manages_weights_file {
            return Ok(());
        }
        // Snapshot the OptionsDict value before stopping/replacing search.
        // px0's `std::string` option value is likewise stable for the whole
        // `UpdateBackendConfig` call (`src/engine.cc:153-167`).
        let path = self.options.weights_file.trim().to_string();
        if path.is_empty() {
            self.stop_and_drop_search()?;
            self.loaded_weights_file = None;
            self.backend_error = Some("WeightsFile is not configured".into());
            return Ok(());
        }
        if self.loaded_weights_file.as_deref() == Some(path.as_str()) && self.search.is_some() {
            return Ok(());
        }

        self.stop_and_drop_search()?;
        match OnnxBackend::from_file(&path) {
            Ok(backend) => {
                let backend: Arc<dyn Backend> = Arc::new(CachingBackend::new(Box::new(backend)));
                let mut search = SearchSession::new(Arc::clone(&backend));
                search.set_responder(self.responder.clone());
                self.search = Some(search);
                self.loaded_weights_file = Some(path.clone());
                self.backend_error = None;
                Ok(())
            }
            Err(error) => {
                self.loaded_weights_file = None;
                self.backend_error = Some(format!("cannot load WeightsFile {path}: {error}"));
                Ok(())
            }
        }
    }

    /// px0 `Engine::EnsureSearchStopped` (`src/engine.cc:149-151`).
    fn stop_and_drop_search(&mut self) -> Result<(), EnginError> {
        if let Some(mut search) = self.search.take() {
            search.abort()?;
        }
        Ok(())
    }
}

impl Engine {
    pub fn options(&self) -> &Options {
        &self.options
    }

    pub fn set_option(&mut self, name: &str, value: &str) -> Result<(), EnginError> {
        self.options.set_uci_option(name, value)
    }

    pub(crate) fn ensure_ready(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    pub(crate) fn new_game(&mut self) -> Result<(), EnginError> {
        if let Some(search) = self.search.as_mut() {
            search.abort()?;
        }
        self.set_position(STARTPOS_FEN, &[])
    }

    pub(crate) fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        // SearchSession stops and joins before replacing its reusable tree.
        // px0 `Engine::SetPosition` updates backend configuration after
        // stopping search and before making the new `GameState`
        // (`src/engine.cc:187-197`). Retrying here also handles a file that
        // appeared after an earlier failed `setoption`.
        self.update_backend_config()?;
        let state = GameState::from_fen_moves(fen, moves)?;
        if let Some(search) = self.search.as_mut() {
            search.set_position(&state)?;
        }
        self.position = Some(state);
        Ok(())
    }

    pub(crate) fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        // px0 rejects `go ponder` unless the Ponder option enabled its full
        // position/ponderhit lifecycle (`src/engine.cc:205-215`). This port
        // has not translated that option or lifecycle, so accepting the token
        // and silently running a normal search would be a false UCI feature.
        if params.ponder {
            return Err(EnginError::Uci("Ponder is not enabled.".into()));
        }
        // px0 `Engine::Go` initializes a missing position through `NewGame`
        // before calling `StartSearch` (`src/engine.cc:206-219`). NewGame in
        // turn runs SetPosition and UpdateBackendConfig, so checking the
        // backend first would incorrectly reject a valid bare `go` command.
        if self.position.is_none() {
            self.new_game()?;
        }
        let Some(search) = self.search.as_mut() else {
            let reason = self.backend_error.as_deref().unwrap_or("WeightsFile is not configured");
            responder.send_raw_response(&format!("info string cannot search: {reason}"));
            return Ok(());
        };
        search.start(params)?;
        Ok(())
    }

    pub(crate) fn ponder_hit(&mut self) -> Result<(), EnginError> {
        // px0 `Engine::PonderHit` rejects this outside an active ponder search
        // (`src/engine.cc:226-235`). Ponder is intentionally unavailable here.
        Err(EnginError::Uci("ponderhit while not pondering".into()))
    }

    pub(crate) fn wait(&mut self) -> Result<(), EnginError> {
        let Some(search) = self.search.as_mut() else {
            return Ok(());
        };
        search.wait()
    }

    pub(crate) fn stop(&mut self) -> Result<(), EnginError> {
        if let Some(search) = self.search.as_mut() {
            search.stop();
        }
        self.wait()
    }
}
