//! px0 `src/engine.cc:81-250` 的 P3/P4 引擎接线。

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use xiangqi_core::{GameState, STARTPOS_FEN};

use crate::callbacks::{BestMoveInfo, SearchResponder, ThinkingInfo};
use crate::neural::backend::{Backend, UniformBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::classic::ClassicSearch;
use crate::search::SearchBase;
use crate::uci_loop::UciOptions;
use crate::uci_loop::{EngineController, GoParams, StringUciResponder};
use crate::EnginError;

/// px0 `Engine::UciPonderForwarder` (`src/engine.cc:81-136`).
///
/// px0 registers the UCI loop's responder with the engine, then classic
/// search invokes that forwarder from its watchdog thread. Rust cannot retain
/// the responder borrow in an `Arc` because `UciLoop` still owns it, so this
/// keeps the same non-owning registration model. The mutex serializes worker
/// output against UCI-loop output. `ClassicEngine::unregister_uci_responder`
/// stops and joins search before clearing the pointer, which is the lifetime
/// invariant that makes the erased borrow sound.
struct UciResponderForwarder {
    responder: Mutex<Option<NonNull<dyn StringUciResponder>>>,
}

// SAFETY: the only pointer is installed by `register`, is protected by the
// mutex, and is cleared only after all search threads joined in `unregister`.
// This is the Rust expression of px0's UciPonderForwarder ownership contract.
unsafe impl Send for UciResponderForwarder {}
// SAFETY: see the `Send` implementation above; responder calls are mutexed.
unsafe impl Sync for UciResponderForwarder {}

impl UciResponderForwarder {
    fn register(&self, responder: &mut dyn StringUciResponder) {
        let mut slot = self.responder.lock().expect("uci responder lock");
        assert!(slot.is_none(), "px0 UciPonderForwarder already has a responder");
        // SAFETY: unregister joins all search threads before clearing this
        // pointer, and UciLoop owns the responder for the registration span.
        let pointer = unsafe {
            std::mem::transmute::<NonNull<dyn StringUciResponder>, NonNull<dyn StringUciResponder + 'static>>(
                NonNull::from(responder),
            )
        };
        *slot = Some(pointer);
    }

    fn unregister(&self, responder: &mut dyn StringUciResponder) {
        let mut slot = self.responder.lock().expect("uci responder lock");
        let expected = NonNull::from(responder).cast::<()>();
        let actual = slot
            .as_ref()
            .map(|pointer| pointer.cast::<()>())
            .expect("px0 UciPonderForwarder has no responder");
        assert_eq!(actual, expected, "px0 UciPonderForwarder responder mismatch");
        *slot = None;
    }
}

impl SearchResponder for UciResponderForwarder {
    fn output_best_move(&self, info: &BestMoveInfo) {
        let slot = self.responder.lock().expect("uci responder lock");
        let Some(mut responder) = *slot else {
            return;
        };
        // SAFETY: `register`/`unregister` maintain the documented pointer
        // lifetime invariant and the mutex serializes all callbacks.
        unsafe { responder.as_mut().output_best_move(info) };
    }

    fn output_thinking_info(&self, infos: &[ThinkingInfo]) {
        let slot = self.responder.lock().expect("uci responder lock");
        let Some(mut responder) = *slot else {
            return;
        };
        // SAFETY: see `output_best_move`.
        unsafe { responder.as_mut().output_thinking_info(infos) };
    }
}

/// px0 `Engine` 的 P3 子集：搜索 + UCI controller。
pub struct ClassicEngine {
    // px0 `Engine` owns a search built by its factory (`src/engine.cc:137-151`).
    // The Rust port only has an ONNX factory, which can fail while a UCI
    // process is already alive; `None` means that factory has not produced a
    // usable search. It is deliberately not a UniformBackend fallback.
    search: Option<ClassicSearch>,
    uci_forwarder: Arc<UciResponderForwarder>,
    position: Option<GameState>,
    uci_options: UciOptions,
    manages_weights_file: bool,
    backend_error: Option<String>,
    loaded_weights_file: Option<String>,
}

impl ClassicEngine {
    /// px0 `Engine::Engine` + `Engine::UpdateBackendConfig`
    /// (`src/engine.cc:137-167`), with the ONNX factory deferred until
    /// `SetPosition`. This is the formal UCI constructor.
    pub fn new() -> Self {
        Self {
            search: None,
            uci_forwarder: Arc::new(UciResponderForwarder {
                responder: Mutex::new(None),
            }),
            position: None,
            uci_options: UciOptions::populate_defaults(),
            manages_weights_file: true,
            backend_error: None,
            loaded_weights_file: None,
        }
    }

    pub fn with_backend(backend: Box<dyn Backend>) -> Self {
        let uci_forwarder = Arc::new(UciResponderForwarder {
            responder: Mutex::new(None),
        });
        let mut search = ClassicSearch::new(backend);
        search.set_uci_responder(Arc::clone(&uci_forwarder) as Arc<dyn SearchResponder>);
        Self {
            search: Some(search),
            uci_forwarder,
            position: None,
            uci_options: UciOptions::populate_defaults(),
            manages_weights_file: false,
            backend_error: None,
            loaded_weights_file: None,
        }
    }

    /// Explicit backend construction for tests and direct callers. Formal UCI
    /// startup uses `new` and receives its model through `WeightsFile`.
    pub fn from_onnx_file(path: impl AsRef<std::path::Path>) -> Result<Self, EnginError> {
        Ok(Self::with_backend(Box::new(OnnxBackend::from_file(path)?)))
    }

    /// Deterministic test-only constructor. It never participates in the
    /// formal UCI `WeightsFile` lifecycle.
    pub fn uniform() -> Self {
        Self::with_backend(Box::new(UniformBackend::default()))
    }

    pub fn search(&self) -> Option<&ClassicSearch> {
        self.search.as_ref()
    }

    /// px0 `Engine::UpdateBackendConfig` (`src/engine.cc:153-167`), restricted
    /// to this port's single formal ONNX backend. A failed factory result
    /// removes the old search, so a changed/broken `WeightsFile` cannot keep
    /// searching with stale weights.
    fn update_backend_config(&mut self) -> Result<(), EnginError> {
        if !self.manages_weights_file {
            return Ok(());
        }
        // Snapshot the OptionsDict value before stopping/replacing search.
        // px0's `std::string` option value is likewise stable for the whole
        // `UpdateBackendConfig` call (`src/engine.cc:153-167`).
        let path = self.uci_options.weights_file.trim().to_string();
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
                let mut search = ClassicSearch::new(Box::new(backend));
                search.set_uci_responder(Arc::clone(&self.uci_forwarder) as Arc<dyn SearchResponder>);
                Self::apply_uci_options(&mut search, &self.uci_options)?;
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

    /// px0 applies all options through its shared `OptionsDict` before a
    /// search starts (`src/engine.cc:153-167`, `search/classic/params.cc:688-703`).
    fn apply_uci_options(search: &mut ClassicSearch, options: &UciOptions) -> Result<(), EnginError> {
        search.set_uci_info_options(
            options.multi_pv,
            options.per_pv_counters,
            options.score_type,
            options.nodes_per_second_limit,
        )?;
        search.set_wdl_options(options)
    }

    /// px0 `Engine::EnsureSearchStopped` (`src/engine.cc:149-151`).
    fn stop_and_drop_search(&mut self) -> Result<(), EnginError> {
        if let Some(mut search) = self.search.take() {
            search.abort_search()?;
        }
        Ok(())
    }
}

impl EngineController for ClassicEngine {
    fn register_uci_responder(&mut self, responder: &mut dyn StringUciResponder) {
        self.uci_forwarder.register(responder);
    }

    fn unregister_uci_responder(&mut self, responder: &mut dyn StringUciResponder) {
        // px0's UCI loop normally outlives Engine. Rust permits either drop
        // order, so finish the active search while the non-owning forwarder is
        // still registered; aborting here would suppress a finite search's
        // required final bestmove (`search.cc:1019-1041`).
        if let Some(search) = self.search.as_mut() {
            let _ = search.finish_for_responder_drop();
        }
        self.uci_forwarder.unregister(responder);
    }

    fn set_uci_options(&mut self, options: &UciOptions) -> Result<(), EnginError> {
        self.uci_options = options.clone();
        if let Some(search) = self.search.as_mut() {
            Self::apply_uci_options(search, options)?;
        }
        Ok(())
    }

    fn ensure_ready(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn new_game(&mut self) -> Result<(), EnginError> {
        if let Some(search) = self.search.as_mut() {
            search.new_game()?;
        }
        self.set_position(STARTPOS_FEN, &[])
    }

    fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        // px0 `Engine::SetPosition` first calls `EnsureSearchStopped()`
        // (`src/engine.cc:187-197`).  The old worker owns mutable tree state;
        // replacing it before all worker threads joined is a UCI race.
        if let Some(search) = self.search.as_mut() {
            search.abort_search()?;
        }
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

    fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
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
        search.start_search(params)?;
        Ok(())
    }

    fn ponder_hit(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn wait(&mut self) -> Result<(), EnginError> {
        match self.search.as_mut() {
            Some(search) => search.wait_search(),
            None => Ok(()),
        }
    }

    fn stop(&mut self, _responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        if let Some(search) = self.search.as_mut() {
            search.stop_search()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uci_loop::VecUciResponder;

    #[test]
    fn deferred_engine_never_searches_with_uniform_when_weights_are_missing() {
        let mut engine = ClassicEngine::new();
        let mut responder = VecUciResponder::default();

        engine
            .go(&GoParams::default(), &mut responder)
            .expect("missing weights is a UCI status, not an engine error");

        assert!(engine.search().is_none());
        assert_eq!(
            responder.responses,
            vec!["info string cannot search: WeightsFile is not configured"]
        );
    }

    #[test]
    fn explicit_test_backend_ignores_empty_weights_file_option() {
        let mut engine = ClassicEngine::uniform();
        let options = UciOptions::populate_defaults();

        engine
            .set_uci_options(&options)
            .expect("display options must not replace an explicit test backend");

        assert!(engine.search().is_some());
    }
}
