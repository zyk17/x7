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
    search: ClassicSearch,
    uci_forwarder: Arc<UciResponderForwarder>,
    position: Option<GameState>,
    unavailable_reason: Option<String>,
    uci_weights_file: Option<String>,
    loaded_weights_file: Option<String>,
}

impl ClassicEngine {
    pub fn with_backend(backend: Box<dyn Backend>) -> Self {
        let uci_forwarder = Arc::new(UciResponderForwarder {
            responder: Mutex::new(None),
        });
        let mut search = ClassicSearch::new(backend);
        search.set_uci_responder(Arc::clone(&uci_forwarder) as Arc<dyn SearchResponder>);
        Self {
            search,
            uci_forwarder,
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
        let uci_forwarder = Arc::new(UciResponderForwarder {
            responder: Mutex::new(None),
        });
        let mut search = ClassicSearch::new(Box::new(UniformBackend::default()));
        search.set_uci_responder(Arc::clone(&uci_forwarder) as Arc<dyn SearchResponder>);
        Self {
            search,
            uci_forwarder,
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
    fn register_uci_responder(&mut self, responder: &mut dyn StringUciResponder) {
        self.uci_forwarder.register(responder);
    }

    fn unregister_uci_responder(&mut self, responder: &mut dyn StringUciResponder) {
        // px0's UCI loop normally outlives Engine. Rust permits either drop
        // order, so make the worker lifetime explicit before invalidating the
        // non-owning forwarder pointer (`engine.cc:127-136,247-250`).
        let _ = self.search.abort_search();
        self.uci_forwarder.unregister(responder);
    }

    fn set_uci_options(&mut self, options: &UciOptions) -> Result<(), EnginError> {
        // px0 always owns a configured backend manager. `ClassicEngine::uniform`
        // is deliberately test-only, so an unrelated UCI display option must
        // not turn its supplied backend into an empty WeightsFile request.
        // Formal `ClassicEngine::unavailable` starts with `None` and remains
        // unavailable until an actual px0-named WeightsFile is provided.
        if self.uci_weights_file.is_some() || !options.weights_file.is_empty() {
            self.uci_weights_file = Some(options.weights_file.clone());
        }
        self.search.set_uci_info_options(
            options.multi_pv,
            options.per_pv_counters,
            options.score_type,
            options.nodes_per_second_limit,
        )?;
        self.search.set_wdl_options(options)
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
        Ok(())
    }

    fn ponder_hit(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn wait(&mut self) -> Result<(), EnginError> {
        self.search.wait_search()
    }

    fn stop(&mut self, _responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        self.search.stop_search()?;
        Ok(())
    }
}
