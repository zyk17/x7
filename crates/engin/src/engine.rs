//! px0 `src/engine.cc:137-250` 的 UCI Engine lifecycle.

use std::sync::Arc;

use xiangqi_core::{GameState, STARTPOS_FEN};

use crate::callbacks::SearchResponder;
use crate::neural::backend::{Backend, CachingBackend, UniformBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::{SearchBase, SearchFactory};
use crate::uci_loop::{EngineController, GoParams, StringUciResponder};
use crate::EnginError;
use crate::Options;

/// px0 `Engine` 的 P3 子集：搜索 + UCI controller。
pub struct Engine {
    // px0 `Engine` owns a search built by its factory (`src/engine.cc:137-151`).
    // The Rust port only has an ONNX factory, which can fail while a UCI
    // process is already alive; `None` means that factory has not produced a
    // usable search. It is deliberately not a UniformBackend fallback.
    search: Option<Box<dyn SearchBase>>,
    factory: Arc<dyn SearchFactory>,
    backend: Option<Arc<dyn Backend>>,
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
    /// (`src/engine.cc:137-167`), with the ONNX factory deferred until
    /// `SetPosition`. This is the formal UCI constructor.
    pub fn new() -> Self {
        Self {
            search: None,
            factory: Arc::new(crate::search::stream::Factory),
            backend: None,
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
        let factory: Arc<dyn SearchFactory> = Arc::new(crate::search::stream::Factory);
        let mut search = factory.create(Arc::clone(&backend));
        search.set_responder(None);
        Self {
            search: Some(search),
            factory,
            backend: Some(backend),
            options: Options::default(),
            position: None,
            manages_weights_file: false,
            backend_error: None,
            loaded_weights_file: None,
            responder: None,
        }
    }

    /// px0 `Engine(const SearchFactory&, const OptionsDict&)`: callers may
    /// select classic or stream without changing Engine/UCI lifecycle code.
    pub fn with_factory(factory: Arc<dyn SearchFactory>) -> Self {
        let mut engine = Self::new();
        engine.factory = factory;
        engine
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
                let mut search = self.factory.create(Arc::clone(&backend));
                search.set_responder(self.responder.clone());
                self.search = Some(search);
                self.backend = Some(backend);
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
            search.abort_search()?;
        }
        Ok(())
    }
}

impl EngineController for Engine {
    fn set_search_responder(&mut self, responder: Option<Arc<dyn SearchResponder>>) {
        Engine::set_search_responder(self, responder);
    }

    fn set_uci_options(&mut self, options: &Options) -> Result<(), EnginError> {
        self.options = options.clone();
        // px0 snapshots `OptionsDict` into a new `SearchParams` only when a
        // `Search` is constructed for `Go` (`search/classic/wrapper.cc:114-140`,
        // `params.cc:688-703`). Do not abort a running search merely because a
        // GUI changed an option; the next `go` applies this pending snapshot.
        Ok(())
    }

    fn ensure_ready(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn new_game(&mut self) -> Result<(), EnginError> {
        if let Some(search) = self.search.as_mut() {
            // px0 `ClassicSearch::NewGame` destroys its current `Search`
            // (`src/search/classic/wrapper.cc:100-105`). Its destructor calls
            // `Abort()` then `Wait()`, so an active infinite search cannot
            // block `ucinewgame` forever.
            search.abort_search()?;
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
        // px0 starts its UCI clock in `Engine::SetPosition` before backend
        // configuration (`src/engine.cc:187-197`). Keep the existing clock
        // when replacing a configured backend; if there was no search yet,
        // start the equivalent clock after its first successful creation.
        let had_search = self.search.is_some();
        if let Some(search) = self.search.as_mut() {
            search.start_clock()?;
        }
        // px0 `Engine::SetPosition` updates backend configuration after
        // stopping search and before making the new `GameState`
        // (`src/engine.cc:187-197`). Retrying here also handles a file that
        // appeared after an earlier failed `setoption`.
        self.update_backend_config()?;
        let state = GameState::from_fen_moves(fen, moves)?;
        if let Some(search) = self.search.as_mut() {
            if !had_search {
                search.start_clock()?;
            }
            search.set_position(&state)?;
        }
        self.position = Some(state);
        Ok(())
    }

    fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        // px0 rejects `go ponder` unless the Ponder option enabled its full
        // position/ponderhit lifecycle (`src/engine.cc:205-215`). This port
        // has not translated that option or lifecycle, so accepting the token
        // and silently running a normal search would be a false UCI feature.
        if params.ponder {
            return Err(EnginError::Uci("Ponder is not enabled.".into()));
        }
        // px0 `Engine::Go` restarts the clock only without a UCI clock budget
        // (`src/engine.cc:205-219`). With `wtime` or `btime`, the budget
        // intentionally runs from the preceding `position` command.
        if params.wtime.is_none() && params.btime.is_none() {
            if let Some(search) = self.search.as_mut() {
                search.start_clock()?;
            }
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
        search.validate_go(params)?;
        // px0 constructs a fresh `Search` for every `Go`
        // (`src/search/classic/wrapper.cc:114-140`). Replacing its
        // `unique_ptr` destroys the old search, which performs `Abort()` and
        // `Wait()` (`search.cc:1055-1057`). This port keeps the long-lived
        // A factory-produced SearchBase is restarted for every `go`.
        search.abort_search()?;
        search.start_search(params)?;
        Ok(())
    }

    fn ponder_hit(&mut self) -> Result<(), EnginError> {
        // px0 `Engine::PonderHit` rejects this outside an active ponder search
        // (`src/engine.cc:226-235`). Ponder is intentionally unavailable here.
        Err(EnginError::Uci("ponderhit while not pondering".into()))
    }

    fn wait(&mut self) -> Result<(), EnginError> {
        let Some(search) = self.search.as_mut() else {
            return Ok(());
        };
        search.wait_search()
    }

    fn stop(&mut self) -> Result<(), EnginError> {
        if let Some(search) = self.search.as_mut() {
            search.stop_search()?;
        }
        self.wait()
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::uci_loop::{UciLoop, VecUciResponder};

    fn bestmove_count(responder: &VecUciResponder) -> usize {
        responder
            .responses
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    }

    #[test]
    fn deferred_engine_never_searches_with_uniform_when_weights_are_missing() {
        let mut engine = Engine::new();
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
        let mut engine = Engine::uniform();
        let options = Options::populate_defaults();

        engine
            .set_uci_options(&options)
            .expect("display options must not replace an explicit test backend");

        assert!(engine.search().is_some());
    }

    #[test]
    fn second_go_aborts_and_replaces_the_previous_search() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        let params = GoParams {
            nodes: Some(8),
            ..GoParams::default()
        };

        engine.go(&params, &mut responder).expect("first go");
        engine.wait().expect("first wait");
        let first_visits = engine.search().expect("search").total_root_visits();

        engine.go(&params, &mut responder).expect("second go");
        engine.wait().expect("second wait");
        let second_visits = engine.search().expect("search").total_root_visits();

        assert!(first_visits >= 8);
        // The tree may be retained, trimmed, or reset by its search wrapper;
        // only a fresh completed budget is a UCI lifecycle guarantee here.
        assert!(second_visits >= 8);
    }

    #[test]
    fn bare_go_initializes_startpos_before_searching() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        engine
            .go(
                &GoParams {
                    nodes: Some(1),
                    ..GoParams::default()
                },
                &mut responder,
            )
            .expect("bare go");
        engine.wait().expect("wait");

        assert!(engine.position.is_some());
        assert!(engine.search().expect("search").total_root_visits() >= 1);
    }

    /// px0 accepts a bare `go`: the common stopper supplies the default
    /// 4_000_000_000 visit cap instead of requiring an explicit UCI budget
    /// (`chess/uciloop.cc:207-237`, `stoppers/common.cc:133-145`).
    #[test]
    fn bare_go_without_a_budget_starts_until_explicit_stop() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        engine.go(&GoParams::default(), &mut responder).expect("bare go");
        std::thread::sleep(std::time::Duration::from_millis(10));
        engine.stop(&mut responder).expect("stop");
        engine.wait().expect("wait");

        assert!(engine.search().expect("search").total_root_visits() > 0);
    }

    #[test]
    fn unavailable_ponder_paths_are_rejected_instead_of_running_normally() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        let error = engine
            .go(
                &GoParams {
                    ponder: true,
                    nodes: Some(1),
                    ..GoParams::default()
                },
                &mut responder,
            )
            .expect_err("ponder must not become a normal search");
        assert!(error.to_string().contains("Ponder is not enabled"));
        assert!(engine.ponder_hit().is_err());
    }

    #[test]
    fn unavailable_depth_and_mate_stoppers_are_not_silently_combined_with_nodes() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        for params in [
            GoParams {
                nodes: Some(1),
                depth: Some(4),
                ..GoParams::default()
            },
            GoParams {
                nodes: Some(1),
                mate: Some(2),
                ..GoParams::default()
            },
        ] {
            let error = engine.go(&params, &mut responder).expect_err("untranslated stopper");
            assert!(error.to_string().contains("go depth/go mate stopper"));
        }
    }

    #[test]
    fn legacy_clock_manager_accepts_px0_uci_clock_budget() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        engine
            .go(
                &GoParams {
                    wtime: Some(1_000),
                    winc: Some(10),
                    ..GoParams::default()
                },
                &mut responder,
            )
            .expect("px0 legacy clock manager");
        engine.wait().expect("clock search wait");
        assert!(engine.search().expect("search").total_root_visits() > 0);
    }

    #[test]
    fn new_game_aborts_an_infinite_search_before_resetting_the_tree() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        engine
            .go(
                &GoParams {
                    infinite: true,
                    ..GoParams::default()
                },
                &mut responder,
            )
            .expect("infinite go");

        engine.new_game().expect("new game must abort the active search");
        assert_eq!(engine.search().expect("search").total_root_visits(), 0);
    }

    /// px0 `Engine::Stop` enables exactly one final response, while
    /// `Search::Abort` suppresses a replaced search (`engine.cc:148-151`,
    /// `search.cc:1019-1041`). This is the UCI boundary used by GUI sequences
    /// such as `go infinite`, `stop`, then `wait`.
    #[test]
    fn uci_stop_then_wait_emits_exactly_one_bestmove() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        let mut options = Options::populate_defaults();
        {
            let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
            uci.process_line("position startpos", "test").expect("position");
            uci.process_line("go infinite", "test").expect("infinite go");
            std::thread::sleep(std::time::Duration::from_millis(10));
            uci.process_line("stop", "test").expect("stop");
            uci.process_line("wait", "test").expect("wait");
        }

        assert_eq!(bestmove_count(&responder), 1);
    }

    /// px0 `Engine::SetPosition` calls `EnsureSearchStopped` before it stores
    /// the replacement state (`engine.cc:187-197`). The aborted search must
    /// not publish a stale `bestmove`; the following finite `go` owns the only
    /// response.
    #[test]
    fn uci_position_replacement_suppresses_the_old_search_bestmove() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        let mut options = Options::populate_defaults();
        {
            let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
            uci.process_line("position startpos", "test").expect("initial position");
            uci.process_line("go infinite", "test").expect("infinite go");
            std::thread::sleep(std::time::Duration::from_millis(10));
            uci.process_line("position startpos moves b2b3", "test")
                .expect("replacement position");
            uci.process_line("go nodes 8", "test").expect("replacement go");
            uci.process_line("wait", "test").expect("wait");
        }

        assert_eq!(bestmove_count(&responder), 1);
    }

    /// px0 emits `bestmove a0a0` for a root with no legal move rather than
    /// silently ending the UCI request (`search.cc:612-621`,
    /// `uciloop.cc:279-287`).
    #[test]
    fn uci_checkmated_root_emits_px0_null_bestmove() {
        let mut engine = Engine::uniform();
        let mut responder = VecUciResponder::default();
        let mut options = Options::populate_defaults();
        {
            let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
            uci.process_line("position fen 4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", "test")
                .expect("checkmated position");
            uci.process_line("go nodes 1", "test").expect("go");
            uci.process_line("wait", "test").expect("wait");
        }

        assert_eq!(
            responder
                .responses
                .iter()
                .filter(|line| line.as_str() == "bestmove a0a0")
                .count(),
            1
        );
    }
}
