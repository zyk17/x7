//! 对照 px0 `src/engine.cc:137-250` 的 UCI Engine 生命周期。

use std::sync::Arc;

use xiangqi_core::{GameState, STARTPOS_FEN};

use crate::EnginError;
use crate::Options;
use crate::callbacks::SearchResponder;
use crate::neural::backend::{Backend, CachingBackend, UniformBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::SearchSession;
use crate::uci_loop::{GoParams, StringUciResponder};

/// px0 `Engine` 的 P3 子集：搜索与 UCI 调度。
pub struct Engine {
    // UCI 进程已启动时 ONNX 初始化仍可能失败；`None` 表示没有可用的 stream
    // session，刻意不回退到 UniformBackend。
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
        let search = SearchSession::new(Arc::clone(&backend));
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
        // 在停止或替换搜索前快照 OptionsDict 值。px0 的 `std::string` option 值同样在
        // 整个 `UpdateBackendConfig` 调用中稳定（`src/engine.cc:153-167`）。
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
                let backend: Arc<dyn Backend> = Arc::new(CachingBackend::with_cache_size_power_of_two(
                    Box::new(backend),
                    self.options.nn_cache_size_power_of_two,
                ));
                let mut search = SearchSession::new(Arc::clone(&backend));
                search.set_multi_pv(self.options.multi_pv);
                search.set_mini_batch_size(self.options.mini_batch_size);
                search.set_search_params(
                    self.options.cpuct,
                    self.options.cpuct_base,
                    self.options.cpuct_factor,
                    self.options.fpu_reduction,
                );
                search.set_worker_counts(
                    self.options.gather_workers,
                    self.options.eval_workers,
                    self.options.backprop_workers,
                );
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
        self.options.set_uci_option(name, value)?;
        let option_name = name.to_ascii_lowercase();
        match option_name.as_str() {
            "multipv" => {
                if let Some(search) = self.search.as_mut() {
                    search.set_multi_pv(self.options.multi_pv);
                }
            }
            "minibatchsize" => {
                if let Some(search) = self.search.as_mut() {
                    search.set_mini_batch_size(self.options.mini_batch_size);
                }
            }
            "nncachesizepoweroftwo" => {
                if let Some(search) = self.search.as_mut() {
                    search.set_nn_cache_size_power_of_two(self.options.nn_cache_size_power_of_two);
                }
            }
            "cpuct" | "cpuctbase" | "cpuctfactor" | "fpureduction" => {
                if let Some(search) = self.search.as_mut() {
                    search.set_search_params(
                        self.options.cpuct,
                        self.options.cpuct_base,
                        self.options.cpuct_factor,
                        self.options.fpu_reduction,
                    );
                }
            }
            "gatherworkers" | "evalworkers" | "backpropworkers" => {
                if let Some(search) = self.search.as_mut() {
                    search.set_worker_counts(
                        self.options.gather_workers,
                        self.options.eval_workers,
                        self.options.backprop_workers,
                    );
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn ensure_ready(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    pub(crate) fn new_game(&mut self) -> Result<(), EnginError> {
        if let Some(search) = self.search.as_mut() {
            search.abort()?;
            search.reset_clock();
        }
        self.set_position(STARTPOS_FEN, &[])
    }

    pub(crate) fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        // SearchSession 先结束当前 job、drain reservation，再替换可复用树。
        // px0 `Engine::SetPosition` 在停止搜索后、新建 `GameState` 前更新 backend
        // 配置（`src/engine.cc:187-197`）。此处重试也处理先前 `setoption` 失败后才出现的
        // 文件。
        self.update_backend_config()?;
        let state = GameState::from_fen_moves(fen, moves)?;
        if let Some(search) = self.search.as_mut() {
            search.set_position(&state)?;
        }
        self.position = Some(state);
        Ok(())
    }

    pub(crate) fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        // px0 只有在 Ponder option 启用完整 position/ponderhit 生命周期时才接受
        // `go ponder`（`src/engine.cc:205-215`）。本实现尚未翻译该 option 与生命周期，
        // 因此接受 token 后静默运行普通搜索会构成伪 UCI 功能。
        if params.ponder {
            return Err(EnginError::Uci("Ponder is not enabled.".into()));
        }
        // px0 `Engine::Go` 调用 `StartSearch` 前会经 `NewGame` 初始化缺失的 position
        // （`src/engine.cc:206-219`）。NewGame 随后运行 SetPosition 和
        // UpdateBackendConfig，因此先检查 backend 会错误拒绝合法的裸 `go` 命令。
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
        // px0 `Engine::PonderHit` 在活跃 ponder 搜索外拒绝此命令（`src/engine.cc:226-235`）。
        // 此处刻意不提供 Ponder。
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
