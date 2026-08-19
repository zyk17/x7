//! UCI Engine：拥有 graph、worker pool 与每次搜索 job。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

use crate::neural::backend::{Backend, CachingBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::{
    Search, SearchConfig, SearchControl, SearchGraph, SearchLimits, SearchParams, Stats, TimeBudget, TimeManager,
    WorkerPool, best_mate_with_params, best_move_filtered_with_params, principal_variation_with_history_and_params,
    root_stats, root_variations,
};
use crate::uci_loop::{
    BestMoveInfo, GoParams, ThinkingInfo, Wdl, write_stdout, write_stdout_best_move, write_stdout_thinking,
};
use crate::{EnginError, Options};

/// UCI、图、worker 与单次搜索的唯一 owner。
pub struct Engine {
    // UCI 进程已启动时 ONNX 初始化仍可能失败；`None` 表示没有可用 backend，刻意不回退到 UniformBackend。
    backend: Option<Arc<dyn Backend>>,
    graph: Option<SearchGraph>,
    graph_reaper: GraphReaper,
    worker_pool: Option<Arc<WorkerPool>>,
    next_generation: u64,
    applied_nn_cache_size: Option<u8>,
    time_manager: TimeManager,
    active: Option<ActiveSearch>,
    options: Options,
    position: Option<GameState>,
    manages_weights_file: bool,
    backend_error: Option<String>,
    loaded_weights_file: Option<String>,
    /// 让 abort 与 owner 的整组输出线性化，避免新命令之后漏出旧 generation 的结果。
    stdout_gate: Arc<Mutex<()>>,
}

struct ActiveSearch {
    control: SearchControl,
    publish_output: Arc<AtomicBool>,
    owner_thread: JoinHandle<Result<(), EnginError>>,
    started: Instant,
    clock_budget: Option<TimeBudget>,
}

/// 后台释放已被整图替换的旧 repository。当前 root 的 sibling prune 在 `position`
/// 的 abort 之后同步做：ContinuationTree 入口不绑定 child，不能边搜边扫。
struct GraphReaper {
    sender: Option<crossbeam_channel::Sender<GraphCleanup>>,
    thread: Option<JoinHandle<()>>,
}

enum GraphCleanup {
    Retire(Arc<crate::search::NodeRepository>),
}

impl GraphReaper {
    fn new() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded::<GraphCleanup>();
        let thread = thread::Builder::new()
            .name("engin-graph-reaper".into())
            .spawn(move || {
                while let Ok(cleanup) = receiver.recv() {
                    match cleanup {
                        GraphCleanup::Retire(repository) => match Arc::try_unwrap(repository) {
                            Ok(repository) => repository.release_incrementally(),
                            Err(repository) => drop(repository),
                        },
                    }
                }
            })
            .expect("graph reaper thread starts");
        Self {
            sender: Some(sender),
            thread: Some(thread),
        }
    }

    fn retire(&self, repository: Arc<crate::search::NodeRepository>) {
        self.sender
            .as_ref()
            .expect("graph reaper sender lives with engine")
            .send(GraphCleanup::Retire(repository))
            .expect("graph reaper thread is alive");
    }
}

impl Drop for GraphReaper {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// UCI info 最小输出间隔（毫秒）。
const UCI_INFO_MINIMUM_FREQUENCY: Duration = Duration::from_secs(5);
/// owner 在 stream 等待在途 event 时，最多每 100ms 检查一次是否值得输出进度。
const OWNER_PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// search owner 持有的只读 root view，不参与 worker 的搜索状态。
#[derive(Clone)]
struct RootSnapshot {
    repository: Arc<crate::search::NodeRepository>,
    root_key: crate::search::NodeKey,
    root_history: Arc<PositionHistory>,
    initial_visits: u64,
    root_is_black: bool,
    root_move_filter: Vec<Move>,
    multi_pv: usize,
    params: SearchParams,
}

/// 仅用于判断是否值得构造完整 UCI `info` 的 root marker。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SearchProgress {
    best_move: Option<Move>,
    depth: i32,
    seldepth: i32,
}

/// 上次已经输出的 root marker。
#[derive(Default)]
struct PublishedInfo {
    progress: Option<SearchProgress>,
    time: i64,
}

impl Default for Engine {
    /// 构造空 Engine；不另设初始化路径。
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// UCI 启动时立即创建空 Engine。
    /// 正式 ONNX backend 则由首次 `set_position` 中的 `UpdateBackendConfig` 加载。
    pub fn new() -> Self {
        Self {
            backend: None,
            graph: None,
            graph_reaper: GraphReaper::new(),
            worker_pool: None,
            next_generation: 0,
            applied_nn_cache_size: None,
            time_manager: TimeManager::default(),
            active: None,
            options: Options::default(),
            position: None,
            manages_weights_file: true,
            backend_error: None,
            loaded_weights_file: None,
            stdout_gate: Arc::new(Mutex::new(())),
        }
    }

    /// 安装新 backend 时丢弃旧图与旧 worker；调用方已先 stop/drain。
    /// 按当前 option 重建或更新 NN backend。
    fn install_backend(&mut self, backend: Arc<dyn Backend>) {
        self.backend = Some(backend);
        self.graph = None;
        self.worker_pool = None;
        self.next_generation = 0;
        self.applied_nn_cache_size = None;
    }

    /// 加载失败时不保留旧权重。
    fn update_backend_config(&mut self) -> Result<(), EnginError> {
        if !self.manages_weights_file {
            return Ok(());
        }
        let path = self.options.weights_file.trim().to_string();
        if path.is_empty() {
            self.stop_and_drop_backend()?;
            self.loaded_weights_file = None;
            self.backend_error = Some("WeightsFile is not configured".into());
            return Ok(());
        }
        if self.loaded_weights_file.as_deref() == Some(path.as_str()) && self.backend.is_some() {
            return Ok(());
        }

        self.stop_and_drop_backend()?;
        match OnnxBackend::from_file(&path) {
            Ok(backend) => {
                self.install_backend(Arc::new(CachingBackend::with_cache_size_power_of_two(
                    Box::new(backend),
                    self.options.nn_cache_size_power_of_two,
                )));
                self.applied_nn_cache_size = Some(self.options.nn_cache_size_power_of_two);
                self.loaded_weights_file = Some(path);
                self.backend_error = None;
            }
            Err(error) => {
                self.loaded_weights_file = None;
                self.backend_error = Some(format!("cannot load WeightsFile {path}: {error}"));
            }
        }
        Ok(())
    }

    /// 换权重前停止当前 job。
    fn stop_and_drop_backend(&mut self) -> Result<(), EnginError> {
        self.abort()?;
        self.backend = None;
        self.graph = None;
        self.worker_pool = None;
        self.applied_nn_cache_size = None;
        Ok(())
    }

    pub fn options(&self) -> &Options {
        &self.options
    }

    /// 更新 Engine 生命周期 option。已启动 job 使用自己创建时的 `SearchConfig` 快照。
    pub fn set_option(&mut self, name: &str, value: &str) -> Result<(), EnginError> {
        self.options.set_uci_option(name, value)
    }

    pub(crate) fn ensure_ready(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    pub(crate) fn new_game(&mut self) -> Result<(), EnginError> {
        self.abort()?;
        self.time_manager.reset();
        self.set_position(STARTPOS_FEN, &[])
    }

    /// 先 stop/drain，再用完整 history 复用或重置图。
    pub(crate) fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        self.update_backend_config()?;
        let state = GameState::from_fen_moves(fen, moves)?;
        let history = Arc::new(state.position_history());
        self.abort()?;
        if let Some(graph) = self.graph.as_mut() {
            if let Some(retired) = graph.reset_to_history_after_drain(Arc::clone(&history))? {
                self.graph_reaper.retire(retired);
            }
            if let Some(root) = graph.take_pending_gc_root() {
                graph.repository().retain_from_root(root);
            }
        } else if self.backend.is_some() {
            self.graph = Some(SearchGraph::new(history));
        }
        self.position = Some(state);
        Ok(())
    }

    /// 检查 stream 已实现的 UCI `go` 子集；未支持项明确拒绝。
    fn validate_go(&self, params: &GoParams) -> Result<(), EnginError> {
        if params.depth.is_some() {
            return Err(EnginError::PortIncomplete("go depth is not supported"));
        }
        if params.mate.is_some() {
            return Err(EnginError::PortIncomplete("go mate is not supported"));
        }
        if params.nodes.is_some_and(|nodes| nodes <= 0) {
            return Err(EnginError::Uci("go nodes must be positive".into()));
        }
        if params.movetime.is_some_and(|time| time < 0) {
            return Err(EnginError::Uci("go movetime must not be negative".into()));
        }
        let has_clock = params.wtime.is_some()
            || params.btime.is_some()
            || params.winc.is_some()
            || params.binc.is_some()
            || params.movestogo.is_some();
        if [params.wtime, params.btime, params.winc, params.binc]
            .into_iter()
            .flatten()
            .any(|value| value < 0)
            || params.movestogo.is_some_and(|value| value <= 0)
        {
            return Err(EnginError::Uci(
                "go clock values must be non-negative and movestogo positive".into(),
            ));
        }
        if has_clock {
            let root = self
                .graph
                .as_ref()
                .ok_or(EnginError::Uci("position is not configured".into()))?
                .root_history()
                .last();
            let side_time = if root.is_black_to_move() {
                params.btime
            } else {
                params.wtime
            };
            if side_time.is_none() {
                return Err(EnginError::Uci("go clock is missing side-to-move time".into()));
            }
        }
        if params.movetime.is_some() && has_clock {
            return Err(EnginError::Uci(
                "go movetime cannot be combined with clock fields".into(),
            ));
        }
        if params.infinite && (params.nodes.is_some() || params.movetime.is_some() || has_clock) {
            return Err(EnginError::Uci(
                "go infinite cannot be combined with nodes, movetime, or clock fields".into(),
            ));
        }
        if !params.infinite && params.nodes.is_none() && params.movetime.is_none() && !has_clock {
            return Err(EnginError::Uci(
                "go requires nodes, movetime, clock fields, or infinite".into(),
            ));
        }
        Ok(())
    }

    /// `go searchmoves` 根着过滤。
    fn root_move_filter(&self, searchmoves: &[String]) -> Result<Vec<Move>, EnginError> {
        let graph = self
            .graph
            .as_ref()
            .ok_or(EnginError::Uci("position is not configured".into()))?;
        let board = graph.root_history().last().board();
        let legal_moves = board.generate_legal_moves();
        let moves: Vec<_> = searchmoves
            .iter()
            .filter_map(|move_text| board.parse_move(move_text).ok())
            .filter(|mv| legal_moves.contains(mv))
            .collect();
        if !searchmoves.is_empty() && moves.is_empty() {
            return Err(EnginError::Uci("No legal searchmoves.".into()));
        }
        Ok(moves)
    }

    /// 启动一个独占 job。worker pool 跨 job 常驻，图和配置均由 Engine 直接持有。
    /// 参考 LC3 Overview 的 "Search" / "Workers"。
    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError> {
        self.validate_go(params)?;
        self.abort()?;
        let root_move_filter = self.root_move_filter(&params.searchmoves)?;
        let backend = Arc::clone(
            self.backend
                .as_ref()
                .ok_or(EnginError::Uci("position is not configured".into()))?,
        );
        if self.applied_nn_cache_size != Some(self.options.nn_cache_size_power_of_two) {
            backend.set_cache_size_power_of_two(self.options.nn_cache_size_power_of_two);
            self.applied_nn_cache_size = Some(self.options.nn_cache_size_power_of_two);
        }
        self.next_generation = self.next_generation.wrapping_add(1);
        let config = SearchConfig {
            root_move_filter: root_move_filter.clone(),
            eval_batch_size: self.options.mini_batch_size,
            params: SearchParams {
                cpuct: self.options.cpuct,
                cpuct_base: self.options.cpuct_base,
                cpuct_factor: self.options.cpuct_factor,
                fpu_reduction: self.options.fpu_reduction,
                virtual_mean_fpu_scale: 1.0,
                lcb_stdevs: self.options.lcb_stdevs,
                lcb_min_visit_fraction: self.options.lcb_min_visit_fraction,
            },
            gather_workers: self.options.threads.div_ceil(2),
            eval_workers: self.options.threads / 2,
            ..SearchConfig::default()
        };
        let decision_params = config.params;
        let pool = match self.worker_pool.as_ref() {
            Some(pool) if pool.matches_config(backend.as_ref(), &config) => Arc::clone(pool),
            _ => {
                let pool = Arc::new(WorkerPool::new(backend.as_ref(), &config));
                self.worker_pool = Some(Arc::clone(&pool));
                pool
            }
        };
        let graph = self.graph.as_ref().expect("position creates a graph with a backend");
        let root_is_black = graph.root_history().last().is_black_to_move();
        let search = Search::new_with_graph_in_pool(backend, self.next_generation, graph, config, pool);
        let snapshot = RootSnapshot {
            repository: Arc::clone(search.repository()),
            root_key: search.root_key(),
            root_history: Arc::clone(graph.root_history()),
            initial_visits: search.initial_visits(),
            root_is_black,
            root_move_filter: root_move_filter.clone(),
            multi_pv: self.options.multi_pv,
            params: decision_params,
        };
        let started = Instant::now();
        let clock_budget = if params.movetime.is_none() {
            self.time_manager.budget(params, graph.root_history().last())
        } else {
            None
        };
        let limits = SearchLimits {
            max_playouts: params.nodes.map(|nodes| nodes.max(1) as u64),
            deadline: params
                .movetime
                .map(|ms| started + Duration::from_millis(ms.max(0) as u64))
                .or_else(|| clock_budget.map(|budget| budget.deadline_after(started))),
        };
        let control = search.control();
        let publish_output = Arc::new(AtomicBool::new(true));
        let owner_publish_output = Arc::clone(&publish_output);
        let output_options = self.options.clone();
        let output_gate = Arc::clone(&self.stdout_gate);
        let owner_thread = std::thread::spawn(move || {
            run_search(
                search,
                snapshot,
                output_options,
                output_gate,
                owner_publish_output,
                limits,
            )
        });
        self.active = Some(ActiveSearch {
            control,
            publish_output,
            owner_thread,
            started,
            clock_budget,
        });
        Ok(())
    }

    pub(crate) fn go(&mut self, params: &GoParams) -> Result<(), EnginError> {
        if params.ponder {
            return Err(EnginError::Uci("Ponder is not enabled.".into()));
        }
        if self.position.is_none() {
            self.new_game()?;
        }
        if self.backend.is_none() {
            let reason = self.backend_error.as_deref().unwrap_or("WeightsFile is not configured");
            write_stdout(&[format!("info string cannot search: {reason}")]);
            let mv = self
                .position
                .as_ref()
                .map(|state| legal_fallback_move(&state.position_history(), &[]))
                .unwrap_or(Move::NULL);
            write_stdout_best_move(&BestMoveInfo::new(mv));
            return Ok(());
        }
        self.start_search(params)
    }

    pub(crate) fn ponder_hit(&mut self) -> Result<(), EnginError> {
        Err(EnginError::Uci("ponderhit while not pondering".into()))
    }

    pub(crate) fn wait(&mut self) -> Result<(), EnginError> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let ActiveSearch {
            owner_thread,
            started,
            clock_budget,
            ..
        } = active;
        let result = owner_thread
            .join()
            .map_err(|_| EnginError::Uci("search owner thread panicked".into()))?;
        if let Some(clock_budget) = clock_budget {
            self.time_manager.finish(clock_budget, started.elapsed());
        }
        result
    }

    pub(crate) fn stop(&mut self) -> Result<(), EnginError> {
        if let Some(active) = &self.active {
            active.control.request_stop();
        }
        self.wait()
    }

    /// 替换 position / backend / 新 `go` 时停止 info，但必须留下 `bestmove` 结束上一次 `go`。
    fn abort(&mut self) -> Result<(), EnginError> {
        if let Some(active) = &self.active {
            let _output = self.stdout_gate.lock();
            active.publish_output.store(false, Ordering::Release);
            active.control.request_stop();
        }
        if let Err(error) = self.wait() {
            eprintln!("info string abort drain ignored previous search error: {error}");
        }
        Ok(())
    }
}

/// 单次 job 的唯一 owner：运行搜索、按进度输出、drain 后输出最终结果并归还 worker。
/// 参考 LC3 Overview 的 Search/Watchdog 角色，但不另开 watchdog 线程。
fn run_search(
    mut search: Search,
    snapshot: RootSnapshot,
    output_options: Options,
    output_gate: Arc<Mutex<()>>,
    publish_output: Arc<AtomicBool>,
    limits: SearchLimits,
) -> Result<(), EnginError> {
    let started = Instant::now();
    let mut published = PublishedInfo::default();
    let result = search.run_with_limits_reporting(limits, Some(OWNER_PROGRESS_INTERVAL), |stats| {
        if !publish_output.load(Ordering::Acquire) {
            return;
        }
        let time = started.elapsed().as_millis() as i64;
        let progress = snapshot.progress(&stats);
        if published.should_publish(progress, time) {
            let infos = snapshot.thinking_infos(stats, started);
            let _output = output_gate.lock();
            if publish_output.load(Ordering::Acquire) {
                write_stdout_thinking(&infos, &output_options);
                published.update(progress, time);
            }
        }
    });
    if let Ok(stats) = &result {
        // path-local repetition/rule60 不能标记共享 board node，但对这次 UCI root 已是
        // 真正终局；不得从旧图的 edge 回退出一着看似合法的棋。
        let (best_move, principal_variation) = if search.root_is_path_terminal() {
            (None, Vec::new())
        } else {
            (
                best_move_filtered_with_params(
                    search.repository(),
                    search.root_key(),
                    snapshot.root_is_black,
                    &snapshot.root_move_filter,
                    &snapshot.params,
                ),
                principal_variation_with_history_and_params(
                    search.repository(),
                    search.root_key(),
                    snapshot.root_history.as_ref(),
                    snapshot.root_is_black,
                    &snapshot.root_move_filter,
                    &snapshot.params,
                ),
            )
        };
        let mut infos = snapshot.thinking_infos(stats.clone(), started);
        if let Some(info) = infos.first_mut() {
            info.pv = principal_variation;
        }
        let best_move = reported_uci_move(
            best_move,
            snapshot.root_history.as_ref(),
            &snapshot.root_move_filter,
        );
        let _output = output_gate.lock();
        if publish_output.load(Ordering::Acquire) {
            write_stdout_thinking(&infos, &output_options);
        }
        write_stdout_best_move(&BestMoveInfo::new(best_move));
    } else if let Err(error) = &result {
        let info = ThinkingInfo {
            comment: format!("stream search failed: {error}"),
            ..ThinkingInfo::default()
        };
        let best_move = reported_uci_move(None, snapshot.root_history.as_ref(), &snapshot.root_move_filter);
        let _output = output_gate.lock();
        if publish_output.load(Ordering::Acquire) {
            write_stdout_thinking(&[info], &output_options);
        }
        write_stdout_best_move(&BestMoveInfo::new(best_move));
    }
    search.stop_and_finish();
    result.map(|_| ())
}

fn reported_uci_move(chosen: Option<Move>, history: &PositionHistory, root_move_filter: &[Move]) -> Move {
    match chosen {
        Some(mv) if !mv.is_null() => mv,
        _ => legal_fallback_move(history, root_move_filter),
    }
}

fn legal_fallback_move(history: &PositionHistory, root_move_filter: &[Move]) -> Move {
    let legal = history.last().board().generate_legal_moves();
    if root_move_filter.is_empty() {
        legal.into_iter().next()
    } else {
        root_move_filter.iter().copied().find(|mv| legal.contains(mv))
    }
    .unwrap_or(Move::NULL)
}

impl RootSnapshot {
    /// info 输出门槛：marker 未变化时不生成 PV/MultiPV。
    fn progress(&self, stats: &Stats) -> SearchProgress {
        SearchProgress {
            best_move: best_move_filtered_with_params(
                &self.repository,
                self.root_key,
                self.root_is_black,
                &self.root_move_filter,
                &self.params,
            ),
            depth: stats.average_depth.min(i32::MAX as u64) as i32,
            seldepth: stats.max_depth.min(i32::MAX as u64) as i32,
        }
    }

    /// 同一 root 快照按根边排序输出 MultiPV。
    fn thinking_infos(&self, stats: Stats, started: Instant) -> Vec<ThinkingInfo> {
        let time = started.elapsed().as_millis() as i64;
        let nodes = self.initial_visits.saturating_add(stats.completed_playouts) as i64;
        let nps = if time == 0 {
            0
        } else {
            (stats.completed_playouts as i64 * 1000 / time) as i32
        };
        let eps = if time == 0 {
            0
        } else {
            (stats.network_evaluations as i64 * 1000 / time) as i32
        };
        let common = ThinkingInfo {
            depth: stats.average_depth.min(i32::MAX as u64) as i32,
            seldepth: stats.max_depth.min(i32::MAX as u64) as i32,
            time,
            nodes,
            nps,
            eps,
            ..ThinkingInfo::default()
        };
        let Some(root) = root_stats(&self.repository, self.root_key) else {
            return vec![common];
        };
        let wl = (-root.q).clamp(-1.0, 1.0);
        let draw = root.draw.clamp(0.0, 1.0);
        let variations = root_variations(
            &self.repository,
            self.root_key,
            Some(self.root_history.as_ref()),
            self.root_is_black,
            &self.root_move_filter,
            self.multi_pv,
            &self.params,
        );
        if variations.is_empty() {
            let win = ((1.0 - draw + wl) * 0.5).clamp(0.0, 1.0);
            let loss = ((1.0 - draw - wl) * 0.5).clamp(0.0, 1.0);
            let mate = best_mate_with_params(&self.repository, self.root_key, &self.root_move_filter, &self.params);
            return vec![ThinkingInfo {
                mate,
                score: mate.is_none().then_some((wl * 1000.0).round() as i32),
                wdl: Some(Wdl {
                    w: (win * 1000.0).round() as i32,
                    d: (draw * 1000.0).round() as i32,
                    l: (loss * 1000.0).round() as i32,
                }),
                pv: principal_variation_with_history_and_params(
                    &self.repository,
                    self.root_key,
                    self.root_history.as_ref(),
                    self.root_is_black,
                    &self.root_move_filter,
                    &self.params,
                ),
                ..common
            }];
        }
        let show_multipv = variations.len() > 1;
        variations
            .into_iter()
            .enumerate()
            .map(|(index, variation)| {
                let win = ((1.0 - variation.draw + variation.wl) * 0.5).clamp(0.0, 1.0);
                let loss = ((1.0 - variation.draw - variation.wl) * 0.5).clamp(0.0, 1.0);
                ThinkingInfo {
                    mate: variation.mate,
                    score: variation
                        .mate
                        .is_none()
                        .then_some((variation.wl * 1000.0).round() as i32),
                    wdl: Some(Wdl {
                        w: (win * 1000.0).round() as i32,
                        d: (variation.draw * 1000.0).round() as i32,
                        l: (loss * 1000.0).round() as i32,
                    }),
                    pv: variation.pv,
                    multipv: if show_multipv { (index + 1) as i32 } else { -1 },
                    ..common.clone()
                }
            })
            .collect()
    }
}

impl PublishedInfo {
    /// 周期性 info 输出门槛。
    fn should_publish(&self, progress: SearchProgress, time: i64) -> bool {
        progress.best_move.is_some()
            && (self.progress != Some(progress)
                || time.saturating_sub(self.time) > UCI_INFO_MINIMUM_FREQUENCY.as_millis() as i64)
    }

    fn update(&mut self, progress: SearchProgress, time: i64) {
        self.progress = Some(progress);
        self.time = time;
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine, legal_fallback_move, reported_uci_move};
    use crate::uci_loop::GoParams;
    use xiangqi_core::{GameState, Move, STARTPOS_FEN};

    fn go_nodes(engine: &mut Engine, nodes: i32) {
        engine
            .go(&GoParams {
                nodes: Some(nodes),
                ..GoParams::default()
            })
            .expect("go");
        engine.wait().expect("search owner");
    }

    #[test]
    fn engine_graph_reuse_survives_cannon_knight_chase() {
        let weights = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/x7.onnx");
        if !std::path::Path::new(weights).is_file() {
            return;
        }
        let mut engine = Engine::new();
        engine.set_option("WeightsFile", weights).expect("weights option");
        let fen = "2bakc3/4a3n/4b4/2C1p4/P8/4P2cN/P8/4B1C2/4A4/4KAB2 b - - 0 1";
        let prefixes: [&[&str]; 8] = [
            &[],
            &["f9f4"],
            &["f9f4", "g2g4"],
            &["f9f4", "g2g4", "f4f3"],
            &["f9f4", "g2g4", "f4f3", "g4g2"],
            &["f9f4", "g2g4", "f4f3", "g4g2", "f3f4"],
            &["f9f4", "g2g4", "f4f3", "g4g2", "f3f4", "g2g4"],
            &["f9f4", "g2g4", "f4f3", "g4g2", "f3f4", "g2g4", "f4f3"],
        ];
        for (ply, moves) in prefixes.into_iter().enumerate() {
            let moves: Vec<String> = moves.iter().map(|mv| (*mv).to_string()).collect();
            engine
                .set_position(fen, &moves)
                .unwrap_or_else(|error| panic!("position at ply {ply}: {error}"));
            go_nodes(&mut engine, 2000);
        }
    }

    #[test]
    fn unsearched_root_reports_a_legal_move_not_null() {
        let history = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str])
            .expect("startpos")
            .position_history();
        let mv = reported_uci_move(None, &history, &[]);
        assert!(!mv.is_null());
        assert!(history.last().board().generate_legal_moves().contains(&mv));
    }

    #[test]
    fn searchmoves_fallback_stays_inside_the_filter() {
        let history = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str])
            .expect("startpos")
            .position_history();
        let filter = vec![history.last().board().parse_move("b2b3").expect("b2b3")];
        assert_eq!(legal_fallback_move(&history, &filter), filter[0]);
        assert_eq!(reported_uci_move(Some(Move::NULL), &history, &filter), filter[0]);
    }
}
