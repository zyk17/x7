//! UCI Engine：拥有 graph、worker pool 与每次搜索 job。

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use xiangqi_core::{GameState, Move, STARTPOS_FEN};

use crate::neural::backend::{Backend, CachingBackend, UniformBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::{
    Search, SearchConfig, SearchControl, SearchGeneration, SearchGraph, SearchLimits, SearchParams, Stats, TimeBudget,
    TimeManager, WorkerPool, best_mate, best_move_filtered, principal_variation_filtered, root_stats, root_variations,
};
use crate::uci_loop::{BestMoveInfo, GoParams, StringUciResponder, ThinkingInfo, UciOutputQueue, Wdl};
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
    output: Option<Arc<UciOutputQueue>>,
}

struct ActiveSearch {
    control: SearchControl,
    completion: Arc<Completion>,
    publish_output: Arc<Mutex<bool>>,
    search_thread: JoinHandle<()>,
    watchdog_thread: JoinHandle<()>,
    started: Instant,
    clock_budget: Option<TimeBudget>,
}

/// 后台回收旧 sibling 图或已被新 position 替换的整张 graph，不占用下一次 `go` 的 UCI 线程。
///
/// 走子后的 prune 与新搜索共享 repository，但由 repository topology 锁保证绑定安全；
/// 无关 position 的 retire 则不与活跃图共享任何 node 或锁。
struct GraphReaper {
    sender: Option<crossbeam_channel::Sender<GraphCleanup>>,
    thread: Option<JoinHandle<()>>,
}

enum GraphCleanup {
    Prune(Arc<crate::search::NodeRepository>, crate::search::NodeKey),
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
                        GraphCleanup::Prune(repository, root) => {
                            repository.retain_from_root(root);
                        }
                        // 正常 reset 已 drain；若还有外部只读 Arc，则保守地交给它最后一次
                        // drop。常规路径下能独占旧图，按 shard 低干扰释放。
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

    fn prune(&self, repository: Arc<crate::search::NodeRepository>, root: crate::search::NodeKey) {
        self.sender
            .as_ref()
            .expect("graph reaper sender lives with engine")
            .send(GraphCleanup::Prune(repository, root))
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

/// 已完成 job 的最小结果；只在 Engine 的两个线程之间交接。
#[derive(Clone, Debug, PartialEq)]
struct CompletedSearch {
    stats: Stats,
    best_move: Option<Move>,
    principal_variation: Vec<Move>,
}

/// watchdog 持有的只读 root view，不参与 worker 的搜索状态。
#[derive(Clone)]
struct RootSnapshot {
    repository: Arc<crate::search::NodeRepository>,
    root_key: crate::search::NodeKey,
    initial_visits: u64,
    root_is_black: bool,
    root_move_filter: Vec<Move>,
    multi_pv: usize,
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

#[derive(Default)]
struct Completion {
    result: Mutex<Option<Result<CompletedSearch, EnginError>>>,
    ready: Condvar,
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
            output: None,
        }
    }

    /// 测试构造器共用的 backend 安装路径；正式 UCI 仍只经 `WeightsFile`。
    fn with_backend(backend: Box<dyn Backend>) -> Self {
        let mut engine = Self::new();
        engine.manages_weights_file = false;
        engine.install_backend(Arc::from(backend));
        engine
    }

    /// 显式 ONNX 构造，正式 UCI 启动仍经 `new` + `WeightsFile`。
    pub fn from_onnx_file(path: impl AsRef<std::path::Path>) -> Result<Self, EnginError> {
        Ok(Self::with_backend(Box::new(CachingBackend::new(Box::new(
            OnnxBackend::from_file(path)?,
        )))))
    }

    /// 确定性测试 backend，不参与正式 UCI 的 `WeightsFile` 生命周期。
    pub fn uniform() -> Self {
        Self::with_backend(Box::new(UniformBackend::default()))
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
                self.graph_reaper.prune(Arc::clone(graph.repository()), root);
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
            let position = self
                .position
                .as_ref()
                .ok_or(EnginError::Uci("position is not configured".into()))?;
            let side_time = if position.current_position().is_black_to_move() {
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
            },
            gather_workers: self.options.gather_workers,
            eval_workers: self.options.eval_workers,
            backprop_workers: self.options.backprop_workers,
            ..SearchConfig::default()
        };
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
        let search =
            Search::new_with_graph_in_pool(backend, SearchGeneration(self.next_generation), graph, config, pool);
        let snapshot = RootSnapshot {
            repository: Arc::clone(search.repository()),
            root_key: search.root_key(),
            initial_visits: search.initial_visits(),
            root_is_black,
            root_move_filter: root_move_filter.clone(),
            multi_pv: self.options.multi_pv,
        };
        let started = Instant::now();
        let clock_budget = if params.movetime.is_none() {
            let position = self
                .position
                .as_ref()
                .expect("validated search position")
                .current_position();
            self.time_manager.budget(params, &position)
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
        let completion = Arc::new(Completion::default());
        let search_completion = Arc::clone(&completion);
        let search_thread = std::thread::spawn(move || {
            let result = run_search(search, root_is_black, root_move_filter, limits);
            *search_completion.result.lock() = Some(result);
            search_completion.ready.notify_all();
        });
        let publish_output = Arc::new(Mutex::new(true));
        let watchdog_thread = std::thread::spawn({
            let completion = Arc::clone(&completion);
            let publish_output = Arc::clone(&publish_output);
            let output = self.output.clone();
            let watchdog_control = control.clone();
            move || watchdog(watchdog_control, snapshot, completion, publish_output, output, started)
        });
        self.active = Some(ActiveSearch {
            control,
            completion,
            publish_output,
            search_thread,
            watchdog_thread,
            started,
            clock_budget,
        });
        Ok(())
    }

    pub(crate) fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        if params.ponder {
            return Err(EnginError::Uci("Ponder is not enabled.".into()));
        }
        if self.position.is_none() {
            self.new_game()?;
        }
        if self.backend.is_none() {
            let reason = self.backend_error.as_deref().unwrap_or("WeightsFile is not configured");
            responder.send_raw_response(&format!("info string cannot search: {reason}"));
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
            completion,
            search_thread,
            watchdog_thread,
            started,
            clock_budget,
            ..
        } = active;
        loop {
            let mut result = completion.result.lock();
            if result.is_some() {
                break;
            }
            completion.ready.wait(&mut result);
        }
        let _ = search_thread.join();
        let _ = watchdog_thread.join();
        let result = completion.result.lock().take().expect("completed search has a result");
        if let Some(clock_budget) = clock_budget {
            self.time_manager.finish(clock_budget, started.elapsed());
        }
        result.map(|_| ())
    }

    pub(crate) fn stop(&mut self) -> Result<(), EnginError> {
        if let Some(active) = &self.active {
            active.control.request_stop();
        }
        self.wait()
    }

    /// 替换 position 或 backend 时取消输出但完整 drain reservation。
    fn abort(&mut self) -> Result<(), EnginError> {
        if let Some(active) = &self.active {
            *active.publish_output.lock() = false;
            active.control.request_stop();
        }
        self.wait()
    }

    /// UCI loop 安装自己的输出队列；Engine 不提供嵌入式 library callback。
    pub(crate) fn set_output_queue(&mut self, output: Option<Arc<UciOutputQueue>>) {
        self.output = output;
    }
}

/// 单次 job 的 owner 运行、取 root 结果并在退出前归还 worker。参考 LC3 Overview 的 "Search"。
fn run_search(
    mut search: Search,
    root_is_black: bool,
    root_move_filter: Vec<Move>,
    limits: SearchLimits,
) -> Result<CompletedSearch, EnginError> {
    let stats = search.run_with_limits(limits)?;
    // path-local repetition/rule60 不能标记共享 board node，但对这次 UCI root 已是
    // 真正终局；不得从旧图的 edge 回退出一着看似合法的棋。
    let (best_move, principal_variation) = if search.root_is_path_terminal() {
        (None, Vec::new())
    } else {
        (
            best_move_filtered(search.repository(), search.root_key(), root_is_black, &root_move_filter),
            principal_variation_filtered(search.repository(), search.root_key(), root_is_black, &root_move_filter),
        )
    };
    search.stop_and_finish();
    Ok(CompletedSearch {
        stats,
        best_move,
        principal_variation,
    })
}

impl RootSnapshot {
    /// info 输出门槛：marker 未变化时不生成 PV/MultiPV。
    fn progress(&self, stats: &Stats) -> SearchProgress {
        SearchProgress {
            best_move: best_move_filtered(
                &self.repository,
                self.root_key,
                self.root_is_black,
                &self.root_move_filter,
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
            self.root_is_black,
            &self.root_move_filter,
            self.multi_pv,
        );
        if variations.is_empty() {
            let win = ((1.0 - draw + wl) * 0.5).clamp(0.0, 1.0);
            let loss = ((1.0 - draw - wl) * 0.5).clamp(0.0, 1.0);
            let mate = best_mate(&self.repository, self.root_key, &self.root_move_filter);
            return vec![ThinkingInfo {
                mate,
                score: mate.is_none().then_some((wl * 1000.0).round() as i32),
                wdl: Some(Wdl {
                    w: (win * 1000.0).round() as i32,
                    d: (draw * 1000.0).round() as i32,
                    l: (loss * 1000.0).round() as i32,
                }),
                pv: principal_variation_filtered(
                    &self.repository,
                    self.root_key,
                    self.root_is_black,
                    &self.root_move_filter,
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

/// LC3 Overview 的 WatchdogWorker：只输出，搜索仍只由 Gather/Eval/NN/Backprop 负责。
fn watchdog(
    control: SearchControl,
    snapshot: RootSnapshot,
    completion: Arc<Completion>,
    publish_output: Arc<Mutex<bool>>,
    output: Option<Arc<UciOutputQueue>>,
    started: Instant,
) {
    let mut published = PublishedInfo::default();
    loop {
        let mut result = completion.result.lock();
        if result.is_none() {
            // stream 可以形成单项 NN batch；最多每 100ms 合并一次，避免 UCI/PV 格式化影响吞吐。
            completion.ready.wait_for(&mut result, Duration::from_millis(100));
        }
        if let Some(result) = result.as_ref() {
            if let Some(output) = output.as_ref() {
                if !*publish_output.lock() {
                    return;
                }
                match result {
                    Ok(result) => {
                        let mut infos = snapshot.thinking_infos(result.stats.clone(), started);
                        if let Some(info) = infos.first_mut() {
                            info.pv = result.principal_variation.clone();
                        }
                        output.push_thinking_info(infos);
                        output.push_best_move(BestMoveInfo::new(result.best_move.unwrap_or(Move::NULL)));
                    }
                    Err(error) => output.push_thinking_info(vec![ThinkingInfo {
                        comment: format!("stream search failed: {error}"),
                        ..ThinkingInfo::default()
                    }]),
                }
            }
            return;
        }
        drop(result);
        if let Some(output) = output.as_ref() {
            if !*publish_output.lock() {
                return;
            }
            let stats = control.stats();
            let time = started.elapsed().as_millis() as i64;
            let progress = snapshot.progress(&stats);
            if published.should_publish(progress, time) {
                output.push_thinking_info(snapshot.thinking_infos(stats, started));
                published.update(progress, time);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use xiangqi_core::initialize_magic_bitboards;

    use super::Engine;
    use crate::GoParams;

    static INIT: Once = Once::new();

    #[test]
    fn engine_reuses_graph_for_the_same_root_node_budget() {
        INIT.call_once(initialize_magic_bitboards);
        let mut engine = Engine::uniform();
        engine.set_position(xiangqi_core::STARTPOS_FEN, &[]).expect("position");
        engine
            .start_search(&GoParams {
                nodes: Some(8),
                ..GoParams::default()
            })
            .expect("first search");
        engine.wait().expect("first wait");
        let root = engine.graph.as_ref().expect("graph").root_key();
        let visits = engine
            .graph
            .as_ref()
            .expect("graph")
            .repository()
            .get(root)
            .expect("root")
            .completed_visits();
        engine
            .start_search(&GoParams {
                nodes: Some(8),
                ..GoParams::default()
            })
            .expect("same root");
        engine.wait().expect("same root wait");
        assert_eq!(
            engine
                .graph
                .as_ref()
                .expect("graph")
                .repository()
                .get(root)
                .expect("root")
                .completed_visits(),
            visits
        );
    }
}
