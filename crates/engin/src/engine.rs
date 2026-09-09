//! UCI Engine：拥有 graph、worker pool 与每次搜索 job。

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

use crate::neural::backend::{Backend, CachingBackend};
use crate::neural::onnx::OnnxBackend;
use crate::search::{
    NodeArena, NodeId, NoopObserver, Search, SearchConfig, SearchLimits, SearchParams, SearchTree, Stats, StopHandle,
    TimeBudget, TimeManager, WorkerPool, best_mate_with_params, best_move_filtered_with_params,
    principal_variation_with_params, root_stats, root_variations,
};
use crate::uci::{
    BestMoveInfo, GoParams, ThinkingInfo, Wdl, write_stdout, write_stdout_best_move, write_stdout_thinking,
};
use crate::{EnginError, Options};

/// UCI、图、worker 与单次搜索的唯一 owner。
pub struct Engine {
    // UCI 进程已启动时 ONNX 初始化仍可能失败；`None` 表示没有可用 backend，刻意不回退到 UniformBackend。
    backend: Option<Arc<dyn Backend>>,
    graph: Option<SearchTree>,
    graph_reaper: GraphReaper,
    worker_pool: Option<Arc<WorkerPool>>,
    applied_nn_cache_size: Option<u8>,
    time_manager: TimeManager,
    running: Option<ActiveSearch>,
    options: Options,
    backend_error: Option<String>,
    loaded_weights_file: Option<String>,
}

/// Engine 在 owner thread 运行期间保留的控制句柄。
struct ActiveSearch {
    stop: StopHandle,
    owner_thread: JoinHandle<(Duration, Result<(), EnginError>)>,
    clock_budget: Option<TimeBudget>,
}

/// owner thread 持有的一次完整搜索：运行、UCI info 与最终 bestmove。
struct SearchOwner {
    search: Search,
    snapshot: RootSnapshot,
    options: Options,
    limits: SearchLimits,
    started: Instant,
}

/// 后台回收旧 sibling 子树，或释放已被新 position 替换的整张 arena。
/// 前进换根后的 prune 只会在旧 job drain 后入队，可与下一手 `go` 重叠。
struct GraphReaper {
    sender: Option<crossbeam_channel::Sender<GraphCleanup>>,
    thread: Option<JoinHandle<()>>,
}

enum GraphCleanup {
    Prune(Arc<crate::search::NodeArena>, Vec<crate::search::NodeId>, Vec<crate::search::NodeId>),
    Retire(Arc<crate::search::NodeArena>),
}

impl GraphReaper {
    fn new() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded::<GraphCleanup>();
        let thread = thread::Builder::new()
            .name("engin-graph-reaper".into())
            .spawn(move || {
                while let Ok(cleanup) = receiver.recv() {
                    match cleanup {
                        GraphCleanup::Prune(arena, roots, nodes) => {
                            let _ = arena.remove_subtrees(roots);
                            arena.remove_nodes(nodes);
                        }
                        GraphCleanup::Retire(arena) => drop(arena),
                    }
                }
            })
            .expect("graph reaper thread starts");
        Self { sender: Some(sender), thread: Some(thread) }
    }

    fn retire(&self, arena: Arc<crate::search::NodeArena>) {
        self.sender
            .as_ref()
            .expect("graph reaper sender lives with engine")
            .send(GraphCleanup::Retire(arena))
            .expect("graph reaper thread is alive");
    }

    fn prune(
        &self,
        arena: Arc<crate::search::NodeArena>,
        roots: Vec<crate::search::NodeId>,
        nodes: Vec<crate::search::NodeId>,
    ) {
        if roots.is_empty() && nodes.is_empty() {
            return;
        }
        self.sender
            .as_ref()
            .expect("graph reaper sender lives with engine")
            .send(GraphCleanup::Prune(arena, roots, nodes))
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

/// 根 marker 不变时，仍重发 UCI info 的间隔。
const UCI_INFO_REPEAT_INTERVAL: Duration = Duration::from_secs(5);
/// owner 在 stream 等待在途 event 时，最多每 100ms 检查一次是否值得输出进度。
const OWNER_PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// search owner 持有的只读 root view，不参与 worker 的搜索状态。
#[derive(Clone)]
struct RootSnapshot {
    arena: Arc<NodeArena>,
    root_id: NodeId,
    root_history: Arc<PositionHistory>,
    initial_visits: u64,
    root_is_black: bool,
    root_move_filter: Vec<Move>,
    multi_pv: usize,
    params: SearchParams,
}

/// 上次 UCI info 已覆盖的根状态；不是搜索进度或统计快照。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InfoMarker {
    best_move: Option<Move>,
    depth: i32,
    seldepth: i32,
}

/// 上次输出的 marker 与时间。
#[derive(Default)]
struct LastInfo {
    marker: Option<InfoMarker>,
    time: i64,
}

impl Default for Engine {
    /// 构造空 Engine；不另设初始化路径。
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// UCI 启动时立即创建空 Engine；backend 在首次 `go` 时加载。
    pub fn new() -> Self {
        Self {
            backend: None,
            graph: None,
            graph_reaper: GraphReaper::new(),
            worker_pool: None,
            applied_nn_cache_size: None,
            time_manager: TimeManager::default(),
            running: None,
            options: Options::default(),
            backend_error: None,
            loaded_weights_file: None,
        }
    }

    /// 丢弃绑定旧 backend 的资源；树由 position 维护，换权重成功后才重建。
    fn clear_backend(&mut self) {
        self.backend = None;
        self.worker_pool = None;
        self.applied_nn_cache_size = None;
    }

    /// 只保留当前局面的 history，换权重后旧 evidence 不可复用。
    fn rebuild_tree(&mut self) {
        let history = self.graph.as_ref().map(|graph| Arc::clone(graph.root_history()));
        self.graph = history.map(SearchTree::new);
    }

    /// 在 `go` 边界按当前 option 更新 backend；换权重后旧 tree evidence 不再有效。
    fn update_backend(&mut self) {
        let path = self.options.weights_file.trim().to_string();
        if path.is_empty() {
            self.clear_backend();
            self.loaded_weights_file = None;
            self.backend_error = Some("WeightsFile is not configured".into());
            return;
        }
        if self.loaded_weights_file.as_ref() == Some(&path) && self.backend.is_some() {
            return;
        }

        self.clear_backend();
        match OnnxBackend::from_file(&path) {
            Ok(backend) => {
                self.backend = Some(Arc::new(CachingBackend::with_cache_size_power_of_two(
                    Box::new(backend),
                    self.options.nn_cache_size_power_of_two,
                )));
                self.applied_nn_cache_size = Some(self.options.nn_cache_size_power_of_two);
                self.loaded_weights_file = Some(path);
                self.backend_error = None;
                self.rebuild_tree();
            }
            Err(error) => {
                self.loaded_weights_file = None;
                self.backend_error = Some(format!("cannot load WeightsFile {path}: {error}"));
            }
        }
    }

    pub fn options(&self) -> &Options {
        &self.options
    }

    /// 更新 Engine 生命周期 option。已启动 job 使用自己创建时的 `SearchConfig` / `SearchParams` 快照。
    pub fn set_option(&mut self, name: &str, value: &str) -> Result<(), EnginError> {
        self.options.set_uci_option(name, value)
    }

    pub(crate) fn new_game(&mut self) -> Result<(), EnginError> {
        self.time_manager.reset();
        self.set_position(STARTPOS_FEN, &[])
    }

    /// 先解析 position，再停止旧 job；只按完整 history 复用或重置树。
    pub(crate) fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        let state = GameState::from_fen_moves(fen, moves)?;
        let history = Arc::new(state.position_history());
        self.abort_search();
        if let Some(graph) = self.graph.as_mut() {
            if let Some(retired) = graph.reset_to_history_after_drain(Arc::clone(&history))? {
                self.graph_reaper.retire(retired);
            }
            let (roots, nodes) = graph.take_pending_gc();
            self.graph_reaper.prune(Arc::clone(graph.arena()), roots, nodes);
        } else {
            self.graph = Some(SearchTree::new(history));
        }
        Ok(())
    }

    /// 检查 stream 已实现的 UCI `go` 子集；未支持项明确拒绝。
    fn validate_go(&self, params: &GoParams) -> Result<(), EnginError> {
        if params.depth.is_some() {
            return Err(EnginError::Uci("go depth is not supported".into()));
        }
        if params.mate.is_some() {
            return Err(EnginError::Uci("go mate is not supported".into()));
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
        if [params.wtime, params.btime, params.winc, params.binc].into_iter().flatten().any(|value| value < 0)
            || params.movestogo.is_some_and(|value| value <= 0)
        {
            return Err(EnginError::Uci("go clock values must be non-negative and movestogo positive".into()));
        }
        if has_clock {
            let root =
                self.graph.as_ref().ok_or(EnginError::Uci("position is not configured".into()))?.root_history().last();
            let side_time = if root.is_black_to_move() { params.btime } else { params.wtime };
            if side_time.is_none() {
                return Err(EnginError::Uci("go clock is missing side-to-move time".into()));
            }
        }
        if params.movetime.is_some() && has_clock {
            return Err(EnginError::Uci("go movetime cannot be combined with clock fields".into()));
        }
        if params.infinite && (params.nodes.is_some() || params.movetime.is_some() || has_clock) {
            return Err(EnginError::Uci("go infinite cannot be combined with nodes, movetime, or clock fields".into()));
        }
        if !params.infinite && params.nodes.is_none() && params.movetime.is_none() && !has_clock {
            return Err(EnginError::Uci("go requires nodes, movetime, clock fields, or infinite".into()));
        }
        Ok(())
    }

    /// `go searchmoves` 根着过滤。
    fn root_move_filter(&self, searchmoves: &[String]) -> Result<Vec<Move>, EnginError> {
        let graph = self.graph.as_ref().ok_or(EnginError::Uci("position is not configured".into()))?;
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

    fn search_config(&self) -> SearchConfig {
        SearchConfig {
            eval_batch_size: self.options.nn_batch_size,
            params: SearchParams {
                cpuct: self.options.cpuct,
                cpuct_base: self.options.cpuct_base,
                cpuct_factor: self.options.cpuct_factor,
                fpu_reduction: self.options.fpu_reduction,
                variance_bonus_scale: self.options.variance_bonus_scale,
                virtual_mean_fpu_scale: self.options.virtual_mean_fpu_scale,
                decision_lcb_stdevs: self.options.decision_lcb_stdevs,
                decision_ucb_stdevs: self.options.decision_ucb_stdevs,
                decision_rule: self.options.decision_rule,
                decision_mix_n_weight: self.options.decision_mix_n_weight,
            },
            nn_window: self.options.nn_window,
            threads: self.options.threads,
        }
    }

    fn ensure_worker_pool(&mut self, backend: &Arc<dyn Backend>, config: &SearchConfig) -> Arc<WorkerPool> {
        match self.worker_pool.as_ref() {
            Some(pool) if pool.matches_config(backend.as_ref(), config) => Arc::clone(pool),
            _ => {
                let pool = Arc::new(WorkerPool::new(backend.as_ref(), config));
                self.worker_pool = Some(Arc::clone(&pool));
                pool
            }
        }
    }

    /// 启动一个独占 job。worker pool 跨 job 常驻，图和配置均由 Engine 直接持有。
    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError> {
        self.validate_go(params)?;
        self.abort_search();
        self.update_backend();
        let Some(backend) = self.backend.as_ref().map(Arc::clone) else {
            self.report_missing_backend();
            return Ok(());
        };
        let root_move_filter = self.root_move_filter(&params.searchmoves)?;
        if self.applied_nn_cache_size != Some(self.options.nn_cache_size_power_of_two) {
            backend.set_cache_size_power_of_two(self.options.nn_cache_size_power_of_two);
            self.applied_nn_cache_size = Some(self.options.nn_cache_size_power_of_two);
        }
        let config = self.search_config();
        let decision_params = config.params;
        let pool = self.ensure_worker_pool(&backend, &config);
        let graph = self.graph.as_ref().expect("position creates a tree");
        let root_is_black = graph.root_history().last().is_black_to_move();
        let search = Search::start_with_pool(backend, graph, config, NoopObserver, pool);
        let snapshot = RootSnapshot {
            arena: Arc::clone(search.arena()),
            root_id: search.root_id(),
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
            root_move_filter,
        };
        let stop = search.stop_handle();
        let owner = SearchOwner { search, snapshot, options: self.options.clone(), limits, started };
        let owner_thread = thread::spawn(move || owner.run());
        self.running = Some(ActiveSearch { stop, owner_thread, clock_budget });
        Ok(())
    }

    pub(crate) fn go(&mut self, params: &GoParams) -> Result<(), EnginError> {
        if params.ponder {
            return Err(EnginError::Uci("Ponder is not enabled.".into()));
        }
        if self.graph.is_none() {
            self.new_game()?;
        }
        self.start_search(params)
    }

    /// 无法搜索时结束当前 `go`，但不伪造一个 legal decision。
    fn report_missing_backend(&self) {
        let reason = self.backend_error.as_deref().unwrap_or("WeightsFile is not configured");
        write_stdout(&[format!("info string cannot search: {reason}")]);
        write_stdout_best_move(&BestMoveInfo::new(Move::NULL));
    }

    pub(crate) fn ponder_hit(&mut self) -> Result<(), EnginError> {
        Err(EnginError::Uci("ponderhit while not pondering".into()))
    }

    /// 等待当前 search job 自然结束或完成 stop 后的 drain。
    pub(crate) fn wait_search(&mut self) -> Result<(), EnginError> {
        let Some(running) = self.running.take() else {
            return Ok(());
        };
        let ActiveSearch { owner_thread, clock_budget, .. } = running;
        let (elapsed, result) =
            owner_thread.join().map_err(|_| EnginError::Uci("search owner thread panicked".into()))?;
        if let Some(clock_budget) = clock_budget {
            self.time_manager.finish(clock_budget, elapsed);
        }
        result
    }

    /// 请求停止当前 search job，并等待其输出 bestmove。
    pub(crate) fn stop_search(&mut self) -> Result<(), EnginError> {
        if let Some(running) = &self.running {
            running.stop.request_stop();
        }
        self.wait_search()
    }

    /// 替换 position / backend / 新 `go` 前停止并 join 上一次 search。
    fn abort_search(&mut self) {
        if let Some(running) = &self.running {
            running.stop.request_stop();
        }
        if let Err(error) = self.wait_search() {
            eprintln!("info string abort drain ignored previous search error: {error}");
        }
    }
}

impl SearchOwner {
    fn run(mut self) -> (Duration, Result<(), EnginError>) {
        let mut last_info = LastInfo::default();
        let snapshot = &self.snapshot;
        let options = &self.options;
        let started = self.started;
        let limits = std::mem::take(&mut self.limits);
        let result = self.search.run_with_report(limits, Some(OWNER_PROGRESS_INTERVAL), |stats| {
            let time = started.elapsed().as_millis() as i64;
            let marker = snapshot.info_marker(&stats);
            if last_info.should_publish(marker, time) {
                let infos = snapshot.thinking_infos(stats, started);
                write_stdout_thinking(&infos, options);
                last_info.update(marker, time);
            }
        });
        // run 已 drain event；此处归还 worker 后，下面只读 graph 并输出最终结果。
        self.search.finish();
        match &result {
            Ok(stats) => self.publish_result(stats),
            Err(error) => self.publish_error(error),
        }
        (self.started.elapsed(), result.map(|_| ()))
    }

    fn publish_result(&self, stats: &Stats) {
        // root 终局：不从旧图 edge 选着。GUI 未必实现完整规则（重复/rule60），
        // UCI 仍可用 legal fallback 回一着；将死无着才是 a0a0。
        let (chosen, principal_variation) = if self.search.root_is_terminal() {
            (None, Vec::new())
        } else {
            (
                best_move_filtered_with_params(
                    &self.snapshot.arena,
                    self.snapshot.root_id,
                    self.snapshot.root_is_black,
                    &self.snapshot.root_move_filter,
                    &self.snapshot.params,
                ),
                principal_variation_with_params(
                    &self.snapshot.arena,
                    self.snapshot.root_id,
                    self.snapshot.root_is_black,
                    &self.snapshot.root_move_filter,
                    &self.snapshot.params,
                ),
            )
        };
        let mut infos = self.snapshot.thinking_infos(stats.clone(), self.started);
        if let Some(info) = infos.first_mut() {
            info.pv = principal_variation;
        }
        let best_move = reported_uci_move(chosen, self.snapshot.root_history.as_ref(), &self.snapshot.root_move_filter);
        self.write_finished(&infos, best_move);
    }

    fn publish_error(&self, error: &EnginError) {
        let info = ThinkingInfo { comment: format!("stream search failed: {error}"), ..ThinkingInfo::default() };
        let best_move = reported_uci_move(None, self.snapshot.root_history.as_ref(), &self.snapshot.root_move_filter);
        self.write_finished(&[info], best_move);
    }

    fn write_finished(&self, infos: &[ThinkingInfo], best_move: Move) {
        write_stdout_thinking(infos, &self.options);
        write_stdout_best_move(&BestMoveInfo::new(best_move));
    }
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
    /// 只在此 marker 变化时构造新的周期性 UCI info。
    fn info_marker(&self, stats: &Stats) -> InfoMarker {
        InfoMarker {
            best_move: best_move_filtered_with_params(
                &self.arena,
                self.root_id,
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
        let nps = if time == 0 { 0 } else { (stats.completed_playouts as i64 * 1000 / time) as i32 };
        let eps = if time == 0 { 0 } else { (stats.network_evaluations as i64 * 1000 / time) as i32 };
        let common = ThinkingInfo {
            depth: stats.average_depth.min(i32::MAX as u64) as i32,
            seldepth: stats.max_depth.min(i32::MAX as u64) as i32,
            time,
            nodes,
            nps,
            eps,
            ..ThinkingInfo::default()
        };
        let Some(root) = root_stats(&self.arena, self.root_id) else {
            return vec![common];
        };
        let wl = (-root.q).clamp(-1.0, 1.0);
        let draw = root.draw.clamp(0.0, 1.0);
        let variations = root_variations(
            &self.arena,
            self.root_id,
            self.root_is_black,
            &self.root_move_filter,
            self.multi_pv,
            &self.params,
        );
        if variations.is_empty() {
            let win = ((1.0_f32 - draw + wl) * 0.5).clamp(0.0, 1.0);
            let loss = ((1.0_f32 - draw - wl) * 0.5).clamp(0.0, 1.0);
            let mate = best_mate_with_params(&self.arena, self.root_id, &self.root_move_filter, &self.params);
            return vec![ThinkingInfo {
                mate,
                score: mate.is_none().then_some((wl * 1000.0).round() as i32),
                wdl: Some(Wdl {
                    w: (win * 1000.0).round() as i32,
                    d: (draw * 1000.0).round() as i32,
                    l: (loss * 1000.0).round() as i32,
                }),
                pv: principal_variation_with_params(
                    &self.arena,
                    self.root_id,
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
                let win = ((1.0_f32 - variation.draw + variation.wl) * 0.5).clamp(0.0, 1.0);
                let loss = ((1.0_f32 - variation.draw - variation.wl) * 0.5).clamp(0.0, 1.0);
                ThinkingInfo {
                    mate: variation.mate,
                    score: variation.mate.is_none().then_some((variation.wl * 1000.0).round() as i32),
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

impl LastInfo {
    /// 周期性 info 输出门槛。
    fn should_publish(&self, marker: InfoMarker, time: i64) -> bool {
        marker.best_move.is_some()
            && (self.marker != Some(marker)
                || time.saturating_sub(self.time) > UCI_INFO_REPEAT_INTERVAL.as_millis() as i64)
    }

    fn update(&mut self, marker: InfoMarker, time: i64) {
        self.marker = Some(marker);
        self.time = time;
    }
}

#[cfg(test)]
mod tests {
    use super::{legal_fallback_move, reported_uci_move};
    use xiangqi_core::{GameState, Move, STARTPOS_FEN};

    #[test]
    fn unsearched_root_reports_a_legal_move_not_null() {
        let history = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos").position_history();
        let mv = reported_uci_move(None, &history, &[]);
        assert!(!mv.is_null());
        assert!(history.last().board().generate_legal_moves().contains(&mv));
    }

    #[test]
    fn searchmoves_fallback_stays_inside_the_filter() {
        let history = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos").position_history();
        let filter = vec![history.last().board().parse_move("b2b3").expect("b2b3")];
        assert_eq!(legal_fallback_move(&history, &filter), filter[0]);
        assert_eq!(reported_uci_move(Some(Move::NULL), &history, &filter), filter[0]);
    }
}
