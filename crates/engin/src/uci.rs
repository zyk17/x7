//! 中国象棋 UCI 子集：`uci` / `isready` / `setoption` / `position` / `go` / `stop` / `quit`。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use xiangqi_core::{uci_to_move, Color, Move, Position, START_FEN};

use crate::benchmark::resolve_data_file;
use crate::history::PositionHistory;
use crate::mcts::{
    allocate_think_time_ms, MctsBudget, MctsConfig, MctsEngine, MctsSearchProgress, MctsTree,
    OnnxPolicyValueEval, SharedPolicy, SearchStats, UciTimeParams, LC0_DEFAULT_MOVE_OVERHEAD_MS,
};
use crate::policy_onnx::{resolved_search_threads, PolicySessionPool};

static UCI_STDOUT_LOCK: Mutex<()> = Mutex::new(());
const UCI_INFO_INTERVAL: Duration = Duration::from_millis(200);
const AUTO_THREADS: usize = 0;
const MAX_THREADS: usize = 128;

fn q_to_cp(q: f32) -> i32 {
    let wl = q.clamp(-0.999, 0.999) as f64;
    (90.0 * (1.5637541897 * wl).tan()).round() as i32
}

struct UciInfoView<'a> {
    best_value: f32,
    best_mate: Option<i32>,
    depth: u32,
    seldepth: u32,
    playouts: u32,
    nodes: usize,
    elapsed_ms: u64,
    nps_elapsed_ms: u64,
    pv: &'a [xiangqi_core::Move],
    multipv: Option<u32>,
}

fn uci_info_line(view: UciInfoView<'_>) -> String {
    let UciInfoView {
        best_value,
        best_mate,
        depth,
        seldepth,
        playouts,
        nodes,
        elapsed_ms,
        nps_elapsed_ms,
        pv,
        multipv,
    } = view;
    let nps = SearchStats::playouts_per_second(playouts, nps_elapsed_ms);
    let score = if let Some(mate) = best_mate {
        format!("score mate {mate}")
    } else {
        let cp = q_to_cp(best_value);
        format!("score cp {cp}")
    };
    // lc0 `StringUciResponder::OutputThinkingInfo` (uciloop.cc:305-329)
    let depth = depth.max(1);
    let mut line = format!(
        "info depth {depth} seldepth {seldepth} time {elapsed_ms} nodes {nodes} {score} nps {nps}"
    );
    if let Some(n) = multipv {
        line.push_str(&format!(" multipv {n}"));
    }
    if !pv.is_empty() {
        line.push_str(" pv ");
        line.push_str(
            &pv.iter()
                .map(|mv| xiangqi_core::move_to_uci(*mv))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    line
}

fn emit_mcts_progress(output: &Arc<dyn Fn(&str) + Send + Sync>, progress: &MctsSearchProgress, elapsed_ms: u64) {
    if progress.playouts == 0 {
        return;
    }
    let common = |best_value: f32, best_mate: Option<i32>, pv: &[Move], multipv: Option<u32>| {
        uci_info_line(UciInfoView {
            best_value,
            best_mate,
            depth: progress.depth,
            seldepth: progress.seldepth,
            playouts: progress.playouts,
            nodes: progress.nodes,
            elapsed_ms,
            nps_elapsed_ms: progress.nps_elapsed_ms,
            pv,
            multipv,
        })
    };

    if progress.multi_pv > 1 && !progress.pv_lines.is_empty() {
        for line in &progress.pv_lines {
            output(&common(
                line.best_value,
                line.best_mate,
                &line.pv,
                Some(line.multipv),
            ));
        }
    } else {
        output(&common(
            progress.best_value,
            progress.best_mate,
            &progress.pv,
            None,
        ));
    }
    if progress.retry_without_playout > 0 {
        output(&format!(
            "info string retry_without_playout {}",
            progress.retry_without_playout
        ));
    }
}

type EngineOutput = Arc<dyn Fn(&str) + Send + Sync + 'static>;
type SharedSearchEngine = Arc<Mutex<MctsEngine<OnnxPolicyValueEval>>>;

fn uci_out_line(line: &str) {
    let _g = UCI_STDOUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    println!("{line}");
}

#[derive(Debug, Clone, Default)]
struct GoParams {
    infinite: bool,
    ponder: bool,
    depth: Option<u32>,
    movetime: Option<u64>,
    nodes: Option<u64>,
    mate: Option<u32>,
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movestogo: Option<u32>,
    searchmoves: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct GoParseOutcome {
    params: GoParams,
    warnings: Vec<String>,
}

const GO_KEYWORDS: &[&str] = &[
    "infinite",
    "ponder",
    "searchmoves",
    "wtime",
    "btime",
    "winc",
    "binc",
    "movestogo",
    "depth",
    "mate",
    "nodes",
    "movetime",
];

fn is_go_keyword(token: &str) -> bool {
    GO_KEYWORDS.contains(&token)
}

fn parse_go(args: &str) -> GoParseOutcome {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let mut params = GoParams::default();
    let mut warnings = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        let tok = tokens[i];
        match tok {
            "infinite" => {
                params.infinite = true;
                i += 1;
            }
            "ponder" => {
                params.ponder = true;
                i += 1;
            }
            "searchmoves" => {
                i += 1;
                while i < tokens.len() && !is_go_keyword(tokens[i]) {
                    params.searchmoves.push(tokens[i].to_string());
                    i += 1;
                }
            }
            "wtime" | "btime" | "winc" | "binc" | "movestogo" | "mate" => {
                i += 1;
                if i < tokens.len() && !is_go_keyword(tokens[i]) {
                    match tok {
                        "wtime" => params.wtime = tokens[i].parse().ok(),
                        "btime" => params.btime = tokens[i].parse().ok(),
                        "winc" => params.winc = tokens[i].parse().ok(),
                        "binc" => params.binc = tokens[i].parse().ok(),
                        "movestogo" => params.movestogo = tokens[i].parse().ok(),
                        "mate" => params.mate = tokens[i].parse().ok(),
                        _ => {}
                    }
                    i += 1;
                }
            }
            "depth" => {
                i += 1;
                if i < tokens.len() {
                    if let Ok(n) = tokens[i].parse::<u32>() {
                        params.depth = Some(n);
                    } else {
                        warnings.push(format!("invalid go depth value: {}", tokens[i]));
                    }
                    i += 1;
                }
            }
            "movetime" => {
                i += 1;
                if i < tokens.len() {
                    if let Ok(n) = tokens[i].parse::<u64>() {
                        params.movetime = Some(n);
                    } else {
                        warnings.push(format!("invalid go movetime value: {}", tokens[i]));
                    }
                    i += 1;
                }
            }
            "nodes" => {
                i += 1;
                if i < tokens.len() {
                    if let Ok(n) = tokens[i].parse::<u64>() {
                        params.nodes = Some(n);
                    } else {
                        warnings.push(format!("invalid go nodes value: {}", tokens[i]));
                    }
                    i += 1;
                }
            }
            other => {
                warnings.push(format!("unknown go token: {other}"));
                i += 1;
            }
        }
    }

    GoParseOutcome { params, warnings }
}

fn time_params_from_go(params: &GoParams) -> UciTimeParams {
    UciTimeParams {
        infinite: params.infinite,
        ponder: params.ponder,
        movetime: params.movetime,
        wtime: params.wtime,
        btime: params.btime,
        winc: params.winc,
        binc: params.binc,
        movestogo: params.movestogo,
    }
}

fn history_for_go(history: &PositionHistory, ponder: bool) -> PositionHistory {
    if !ponder || history.game_moves().is_empty() {
        return history.clone_for_search();
    }
    let mut h = PositionHistory::from_position(history.game_start().clone_for_search());
    for &mv in history.game_moves().iter().take(history.game_moves().len().saturating_sub(1)) {
        h.push_move(mv);
    }
    h
}

fn resolve_searchmoves(history: &PositionHistory, uci_moves: &[String]) -> Result<Vec<Move>, String> {
    if uci_moves.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(uci_moves.len());
    for uci in uci_moves {
        let mv = uci_to_move(history.current(), uci)
            .ok_or_else(|| format!("非法 searchmove: {uci}"))?;
        out.push(mv);
    }
    Ok(out)
}

fn parse_setoption(line: &str) -> Option<(String, Option<String>)> {
    let rest = line.strip_prefix("setoption")?.trim();
    let rest = rest.strip_prefix("name")?.trim();
    if let Some(idx) = rest.find(" value ") {
        let name = rest[..idx].trim().to_string();
        let val = rest[idx + " value ".len()..].to_string();
        return Some((name, Some(val)));
    }
    Some((rest.trim().to_string(), None))
}

pub fn parse_position_uci(line: &str) -> Result<Position, String> {
    Ok(parse_position_history(line)?.current().clone_for_search())
}

pub fn parse_position_history_uci(line: &str) -> Result<PositionHistory, String> {
    parse_position_history(line)
}

fn parse_position_history(line: &str) -> Result<PositionHistory, String> {
    let rest = line
        .strip_prefix("position")
        .ok_or_else(|| "internal".to_string())?
        .trim();
    let (mut history, moves_tail) = if let Some(tail) = rest.strip_prefix("startpos") {
        let tail = tail.trim_start();
        let moves_tail = parse_moves_suffix(tail)?;
        (
            PositionHistory::from_position(Position::from_fen(START_FEN).map_err(|e| e.to_string())?),
            moves_tail,
        )
    } else if let Some(fen_part) = rest.strip_prefix("fen") {
        let fen_part = fen_part.trim();
        let (fen_str, moves_tail) = split_fen_and_moves(fen_part)?;
        (
            PositionHistory::from_position(Position::from_fen(fen_str.trim()).map_err(|e| e.to_string())?),
            moves_tail,
        )
    } else {
        return Err("position 需要 startpos 或 fen".into());
    };

    if let Some(mvs) = moves_tail {
        for mv in mvs.split_whitespace() {
            let Some(m) = uci_to_move(history.current(), mv) else {
                return Err(format!("非法或不可执行着法: {mv}"));
            };
            history.push_move(m);
        }
    }
    Ok(history)
}

fn parse_moves_suffix(tail: &str) -> Result<Option<String>, String> {
    let tail = tail.trim_start();
    if tail.is_empty() {
        return Ok(None);
    }
    let tail = tail
        .strip_prefix("moves")
        .ok_or_else(|| "position startpos 后应为 moves".to_string())?;
    let s = tail.trim();
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s.to_string()))
    }
}

fn split_fen_and_moves(s: &str) -> Result<(&str, Option<String>), String> {
    let s = s.trim();
    if let Some(i) = s.find(" moves ") {
        let fen = &s[..i];
        let rest = s[i + " moves ".len()..].trim();
        if rest.is_empty() {
            Ok((fen, None))
        } else {
            Ok((fen, Some(rest.to_string())))
        }
    } else {
        Ok((s, None))
    }
}

struct Engine {
    history: PositionHistory,
    policy: SharedPolicy,
    policy_path: Option<PathBuf>,
    config: MctsConfig,
    threads: usize,
    search_engine: SharedSearchEngine,
    default_playouts: u32,
    search_stop: Option<Arc<AtomicBool>>,
    search_join: Option<JoinHandle<()>>,
    last_go: Option<GoParseOutcome>,
    output: EngineOutput,
}

impl Engine {
    fn new() -> Self {
        Self::new_with_output(Arc::new(uci_out_line))
    }

    fn new_with_output(output: EngineOutput) -> Self {
        let policy_path = resolve_data_file("x7.onnx");
        let policy = policy_path
            .as_ref()
            .and_then(|path| PolicySessionPool::from_file(path).ok())
            .map(Arc::new);
        Self {
            history: PositionHistory::new_startpos(),
            policy: policy.clone(),
            policy_path,
            config: MctsConfig::default(),
            threads: AUTO_THREADS,
            search_engine: Arc::new(Mutex::new(MctsEngine::new(
                MctsConfig::default(),
                OnnxPolicyValueEval::new(policy, MctsConfig::default().nn_cache_size),
            ))),
            default_playouts: 256,
            search_stop: None,
            search_join: None,
            last_go: None,
            output,
        }
    }

    fn emit(&self, line: &str) {
        (self.output)(line);
    }

    fn stop_and_join(&mut self) {
        if let Some(s) = self.search_stop.take() {
            s.store(true, Ordering::SeqCst);
        }
        if let Some(h) = self.search_join.take() {
            let _ = h.join();
        }
    }

    /// lc0 `Engine::NewGame` (engine.cc:226-229) + `SetPosition(startpos)`.
    fn new_game(&mut self) {
        self.stop_and_join();
        {
            let mut mcts = self
                .search_engine
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            mcts.evaluator.clear_cache();
            mcts.tree.clear();
            mcts.root_id = None;
            mcts.root_history = None;
        }
        self.history = PositionHistory::new_startpos();
    }

    fn reload_policy(&mut self) -> Result<(), String> {
        self.policy = None;
        if let Some(ref p) = self.policy_path {
            if p.is_file() {
                let pool = PolicySessionPool::from_file(p).map_err(|e| e.to_string())?;
                self.policy = Some(Arc::new(pool));
            }
        }
        self.rebuild_search_engine(false);
        Ok(())
    }

    /// `preserve_tree=false`：换权重后必须丢弃旧树（prior/Q/visit 来自旧模型）。
    fn rebuild_search_engine(&mut self, preserve_tree: bool) {
        let mut engine = self
            .search_engine
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let preserved = if preserve_tree {
            (
                std::mem::take(&mut engine.tree),
                engine.root_id,
                engine.root_history.take(),
            )
        } else {
            (MctsTree::new(), None, None)
        };
        drop(engine);
        let mut fresh = MctsEngine::new(
            self.config,
            OnnxPolicyValueEval::new(self.policy.clone(), self.config.nn_cache_size),
        );
        if preserve_tree {
            fresh.tree = preserved.0;
            fresh.root_id = preserved.1;
            fresh.root_history = preserved.2;
        }
        self.search_engine = Arc::new(Mutex::new(fresh));
    }

    fn resolved_threads(&self) -> usize {
        if self.threads == AUTO_THREADS {
            self.policy
                .as_ref()
                .map(|pool| resolved_search_threads(0, &pool.backend_attributes()))
                .unwrap_or(1)
        } else {
            self.threads.clamp(1, MAX_THREADS)
        }
    }

    fn send_uci_ident(&self) {
        self.emit("id name 77xiangqi_engine");
        self.emit("id author github.com/77xiangqi_engine");
        let policy_default = self
            .policy_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<empty>".to_string());
        self.emit(&format!("option name PolicyFile type string default {policy_default}"));
        self.emit("option name MctsPlayouts type spin default 256 min 1 max 1000000");
        self.emit("option name CPuct type string default 1.745");
        self.emit("option name FpuValue type string default 0.330");
        self.emit("option name MinibatchSize type spin default 0 min 0 max 1024");
        self.emit("option name NNCacheSize type spin default 200000 min 0 max 10000000");
        self.emit("option name Threads type spin default 0 min 0 max 128");
        self.emit("option name MultiPV type spin default 1 min 1 max 500");
        self.emit(&format!(
            "info string policy {}",
            if self.policy.is_some() { "loaded" } else { "not_loaded" }
        ));
        self.emit("uciok");
    }

    fn handle_setoption(&mut self, line: &str) -> Result<(), String> {
        let Some((name, value)) = parse_setoption(line) else {
            return Ok(());
        };
        match name.as_str() {
            "PolicyFile" => {
                self.policy_path = value
                    .as_ref()
                    .map(|s| PathBuf::from(s.trim()))
                    .filter(|p| !p.as_os_str().is_empty());
                self.reload_policy()?;
            }
            "MctsPlayouts" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<u32>() {
                        self.default_playouts = n.clamp(1, 1_000_000);
                    }
                }
            }
            "CPuct" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<f32>() {
                        let cpuct = n.clamp(0.01, 100.0);
                        self.config.cpuct = cpuct;
                        self.config.cpuct_root = cpuct;
                        self.rebuild_search_engine(true);
                    }
                }
            }
            "FpuValue" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<f32>() {
                        let fpu = n.clamp(0.0, 2.0);
                        self.config.fpu_reduction = fpu;
                        self.config.fpu_reduction_root = fpu;
                        self.rebuild_search_engine(true);
                    }
                }
            }
            "MinibatchSize" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<i32>() {
                        self.config.minibatch_size = n.clamp(0, 1024);
                        self.rebuild_search_engine(true);
                    }
                }
            }
            "NNCacheSize" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<usize>() {
                        self.config.nn_cache_size = n.min(10_000_000);
                        self.rebuild_search_engine(false);
                    }
                }
            }
            "Threads" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<usize>() {
                        self.threads = n.min(MAX_THREADS);
                    }
                }
            }
            "MultiPV" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<u32>() {
                        self.config.multi_pv = n.clamp(1, 500);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// lc0 `PopulateCommonUciStoppers`（stoppers/common.cc:118-165）：多个 stopper 并行，先触发先停。
    fn budget_from_go(&self, params: &GoParams) -> Result<MctsBudget, String> {
        // lc0: movetime / clock stopper 在 infinite/ponder/mate 模式下不启用（common.cc:123,148）。
        let skip_timed_stopper = params.infinite || params.ponder || params.mate.is_some();

        let mut budget = MctsBudget {
            max_playouts: None,
            max_nodes: None,
            max_depth: None,
            max_mate: None,
            deadline: None,
            stop: None,
        };

        if let Some(ms) = params.movetime {
            if !skip_timed_stopper {
                budget.deadline =
                    Some(Instant::now() + Duration::from_millis(ms.saturating_sub(LC0_DEFAULT_MOVE_OVERHEAD_MS as u64)));
            }
        } else if !skip_timed_stopper {
            let pos = self.history.current();
            let is_black = pos.side_to_move == Color::Black;
            let time_params = time_params_from_go(params);
            if time_params.has_clock() {
                if let Some(ms) = allocate_think_time_ms(&time_params, is_black, pos.game_ply) {
                    budget.deadline = Some(Instant::now() + Duration::from_millis(ms));
                }
            }
        }

        if let Some(nodes) = params.nodes {
            // lc0 `VisitsStopper`（common.cc:133-145）
            budget.max_nodes = Some(nodes.min(u64::from(u32::MAX)) as u32);
        }

        if let Some(depth) = params.depth {
            budget.max_depth = Some(depth);
        }

        if let Some(mate) = params.mate {
            budget.max_mate = Some(mate.max(1));
        }

        if !params.infinite
            && !params.ponder
            && budget.max_playouts.is_none()
            && budget.max_nodes.is_none()
            && budget.max_depth.is_none()
            && budget.max_mate.is_none()
            && budget.deadline.is_none()
        {
            budget.max_playouts = Some(self.default_playouts.max(1));
        }

        Ok(budget)
    }

    fn spawn_go(&mut self, outcome: GoParseOutcome) {
        for warning in &outcome.warnings {
            self.emit(&format!("info string {warning}"));
        }
        self.last_go = Some(outcome.clone());

        let searchmoves = match resolve_searchmoves(&self.history, &outcome.params.searchmoves) {
            Ok(moves) => moves,
            Err(err) => {
                self.emit(&format!("info string {err}"));
                self.emit("bestmove (none)");
                return;
            }
        };
        if !outcome.params.searchmoves.is_empty() && searchmoves.is_empty() {
            self.emit("info string searchmoves 无合法着法");
            self.emit("bestmove (none)");
            return;
        }

        let history = history_for_go(&self.history, outcome.params.ponder);
        let mut budget = match self.budget_from_go(&outcome.params) {
            Ok(budget) => budget,
            Err(err) => {
                self.emit(&format!("info string {err}"));
                self.emit("bestmove (none)");
                return;
            }
        };
        let config = self.config;
        let threads = self.resolved_threads();
        let stop = Arc::new(AtomicBool::new(false));
        let output = Arc::clone(&self.output);
        let search_engine = Arc::clone(&self.search_engine);
        budget.stop = Some(stop.clone());
        self.search_stop = Some(stop.clone());

        self.search_join = Some(thread::spawn(move || {
            let started_at = Instant::now();
            if stop.load(Ordering::SeqCst) {
                output("bestmove (none)");
                return;
            }

            let mut engine = search_engine.lock().unwrap_or_else(|e| e.into_inner());
            engine.config = config;

            let root_moves = if searchmoves.is_empty() {
                None
            } else {
                Some(searchmoves.as_slice())
            };

            let search_result = if threads > 1 {
                engine.search_root_history_parallel_with_progress(
                    &history,
                    budget,
                    threads,
                    UCI_INFO_INTERVAL,
                    root_moves,
                    |progress| {
                        let elapsed_ms = started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                        emit_mcts_progress(&output, progress, elapsed_ms);
                    },
                )
            } else {
                engine.search_root_history_with_progress(
                    &history,
                    budget,
                    UCI_INFO_INTERVAL,
                    root_moves,
                    |progress| {
                        let elapsed_ms = started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                        emit_mcts_progress(&output, progress, elapsed_ms);
                    },
                )
            };

            match search_result {
                Ok(result) => {
                    let elapsed_ms = started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                    emit_mcts_progress(
                        &output,
                        &MctsSearchProgress {
                            best_move: result.best_move,
                            pv: result.pv,
                            pv_lines: result.pv_lines,
                            multi_pv: result.multi_pv,
                            playouts: result.playouts,
                            root_visits: result.root_visits,
                            nodes: result.nodes,
                            tree_nodes: result.tree_nodes,
                            depth: result.depth,
                            seldepth: result.seldepth,
                            root_value: result.root_value,
                            best_value: result.best_value,
                            best_mate: result.best_mate,
                            nps_elapsed_ms: result.nps_elapsed_ms,
                            retry_without_playout: result.retry_without_playout,
                            moves: result.moves,
                        },
                        elapsed_ms,
                    );
                    if let Some(best_move) = result.best_move {
                        let best_uci = xiangqi_core::move_to_uci(best_move);
                        if let Some(pm) = engine.ponder_move_for(best_move) {
                            output(&format!(
                                "bestmove {best_uci} ponder {}",
                                xiangqi_core::move_to_uci(pm)
                            ));
                        } else {
                            output(&format!("bestmove {best_uci}"));
                        }
                    } else {
                        output("bestmove (none)");
                    }
                }
                Err(err) => {
                    output(&format!("info string {err}"));
                    output("bestmove (none)");
                }
            }
        }));
    }

    /// lc0 `Engine::PonderHit`（engine.cc:249-256）。
    fn handle_ponderhit(&mut self) {
        let Some(mut last) = self.last_go.clone() else {
            return;
        };
        if !last.params.ponder {
            return;
        }
        self.stop_and_join();
        last.params.ponder = false;
        self.spawn_go(last);
    }

    fn handle_line(&mut self, line: &str) -> bool {
        match line {
            "uci" => self.send_uci_ident(),
            "isready" => {
                // lc0 `UciLoop::DispatchCommand` isready (uciloop.cc:188-190): EnsureReady only.
                self.emit("readyok");
            }
            "ucinewgame" => self.new_game(),
            "stop" => {
                self.stop_and_join();
            }
            "ponderhit" => self.handle_ponderhit(),
            "quit" => {
                self.stop_and_join();
                return false;
            }
            _ if line.starts_with("setoption") => {
                self.stop_and_join();
                if let Err(err) = self.handle_setoption(line) {
                    self.emit(&format!("info string {err}"));
                }
            }
            _ if line.starts_with("position") => {
                // lc0 `Engine::SetPosition` (engine.cc:215-224): stop search, store position; tree on go.
                self.stop_and_join();
                match parse_position_history(line) {
                    Ok(history) => self.history = history,
                    Err(err) => self.emit(&format!("info string {err}")),
                }
            }
            _ if line.starts_with("go") => {
                self.stop_and_join();
                let args = line.strip_prefix("go").unwrap_or("").trim();
                self.spawn_go(parse_go(args));
            }
            _ => self.emit(&format!("info string unknown command (ignored): {line}")),
        }
        true
    }
}

pub fn run_uci_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut engine = Engine::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            engine.stop_and_join();
            break;
        }
        let line = buf.trim_end_matches(['\r', '\n']).trim();
        if line.is_empty() {
            continue;
        }
        if !engine.handle_line(line) {
            break;
        }
    }
    Ok(())
}

pub fn run_uci_for_test<R: BufRead, W: Write>(reader: R, writer: &mut W) -> io::Result<()> {
    let out = Arc::new(Mutex::new(Vec::<String>::new()));
    let out_sink = {
        let out = Arc::clone(&out);
        Arc::new(move |line: &str| {
            out.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
        }) as EngineOutput
    };
    let mut engine = Engine::new_with_output(out_sink);
    let mut reader = reader;
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            engine.stop_and_join();
            break;
        }
        let line = buf.trim_end_matches(['\r', '\n']).trim();
        if line.is_empty() {
            continue;
        }
        if !engine.handle_line(line) {
            break;
        }
    }
    engine.stop_and_join();
    for line in out.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        writeln!(writer, "{line}")?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::time::Duration;
    use xiangqi_core::legal_moves_uci;

    #[test]
    fn parse_setoption_name_value() {
        let (n, v) = parse_setoption("setoption name PolicyFile value C:/m.onnx").unwrap();
        assert_eq!(n, "PolicyFile");
        assert_eq!(v.as_deref(), Some("C:/m.onnx"));
    }

    #[test]
    fn parse_position_startpos_moves() {
        let pos0 = Position::from_fen(START_FEN).unwrap();
        let first = legal_moves_uci(&pos0).into_iter().next().expect("mv");
        let line = format!("position startpos moves {first}");
        let pos = parse_position_uci(&line).expect("ok");
        let legals = legal_moves_uci(&pos);
        assert!(!legals.is_empty());
    }

    #[test]
    fn parse_position_history_keeps_real_prefix() {
        let pos0 = Position::from_fen(START_FEN).unwrap();
        let first = legal_moves_uci(&pos0).into_iter().next().expect("mv");
        let line = format!("position startpos moves {first}");
        let history = parse_position_history_uci(&line).expect("ok");
        assert_eq!(history.len(), 2);
        assert_eq!(history.positions().next().expect("root").fen(), START_FEN);
    }

    #[test]
    fn parse_position_rejects_illegal_history_move() {
        let line = "position fen k3r4/9/9/9/9/9/9/9/9/3AK4 w - - 0 1 moves d0d1";
        let err = parse_position_history_uci(line)
            .err()
            .expect("must reject illegal move");
        assert!(err.contains("非法"));
    }

    #[test]
    fn uci_info_line_depth_at_least_one() {
        let line = uci_info_line(UciInfoView {
            best_value: 0.0,
            best_mate: None,
            depth: 0,
            seldepth: 1,
            playouts: 1,
            nodes: 1,
            elapsed_ms: 10,
            nps_elapsed_ms: 10,
            pv: &[],
            multipv: None,
        });
        assert!(line.contains("depth 1"));
        assert!(!line.contains("depth 0"));
    }

    #[test]
    fn uci_info_line_lc0_field_order() {
        let line = uci_info_line(UciInfoView {
            best_value: 0.1,
            best_mate: None,
            depth: 2,
            seldepth: 3,
            playouts: 8,
            nodes: 8,
            elapsed_ms: 100,
            nps_elapsed_ms: 100,
            pv: &[],
            multipv: None,
        });
        let depth_pos = line.find("depth ").expect("depth");
        let time_pos = line.find("time ").expect("time");
        let nodes_pos = line.find("nodes ").expect("nodes");
        let nps_pos = line.find("nps ").expect("nps");
        assert!(depth_pos < time_pos);
        assert!(time_pos < nodes_pos);
        assert!(nodes_pos < nps_pos);
    }

    #[test]
    fn ucinewgame_clears_search_tree_and_history() {
        let mut engine = Engine::new();
        engine.handle_line("setoption name Threads value 1");
        engine.handle_line("position startpos moves h2e2");
        engine.handle_line("go nodes 16");
        std::thread::sleep(Duration::from_millis(200));
        engine.stop_and_join();

        {
            let mcts = engine.search_engine.lock().unwrap_or_else(|e| e.into_inner());
            assert!(mcts.tree.len() > 0, "search should materialize tree nodes");
        }
        assert!(engine.history.len() > 1);

        engine.handle_line("ucinewgame");

        let mcts = engine.search_engine.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(mcts.tree.len(), 0);
        assert!(mcts.root_id.is_none());
        assert!(mcts.root_history.is_none());
        assert_eq!(engine.history.len(), 1);
        assert_eq!(engine.history.current().fen(), START_FEN);
    }

    #[test]
    fn go_infinite_means_no_fixed_visit_cap() {
        let engine = Engine::new();
        let budget = engine
            .budget_from_go(&parse_go("infinite").params)
            .expect("budget");
        assert!(budget.max_playouts.is_none());
        assert!(budget.max_nodes.is_none());
        assert!(budget.deadline.is_none());
    }

    #[test]
    fn go_movetime_and_nodes_combine_limits() {
        let engine = Engine::new();
        let budget = engine
            .budget_from_go(&parse_go("movetime 1000 nodes 500").params)
            .expect("budget");
        assert!(budget.deadline.is_some());
        assert_eq!(budget.max_nodes, Some(500));
        assert!(budget.max_depth.is_none());
        assert!(budget.max_playouts.is_none());
    }

    #[test]
    fn go_depth_and_nodes_combine_limits() {
        let engine = Engine::new();
        let budget = engine
            .budget_from_go(&parse_go("depth 8 nodes 2000").params)
            .expect("budget");
        assert_eq!(budget.max_depth, Some(8));
        assert_eq!(budget.max_nodes, Some(2000));
        assert!(budget.deadline.is_none());
        assert!(budget.max_playouts.is_none());
    }

    #[test]
    fn go_wtime_allocates_think_time() {
        let engine = Engine::new();
        let budget = engine
            .budget_from_go(&parse_go("wtime 60000 btime 60000").params)
            .expect("wtime/btime should produce budget");
        assert!(budget.deadline.is_some());
        assert!(budget.max_playouts.is_none());
    }

    #[test]
    fn go_wtime_with_nodes_combines_limits() {
        let engine = Engine::new();
        let budget = engine
            .budget_from_go(&parse_go("wtime 60000 btime 60000 nodes 8").params)
            .expect("budget");
        assert!(budget.deadline.is_some());
        assert_eq!(budget.max_nodes, Some(8));
    }

    #[test]
    fn go_wtime_side_to_move_uses_matching_clock() {
        let mut engine = Engine::new();
        engine.history = PositionHistory::new_startpos();
        let white_budget = engine
            .budget_from_go(&parse_go("wtime 10000 btime 90000").params)
            .expect("budget");
        // 切换到黑方
        let pos0 = Position::from_fen(START_FEN).unwrap();
        let mv = legal_moves_uci(&pos0).into_iter().next().expect("mv");
        engine.handle_line(&format!("position startpos moves {mv}"));
        let black_budget = engine
            .budget_from_go(&parse_go("wtime 10000 btime 90000").params)
            .expect("budget");
        assert!(white_budget.deadline.is_some());
        assert!(black_budget.deadline.is_some());
    }

    #[test]
    fn parse_go_unknown_token_emits_warning() {
        let outcome = parse_go("nodes 16 not_a_uci_flag");
        assert_eq!(outcome.params.nodes, Some(16));
        assert!(outcome
            .warnings
            .iter()
            .any(|w| w.contains("unknown go token")));
    }

    #[test]
    fn uci_info_line_prefers_mate_over_cp() {
        let line = uci_info_line(UciInfoView {
            best_value: 0.25,
            best_mate: Some(3),
            depth: 6,
            seldepth: 8,
            playouts: 64,
            nodes: 128,
            elapsed_ms: 50,
            nps_elapsed_ms: 50,
            pv: &[],
            multipv: None,
        });
        assert!(line.contains("score mate 3"));
        assert!(!line.contains("score cp"));
    }

    #[test]
    fn uci_info_line_nps_zero_before_nn_timing() {
        let line = uci_info_line(UciInfoView {
            best_value: 0.0,
            best_mate: None,
            depth: 4,
            seldepth: 4,
            playouts: 32,
            nodes: 32,
            elapsed_ms: 200,
            nps_elapsed_ms: 0,
            pv: &[],
            multipv: None,
        });
        assert!(line.contains("nps 0"));
        assert!(line.contains("time 200"));
    }

    #[test]
    fn go_nodes_maps_to_visits_budget() {
        let engine = Engine::new();
        let budget = engine
            .budget_from_go(&parse_go("nodes 1234").params)
            .expect("budget");
        assert_eq!(budget.max_nodes, Some(1234));
        assert!(budget.max_playouts.is_none());
    }

    #[test]
    fn mcts_playouts_updates_playouts_budget() {
        let mut engine = Engine::new();
        engine.handle_line("setoption name MctsPlayouts value 512");
        let budget = engine.budget_from_go(&parse_go("").params).expect("budget");
        assert_eq!(budget.max_playouts, Some(512));
    }

    #[test]
    fn split_fen_moves() {
        let (fen, m) =
            split_fen_and_moves("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1 moves b6b5")
                .unwrap();
        assert!(fen.contains("b -"));
        assert_eq!(m.as_deref(), Some("b6b5"));
    }

    #[test]
    fn uci_dialog_smoke() {
        let input = b"uci\nsetoption name MctsPlayouts value 64\nisready\nposition startpos\ngo nodes 1\nquit\n";
        let mut out = Vec::new();
        run_uci_for_test(Cursor::new(&input[..]), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("uciok"));
        assert!(s.contains("readyok"));
        assert!(s.contains("bestmove"));
        assert!(s.contains("MctsPlayouts"));
        assert!(s.contains("CPuct"));
        assert!(s.contains("FpuValue"));
        assert!(s.contains("MinibatchSize"));
        assert!(s.contains("NNCacheSize"));
        assert!(s.contains("Threads"));
        assert!(!s.contains("option name Playouts "));
        assert!(!s.contains("option name Visits "));
        assert!(!s.contains("option name MctsCpuct "));
        assert!(!s.contains("option name MctsWorkers "));
    }

    #[test]
    fn go_depth_stops_at_target_depth() {
        let lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let lines = Arc::clone(&lines);
            Arc::new(move |line: &str| {
                lines.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
            }) as EngineOutput
        };
        let mut engine = Engine::new_with_output(sink);
        engine.handle_line("setoption name Threads value 1");
        engine.handle_line("position startpos");
        engine.handle_line("go depth 2");
        std::thread::sleep(Duration::from_millis(200));
        engine.stop_and_join();

        let out = lines.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
        assert!(out.contains("info depth"));
        assert!(out.contains("bestmove"));
        assert!(!out.contains("暂不支持"));
    }

    #[test]
    fn go_infinite_can_be_stopped() {
        let lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let lines = Arc::clone(&lines);
            Arc::new(move |line: &str| {
                lines.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
            }) as EngineOutput
        };
        let mut engine = Engine::new_with_output(sink);
        engine.handle_line("position startpos");
        engine.handle_line("go infinite");
        std::thread::sleep(Duration::from_millis(50));
        engine.handle_line("stop");
        engine.stop_and_join();

        let out = lines.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
        assert!(out.contains("bestmove"));
    }

    #[test]
    fn go_infinite_emits_progress_info() {
        let lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let lines = Arc::clone(&lines);
            Arc::new(move |line: &str| {
                lines.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
            }) as EngineOutput
        };
        let mut engine = Engine::new_with_output(sink);
        engine.handle_line("setoption name Threads value 1");
        engine.handle_line("position startpos");
        engine.handle_line("go infinite");
        std::thread::sleep(UCI_INFO_INTERVAL + Duration::from_millis(50));
        engine.handle_line("stop");
        engine.stop_and_join();

        let out = lines.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
        assert!(out.contains("info depth"));
        assert!(out.contains("bestmove"));
    }

    #[test]
    fn stop_then_position_then_go_still_searches() {
        let lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let lines = Arc::clone(&lines);
            Arc::new(move |line: &str| {
                lines.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
            }) as EngineOutput
        };
        let mut engine = Engine::new_with_output(sink);
        engine.handle_line("setoption name Threads value 1");
        engine.handle_line("position startpos");
        engine.handle_line("go infinite");
        std::thread::sleep(Duration::from_millis(50));
        engine.handle_line("stop");
        engine.handle_line("position startpos");
        engine.handle_line("setoption name MultiPV value 2");
        engine.handle_line("go nodes 128");
        std::thread::sleep(Duration::from_millis(400));
        engine.stop_and_join();

        let out = lines.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
        assert!(out.contains("bestmove"));
        assert!(!out.ends_with("bestmove (none)"));
        assert!(out.contains("multipv 1"));
        assert!(out.contains("multipv 2"));
    }

    #[test]
    fn multipv_one_omits_multipv_field() {
        let lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let lines = Arc::clone(&lines);
            Arc::new(move |line: &str| {
                lines.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
            }) as EngineOutput
        };
        let mut engine = Engine::new_with_output(sink);
        engine.handle_line("setoption name Threads value 1");
        engine.handle_line("position startpos");
        engine.handle_line("go nodes 16");
        std::thread::sleep(Duration::from_millis(200));
        engine.stop_and_join();

        let out = lines.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
        assert!(out.contains("info depth"));
        assert!(!out.contains("multipv"));
    }

    #[test]
    fn uci_info_line_includes_multipv_when_set() {
        let line = uci_info_line(UciInfoView {
            best_value: 0.0,
            best_mate: None,
            depth: 4,
            seldepth: 4,
            playouts: 32,
            nodes: 32,
            elapsed_ms: 100,
            nps_elapsed_ms: 100,
            pv: &[],
            multipv: Some(2),
        });
        assert!(line.contains("multipv 2"));
    }

    #[test]
    fn threads_option_searches_without_hanging() {
        let lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let lines = Arc::clone(&lines);
            Arc::new(move |line: &str| {
                lines.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
            }) as EngineOutput
        };
        let mut engine = Engine::new_with_output(sink);
        engine.policy = None;
        engine.rebuild_search_engine(true);
        engine.handle_line("setoption name Threads value 2");
        engine.handle_line("position startpos");
        engine.handle_line("go nodes 16");
        std::thread::sleep(Duration::from_millis(150));
        engine.stop_and_join();

        let out = lines.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
        assert!(out.contains("bestmove"));
        assert!(!out.ends_with("bestmove (none)"));
    }

    #[test]
    fn policy_reload_discards_search_tree() {
        let mut engine = Engine::new();
        engine.handle_line("setoption name Threads value 1");
        engine.handle_line("position startpos");
        engine.handle_line("go nodes 16");
        std::thread::sleep(Duration::from_millis(200));
        engine.stop_and_join();

        {
            let mcts = engine.search_engine.lock().unwrap_or_else(|e| e.into_inner());
            assert!(mcts.tree.len() > 0, "search should materialize tree nodes");
            assert!(mcts.root_id.is_some());
        }

        engine.reload_policy().expect("reload");

        let mcts = engine.search_engine.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(mcts.tree.len(), 0);
        assert!(mcts.root_id.is_none());
        assert!(mcts.root_history.is_none());
    }

    #[test]
    fn threads_zero_means_auto() {
        let mut engine = Engine::new();
        assert_eq!(engine.threads, AUTO_THREADS);
        engine.handle_line("setoption name Threads value 0");
        assert_eq!(engine.threads, AUTO_THREADS);
        assert!(engine.resolved_threads() >= 1);
    }
}
