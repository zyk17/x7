//! 中国象棋 UCI 子集：`uci` / `isready` / `setoption` / `position` / `go` / `stop` / `quit`。

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use xiangqi_core::{uci_to_move, Position, START_FEN};

use crate::mcts::{MctsBudget, MctsConfig, MctsEngine, MctsMoveStat, MctsSearchProgress, OnnxPolicyValueEval, SharedPolicy};
use crate::policy_onnx::PolicyOnnx;

static UCI_STDOUT_LOCK: Mutex<()> = Mutex::new(());
const UCI_INFO_INTERVAL: Duration = Duration::from_millis(200);
const UCI_ROOT_MOVE_TOP_K: usize = 5;

fn approximate_depth(nodes: usize) -> u32 {
    if nodes <= 1 {
        0
    } else {
        (f64::ln(nodes as f64) / f64::ln(1.8)).floor().max(0.0) as u32
    }
}

fn uci_info_line(
    root_value: f32,
    playouts: u32,
    nodes: usize,
    elapsed_ms: u64,
    pv: Option<xiangqi_core::Move>,
) -> String {
    let cp = (root_value * 100.0).round() as i32;
    let depth = approximate_depth(nodes);
    let nps = if elapsed_ms > 0 {
        (playouts as u128 * 1000 / u128::from(elapsed_ms)) as u64
    } else {
        0
    };
    let mut line = format!("info depth {depth} score cp {cp} nodes {nodes} nps {nps} time {elapsed_ms}");
    if let Some(mv) = pv {
        line.push_str(" pv ");
        line.push_str(&xiangqi_core::move_to_uci(mv));
    }
    line
}

fn uci_root_moves_line(moves: &[MctsMoveStat]) -> String {
    let mut stats = moves.to_vec();
    stats.sort_by(|a, b| b.visits.cmp(&a.visits).then_with(|| b.q.partial_cmp(&a.q).unwrap_or(std::cmp::Ordering::Equal)));
    let body = stats
        .into_iter()
        .take(UCI_ROOT_MOVE_TOP_K)
        .map(|s| format!("{}:{}:{:.3}", xiangqi_core::move_to_uci(s.mv), s.visits, s.q))
        .collect::<Vec<_>>()
        .join(",");
    format!("info string root_moves {body}")
}

fn emit_mcts_progress(
    output: &Arc<dyn Fn(&str) + Send + Sync>,
    progress: &MctsSearchProgress,
    elapsed_ms: u64,
) {
    output(&uci_info_line(
        progress.root_value,
        progress.playouts,
        progress.nodes,
        elapsed_ms,
        progress.best_move,
    ));
    if let Some(best_move) = progress.best_move {
        let best_uci = xiangqi_core::move_to_uci(best_move);
        output(&format!(
            "info string mcts playouts={} root_visits={} nodes={} root_value={:.4} bestmove={}",
            progress.playouts, progress.root_visits, progress.nodes, progress.root_value, best_uci
        ));
    }
}

type EngineOutput = Arc<dyn Fn(&str) + Send + Sync + 'static>;

fn uci_out_line(line: &str) {
    let _g = UCI_STDOUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    println!("{line}");
}

#[derive(Debug, Clone, Default)]
struct GoParams {
    infinite: bool,
    depth: Option<u32>,
    movetime: Option<u64>,
    nodes: Option<u64>,
}

fn parse_go(args: &str) -> GoParams {
    let mut p = GoParams::default();
    let mut it = args.split_whitespace();
    while let Some(tok) = it.next() {
        match tok {
            "infinite" => p.infinite = true,
            "depth" => {
                if let Some(n) = it.next().and_then(|s| s.parse().ok()) {
                    p.depth = Some(n);
                }
            }
            "movetime" => {
                if let Some(n) = it.next().and_then(|s| s.parse().ok()) {
                    p.movetime = Some(n);
                }
            }
            "nodes" => {
                if let Some(n) = it.next().and_then(|s| s.parse().ok()) {
                    p.nodes = Some(n);
                }
            }
            _ => {}
        }
    }
    p
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
    parse_position(line)
}

fn parse_position(line: &str) -> Result<Position, String> {
    let rest = line
        .strip_prefix("position")
        .ok_or_else(|| "internal".to_string())?
        .trim();
    let (mut pos, moves_tail) = if let Some(tail) = rest.strip_prefix("startpos") {
        let tail = tail.trim_start();
        let moves_tail = parse_moves_suffix(tail)?;
        (Position::from_fen(START_FEN).map_err(|e| e.to_string())?, moves_tail)
    } else if let Some(fen_part) = rest.strip_prefix("fen") {
        let fen_part = fen_part.trim();
        let (fen_str, moves_tail) = split_fen_and_moves(fen_part)?;
        (
            Position::from_fen(fen_str.trim()).map_err(|e| e.to_string())?,
            moves_tail,
        )
    } else {
        return Err("position 需要 startpos 或 fen".into());
    };

    if let Some(mvs) = moves_tail {
        for mv in mvs.split_whitespace() {
            let Some(m) = uci_to_move(&pos, mv) else {
                return Err(format!("非法或不可执行着法: {mv}"));
            };
            pos.do_move(m);
        }
    }
    Ok(pos)
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
    pos: Position,
    policy: SharedPolicy,
    policy_path: Option<PathBuf>,
    vocab_path: Option<PathBuf>,
    vocab: Arc<HashMap<String, usize>>,
    config: MctsConfig,
    default_playouts: u32,
    search_stop: Option<Arc<AtomicBool>>,
    search_join: Option<JoinHandle<()>>,
    output: EngineOutput,
}

impl Engine {
    fn new() -> Self {
        Self::new_with_output(Arc::new(uci_out_line))
    }

    fn new_with_output(output: EngineOutput) -> Self {
        Self {
            pos: Position::from_fen(START_FEN).expect("startpos"),
            policy: None,
            policy_path: None,
            vocab_path: None,
            vocab: Arc::new(HashMap::new()),
            config: MctsConfig::default(),
            default_playouts: 256,
            search_stop: None,
            search_join: None,
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

    fn reload_policy(&mut self) -> Result<(), String> {
        self.policy = None;
        if let Some(ref p) = self.policy_path {
            if p.is_file() {
                let net = PolicyOnnx::from_file(p).map_err(|e| e.to_string())?;
                self.policy = Some(Arc::new(Mutex::new(net)));
            }
        }
        Ok(())
    }

    fn reload_vocab(&mut self) -> Result<(), String> {
        self.vocab = Arc::new(HashMap::new());
        if let Some(ref p) = self.vocab_path {
            if p.is_file() {
                let (m, _) = crate::vocab::load_move_vocab(p)?;
                self.vocab = Arc::new(m);
            }
        }
        Ok(())
    }

    fn send_uci_ident(&self) {
        self.emit("id name 77xiangqi_engine");
        self.emit("id author github.com/77xiangqi_engine");
        self.emit("option name PolicyFile type string default <empty>");
        self.emit("option name VocabFile type string default <empty>");
        self.emit("option name Playouts type spin default 256 min 1 max 1000000");
        self.emit("option name Visits type spin default 256 min 1 max 1000000");
        self.emit("option name Cpuct type string default 1.25");
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
            "VocabFile" => {
                self.vocab_path = value
                    .as_ref()
                    .map(|s| PathBuf::from(s.trim()))
                    .filter(|p| !p.as_os_str().is_empty());
                self.reload_vocab()?;
            }
            "Playouts" | "Visits" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<u32>() {
                        self.default_playouts = n.clamp(1, 1_000_000);
                    }
                }
            }
            "Cpuct" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<f32>() {
                        self.config.cpuct = n.clamp(0.01, 100.0);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn budget_from_go(&self, params: &GoParams) -> Result<MctsBudget, String> {
        if let Some(ms) = params.movetime {
            return Ok(MctsBudget::from_movetime_ms(ms));
        }

        if let Some(nodes) = params.nodes {
            return Ok(MctsBudget {
                max_playouts: None,
                max_nodes: Some(nodes.min(u64::from(u32::MAX)) as u32),
                deadline: None,
                stop: None,
            });
        }

        if params.infinite {
            return Ok(MctsBudget {
                max_playouts: None,
                max_nodes: None,
                deadline: None,
                stop: None,
            });
        }

        if let Some(depth) = params.depth {
            return Err(format!("go depth {depth} 暂不支持；请使用 go nodes / go movetime / go infinite"));
        }

        Ok(MctsBudget {
            max_playouts: Some(self.default_playouts.max(1)),
            max_nodes: None,
            deadline: None,
            stop: None,
        })
    }

    fn spawn_go(&mut self, params: GoParams) {
        let pos = self.pos.clone();
        let policy = self.policy.clone();
        let vocab = Arc::clone(&self.vocab);
        let mut budget = match self.budget_from_go(&params) {
            Ok(budget) => budget,
            Err(err) => {
                self.emit(&format!("info string {err}"));
                self.emit("bestmove (none)");
                return;
            }
        };
        let config = self.config;
        let stop = Arc::new(AtomicBool::new(false));
        let output = Arc::clone(&self.output);
        budget.stop = Some(stop.clone());
        self.search_stop = Some(stop.clone());

        self.search_join = Some(thread::spawn(move || {
            let started_at = Instant::now();
            if stop.load(Ordering::SeqCst) {
                output("bestmove (none)");
                return;
            }

            let mut engine = MctsEngine::new(
                config,
                OnnxPolicyValueEval {
                    policy: &policy,
                    vocab: &vocab,
                },
            );

            match engine.search_root_with_progress(&pos, budget, UCI_INFO_INTERVAL, |progress| {
                let elapsed_ms = started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                emit_mcts_progress(&output, progress, elapsed_ms);
            }) {
                Ok(result) => {
                    let elapsed_ms = started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                    output(&uci_info_line(
                        result.root_value,
                        result.playouts,
                        result.nodes,
                        elapsed_ms,
                        result.best_move,
                    ));
                    if !result.moves.is_empty() {
                        output(&uci_root_moves_line(&result.moves));
                    }
                    if let Some(best_move) = result.best_move {
                        let best_uci = xiangqi_core::move_to_uci(best_move);
                        output(&format!(
                            "info string mcts playouts={} root_visits={} nodes={} root_value={:.4}",
                            result.playouts, result.root_visits, result.nodes, result.root_value
                        ));
                        output(&format!("bestmove {best_uci}"));
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

    fn handle_line(&mut self, line: &str) -> bool {
        match line {
            "uci" => self.send_uci_ident(),
            "isready" => {
                self.stop_and_join();
                self.emit("readyok");
            }
            "ucinewgame" => self.stop_and_join(),
            "stop" => {
                if let Some(s) = self.search_stop.as_ref() {
                    s.store(true, Ordering::SeqCst);
                }
            }
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
                self.stop_and_join();
                match parse_position(line) {
                    Ok(pos) => self.pos = pos,
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
        let pos = parse_position(&line).expect("ok");
        let legals = legal_moves_uci(&pos);
        assert!(!legals.is_empty());
    }

    #[test]
    fn go_infinite_means_no_fixed_visit_cap() {
        let engine = Engine::new();
        let budget = engine.budget_from_go(&parse_go("infinite")).expect("budget");
        assert!(budget.max_playouts.is_none());
        assert!(budget.max_nodes.is_none());
        assert!(budget.deadline.is_none());
    }

    #[test]
    fn go_nodes_maps_to_node_budget() {
        let engine = Engine::new();
        let budget = engine.budget_from_go(&parse_go("nodes 1234")).expect("budget");
        assert_eq!(budget.max_nodes, Some(1234));
        assert!(budget.max_playouts.is_none());
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
        let input = b"uci\nsetoption name Playouts value 64\nisready\nposition startpos\ngo nodes 1\nquit\n";
        let mut out = Vec::new();
        run_uci_for_test(Cursor::new(&input[..]), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("uciok"));
        assert!(s.contains("readyok"));
        assert!(s.contains("bestmove"));
        assert!(s.contains("Playouts"));
        assert!(s.contains("Visits"));
    }

    #[test]
    fn go_depth_reports_unsupported() {
        let input = b"uci\nposition startpos\ngo depth 2\nquit\n";
        let mut out = Vec::new();
        run_uci_for_test(Cursor::new(&input[..]), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("暂不支持"));
        assert!(s.contains("bestmove (none)"));
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
        engine.handle_line("position startpos");
        engine.handle_line("go infinite");
        std::thread::sleep(UCI_INFO_INTERVAL + Duration::from_millis(50));
        engine.handle_line("stop");
        engine.stop_and_join();

        let out = lines.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
        assert!(out.contains("info string mcts"));
        assert!(out.contains("bestmove"));
    }
}
