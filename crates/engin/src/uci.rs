//! 中国象棋 UCI 子集：`uci` / `isready` / `setoption` / `position` / `go` / `stop` / `quit`。
//!
//! - `go`：迭代加深 + 静止搜索；`movetime` / `nodes` 在思考中检查；`infinite` 直至 `stop`。
//! - 根节点在已加载 **PolicyFile** + **VocabFile** 且维数一致时按 policy logit 排序着法。

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use xiangqi_core::{uci_to_move, Position, START_FEN};

use crate::policy_onnx::PolicyOnnx;
use crate::eval::{NNLeafMode, NnEvalSession};
use crate::search::{root_search_iterative, RootSearchShared, SearchAblation, SearchLimits};
use crate::tt::TranspositionTable;

static UCI_STDOUT_LOCK: Mutex<()> = Mutex::new(());

fn uci_out_line(line: &str) {
    let _g = UCI_STDOUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    println!("{line}");
}

/// `go` 参数（未列出的字段忽略）。
#[derive(Debug, Clone, Default)]
struct GoParams {
    infinite: bool,
    ponder: bool,
    depth: Option<u32>,
    movetime: Option<u64>,
    nodes: Option<u64>,
}

fn parse_uci_check(s: &str) -> bool {
    matches!(s.trim().to_ascii_lowercase().as_str(), "true" | "yes" | "1" | "on")
}

fn parse_go(args: &str) -> GoParams {
    let mut p = GoParams::default();
    let mut it = args.split_whitespace();
    while let Some(tok) = it.next() {
        match tok {
            "infinite" => p.infinite = true,
            "ponder" => p.ponder = true,
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

/// `setoption name <id> [value <x>]`；`value` 之后整段为选项值（可含空格）。
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

/// P3 联调 / 集成测试：解析完整 `position …` 行（与 UCI 状态机内逻辑一致）。
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

/// 从 `fen` 串中分出局面与 `moves ...` 后缀（FEN 可含空格）。
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
    policy: Arc<Mutex<Option<PolicyOnnx>>>,
    policy_path: Option<PathBuf>,
    vocab_path: Option<PathBuf>,
    vocab: HashMap<String, usize>,
    vocab_size: usize,
    /// 置换表（`Hash` / `Clear Hash`）；与搜索线程间共享。
    tt: Arc<Mutex<TranspositionTable>>,
    hash_mb: u32,
    threads: u32,
    multipv: u32,
    /// ONNX 在搜索中的消融（`UsePolicyOrdering` / `NNLeafMode`）。
    ablation: SearchAblation,
    search_stop: Option<Arc<AtomicBool>>,
    search_join: Option<JoinHandle<()>>,
}

impl Engine {
    fn new() -> Self {
        Self {
            pos: Position::from_fen(START_FEN).expect("startpos"),
            policy: Arc::new(Mutex::new(None)),
            policy_path: None,
            vocab_path: None,
            vocab: HashMap::new(),
            vocab_size: 0,
            tt: Arc::new(Mutex::new(TranspositionTable::new(16))),
            hash_mb: 16,
            threads: 1,
            multipv: 1,
            ablation: SearchAblation::ALL_ON,
            search_stop: None,
            search_join: None,
        }
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
        let mut g = self.policy.lock().map_err(|_| "policy 锁中毒".to_string())?;
        *g = None;
        if let Some(ref p) = self.policy_path {
            if p.is_file() {
                let net = PolicyOnnx::from_file(p).map_err(|e| e.to_string())?;
                *g = Some(net);
            }
        }
        Ok(())
    }

    fn reload_vocab(&mut self) -> Result<(), String> {
        self.vocab.clear();
        self.vocab_size = 0;
        if let Some(ref p) = self.vocab_path {
            if p.is_file() {
                let (m, n) = crate::vocab::load_move_vocab(p)?;
                self.vocab = m;
                self.vocab_size = n;
            }
        }
        Ok(())
    }

    fn send_uci_ident(&self) {
        uci_out_line("id name 77xiangqi_engine");
        uci_out_line("id author github.com/77xiangqi_engine");
        uci_out_line("option name PolicyFile type string default <empty>");
        uci_out_line("option name VocabFile type string default <empty>");
        uci_out_line("option name Hash type spin default 16 min 1 max 65536");
        uci_out_line("option name Threads type spin default 1 min 1 max 512");
        uci_out_line("option name MultiPV type spin default 1 min 1 max 16");
        uci_out_line("option name Clear Hash type button");
        uci_out_line("option name UsePolicyOrdering type check default true");
        uci_out_line("option name NNLeafMode type combo var Off var MainLeafOnly var AllLeaf default MainLeafOnly");
        uci_out_line("uciok");
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
            "Hash" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<u32>() {
                        self.hash_mb = n.clamp(1, 65536);
                        if let Ok(mut g) = self.tt.lock() {
                            *g = TranspositionTable::new(self.hash_mb as usize);
                        }
                    }
                }
            }
            "Threads" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<u32>() {
                        self.threads = n.clamp(1, 512);
                    }
                }
            }
            "MultiPV" => {
                if let Some(ref v) = value {
                    if let Ok(n) = v.trim().parse::<u32>() {
                        self.multipv = n.clamp(1, 16);
                    }
                }
            }
            "Clear Hash" => {
                if let Ok(mut g) = self.tt.lock() {
                    g.clear();
                }
            }
            "UsePolicyOrdering" => {
                if let Some(ref v) = value {
                    self.ablation.policy_ordering = parse_uci_check(v);
                }
            }
            "NNLeafMode" => {
                if let Some(ref v) = value {
                    if let Some(m) = NNLeafMode::parse_uci(v) {
                        self.ablation.nn_leaf_mode = m;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn spawn_go(&mut self, params: GoParams) {
        let fen = self.pos.fen();
        let policy = Arc::clone(&self.policy);
        let tt = Arc::clone(&self.tt);
        let vocab = self.vocab.clone();
        let vocab_size = self.vocab_size;
        let ablation = self.ablation;
        let stop = Arc::new(AtomicBool::new(false));
        self.search_stop = Some(stop.clone());
        let multipv = self.multipv.max(1);

        let handle = thread::spawn(move || {
            let search_start = Instant::now();
            let Ok(mut pos) = Position::from_fen(&fen) else {
                uci_out_line("bestmove (none)");
                return;
            };

            let max_depth = params
                .depth
                .unwrap_or(
                    if params.movetime.is_some() || params.infinite || params.ponder || params.nodes.is_some() {
                        64
                    } else {
                        4
                    },
                )
                .clamp(1, 64);

            let deadline = params.movetime.map(|ms| search_start + Duration::from_millis(ms));

            let limits = SearchLimits {
                deadline,
                max_nodes: params.nodes,
            };

            let mut tt_guard = match tt.lock() {
                Ok(g) => g,
                Err(_) => {
                    uci_out_line("bestmove (none)");
                    return;
                }
            };
            let mut nn_eval = NnEvalSession::default();
            let mut shared = RootSearchShared {
                policy: &policy,
                vocab: &vocab,
                vocab_size,
                tt: &mut tt_guard,
                stop: Some(&stop),
                ablation,
                nn_eval: &mut nn_eval,
            };
            let bm = match root_search_iterative(&mut pos, max_depth, &mut shared, limits) {
                Some(r) => {
                    let elapsed_ms = search_start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                    uci_out_line(&format!(
                        "info depth {} seldepth {} multipv 1 score cp {} nodes {} time {} pv {}",
                        r.main_depth, r.seldepth, r.score_cp, r.nodes, elapsed_ms, r.best_uci
                    ));
                    r.best_uci
                }
                None => "(none)".to_string(),
            };
            let _ = multipv; // 将来 MultiPV 多行 info
            uci_out_line(&format!("bestmove {bm}"));
        });

        self.search_join = Some(handle);
    }
}

/// 自 stdin 读行并处理 UCI，应答至 stdout。
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
        match line {
            "uci" => engine.send_uci_ident(),
            "isready" => {
                engine.stop_and_join();
                uci_out_line("readyok");
            }
            "ucinewgame" => {
                engine.stop_and_join();
            }
            "stop" => {
                if let Some(s) = engine.search_stop.as_ref() {
                    s.store(true, Ordering::SeqCst);
                }
            }
            "quit" => {
                engine.stop_and_join();
                break;
            }
            _ if line.starts_with("setoption") => {
                engine.stop_and_join();
                if let Err(e) = engine.handle_setoption(line) {
                    uci_out_line(&format!("info string {e}"));
                }
            }
            _ if line.starts_with("position") => {
                engine.stop_and_join();
                match parse_position(line) {
                    Ok(pos) => engine.pos = pos,
                    Err(e) => uci_out_line(&format!("info string {e}")),
                }
            }
            _ if line.starts_with("go") => {
                engine.stop_and_join();
                let args = line.strip_prefix("go").unwrap_or("").trim();
                let params = parse_go(args);
                engine.spawn_go(params);
            }
            _ if line == "ponderhit" => {}
            _ => {
                uci_out_line(&format!("info string unknown command (ignored): {line}"));
            }
        }
    }
    Ok(())
}

/// 用于测试：将应答写入 `writer`（不含互斥锁，与 `run_uci_stdio` 行为略异）。
pub fn run_uci_for_test<R: BufRead, W: Write>(reader: R, writer: &mut W) -> io::Result<()> {
    let mut engine = Engine::new();
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
        let mut reply = |s: &str| {
            writeln!(writer, "{s}")?;
            writer.flush()?;
            io::Result::Ok(())
        };
        match line {
            "uci" => {
                reply("id name 77xiangqi_engine")?;
                reply("id author github.com/77xiangqi_engine")?;
                reply("option name PolicyFile type string default <empty>")?;
                reply("option name VocabFile type string default <empty>")?;
                reply("option name Hash type spin default 16 min 1 max 65536")?;
                reply("option name Threads type spin default 1 min 1 max 512")?;
                reply("option name MultiPV type spin default 1 min 1 max 16")?;
                reply("option name Clear Hash type button")?;
                reply("option name UsePolicyOrdering type check default true")?;
                reply("option name NNLeafMode type combo var Off var MainLeafOnly var AllLeaf default MainLeafOnly")?;
                reply("uciok")?;
            }
            "isready" => {
                engine.stop_and_join();
                reply("readyok")?;
            }
            "quit" => {
                engine.stop_and_join();
                break;
            }
            _ if line.starts_with("setoption") => {
                engine.stop_and_join();
                let _ = engine.handle_setoption(line);
            }
            _ if line.starts_with("position") => {
                engine.stop_and_join();
                if let Ok(pos) = parse_position(line) {
                    engine.pos = pos;
                }
            }
            _ if line.starts_with("go") => {
                engine.stop_and_join();
                let args = line.strip_prefix("go").unwrap_or("").trim();
                let params = parse_go(args);
                let max_depth = params.depth.unwrap_or(4).clamp(1, 64);
                let Ok(mut pos) = Position::from_fen(&engine.pos.fen()) else {
                    reply("bestmove (none)")?;
                    continue;
                };
                let mut tt_guard = engine.tt.lock().unwrap();
                let mut nn_eval = NnEvalSession::default();
                let mut shared = RootSearchShared {
                    policy: &engine.policy,
                    vocab: &engine.vocab,
                    vocab_size: engine.vocab_size,
                    tt: &mut tt_guard,
                    stop: None,
                    ablation: engine.ablation,
                    nn_eval: &mut nn_eval,
                };
                let bm = match root_search_iterative(&mut pos, max_depth, &mut shared, SearchLimits::none()) {
                    Some(r) => {
                        reply(&format!(
                            "info depth {} seldepth {} multipv 1 score cp {} nodes {} time 0 pv {}",
                            r.main_depth, r.seldepth, r.score_cp, r.nodes, r.best_uci
                        ))?;
                        r.best_uci
                    }
                    None => "(none)".to_string(),
                };
                reply(&format!("bestmove {bm}"))?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use xiangqi_core::legal_moves_uci;

    #[test]
    fn parse_setoption_name_value() {
        let (n, v) = parse_setoption("setoption name PolicyFile value C:/m.onnx").unwrap();
        assert_eq!(n, "PolicyFile");
        assert_eq!(v.as_deref(), Some("C:/m.onnx"));
    }

    #[test]
    fn parse_setoption_clear_hash_button() {
        let (n, v) = parse_setoption("setoption name Clear Hash").unwrap();
        assert_eq!(n, "Clear Hash");
        assert!(v.is_none());
    }

    #[test]
    fn parse_setoption_value_with_spaces() {
        let (n, v) = parse_setoption("setoption name PolicyFile value C:/a b/model.onnx").unwrap();
        assert_eq!(n, "PolicyFile");
        assert_eq!(v.as_deref(), Some("C:/a b/model.onnx"));
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
    fn split_fen_moves() {
        let (fen, m) =
            split_fen_and_moves("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1 moves b7b6")
                .unwrap();
        assert!(fen.contains("b -"));
        assert_eq!(m.as_deref(), Some("b7b6"));
    }

    #[test]
    fn uci_dialog_smoke() {
        let input = b"uci\nsetoption name Hash value 32\nsetoption name Threads value 4\nsetoption name Clear Hash\nisready\nposition startpos\ngo depth 1\nquit\n";
        let mut out = Vec::new();
        run_uci_for_test(Cursor::new(&input[..]), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("uciok"));
        assert!(s.contains("readyok"));
        assert!(s.contains("bestmove"));
    }
}
