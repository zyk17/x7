//! 固定局面 MCTS 基准，输出 NDJSON。

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use serde_json::json;
use xiangqi_core::{legal_moves_uci, uci_to_move, Position, START_FEN};

use crate::history::PositionHistory;
use crate::mcts::{MctsBudget, MctsConfig, MctsEngine, MctsMoveStat, OnnxPolicyValueEval, SharedPolicy};

static DEFAULT_BENCHMARK_FEN_STRINGS: OnceLock<Vec<String>> = OnceLock::new();

pub fn default_benchmark_fen_strings() -> &'static [String] {
    DEFAULT_BENCHMARK_FEN_STRINGS.get_or_init(|| {
        let mut v = vec![START_FEN.to_string()];
        let mut p = Position::from_fen(START_FEN).expect("START_FEN");
        let mut ms = legal_moves_uci(&p);
        ms.sort();
        let u = ms.first().expect("startpos has legal moves").as_str();
        let m = uci_to_move(&p, u).expect("sorted legal uci");
        p.do_move(m);
        v.push(p.fen());
        v.push(greedy_sorted_plies_fen(START_FEN, 10));
        v.push(greedy_sorted_plies_fen(START_FEN, 28));
        v
    })
}

fn greedy_sorted_plies_fen(start_fen: &str, plies: usize) -> String {
    let mut p = Position::from_fen(start_fen).expect("greedy base");
    for _ in 0..plies {
        let mut ms = legal_moves_uci(&p);
        if ms.is_empty() {
            break;
        }
        ms.sort();
        let u = ms.first().expect("non-empty").as_str();
        let m = uci_to_move(&p, u).expect("uci");
        p.do_move(m);
    }
    p.fen()
}

pub fn resolve_data_file(rel: impl AsRef<Path>) -> Option<PathBuf> {
    let rel = rel.as_ref();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("ENGIN_DATA_DIR") {
        candidates.push(PathBuf::from(dir).join(rel));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("data").join(rel));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("data").join(rel));
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data").join(rel));
    candidates.into_iter().find(|p| p.is_file())
}

#[derive(Debug, Clone, Default)]
pub struct BenchJsonMeta {
    pub onnx_path: Option<String>,
    pub policy_session_loaded: bool,
}

pub struct BenchSessionParams<'a> {
    pub budget: MctsBudget,
    pub config: MctsConfig,
    pub threads: usize,
    pub policy: &'a SharedPolicy,
    pub meta: &'a BenchJsonMeta,
}

pub fn bench_one_json(fen: &str, session: &BenchSessionParams<'_>) -> serde_json::Value {
    let t0 = Instant::now();
    let history = match PositionHistory::from_fen(fen) {
        Ok(history) => history,
        Err(err) => {
            return json!({
                "fen": fen,
                "error": err.to_string(),
            });
        }
    };

    let mut engine = MctsEngine::new(session.config, OnnxPolicyValueEval::new(session.policy.clone()));

    let search_result = if session.threads > 1 {
        engine.search_root_history_parallel_with_progress(
            &history,
            session.budget.clone(),
            session.threads,
            std::time::Duration::ZERO,
            |_| {},
        )
    } else {
        engine.search_root_history(&history, session.budget.clone())
    };

    match search_result {
        Ok(result) => {
            let elapsed_ms = t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
            let nps = if elapsed_ms > 0 {
                (result.playouts as u128 * 1000 / u128::from(elapsed_ms)) as u64
            } else {
                0
            };
            json!({
                "fen": fen,
                "bestmove": result.best_move.map(xiangqi_core::move_to_uci),
                "root_value": result.root_value,
                "playouts": result.playouts,
                "root_visits": result.root_visits,
                "nodes": result.nodes,
                "depth": result.depth,
                "seldepth": result.seldepth,
                "time_ms": elapsed_ms,
                "nps": nps,
                "root_moves": result.moves.iter().map(|stat: &MctsMoveStat| {
                    json!({
                        "move": xiangqi_core::move_to_uci(stat.mv),
                        "visits": stat.visits,
                        "q": stat.q,
                        "prior": stat.prior,
                    })
                }).collect::<Vec<_>>(),
                "mcts_config": {
                    "cpuct": session.config.cpuct,
                    "cpuct_root": session.config.cpuct_root,
                    "cpuct_base": session.config.cpuct_base,
                    "cpuct_factor": session.config.cpuct_factor,
                    "fpu_reduction": session.config.fpu_reduction,
                    "fpu_reduction_root": session.config.fpu_reduction_root,
                    "root_temperature": session.config.root_temperature,
                    "search_batch_size": session.config.search_batch_size,
                    "threads": session.threads,
                },
                "budget": {
                    "max_playouts": session.budget.max_playouts,
                    "max_nodes": session.budget.max_nodes,
                    "has_deadline": session.budget.deadline.is_some(),
                },
                "bench_config": {
                    "onnx_path": session.meta.onnx_path,
                    "policy_session_loaded": session.meta.policy_session_loaded,
                },
            })
        }
        Err(err) => json!({
            "fen": fen,
            "error": err,
        }),
    }
}

pub fn write_benchmark_ndjson<W: Write, S: AsRef<str>>(
    w: &mut W,
    fens: &[S],
    session: &BenchSessionParams<'_>,
) -> io::Result<()> {
    for fen in fens {
        let v = bench_one_json(fen.as_ref(), session);
        writeln!(w, "{}", serde_json::to_string(&v).map_err(io::Error::other)?)?;
    }
    Ok(())
}
