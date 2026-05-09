//! P3：固定局面基准，JSON 行输出，便于回归与对比实验。

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde_json::json;
use xiangqi_core::{legal_moves_uci, uci_to_move, Position, START_FEN};

use crate::policy_onnx::PolicyOnnx;
use crate::eval::NnEvalSession;
use crate::search::{root_search_iterative, RootSearchShared, SearchAblation, SearchLimits};
use crate::tt::TranspositionTable;

static DEFAULT_BENCHMARK_FEN_STRINGS: OnceLock<Vec<String>> = OnceLock::new();

/// 多局面基线：开局 / 短变 / 贪心多步后的中局型 / 更深步后的子力较少型（均为对局可达）。
///
/// 说明：「全盘初始 + 轮到黑走（`b`）」在棋规上虽可解析，但**非对局可达**，故不用纯 FEN 伪造；中残样本由 **双方始终走字典序首着** 的贪心链生成，保证可复现。
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

/// 解析 `data` 资源路径（ONNX / 词表）：便于开发与打包后都能找到 `data/`。
///
/// 依次尝试（首个**存在且为文件**者胜出）：
/// 1. `ENGIN_DATA_DIR/<rel>`
/// 2. `std::env::current_dir()/data/<rel>`（在仓库根 `cargo run` 时命中）
/// 3. `current_exe` 父目录下的 `data/<rel>`（与 exe 同目录的 `data`）
/// 4. 编译时 `crates/engin/../../data/<rel>`（即仓库根 `data/`，开发机固定回退）
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

/// 写入每条 NDJSON 的模型/词表解析信息（归因用）。
#[derive(Debug, Clone, Default)]
pub struct BenchJsonMeta {
    /// 实际使用的 ONNX 路径（无则 `null`）
    pub onnx_path: Option<String>,
    /// 实际使用的词表路径（无则 `null`）
    pub vocab_path: Option<String>,
    /// 会话中是否成功加载 ONNX 会话
    pub policy_session_loaded: bool,
    /// 词表着法条数（与 `vocab_size` 一致）
    pub vocab_entries: usize,
}

/// 单次基准运行参数（避免 `bench_one_json` 参数过长）。
pub struct BenchSessionParams<'a> {
    pub max_depth: u32,
    /// `Some(N)`：迭代加深过程中累计节点上限（与 UCI `go nodes` 口径一致）；`None` 不限制。
    pub max_nodes: Option<u64>,
    pub policy: &'a Arc<Mutex<Option<PolicyOnnx>>>,
    pub vocab: &'a HashMap<String, usize>,
    pub vocab_size: usize,
    pub ablation: SearchAblation,
    pub hash_mb: usize,
    /// ONNX 前向预算（`0` = 不限制）；写入 [`NnEvalSession::nn_eval_budget`]。
    pub nn_eval_budget: u64,
    pub meta: &'a BenchJsonMeta,
}

/// 对单局面跑迭代加深搜索，返回一行 JSON（`serde_json::Value`）。
pub fn bench_one_json(fen: &str, session: &BenchSessionParams<'_>) -> serde_json::Value {
    let t0 = Instant::now();
    let meta = session.meta;
    let mut pos = match Position::from_fen(fen) {
        Ok(p) => p,
        Err(e) => {
            return json!({
                "fen": fen,
                "error": e.to_string(),
                "nn_leaf_mode": session.ablation.nn_leaf_mode.as_str(),
                "policy_ordering": session.ablation.policy_ordering,
                "nn_eval_calls": 0u64,
                "nn_eval_cache_hits": 0u64,
                "nn_eval_cache_misses": 0u64,
                "nn_eval_budget": session.nn_eval_budget,
                "nn_eval_budget_used": 0u64,
                "nn_eval_budget_exhausted": false,
                "nn_eval_main_leaf_calls": 0u64,
                "nn_eval_qsearch_calls": 0u64,
                "bench_config": {
                    "onnx_path": meta.onnx_path,
                    "vocab_path": meta.vocab_path,
                    "policy_session_loaded": meta.policy_session_loaded,
                    "vocab_entries": meta.vocab_entries,
                },
            });
        }
    };
    let mut tt = TranspositionTable::new(session.hash_mb);
    let mut nn_eval = NnEvalSession {
        nn_eval_budget: session.nn_eval_budget,
        ..Default::default()
    };
    let mut shared = RootSearchShared {
        policy: session.policy,
        vocab: session.vocab,
        vocab_size: session.vocab_size,
        tt: &mut tt,
        stop: None,
        ablation: session.ablation,
        nn_eval: &mut nn_eval,
    };
    let limits = SearchLimits {
        max_nodes: session.max_nodes,
        ..SearchLimits::none()
    };
    let r = root_search_iterative(&mut pos, session.max_depth, &mut shared, limits);
    let elapsed_ms = t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    match r {
        Some(res) => {
            let nps = if elapsed_ms > 0 {
                (res.nodes as u128 * 1000 / u128::from(elapsed_ms)) as u64
            } else {
                0
            };
            json!({
                "fen": fen,
                "bestmove": res.best_uci,
                "score_cp": res.score_cp,
                "depth": res.main_depth,
                "seldepth": res.seldepth,
                "nodes": res.nodes,
                "time_ms": elapsed_ms,
                "nps": nps,
                "bench_profile": {
                    "threads": 1u32,
                    "hash_mb": session.hash_mb,
                    "max_depth": session.max_depth,
                    "max_nodes": session.max_nodes,
                    "nn_leaf_mode": session.ablation.nn_leaf_mode.as_str(),
                    "policy_ordering": session.ablation.policy_ordering,
                },
                "nn_leaf_mode": session.ablation.nn_leaf_mode.as_str(),
                "policy_ordering": session.ablation.policy_ordering,
                "nn_eval_calls": res.nn_eval_calls,
                "nn_eval_cache_hits": res.nn_eval_cache_hits,
                "nn_eval_cache_misses": res.nn_eval_cache_misses,
                "nn_eval_budget": res.nn_eval_budget,
                "nn_eval_budget_used": res.nn_eval_budget_used,
                "nn_eval_budget_exhausted": res.nn_eval_budget_exhausted,
                "nn_eval_main_leaf_calls": res.nn_eval_main_leaf_calls,
                "nn_eval_qsearch_calls": res.nn_eval_qsearch_calls,
                "bench_config": {
                    "onnx_path": meta.onnx_path,
                    "vocab_path": meta.vocab_path,
                    "policy_session_loaded": meta.policy_session_loaded,
                    "vocab_entries": meta.vocab_entries,
                },
            })
        }
        None => json!({
            "fen": fen,
            "error": "no_result",
            "time_ms": elapsed_ms,
            "nn_leaf_mode": session.ablation.nn_leaf_mode.as_str(),
            "policy_ordering": session.ablation.policy_ordering,
            "nn_eval_calls": 0u64,
            "nn_eval_cache_hits": 0u64,
            "nn_eval_cache_misses": 0u64,
            "nn_eval_budget": session.nn_eval_budget,
            "nn_eval_budget_used": 0u64,
            "nn_eval_budget_exhausted": false,
            "nn_eval_main_leaf_calls": 0u64,
            "nn_eval_qsearch_calls": 0u64,
            "bench_config": {
                "onnx_path": meta.onnx_path,
                "vocab_path": meta.vocab_path,
                "policy_session_loaded": meta.policy_session_loaded,
                "vocab_entries": meta.vocab_entries,
            },
        }),
    }
}

/// 跑 `fens` 中每个局面，向 `w` 写入 **一行一个 JSON**（NDJSON）。
pub fn write_benchmark_ndjson<W: Write, S: AsRef<str>>(
    w: &mut W,
    fens: &[S],
    session: &BenchSessionParams<'_>,
) -> io::Result<()> {
    for fen in fens {
        let v = bench_one_json(fen.as_ref(), session);
        writeln!(
            w,
            "{}",
            serde_json::to_string(&v).map_err(io::Error::other)?
        )?;
    }
    Ok(())
}
