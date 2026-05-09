//! 用户侧引擎：默认 **UCI（stdin/stdout）**；`--onnx-smoke` 为 ONNX 冒烟；`--bench` 为 P3 搜索基准（NDJSON）。

use std::collections::HashMap;
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, Mutex};

use xiangqi_core::{legal_moves_uci, Position};

use engin::benchmark::{
    default_benchmark_fen_strings, resolve_data_file, write_benchmark_ndjson, BenchJsonMeta,
    BenchSessionParams,
};
use engin::value_probe::{markdown_table_off_vs_main, ValueProbeTableArgs};
use engin::vocab::{load_move_vocab, load_move_vocab_ordered};
use engin::{run_uci_stdio, PolicyOnnx, NNLeafMode, SearchAblation, START_FEN};

fn default_policy_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/policy.onnx")
}

fn default_vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/move_vocab.json")
}

fn print_usage() {
    eprintln!(
        "用法:\n  engin                         UCI 模式（stdin/stdout）\n  \
         engin --onnx-smoke [ONNX] [FEN] [VOCAB]  冒烟；缺省 ONNX=data/policy.onnx、FEN=起始局面；VOCAB 缺省且存在 data/move_vocab.json 则自动加载\n  \
         engin --bench [选项]            P3 基准：NDJSON；默认 **吞吐基线**（nn-leaf off、无 policy 排序、hash 16）；可加 `--policy-ordering` / `--nn-leaf main` 等覆盖\n  \
         engin --value-probe [选项]     小评测集：`nn-leaf off` vs `main`，输出 Markdown 表（见 docs/value-probe.md）\n  \
         --bench / --value-probe 共用: --depth N  --nodes N  --onnx  --vocab  --data-dir  --hash  --require-onnx  --policy-ordering  --no-policy-ordering  --nn-eval-budget N  --nn-leaf …  --no-nn-leaf\n  \
         伪标签真值: cargo test -p xiangqi_dataset user_fen_black_down_material_attack_below_half"
    );
}

#[derive(Debug)]
struct BenchCli {
    depth: u32,
    max_nodes: Option<u64>,
    onnx: Option<PathBuf>,
    vocab: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    require_onnx: bool,
    ablation: SearchAblation,
    hash_mb: usize,
    nn_eval_budget: u64,
    policy_ordering_explicit: bool,
    nn_leaf_explicit: bool,
}

fn parse_bench_cli(rest: &[String]) -> BenchCli {
    let mut depth = 4u32;
    let mut max_nodes: Option<u64> = None;
    let mut onnx: Option<PathBuf> = None;
    let mut vocab: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut require_onnx = false;
    let mut no_policy_ordering = false;
    let mut policy_ordering_explicit = false;
    let mut nn_leaf_mode: Option<NNLeafMode> = None;
    let mut nn_leaf_explicit = false;
    let mut hash_mb = 16usize;
    let mut nn_eval_budget = 0u64;
    let mut i = 0usize;
    while i < rest.len() {
        match rest[i].as_str() {
            "--depth" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<u32>() {
                    depth = n.clamp(1, 64);
                }
                i += 2;
            }
            "--nodes" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<u64>() {
                    max_nodes = (n > 0).then_some(n);
                }
                i += 2;
            }
            "--onnx" if i + 1 < rest.len() => {
                onnx = Some(PathBuf::from(&rest[i + 1]));
                i += 2;
            }
            "--vocab" if i + 1 < rest.len() => {
                vocab = Some(PathBuf::from(&rest[i + 1]));
                i += 2;
            }
            "--data-dir" if i + 1 < rest.len() => {
                data_dir = Some(PathBuf::from(&rest[i + 1]));
                i += 2;
            }
            "--require-onnx" => {
                require_onnx = true;
                i += 1;
            }
            "--policy-ordering" => {
                no_policy_ordering = false;
                policy_ordering_explicit = true;
                i += 1;
            }
            "--no-policy-ordering" => {
                no_policy_ordering = true;
                policy_ordering_explicit = true;
                i += 1;
            }
            "--no-nn-leaf" => {
                nn_leaf_mode = Some(NNLeafMode::Off);
                nn_leaf_explicit = true;
                i += 1;
            }
            "--nn-leaf" if i + 1 < rest.len() => {
                if let Some(m) = NNLeafMode::parse_uci(&rest[i + 1]) {
                    nn_leaf_mode = Some(m);
                    nn_leaf_explicit = true;
                } else {
                    eprintln!("--nn-leaf: 无效值 {:?}，应为 off|main|all", rest[i + 1]);
                }
                i += 2;
            }
            "--nn-eval-budget" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<u64>() {
                    nn_eval_budget = n;
                }
                i += 2;
            }
            "--hash" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<usize>() {
                    hash_mb = n.max(1);
                }
                i += 2;
            }
            other => {
                eprintln!("--bench: 忽略未知参数 {other}");
                i += 1;
            }
        }
    }
    BenchCli {
        depth,
        max_nodes,
        onnx,
        vocab,
        data_dir,
        require_onnx,
        ablation: SearchAblation {
            policy_ordering: !no_policy_ordering,
            nn_leaf_mode: nn_leaf_mode.unwrap_or(NNLeafMode::MainLeafOnly),
        },
        hash_mb,
        nn_eval_budget,
        policy_ordering_explicit,
        nn_leaf_explicit,
    }
}

/// `--bench` 默认对齐单线程 NPS 基线：无 NN 叶值、无 policy 排序（除非显式传参）。
fn apply_bench_throughput_defaults(cli: &mut BenchCli) {
    if !cli.policy_ordering_explicit {
        cli.ablation.policy_ordering = false;
    }
    if !cli.nn_leaf_explicit {
        cli.ablation.nn_leaf_mode = NNLeafMode::Off;
    }
}

fn resolve_bench_onnx_vocab(cli: &BenchCli) -> (Option<PathBuf>, Option<PathBuf>) {
    let onnx = cli
        .onnx
        .clone()
        .or_else(|| {
            cli.data_dir
                .as_ref()
                .map(|d| d.join("policy.onnx"))
                .filter(|p| p.is_file())
        })
        .or_else(|| resolve_data_file("policy.onnx"));

    let vocab = cli
        .vocab
        .clone()
        .or_else(|| {
            cli.data_dir
                .as_ref()
                .map(|d| d.join("move_vocab.json"))
                .filter(|p| p.is_file())
        })
        .or_else(|| resolve_data_file("move_vocab.json"));

    (onnx, vocab)
}

fn run_bench_cli(rest: &[String]) -> io::Result<()> {
    let mut cli = parse_bench_cli(rest);
    apply_bench_throughput_defaults(&mut cli);
    let (onnx_path, vocab_path) = resolve_bench_onnx_vocab(&cli);

    if let Some(ref p) = cli.onnx {
        if !p.is_file() {
            eprintln!("--bench: --onnx 路径不存在: {}", p.display());
            process::exit(1);
        }
    }

    if cli.require_onnx && onnx_path.is_none() {
        eprintln!("--bench: 未找到 policy.onnx（请放入 ./data/、设置 ENGIN_DATA_DIR，或使用 --onnx）");
        process::exit(1);
    }

    let policy = Arc::new(Mutex::new(None));
    let mut meta = BenchJsonMeta::default();

    if let Some(ref op) = onnx_path {
        meta.onnx_path = Some(op.display().to_string());
        match PolicyOnnx::from_file(op) {
            Ok(net) => {
                *policy.lock().unwrap() = Some(net);
                meta.policy_session_loaded = true;
            }
            Err(e) => {
                eprintln!("--bench: 加载 ONNX 失败: {e}");
                if cli.require_onnx {
                    process::exit(1);
                }
            }
        }
    }

    let (vocab, vocab_size) = if let Some(ref vp) = vocab_path {
        meta.vocab_path = Some(vp.display().to_string());
        load_move_vocab(vp).unwrap_or_else(|e| {
            eprintln!("--bench: 词表加载失败: {e}");
            (HashMap::new(), 0)
        })
    } else {
        (HashMap::new(), 0)
    };
    meta.vocab_entries = vocab_size;

    let session = BenchSessionParams {
        max_depth: cli.depth,
        max_nodes: cli.max_nodes,
        policy: &policy,
        vocab: &vocab,
        vocab_size,
        ablation: cli.ablation,
        hash_mb: cli.hash_mb,
        nn_eval_budget: cli.nn_eval_budget,
        meta: &meta,
    };
    let stdout = io::stdout();
    let mut out = stdout.lock();
    write_benchmark_ndjson(&mut out, default_benchmark_fen_strings(), &session)
}

fn run_value_probe_cli(rest: &[String]) {
    let cli = parse_bench_cli(rest);
    let (onnx_path, vocab_path) = resolve_bench_onnx_vocab(&cli);

    let Some(ref op) = onnx_path else {
        eprintln!(
            "--value-probe: 必须能找到并加载 policy.onnx（请放入 ./data/、设置 ENGIN_DATA_DIR、--data-dir 或 --onnx）"
        );
        process::exit(1);
    };
    if !op.is_file() {
        eprintln!("--value-probe: ONNX 路径无效: {}", op.display());
        process::exit(1);
    }

    let policy = Arc::new(Mutex::new(None));
    match PolicyOnnx::from_file(op) {
        Ok(net) => *policy.lock().unwrap() = Some(net),
        Err(e) => {
            eprintln!("--value-probe: 加载 ONNX 失败: {e}");
            process::exit(1);
        }
    }

    let (vocab, vocab_size) = if let Some(ref vp) = vocab_path {
        load_move_vocab(vp).unwrap_or_else(|e| {
            eprintln!("--value-probe: 词表加载失败: {e}");
            (HashMap::new(), 0)
        })
    } else {
        (HashMap::new(), 0)
    };

    let table_args = ValueProbeTableArgs {
        depth: cli.depth,
        hash_mb: cli.hash_mb,
        policy_ordering: cli.ablation.policy_ordering,
        nn_eval_budget: cli.nn_eval_budget,
        onnx_path: op.as_path(),
        policy: &policy,
        vocab: &vocab,
        vocab_size,
    };
    match markdown_table_off_vs_main(&table_args) {
        Ok(md) => print!("{md}"),
        Err(e) => {
            eprintln!("--value-probe: {e}");
            process::exit(1);
        }
    }
}

fn legal_top_lines(logits: &[f32], vocab: &[String], fen: &str, k: usize) -> Result<String, String> {
    let pos = Position::from_fen(fen).map_err(|e| e.to_string())?;
    let mut m: Vec<(usize, String, f32)> = Vec::new();
    for u in legal_moves_uci(&pos) {
        let Some(idx) = vocab.iter().position(|x| x == &u) else {
            continue;
        };
        if idx < logits.len() {
            m.push((idx, u, logits[idx]));
        }
    }
    m.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok(m.into_iter()
        .take(k)
        .map(|(i, u, v)| format!("uci={u} idx={i} logit={v:.6}"))
        .collect::<Vec<_>>()
        .join(" | "))
}

/// 解析 `--onnx-smoke` 后的参数：若首参为已有文件则视为 ONNX，否则视为 FEN。
fn parse_onnx_smoke_args(rest: &[String]) -> (PathBuf, String, Option<PathBuf>) {
    match rest {
        [] => (default_policy_path(), START_FEN.to_string(), None),
        [a] => {
            if Path::new(a).is_file() {
                (PathBuf::from(a), START_FEN.to_string(), None)
            } else {
                (default_policy_path(), a.clone(), None)
            }
        }
        [a, b] => {
            if Path::new(a).is_file() {
                (PathBuf::from(a), b.clone(), None)
            } else {
                (default_policy_path(), a.clone(), Some(PathBuf::from(b)))
            }
        }
        [a, b, c, ..] => {
            if Path::new(a).is_file() {
                let v = Path::new(c).is_file().then(|| PathBuf::from(c));
                (PathBuf::from(a), b.clone(), v)
            } else {
                (default_policy_path(), a.clone(), Some(PathBuf::from(b)))
            }
        }
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(first) = args.next() else {
        if let Err(e) = run_uci_stdio() {
            eprintln!("UCI I/O 错误: {e}");
            process::exit(1);
        }
        return;
    };

    if first == "--help" || first == "-h" {
        print_usage();
        return;
    }

    if first == "--bench" {
        let rest: Vec<String> = args.collect();
        if let Err(e) = run_bench_cli(&rest) {
            eprintln!("--bench I/O: {e}");
            process::exit(1);
        }
        return;
    }

    if first == "--value-probe" {
        let rest: Vec<String> = args.collect();
        run_value_probe_cli(&rest);
        return;
    }

    if first == "--onnx-smoke" {
        let rest: Vec<String> = args.collect();
        let (onnx_path, fen, vocab_arg) = parse_onnx_smoke_args(&rest);
        let vocab_path = vocab_arg.or_else(|| {
            let p = default_vocab_path();
            p.is_file().then_some(p)
        });

        if !onnx_path.is_file() {
            eprintln!("找不到 ONNX 文件: {}", onnx_path.display());
            eprintln!("请先导出 policy.onnx 到仓库根目录 data/，或把 .onnx 路径作为首参。");
            process::exit(1);
        }
        let mut net = match PolicyOnnx::from_file(&onnx_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("加载 ONNX 失败: {e}");
                process::exit(1);
            }
        };
        let fen = fen.as_str();
        let out = match net.eval_fen(fen) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("推理失败: {e}");
                process::exit(1);
            }
        };

        println!("model: {}", onnx_path.display());
        println!("fen: {fen}");
        println!("logits_len: {}", out.logits.len());

        if let Some(ref vp) = vocab_path {
            match load_move_vocab_ordered(vp) {
                Ok(vocab) => {
                    if vocab.len() != out.logits.len() {
                        eprintln!(
                            "警告: 词表长度 {} 与 logits {} 不一致，跳过 UCI 解码",
                            vocab.len(),
                            out.logits.len()
                        );
                    } else {
                        println!("vocab: {}", vp.display());
                        match legal_top_lines(&out.logits, &vocab, fen, 8) {
                            Ok(s) => {
                                println!("policy_legal_top8: {s}");
                            }
                            Err(e) => eprintln!("policy_legal_top8: (skip) {e}"),
                        }
                    }
                }
                Err(e) => eprintln!("词表加载失败: {e}"),
            }
        } else {
            eprintln!("未找到词表；下列为全词表 logit 最高的下标（常含非法着，仅调试用）。");
            let mut top: Vec<(usize, f32)> = out.logits.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let s: Vec<String> = top
                .iter()
                .take(5)
                .map(|(i, v)| format!("idx={i} logit={v:.6}"))
                .collect();
            println!("policy_raw_top5_no_vocab: {}", s.join(" | "));
        }

        print!("aux(onnx): ");
        if let (Some(a), Some(d), Some(t)) = (out.attack, out.danger, out.tactical) {
            println!("attack={a:.6} danger={d:.6} tactical={t:.6}");
        } else {
            println!("{:?}", (out.attack, out.danger, out.tactical));
        }
        print!("value(onnx): ");
        if let Some(v) = out.value {
            println!("{v:.6}");
        } else {
            println!("None");
        }
        return;
    }

    eprintln!("未知参数: {first}");
    print_usage();
    process::exit(1);
}
