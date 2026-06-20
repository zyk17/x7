//! 用户侧引擎：默认 UCI；`--onnx-smoke` 为 ONNX 冒烟；`--bench` 为 MCTS 基准。

use std::collections::HashMap;
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, Mutex};

use xiangqi_core::{legal_moves_uci, Position};

use engin::benchmark::{
    default_benchmark_fen_strings, resolve_data_file, write_benchmark_ndjson, BenchJsonMeta, BenchSessionParams,
};
use engin::mcts::{MctsBudget, MctsConfig};
use engin::vocab::{load_move_vocab, load_move_vocab_ordered};
use engin::{run_uci_stdio, PolicyOnnx, START_FEN};

fn default_policy_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/policy.onnx")
}

fn default_vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/move_vocab.json")
}

fn print_usage() {
    eprintln!(
        "用法:\n  engin                         UCI 模式（stdin/stdout）\n  \
         engin --onnx-smoke [ONNX] [FEN] [VOCAB]  冒烟；缺省 ONNX=data/policy.onnx、FEN=起始局面\n  \
         engin --bench [选项]            MCTS 基准（NDJSON）\n  \
         --bench 选项: --visits N  --nodes N  --movetime MS  --cpuct F  --onnx PATH  --vocab PATH  --data-dir PATH  --require-onnx"
    );
}

#[derive(Debug)]
struct BenchCli {
    budget: MctsBudget,
    config: MctsConfig,
    onnx: Option<PathBuf>,
    vocab: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    require_onnx: bool,
}

fn parse_bench_cli(rest: &[String]) -> BenchCli {
    let mut budget = MctsBudget {
        max_visits: Some(256),
        max_nodes: None,
        deadline: None,
        stop: None,
    };
    let mut config = MctsConfig::default();
    let mut onnx: Option<PathBuf> = None;
    let mut vocab: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut require_onnx = false;
    let mut i = 0usize;
    while i < rest.len() {
        match rest[i].as_str() {
            "--visits" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<u32>() {
                    budget.max_visits = Some(n.max(1));
                }
                i += 2;
            }
            "--nodes" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<u32>() {
                    budget.max_nodes = Some(n.max(1));
                }
                i += 2;
            }
            "--movetime" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<u64>() {
                    budget = MctsBudget::from_movetime_ms(n.max(1));
                }
                i += 2;
            }
            "--cpuct" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<f32>() {
                    config.cpuct = n.clamp(0.01, 100.0);
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
            other => {
                eprintln!("--bench: 忽略未知参数 {other}");
                i += 1;
            }
        }
    }
    BenchCli {
        budget,
        config,
        onnx,
        vocab,
        data_dir,
        require_onnx,
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
    let cli = parse_bench_cli(rest);
    let (onnx_path, vocab_path) = resolve_bench_onnx_vocab(&cli);

    if let Some(ref p) = cli.onnx {
        if !p.is_file() {
            eprintln!("--bench: --onnx 路径不存在: {}", p.display());
            process::exit(1);
        }
    }
    if cli.require_onnx && onnx_path.is_none() {
        eprintln!("--bench: 未找到 policy.onnx");
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
            Err(err) => {
                eprintln!("--bench: 加载 ONNX 失败: {err}");
                if cli.require_onnx {
                    process::exit(1);
                }
            }
        }
    }

    let vocab = if let Some(ref vp) = vocab_path {
        meta.vocab_path = Some(vp.display().to_string());
        let (vocab, size) = load_move_vocab(vp).unwrap_or_else(|err| {
            eprintln!("--bench: 词表加载失败: {err}");
            (HashMap::new(), 0)
        });
        meta.vocab_entries = size;
        vocab
    } else {
        HashMap::new()
    };

    let session = BenchSessionParams {
        budget: cli.budget,
        config: cli.config,
        policy: &policy,
        vocab: &vocab,
        meta: &meta,
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    write_benchmark_ndjson(&mut out, default_benchmark_fen_strings(), &session)
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

    if first == "--onnx-smoke" {
        let rest: Vec<String> = args.collect();
        let (onnx_path, fen, vocab_arg) = parse_onnx_smoke_args(&rest);
        let vocab_path = vocab_arg.or_else(|| {
            let p = default_vocab_path();
            p.is_file().then_some(p)
        });

        if !onnx_path.is_file() {
            eprintln!("找不到 ONNX 文件: {}", onnx_path.display());
            process::exit(1);
        }
        let mut net = match PolicyOnnx::from_file(&onnx_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("加载 ONNX 失败: {e}");
                process::exit(1);
            }
        };
        let out = match net.eval_fen(&fen) {
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
                        match legal_top_lines(&out.logits, &vocab, &fen, 8) {
                            Ok(s) => println!("policy_legal_top8: {s}"),
                            Err(e) => eprintln!("policy_legal_top8: (skip) {e}"),
                        }
                    }
                }
                Err(e) => eprintln!("词表加载失败: {e}"),
            }
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
