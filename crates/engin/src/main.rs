//! 用户侧引擎：默认 UCI；`--onnx-smoke` 为 ONNX 冒烟；`--bench` 为 MCTS 基准。

use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, Mutex};

use xiangqi_core::{legal_moves_uci, Position};

use engin::benchmark::{
    default_benchmark_fen_strings, resolve_data_file, write_benchmark_ndjson, BenchJsonMeta, BenchSessionParams,
};
use engin::mcts::{MctsBudget, MctsConfig, SharedPolicy};
use engin::{run_uci_stdio, PolicyOnnx, START_FEN};

fn default_policy_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/policy.onnx")
}

fn print_usage() {
    eprintln!(
        "用法:\n  engin                         UCI 模式（stdin/stdout）\n  \
         engin --onnx-smoke [ONNX] [FEN]  冒烟；缺省 ONNX=data/policy.onnx、FEN=起始局面\n  \
         engin --bench [选项]            MCTS 基准（NDJSON）\n  \
         --bench 选项: --playouts N  --nodes N  --movetime MS  --cpuct F  --search-batch-size N  --onnx PATH  --fen FEN  --data-dir PATH  --require-onnx"
    );
}

#[derive(Debug)]
struct BenchCli {
    budget: MctsBudget,
    config: MctsConfig,
    onnx: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    require_onnx: bool,
    fens: Vec<String>,
}

fn parse_bench_cli(rest: &[String]) -> BenchCli {
    let mut budget = MctsBudget {
        max_playouts: Some(256),
        max_nodes: None,
        deadline: None,
        stop: None,
    };
    let mut config = MctsConfig::default();
    let mut onnx: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut require_onnx = false;
    let mut fens = Vec::new();
    let mut i = 0usize;
    while i < rest.len() {
        match rest[i].as_str() {
            "--playouts" | "--visits" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<u32>() {
                    budget.max_playouts = Some(n.max(1));
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
            "--search-batch-size" if i + 1 < rest.len() => {
                if let Ok(n) = rest[i + 1].parse::<usize>() {
                    config.search_batch_size = n.clamp(1, 64);
                }
                i += 2;
            }
            "--onnx" if i + 1 < rest.len() => {
                onnx = Some(PathBuf::from(&rest[i + 1]));
                i += 2;
            }
            "--data-dir" if i + 1 < rest.len() => {
                data_dir = Some(PathBuf::from(&rest[i + 1]));
                i += 2;
            }
            "--fen" if i + 1 < rest.len() => {
                fens.push(rest[i + 1].clone());
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
        data_dir,
        require_onnx,
        fens,
    }
}

fn resolve_bench_onnx(cli: &BenchCli) -> Option<PathBuf> {
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
    onnx
}

fn run_bench_cli(rest: &[String]) -> io::Result<()> {
    let cli = parse_bench_cli(rest);
    let onnx_path = resolve_bench_onnx(&cli);

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

    let mut policy: SharedPolicy = None;
    let mut meta = BenchJsonMeta::default();

    if let Some(ref op) = onnx_path {
        meta.onnx_path = Some(op.display().to_string());
        match PolicyOnnx::from_file(op) {
            Ok(net) => {
                policy = Some(Arc::new(Mutex::new(net)));
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

    let session = BenchSessionParams {
        budget: cli.budget,
        config: cli.config,
        policy: &policy,
        meta: &meta,
    };
    let default_fens;
    let fens: &[String] = if cli.fens.is_empty() {
        default_fens = default_benchmark_fen_strings().to_vec();
        &default_fens
    } else {
        &cli.fens
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    write_benchmark_ndjson(&mut out, fens, &session)
}

fn legal_top_lines(logits: &[f32], fen: &str, k: usize) -> Result<String, String> {
    let pos = Position::from_fen(fen).map_err(|e| e.to_string())?;
    let black_to_move = pos.side_to_move == xiangqi_core::types::Color::Black;
    let mut m: Vec<(usize, String, f32)> = Vec::new();
    for u in legal_moves_uci(&pos) {
        let Some(mv) = xiangqi_core::uci_to_move(&pos, &u) else {
            continue;
        };
        let Some(idx) = engin::px0_policy::px0_policy_index(mv, black_to_move) else {
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

fn parse_onnx_smoke_args(rest: &[String]) -> (PathBuf, String) {
    match rest {
        [] => (default_policy_path(), START_FEN.to_string()),
        [a] => {
            if Path::new(a).is_file() {
                (PathBuf::from(a), START_FEN.to_string())
            } else {
                (default_policy_path(), a.clone())
            }
        }
        [a, b, ..] if Path::new(a).is_file() => (PathBuf::from(a), b.clone()),
        [a, ..] => (default_policy_path(), a.clone()),
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
        let (onnx_path, fen) = parse_onnx_smoke_args(&rest);

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
        println!("ep_chain: {}", net.provider_chain());
        println!("fen: {fen}");
        println!("logits_len: {}", out.logits.len());

        match legal_top_lines(&out.logits, &fen, 8) {
            Ok(s) => println!("policy_legal_top8: {s}"),
            Err(e) => eprintln!("policy_legal_top8: (skip) {e}"),
        }

        print!("wdl(onnx): ");
        if let Some(wdl) = out.wdl {
            println!(
                "w={:.6} d={:.6} l={:.6} q={:.6}",
                wdl[0],
                wdl[1],
                wdl[2],
                wdl[0] - wdl[2]
            );
        } else {
            println!("None");
        }
        return;
    }

    eprintln!("未知参数: {first}");
    print_usage();
    process::exit(1);
}
