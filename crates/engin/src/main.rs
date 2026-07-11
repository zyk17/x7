//! 用户侧引擎：默认最小 UCI；`--onnx-smoke` 为 ONNX 冒烟。

use std::env;
use std::path::{Path, PathBuf};
use std::process;
use xiangqi_core::{legal_moves_uci, Position};

use engin::{run_uci_stdio, PolicyOnnx, START_FEN};

fn default_policy_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/x7.onnx")
}

fn print_usage() {
    eprintln!(
        "用法:\n  engin                         UCI 模式（stdin/stdout）\n  \
         engin --onnx-smoke [ONNX] [FEN]  冒烟；缺省 ONNX=data/x7.onnx、FEN=起始局面\n  \
         搜索核心正在按 lc0 classic 重建；UCI `go` 当前只返回 bestmove 0000。"
    );
}

fn legal_top_lines(logits: &[f32], fen: &str, k: usize) -> Result<String, String> {
    let pos = Position::from_fen(fen).map_err(|e| e.to_string())?;
    let black_to_move = pos.side_to_move == xiangqi_core::types::Color::Black;
    let mut m: Vec<(usize, String, f32)> = Vec::new();
    for u in legal_moves_uci(&pos) {
        let Some(mv) = xiangqi_core::uci_to_move(&pos, &u) else {
            continue;
        };
        let Some(idx) = engin::move_vocab::move_vocab_index(mv, black_to_move) else {
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
