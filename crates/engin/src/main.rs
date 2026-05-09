//! 用户侧引擎：默认 **UCI（stdin/stdout）**；`--onnx-smoke` 为 ONNX 冒烟。

use std::env;
use std::path::{Path, PathBuf};
use std::process;

use xiangqi_core::{legal_moves_uci, Position};

use engin::vocab::load_move_vocab_ordered;
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
         engin --onnx-smoke [ONNX] [FEN] [VOCAB]  冒烟；缺省 ONNX=data/policy.onnx、FEN=起始局面；VOCAB 缺省且存在 data/move_vocab.json 则自动加载\n  \
         伪标签真值: cargo test -p xiangqi_dataset user_fen_black_down_material_attack_below_half"
    );
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
