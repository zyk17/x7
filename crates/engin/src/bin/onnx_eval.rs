use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process;

use engin::{parse_position_history_uci, px0_policy, PolicyOnnx, PositionHistory};
use serde_json::json;
use xiangqi_core::{legal_moves_uci, uci_to_move, Position};

#[derive(Clone, Debug)]
struct PolicyEntry {
    idx: usize,
    uci: String,
    logit: f32,
}

struct Args {
    onnx: PathBuf,
    inputs: Vec<String>,
    topk: usize,
    out: Option<PathBuf>,
}

fn print_usage() {
    eprintln!(
        "用法:\n  \
         cargo run --release -p engin --bin onnx_eval -- --onnx MODEL --fen \"FEN\"\n  \
         cargo run --release -p engin --bin onnx_eval -- --onnx MODEL --input positions.txt [--topk 8] [--out out.ndjson]\n\n  \
         输入支持：\n  \
         - 单行 FEN\n  \
         - `position startpos moves ...`\n  \
         - `position fen ... moves ...`"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut onnx = None::<PathBuf>;
    let mut fens = Vec::<String>::new();
    let mut input = None::<PathBuf>;
    let mut topk = 8usize;
    let mut out = None::<PathBuf>;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--onnx" => onnx = args.next().map(PathBuf::from),
            "--fen" => {
                let Some(fen) = args.next() else {
                    return Err("--fen 缺少值".into());
                };
                fens.push(fen);
            }
            "--input" => input = args.next().map(PathBuf::from),
            "--topk" => {
                let Some(v) = args.next() else {
                    return Err("--topk 缺少值".into());
                };
                topk = v.parse::<usize>().map_err(|_| format!("非法 --topk: {v}"))?.max(1);
            }
            "--out" => out = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("未知参数: {other}")),
        }
    }

    let Some(onnx) = onnx else {
        return Err("缺少 --onnx".into());
    };
    if let Some(path) = input {
        let file = File::open(&path).map_err(|e| format!("打开输入文件失败 {}: {e}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| format!("读取输入文件失败: {e}"))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            fens.push(trimmed.to_string());
        }
    }
    if fens.is_empty() {
        return Err("至少提供一个 --fen 或 --input".into());
    }
    Ok(Args {
        onnx,
        inputs: fens,
        topk,
        out,
    })
}

fn parse_history(line: &str) -> Result<PositionHistory, String> {
    let trimmed = line.trim();
    if trimmed.starts_with("position ") {
        parse_position_history_uci(trimmed)
    } else {
        PositionHistory::from_fen(trimmed)
    }
}

fn legal_top_entries(pos: &Position, logits: &[f32], k: usize) -> Vec<PolicyEntry> {
    let black_to_move = pos.side_to_move == xiangqi_core::types::Color::Black;
    let mut entries = legal_moves_uci(pos)
        .into_iter()
        .filter_map(|uci| {
            let mv = uci_to_move(pos, &uci)?;
            let idx = px0_policy::px0_policy_index(mv, black_to_move)?;
            let logit = *logits.get(idx)?;
            Some(PolicyEntry { idx, uci, logit })
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(k);
    entries
}

fn entry_json(entries: Vec<PolicyEntry>) -> Vec<serde_json::Value> {
    entries
        .into_iter()
        .map(|entry| {
            json!({
                "idx": entry.idx,
                "uci": entry.uci,
                "logit": entry.logit,
            })
        })
        .collect()
}

fn main() -> io::Result<()> {
    let Args {
        onnx,
        inputs,
        topk,
        out,
    } = match parse_args() {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{err}");
            print_usage();
            process::exit(1);
        }
    };
    if !onnx.is_file() {
        eprintln!("找不到 ONNX 文件: {}", onnx.display());
        process::exit(1);
    }

    let mut net = match PolicyOnnx::from_file(&onnx) {
        Ok(net) => net,
        Err(err) => {
            eprintln!("加载 ONNX 失败: {err}");
            process::exit(1);
        }
    };
    let mut writer: Box<dyn Write> = match out {
        Some(path) => Box::new(File::create(path)?),
        None => Box::new(io::stdout()),
    };

    eprintln!("model={} ep_chain={}", onnx.display(), net.provider_chain());
    for (line_no, input) in inputs.into_iter().enumerate() {
        let history = match parse_history(&input) {
            Ok(history) => history,
            Err(err) => {
                writeln!(
                    writer,
                    "{}",
                    json!({
                        "line": line_no + 1,
                        "input": input,
                        "error": err,
                    })
                )?;
                continue;
            }
        };
        let pos = history.current().clone_for_search();
        let board = match engin::fen_tensor::history_to_planes(&history) {
            Ok(board) => board,
            Err(err) => {
                writeln!(
                    writer,
                    "{}",
                    json!({
                        "line": line_no + 1,
                        "input": input,
                        "fen": pos.fen(),
                        "error": err,
                    })
                )?;
                continue;
            }
        };
        let eval = match net.eval_board(&board) {
            Ok(eval) => eval,
            Err(err) => {
                writeln!(
                    writer,
                    "{}",
                    json!({
                        "line": line_no + 1,
                        "input": input,
                        "fen": pos.fen(),
                        "error": err.to_string(),
                    })
                )?;
                continue;
            }
        };
        let legal_top = entry_json(legal_top_entries(&pos, &eval.logits, topk));
        let wdl = eval.wdl.map(|wdl| {
            json!({
                "w": wdl[0],
                "d": wdl[1],
                "l": wdl[2],
            })
        });
        let q = eval.wdl.map(|wdl| wdl[0] - wdl[2]);
        writeln!(
            writer,
            "{}",
            json!({
                "line": line_no + 1,
                "input": input,
                "fen": pos.fen(),
                "side_to_move": if pos.side_to_move == xiangqi_core::types::Color::Black { "b" } else { "w" },
                "provider_chain": net.provider_chain(),
                "policy_topk_legal": legal_top,
                "wdl": wdl,
                "q": q,
            })
        )?;
    }
    Ok(())
}
