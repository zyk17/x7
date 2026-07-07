use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process;

use engin::{fen_tensor, parse_position_history_uci, PositionHistory};
use serde_json::json;

fn print_usage() {
    eprintln!(
        "用法:\n  \
         cargo run --release -p engin --bin dump_planes -- --fen \"FEN\"\n  \
         cargo run --release -p engin --bin dump_planes -- --input positions.txt [--out out.ndjson]\n\n  \
         输入支持：\n  \
         - 单行 FEN\n  \
         - `position startpos moves ...`\n  \
         - `position fen ... moves ...`"
    );
}

fn parse_args() -> Result<(Vec<String>, Option<PathBuf>), String> {
    let mut inputs = Vec::<String>::new();
    let mut input_path = None::<PathBuf>;
    let mut out = None::<PathBuf>;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fen" => {
                let Some(fen) = args.next() else {
                    return Err("--fen 缺少值".into());
                };
                inputs.push(fen);
            }
            "--input" => input_path = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("未知参数: {other}")),
        }
    }

    if let Some(path) = input_path {
        let file = File::open(&path).map_err(|e| format!("打开输入文件失败 {}: {e}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| format!("读取输入文件失败: {e}"))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            inputs.push(trimmed.to_string());
        }
    }

    if inputs.is_empty() {
        return Err("至少提供一个 --fen 或 --input".into());
    }

    Ok((inputs, out))
}

fn parse_history(line: &str) -> Result<PositionHistory, String> {
    let trimmed = line.trim();
    if trimmed.starts_with("position ") {
        parse_position_history_uci(trimmed)
    } else {
        PositionHistory::from_fen(trimmed)
    }
}

fn nested_planes(planes: ndarray::ArrayView4<'_, f32>) -> Vec<Vec<Vec<f32>>> {
    let channels = planes.shape()[1];
    let rows = planes.shape()[2];
    let cols = planes.shape()[3];
    let mut out = Vec::with_capacity(channels);
    for c in 0..channels {
        let mut plane = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for col in 0..cols {
                row.push(planes[[0, c, r, col]]);
            }
            plane.push(row);
        }
        out.push(plane);
    }
    out
}

fn plane_nonzero_counts(planes: ndarray::ArrayView4<'_, f32>) -> Vec<usize> {
    let channels = planes.shape()[1];
    let rows = planes.shape()[2];
    let cols = planes.shape()[3];
    let mut counts = Vec::with_capacity(channels);
    for c in 0..channels {
        let mut count = 0usize;
        for r in 0..rows {
            for col in 0..cols {
                if planes[[0, c, r, col]] != 0.0 {
                    count += 1;
                }
            }
        }
        counts.push(count);
    }
    counts
}

fn main() -> io::Result<()> {
    let (inputs, out) = match parse_args() {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{err}");
            print_usage();
            process::exit(1);
        }
    };
    let mut writer: Box<dyn Write> = match out {
        Some(path) => Box::new(File::create(path)?),
        None => Box::new(io::stdout()),
    };

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
        let planes = match fen_tensor::history_to_planes(&history) {
            Ok(planes) => planes,
            Err(err) => {
                writeln!(
                    writer,
                    "{}",
                    json!({
                        "line": line_no + 1,
                        "input": input,
                        "fen": history.current().fen(),
                        "error": err,
                    })
                )?;
                continue;
            }
        };
        let entries = history
            .debug_entries()
            .into_iter()
            .map(|entry| {
                json!({
                    "fen": entry.fen,
                    "repeated": entry.repeated,
                    "side_to_move": entry.side_to_move.to_string(),
                    "rule60": entry.rule60,
                })
            })
            .collect::<Vec<_>>();
        writeln!(
            writer,
            "{}",
            json!({
                "line": line_no + 1,
                "input": input,
                "fen": history.current().fen(),
                "history_len": history.len(),
                "history_entries": entries,
                "shape": [planes.shape()[1], planes.shape()[2], planes.shape()[3]],
                "plane_nonzero_counts": plane_nonzero_counts(planes.view()),
                "planes": nested_planes(planes.view()),
            })
        )?;
    }

    Ok(())
}
