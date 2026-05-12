//! 将当前局面的合法 UCI（常见约定：`a0`～`i9`）排序后逐行打印，便于与其它引擎或库对拍。
//!
//! 用法：`cargo run -p xiangqi_core --bin legal_moves_dump -- [--fen FEN] [--prefix "uci1 uci2 ..."]`
//! 或：`--stdin` 读取一行：`FEN<TAB>前缀`（前缀可为空）。

use std::env;
use std::io::{self, BufRead};
use std::process;

use xiangqi_core::{legal_moves_uci, uci_to_move, Position, START_FEN};

fn usage() -> ! {
    eprintln!(
        "用法: legal_moves_dump [--fen FEN] [--prefix \"m1 m2 ...\"] [--stdin]\n\
         默认 FEN 为起始局面；前缀为从该根局面依次执行的 UCI（空格分隔，纵坐标 0～9）。\n\
         --stdin：从标准输入读一行 `FEN\\t前缀`。"
    );
    process::exit(2);
}

fn parse_cli() -> (String, String) {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return (START_FEN.to_string(), String::new());
    }
    let mut fen: Option<String> = None;
    let mut prefix = String::new();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => usage(),
            "--fen" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("错误: --fen 缺少参数");
                    process::exit(2);
                };
                fen = Some(v.clone());
                i += 1;
            }
            "--prefix" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("错误: --prefix 缺少参数");
                    process::exit(2);
                };
                prefix = v.clone();
                i += 1;
            }
            "--stdin" => {
                let mut line = String::new();
                io::stdin().lock().read_line(&mut line).unwrap_or_default();
                let line = line.trim_end_matches(['\r', '\n']);
                if let Some(tab) = line.find('\t') {
                    fen = Some(line[..tab].trim().to_string());
                    prefix = line[tab + 1..].trim().to_string();
                } else {
                    fen = Some(line.trim().to_string());
                }
                i += 1;
            }
            other => {
                eprintln!("未知参数: {other}");
                usage();
            }
        }
    }
    (fen.unwrap_or_else(|| START_FEN.to_string()), prefix)
}

fn apply_prefix(pos: &mut Position, prefix: &str) -> Result<(), String> {
    let prefix = prefix.trim();
    if prefix.is_empty() {
        return Ok(());
    }
    for tok in prefix.split_whitespace() {
        let mv = uci_to_move(pos, tok).ok_or_else(|| format!("无法解析或非当前行棋方着法: {tok}"))?;
        if !pos.legal(mv) {
            return Err(format!("非法着法（含将军未解等）: {tok}"));
        }
        pos.do_move(mv);
    }
    Ok(())
}

fn main() {
    let (fen, prefix) = parse_cli();
    let mut pos = match Position::from_fen(&fen) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FEN 解析失败: {e}");
            process::exit(1);
        }
    };
    if let Err(e) = apply_prefix(&mut pos, &prefix) {
        eprintln!("前缀走子失败: {e}");
        process::exit(1);
    }
    let mut moves = legal_moves_uci(&pos);
    moves.sort();
    for m in moves {
        println!("{m}");
    }
}
