//! 最小 UCI 外壳。
//!
//! 这里刻意不保留旧搜索的 budget、统计和异步控制。MCTS 重建前，`go` 明确
//! 返回空着，避免 GUI 将旧搜索伪装成可用结果。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use xiangqi_core::{uci_to_move, Position, START_FEN};

use crate::history::PositionHistory;

fn default_policy_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/x7.onnx")
}

pub fn parse_position_uci(line: &str) -> Result<Position, String> {
    Ok(parse_position_history(line)?.current().clone_for_search())
}

pub fn parse_position_history_uci(line: &str) -> Result<PositionHistory, String> {
    parse_position_history(line)
}

fn parse_position_history(line: &str) -> Result<PositionHistory, String> {
    let rest = line
        .strip_prefix("position")
        .ok_or_else(|| "internal: position prefix missing".to_string())?
        .trim();

    let (mut history, moves) = if let Some(tail) = rest.strip_prefix("startpos") {
        (
            PositionHistory::from_position(Position::from_fen(START_FEN).map_err(|e| e.to_string())?),
            parse_moves_suffix(tail)?,
        )
    } else if let Some(tail) = rest.strip_prefix("fen") {
        let (fen, moves) = split_fen_and_moves(tail.trim())?;
        (
            PositionHistory::from_position(Position::from_fen(fen).map_err(|e| e.to_string())?),
            moves,
        )
    } else {
        return Err("position requires startpos or fen".into());
    };

    if let Some(moves) = moves {
        for text in moves.split_whitespace() {
            let mv = uci_to_move(history.current(), text).ok_or_else(|| format!("illegal position move: {text}"))?;
            history.push_move(mv);
        }
    }
    Ok(history)
}

fn parse_moves_suffix(tail: &str) -> Result<Option<&str>, String> {
    let tail = tail.trim();
    if tail.is_empty() {
        return Ok(None);
    }
    let moves = tail
        .strip_prefix("moves")
        .ok_or_else(|| "position startpos must be followed by moves".to_string())?
        .trim();
    Ok((!moves.is_empty()).then_some(moves))
}

fn split_fen_and_moves(text: &str) -> Result<(&str, Option<&str>), String> {
    if let Some((fen, moves)) = text.split_once(" moves ") {
        let moves = moves.trim();
        return Ok((fen.trim(), (!moves.is_empty()).then_some(moves)));
    }
    if text.is_empty() {
        Err("position fen is empty".into())
    } else {
        Ok((text, None))
    }
}

fn parse_setoption(line: &str) -> Option<(&str, Option<&str>)> {
    let rest = line.strip_prefix("setoption")?.trim().strip_prefix("name")?.trim();
    if let Some((name, value)) = rest.split_once(" value ") {
        Some((name.trim(), Some(value.trim())))
    } else {
        Some((rest, None))
    }
}

struct UciShell {
    history: PositionHistory,
    policy_file: PathBuf,
}

impl UciShell {
    fn new() -> Self {
        Self {
            history: PositionHistory::new_startpos(),
            policy_file: default_policy_path(),
        }
    }

    fn handle<W: Write>(&mut self, line: &str, out: &mut W) -> io::Result<bool> {
        match line.trim() {
            "uci" => {
                writeln!(out, "id name 77xiangqi_engine")?;
                writeln!(out, "id author 77")?;
                writeln!(
                    out,
                    "option name PolicyFile type string default {}",
                    self.policy_file.display()
                )?;
                writeln!(out, "uciok")?;
            }
            "isready" => writeln!(out, "readyok")?,
            "ucinewgame" => self.history = PositionHistory::new_startpos(),
            "stop" | "ponderhit" => {}
            "quit" => return Ok(false),
            command if command.starts_with("position ") => match parse_position_history(command) {
                Ok(history) => self.history = history,
                Err(error) => writeln!(out, "info string invalid position: {error}")?,
            },
            command if command.starts_with("setoption") => {
                if let Some(("PolicyFile", Some(value))) = parse_setoption(command) {
                    self.policy_file = PathBuf::from(value);
                    writeln!(out, "info string policy path set: {}", self.policy_file.display())?;
                }
            }
            command if command.starts_with("go") => {
                writeln!(out, "info string search unavailable: rebuilding MCTS from lc0 classic")?;
                writeln!(out, "bestmove 0000")?;
            }
            _ => {}
        }
        Ok(true)
    }
}

pub fn run_uci_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_uci(stdin.lock(), stdout.lock())
}

fn run_uci<R: BufRead, W: Write>(input: R, mut out: W) -> io::Result<()> {
    let mut shell = UciShell::new();
    for line in input.lines() {
        let line = line?;
        if !shell.handle(&line, &mut out)? {
            break;
        }
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
pub fn run_uci_for_test<R: BufRead, W: Write>(input: R, output: W) -> io::Result<()> {
    run_uci(input, output)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn position_moves_preserve_history() {
        let history = parse_position_history_uci("position startpos moves h0g2 h9g7").expect("position");
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn invalid_position_move_is_rejected() {
        assert!(parse_position_history_uci("position startpos moves z0z9").is_err());
    }

    #[test]
    fn uci_shell_never_returns_a_legacy_search_move() {
        let input = b"uci\nisready\nposition startpos\ngo nodes 10000\nquit\n";
        let mut output = Vec::new();
        run_uci_for_test(Cursor::new(&input[..]), &mut output).expect("uci");
        let output = String::from_utf8(output).expect("utf8");
        assert!(output.contains("uciok"));
        assert!(output.contains("readyok"));
        assert!(output.contains("bestmove 0000"));
    }
}
