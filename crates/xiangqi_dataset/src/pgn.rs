//! PGN 读取与 movetext 清理（对齐 `nn/src/pgn.py`）。

use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct ParsedGame {
    pub headers: std::collections::HashMap<String, String>,
    pub movetext_raw: String,
}

fn header_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#"^\[(\w+)\s+"((?:\\.|[^"])*)"\s*\]$"#).expect("header re"))
}

fn iccs_move_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b([A-Ia-i]\d+)\s*-\s*([A-Ia-i]\d+)\b").expect("iccs re"))
}

fn uci_token_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[a-i][0-9][a-i][0-9][a-z]?$").expect("uci re"))
}

fn move_num_dot_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^\d+\.").expect("move num"))
}

fn black_move_num_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^\d+\.\.\.").expect("black move num"))
}

fn brace_comment_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\{[^{}]*\}").expect("brace"))
}

fn event_line_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?m)^\[Event\s").expect("event"))
}

/// 去掉 `{ }` 与 `( )` 变着。
pub fn strip_comments_and_variations(movetext: &str) -> String {
    let mut s = movetext.to_string();
    while s.contains('{') {
        s = brace_comment_re().replace_all(&s, " ").to_string();
    }
    let mut depth = 0i32;
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth = (depth - 1).max(0),
            _ => {
                if depth == 0 {
                    out.push(ch);
                }
            }
        }
    }
    out
}

pub fn movetext_uci_tokens(movetext: &str) -> Vec<String> {
    let clean = strip_comments_and_variations(movetext);
    let mut tokens = Vec::new();
    let uci_re = uci_token_re();
    let md = move_num_dot_re();
    let bd = black_move_num_re();
    for raw in clean.split_whitespace() {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        if matches!(t, "1-0" | "0-1" | "1/2-1/2" | "*") {
            continue;
        }
        if bd.is_match(t) || md.is_match(t) {
            continue;
        }
        if uci_re.is_match(t) {
            tokens.push(t.to_string());
        }
    }
    tokens
}

pub fn movetext_iccs_pairs(movetext: &str) -> Vec<String> {
    let clean = strip_comments_and_variations(movetext);
    let mut out = Vec::new();
    for m in iccs_move_re().captures_iter(&clean) {
        let a = m.get(1).unwrap().as_str().to_lowercase();
        let b = m.get(2).unwrap().as_str().to_lowercase();
        out.push(format!("{a}-{b}"));
    }
    out
}

pub fn pgn_format(headers: &std::collections::HashMap<String, String>) -> String {
    headers
        .get("Format")
        .map(|s| s.trim().to_uppercase())
        .unwrap_or_default()
}

/// 与 Python 一致：以 `[Event` 作为新对局起点。
pub fn read_pgn_games(raw: &str) -> Vec<ParsedGame> {
    let raw = raw.replace("\r\n", "\n").replace('\r', "\n");
    let event_starts: Vec<usize> = event_line_re()
        .find_iter(&raw)
        .map(|m| m.start())
        .collect();
    if event_starts.is_empty() {
        return Vec::new();
    }
    let mut games = Vec::new();
    for i in 0..event_starts.len() {
        let start = event_starts[i];
        let end = event_starts
            .get(i + 1)
            .copied()
            .unwrap_or(raw.len());
        let chunk = raw[start..end].trim();
        if let Some(g) = parse_one_game_chunk(chunk) {
            games.push(g);
        }
    }
    games
}

fn parse_one_game_chunk(chunk: &str) -> Option<ParsedGame> {
    let lines: Vec<&str> = chunk.lines().collect();
    let mut headers = std::collections::HashMap::new();
    let mut movetext_lines: Vec<String> = Vec::new();
    let mut phase = "headers";
    let hr = header_re();
    for line in lines {
        let s = line.trim();
        if phase == "headers" {
            if s.is_empty() {
                phase = "moves";
                continue;
            }
            if s.starts_with('[') {
                if let Some(caps) = hr.captures(s) {
                    let key = caps.get(1)?.as_str().to_string();
                    let mut val = caps.get(2)?.as_str().replace("\\\"", "\"");
                    val = val.replace("\\\\", "\\");
                    headers.insert(key, val);
                }
                continue;
            }
            phase = "moves";
        }
        if phase == "moves" && !s.is_empty() {
            movetext_lines.push(s.to_string());
        }
    }
    if headers.is_empty() {
        return None;
    }
    Some(ParsedGame {
        headers,
        movetext_raw: movetext_lines.join(" "),
    })
}
