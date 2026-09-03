//! UCI 协议解析与输出。
//!
//! 命令形状与 info 字段历史上参考过 px0 `uciloop`；本模块由 X7 维护，未支持的
//! 命令必须明确拒绝。

use std::collections::HashMap;

use xiangqi_core::STARTPOS_FEN;

use crate::error::EnginError;
use crate::{Engine, Options};

/// bestmove（及可选 ponder）输出。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BestMoveInfo {
    pub bestmove: xiangqi_core::Move,
    pub ponder: xiangqi_core::Move,
}

impl BestMoveInfo {
    pub const fn new(bestmove: xiangqi_core::Move) -> Self {
        Self {
            bestmove,
            ponder: xiangqi_core::Move::NULL,
        }
    }
}

/// UCI info 中的 WDL 三分量。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Wdl {
    pub w: i32,
    pub d: i32,
    pub l: i32,
}

/// UCI `info` 行字段。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThinkingInfo {
    pub depth: i32,
    pub seldepth: i32,
    pub time: i64,
    pub nodes: i64,
    pub nps: i32,
    pub eps: i32,
    pub mate: Option<i32>,
    pub score: Option<i32>,
    pub wdl: Option<Wdl>,
    pub pv: Vec<xiangqi_core::Move>,
    pub multipv: i32,
    pub comment: String,
}

impl Default for ThinkingInfo {
    /// 未出现字段保持默认哨兵值（负数 / None）。
    fn default() -> Self {
        Self {
            depth: -1,
            seldepth: -1,
            time: -1,
            nodes: -1,
            nps: -1,
            eps: -1,
            mate: None,
            score: None,
            wdl: None,
            pv: Vec::new(),
            multipv: -1,
            comment: String::new(),
        }
    }
}

/// UCI `go` 参数。
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GoParams {
    pub wtime: Option<i64>,
    pub btime: Option<i64>,
    pub winc: Option<i64>,
    pub binc: Option<i64>,
    pub movestogo: Option<i32>,
    pub depth: Option<i32>,
    pub mate: Option<i32>,
    pub nodes: Option<i32>,
    pub movetime: Option<i64>,
    pub infinite: bool,
    pub searchmoves: Vec<String>,
    pub ponder: bool,
}

/// UCI 主循环：解析命令并驱动 Engine。
pub struct UciLoop<'a> {
    engine: &'a mut Engine,
}

impl<'a> UciLoop<'a> {
    pub fn new(engine: &'a mut Engine) -> Self {
        Self { engine }
    }

    fn dispatch_command(&mut self, command: &str, params: &HashMap<String, String>) -> Result<bool, EnginError> {
        match command {
            "uci" => {
                let mut response = vec![
                    format!("id name x7 v{}", env!("CARGO_PKG_VERSION")),
                    "id author 777".into(),
                    String::new(),
                ];
                response.extend(self.engine.options().list_options_uci());
                response.push("uciok".into());
                write_stdout(&response);
            }
            "isready" => {
                self.engine.ensure_ready()?;
                write_stdout(&["readyok".into()]);
            }
            "setoption" => {
                if get_or_empty(params, "name").is_empty() {
                    return Err(EnginError::Uci("setoption requires name".into()));
                }
                self.engine
                    .set_option(get_or_empty(params, "name"), get_or_empty(params, "value"))?;
            }
            "ucinewgame" => self.engine.new_game()?,
            "position" => {
                if contains_key(params, "fen") == contains_key(params, "startpos") {
                    return Err(EnginError::Uci("Position requires either fen or startpos".into()));
                }
                let moves = split_at_whitespace(get_or_empty(params, "moves"));
                let fen = get_or_empty(params, "fen");
                self.engine
                    .set_position(if fen.is_empty() { STARTPOS_FEN } else { fen }, &moves)?;
            }
            "go" => {
                let mut go_params = GoParams::default();
                // px0 只接受 `infinite`（`uciloop.cc:70,209-213`）。`infinity` 是本地别名，
                // 设置相同的 `GoParams::infinite` flag。
                for flag in ["infinite", "infinity"] {
                    if !contains_key(params, flag) {
                        continue;
                    }
                    let value = get_or_empty(params, flag);
                    if !value.is_empty() {
                        return Err(EnginError::Uci(format!("Unexpected token {value}")));
                    }
                    go_params.infinite = true;
                }
                if contains_key(params, "searchmoves") {
                    go_params.searchmoves = split_at_whitespace(get_or_empty(params, "searchmoves"));
                }
                if contains_key(params, "ponder") {
                    let value = get_or_empty(params, "ponder");
                    if !value.is_empty() {
                        return Err(EnginError::Uci(format!("Unexpected token {value}")));
                    }
                    go_params.ponder = true;
                }
                macro_rules! ucigooption {
                    ($field:ident) => {
                        if contains_key(params, stringify!($field)) {
                            go_params.$field = Some(get_numeric(params, stringify!($field))? as i64);
                        }
                    };
                    ($field:ident, i32) => {
                        if contains_key(params, stringify!($field)) {
                            go_params.$field = Some(get_numeric(params, stringify!($field))?);
                        }
                    };
                }
                ucigooption!(wtime);
                ucigooption!(btime);
                ucigooption!(winc);
                ucigooption!(binc);
                ucigooption!(movestogo, i32);
                ucigooption!(depth, i32);
                ucigooption!(mate, i32);
                ucigooption!(nodes, i32);
                ucigooption!(movetime);
                self.engine.go(&go_params)?;
            }
            "wait" => self.engine.wait()?,
            "stop" => self.engine.stop()?,
            "ponderhit" => self.engine.ponder_hit()?,
            "quit" => return Ok(false),
            _ => return Err(EnginError::Uci(format!("Unknown command: {command}"))),
        }
        Ok(true)
    }

    /// 处理一行 UCI 输入；返回 false 表示 quit。
    pub fn process_line(&mut self, line: &str) -> Result<bool, EnginError> {
        let (command, params) = parse_command(line)?;
        if command.is_empty() {
            return Ok(true);
        }
        self.dispatch_command(&command, &params)
    }
}

impl Drop for UciLoop<'_> {
    /// 退出前确保搜索已停止。
    fn drop(&mut self) {
        let _ = self.engine.stop();
    }
}

/// 将一行 UCI 文本解析为 command + kv。
pub fn parse_command(line: &str) -> Result<(String, HashMap<String, String>), EnginError> {
    // PowerShell 管道会在第一行附带 UTF-8 BOM；把它视为传输层前缀而非 UCI token。
    let line = line.trim_start_matches('\u{feff}');
    let mut params = HashMap::new();
    let mut value: Option<&mut String> = None;

    let mut chars = line.chars().peekable();
    let token = read_token(&mut chars);
    if token.is_empty() {
        return Ok((String::new(), params));
    }

    let mut command_token = token;
    if command_token == "fen" || command_token == "startpos" {
        command_token = "position".to_string();
        chars = line.chars().peekable();
    }

    if !is_known_command(&command_token) {
        return Err(EnginError::Uci(format!("Unknown command: {line}")));
    }

    if command_token == "setoption" {
        return parse_setoption(line).map(|params| (command_token, params));
    }

    let mut whitespace = "";
    while let Some(token) = {
        let t = read_token(&mut chars);
        if t.is_empty() { None } else { Some(t) }
    } {
        if !is_command_key(&command_token, &token) {
            let Some(current) = value.as_mut() else {
                return Err(EnginError::Uci(format!("Unexpected token: {token}")));
            };
            current.push_str(whitespace);
            current.push_str(&token);
            whitespace = " ";
        } else {
            let entry = params.entry(token).or_default();
            value = Some(entry);
            whitespace = "";
        }
    }

    Ok((command_token, params))
}

/// 缺 key 时返回空串借用，避免无谓 `String` 拷贝。
pub fn get_or_empty<'a>(params: &'a HashMap<String, String>, key: &str) -> &'a str {
    params.get(key).map(String::as_str).unwrap_or("")
}

/// 读取数值型 UCI 参数。
pub fn get_numeric(params: &HashMap<String, String>, key: &str) -> Result<i32, EnginError> {
    let Some(value) = params.get(key) else {
        return Err(EnginError::Uci("Unexpected error".into()));
    };
    if value.is_empty() {
        return Err(EnginError::Uci(format!("expected value after {key}")));
    }
    value
        .parse::<i32>()
        .map_err(|_| EnginError::Uci(format!("invalid value {value}")))
}

/// 判断命令是否包含指定 key。
pub fn contains_key(params: &HashMap<String, String>, key: &str) -> bool {
    params.contains_key(key)
}

/// 格式化 `bestmove` 行。
pub fn format_best_move(info: &BestMoveInfo) -> String {
    let mut res = format!("bestmove {}", info.bestmove);
    if !info.ponder.is_null() {
        res.push_str(&format!(" ponder {}", info.ponder));
    }
    res
}

/// 格式化 `info` 行。
pub fn format_thinking_info(info: &ThinkingInfo, options: &Options) -> String {
    let mut res = String::from("info");
    if info.depth >= 0 {
        res.push_str(&format!(" depth {}", info.depth.max(1)));
    }
    if info.seldepth >= 0 {
        res.push_str(&format!(" seldepth {}", info.seldepth));
    }
    if info.time >= 0 {
        res.push_str(&format!(" time {}", info.time));
    }
    if info.nodes >= 0 {
        res.push_str(&format!(" nodes {}", info.nodes));
    }
    if let Some(mate) = info.mate {
        res.push_str(&format!(" score mate {mate}"));
    }
    if let Some(score) = info.score {
        res.push_str(&format!(" score cp {score}"));
    }
    if options.show_wdl
        && let Some(wdl) = info.wdl
    {
        res.push_str(&format!(" wdl {} {} {}", wdl.w, wdl.d, wdl.l));
    }
    if info.nps >= 0 {
        res.push_str(&format!(" nps {}", info.nps));
    }
    if info.eps >= 0 && options.show_eps {
        res.push_str(&format!(" eps {}", info.eps));
    }
    if info.multipv >= 0 {
        res.push_str(&format!(" multipv {}", info.multipv));
    }
    if !info.pv.is_empty() {
        res.push_str(" pv");
        for mv in &info.pv {
            res.push(' ');
            res.push_str(&mv.to_string());
        }
    }
    if !info.comment.is_empty() {
        res.push_str(&format!(" string {}", info.comment));
    }
    res
}

/// 主线程与 search owner 共用的 stdout 边界。一次调用持有同一把 stdout 锁，避免
/// `info` / `bestmove` 与同步 UCI 回复交织成半行。
pub(crate) fn write_stdout(responses: &[String]) {
    use std::io::{self, Write};

    let stdout = io::stdout();
    let mut lock = stdout.lock();
    for response in responses {
        let _ = writeln!(lock, "{response}");
    }
    let _ = lock.flush();
}

pub(crate) fn write_stdout_thinking(infos: &[ThinkingInfo], options: &Options) {
    let lines: Vec<_> = infos.iter().map(|info| format_thinking_info(info, options)).collect();
    write_stdout(&lines);
}

pub(crate) fn write_stdout_best_move(info: &BestMoveInfo) {
    write_stdout(&[format_best_move(info)]);
}

fn is_known_command(command: &str) -> bool {
    matches!(
        command,
        "uci"
            | "isready"
            | "setoption"
            | "ucinewgame"
            | "position"
            | "go"
            | "stop"
            | "ponderhit"
            | "quit"
            | "fen"
            | "wait"
    )
}

fn is_command_key(command: &str, key: &str) -> bool {
    match command {
        "setoption" => matches!(key, "name" | "value"),
        "position" => matches!(key, "fen" | "startpos" | "moves"),
        "go" => matches!(
            key,
            "infinite"
                | "infinity"
                | "wtime"
                | "btime"
                | "winc"
                | "binc"
                | "movestogo"
                | "depth"
                | "mate"
                | "nodes"
                | "movetime"
                | "searchmoves"
                | "ponder"
        ),
        _ => false,
    }
}

fn read_token(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    while chars.peek().is_some_and(|c| c.is_whitespace()) {
        chars.next();
    }
    let mut token = String::new();
    while let Some(c) = chars.peek() {
        if c.is_whitespace() {
            break;
        }
        token.push(*c);
        chars.next();
    }
    token
}

fn split_at_whitespace(value: &str) -> Vec<String> {
    if value.is_empty() {
        Vec::new()
    } else {
        value.split_whitespace().map(str::to_string).collect()
    }
}

fn parse_setoption(line: &str) -> Result<HashMap<String, String>, EnginError> {
    let rest = line
        .trim()
        .strip_prefix("setoption")
        .ok_or_else(|| EnginError::Uci("setoption must be followed by name".into()))?
        .trim_start();
    if !rest.starts_with("name ") {
        return Err(EnginError::Uci("setoption must be followed by name".into()));
    }
    let rest = &rest[5..];
    let value_at = token_offset(rest, "value");
    let (name, value) = match value_at {
        Some((start, end)) => (&rest[..start], Some(&rest[end..])),
        None => (rest, None),
    };
    let mut params = HashMap::new();
    params.insert("name".to_string(), name.trim().to_string());
    if let Some(value) = value {
        params.insert("value".to_string(), value.trim().to_string());
    }
    Ok(params)
}

/// 从左到右扫描 `setoption` token。
fn token_offset(text: &str, needle: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut start = 0;
    while start < bytes.len() {
        while start < bytes.len() && bytes[start].is_ascii_whitespace() {
            start += 1;
        }
        let end = bytes[start..]
            .iter()
            .position(|byte| byte.is_ascii_whitespace())
            .map_or(bytes.len(), |offset| start + offset);
        if &text[start..end] == needle {
            return Some((start, end));
        }
        start = end.saturating_add(1);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::Wdl;
    use super::*;
    use xiangqi_core::{Move, Square};

    #[test]
    fn parse_position_startpos_moves() {
        let (command, params) = parse_command("position startpos moves h2h4 h9h7").expect("parse position");
        assert_eq!(command, "position");
        assert!(contains_key(&params, "startpos"));
        assert!(!contains_key(&params, "fen"));
        assert_eq!(get_or_empty(&params, "moves"), "h2h4 h9h7");
    }

    #[test]
    fn parse_fen_shorthand() {
        let (command, params) =
            parse_command("fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1 moves h2h4")
                .expect("parse fen shorthand");
        assert_eq!(command, "position");
        assert!(contains_key(&params, "fen"));
        assert_eq!(get_or_empty(&params, "moves"), "h2h4");
    }

    #[test]
    fn parse_setoption_preserves_spaces() {
        let (command, params) = parse_command("setoption name UCI_ShowWDL value true").expect("parse setoption");
        assert_eq!(command, "setoption");
        assert_eq!(get_or_empty(&params, "name"), "UCI_ShowWDL");
        assert_eq!(get_or_empty(&params, "value"), "true");
    }

    #[test]
    fn parse_setoption_uses_first_value_token() {
        let (_, params) = parse_command("setoption name A value B value C").expect("parse option");
        assert_eq!(get_or_empty(&params, "name"), "A");
        assert_eq!(get_or_empty(&params, "value"), "B value C");

        let (_, params) = parse_command("setoption name UCI_ShowWDL").expect("parse option");
        assert_eq!(get_or_empty(&params, "name"), "UCI_ShowWDL");
        assert!(!contains_key(&params, "value"));
    }

    #[test]
    fn weights_file_option_matches_px0_name() {
        let mut options = Options::default();
        options
            .set_uci_option("WeightsFile", "data/x7.onnx")
            .expect("weights option");
        assert_eq!(options.weights_file, "data/x7.onnx");
        assert!(
            options
                .list_options_uci()
                .iter()
                .any(|line| line == "option name WeightsFile type string default data/x7.onnx")
        );
    }

    #[test]
    fn minibatch_size_option_matches_px0_range() {
        let mut options = Options::default();
        options
            .set_uci_option("NnBatchSize", "128")
            .expect("minibatch-size option");
        assert_eq!(options.nn_batch_size, 128);
        assert!(options.set_uci_option("NnBatchSize", "1025").is_err());
    }

    #[test]
    fn multipv_option_matches_px0_range() {
        let mut options = Options::default();
        options.set_uci_option("MultiPV", "3").expect("multipv option");
        assert_eq!(options.multi_pv, 3);
        assert!(options.set_uci_option("MultiPV", "0").is_err());
        assert!(options.set_uci_option("MultiPV", "501").is_err());
        assert!(
            options
                .list_options_uci()
                .iter()
                .any(|line| line == "option name MultiPV type spin default 3 min 1 max 500")
        );
    }

    #[test]
    fn search_options_accept_puct_params_and_threads() {
        let mut options = Options::default();
        options.set_uci_option("cpuct", "1.5").expect("cpuct");
        options.set_uci_option("CPUctBase", "20000").expect("cpuct base");
        options.set_uci_option("cpuctfactor", "2.5").expect("cpuct factor");
        options.set_uci_option("fpureduction", "0.35").expect("fpu reduction");
        options
            .set_uci_option("VarianceBonusScale", "0.4")
            .expect("variance bonus scale");
        options.set_uci_option("DecisionLcbStdevs", "4").expect("lcb stdevs");
        options.set_uci_option("DecisionUcbStdevs", "3").expect("ucb stdevs");
        options.set_uci_option("NnWindow", "2.25").expect("nn window");
        options
            .set_uci_option("VirtualMeanFpuScale", "0.75")
            .expect("virtual mean FPU scale");
        options.set_uci_option("DecisionRule", "MixNQ").expect("decision rule");
        options
            .set_uci_option("DecisionMixNWeight", "0.3")
            .expect("decision mix N weight");
        options.set_uci_option("threads", "7").expect("threads");
        options
            .set_uci_option("nncachesizepoweroftwo", "20")
            .expect("cache size power");
        assert_eq!(options.cpuct, 1.5);
        assert_eq!(options.cpuct_base, 20_000.0);
        assert_eq!(options.cpuct_factor, 2.5);
        assert_eq!(options.fpu_reduction, 0.35);
        assert_eq!(options.variance_bonus_scale, 0.4);
        assert_eq!(options.decision_lcb_stdevs, 4.0);
        assert_eq!(options.decision_ucb_stdevs, 3.0);
        assert_eq!(options.nn_window, 2.25);
        assert_eq!(options.virtual_mean_fpu_scale, 0.75);
        assert_eq!(options.decision_rule, crate::search::DecisionRule::MixNQ);
        assert_eq!(options.decision_mix_n_weight, 0.3);
        assert_eq!(options.nn_cache_size_power_of_two, 20);
        assert_eq!(options.threads, 7);
        assert!(options.set_uci_option("CPuct", "NaN").is_err());
        assert!(options.set_uci_option("CPuctBase", "0").is_err());
        assert!(options.set_uci_option("CPuctFactor", "NaN").is_err());
        assert!(options.set_uci_option("FpuReduction", "-0.1").is_err());
        assert!(options.set_uci_option("VarianceBonusScale", "NaN").is_err());
        assert!(options.set_uci_option("DecisionLcbStdevs", "NaN").is_err());
        assert!(options.set_uci_option("DecisionUcbStdevs", "NaN").is_err());
        assert!(options.set_uci_option("NnWindow", "0").is_err());
        assert!(options.set_uci_option("VirtualMeanFpuScale", "-0.1").is_err());
        assert!(options.set_uci_option("DecisionRule", "bad-rule").is_err());
        assert!(options.set_uci_option("DecisionMixNWeight", "-0.1").is_err());
        options
            .set_uci_option("Threads", "1")
            .expect("threads below minimum clamp");
        assert_eq!(options.threads, 2);
        options.set_uci_option("Threads", "0").expect("zero threads clamp");
        assert_eq!(options.threads, 2);
        assert!(options.set_uci_option("Threads", "129").is_err());
        assert!(options.set_uci_option("NnCacheSizePowerOfTwo", "49").is_err());
    }

    #[test]
    fn default_thinking_info_matches_px0_sentinels() {
        assert_eq!(
            format_thinking_info(&ThinkingInfo::default(), &Options::default()),
            "info"
        );
    }

    #[test]
    fn format_thinking_info_matches_px0_fields() {
        let options = Options {
            show_wdl: true,
            show_eps: true,
            ..Options::default()
        };
        let info = ThinkingInfo {
            depth: 0,
            seldepth: 3,
            time: 12,
            nodes: 100,
            nps: 8000,
            eps: 42,
            score: Some(15),
            wdl: Some(Wdl { w: 100, d: 200, l: 300 }),
            pv: vec![Move::new(
                Square::parse("h2").expect("h2"),
                Square::parse("h4").expect("h4"),
            )],
            multipv: 1,
            comment: "note".into(),
            ..ThinkingInfo::default()
        };
        let formatted = format_thinking_info(&info, &options);
        assert!(formatted.contains("depth 1"));
        assert!(formatted.contains("seldepth 3"));
        assert!(formatted.contains("score cp 15"));
        assert!(formatted.contains("wdl 100 200 300"));
        assert!(formatted.contains("eps 42"));
        assert!(formatted.contains(" pv h2h4"));
        assert!(formatted.contains(" string note"));
    }
}
