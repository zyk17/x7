//! px0 `src/chess/uciloop.h:42-127` 与 `uciloop.cc:45-337`。

use std::collections::{HashMap, HashSet};

use xiangqi_core::{GameState, STARTPOS_FEN};

use crate::callbacks::{BestMoveInfo, ThinkingInfo};
use crate::error::EnginError;
use crate::neural::cache::DEFAULT_NN_CACHE_SIZE;
use crate::search::classic::{ContemptMode, ScoreType};

/// px0 `GoParams` (`uciloop.h:42-55`)。
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

/// px0 `StringUciResponder::PopulateParams`、classic search display options
/// 与 `SharedBackendParams` 的 Rust ONNX 子集。
#[derive(Clone, Debug, PartialEq)]
pub struct UciOptions {
    pub show_wdl: bool,
    pub show_eps: bool,
    pub show_moves_left: bool,
    /// px0 `MultiPV` / `PerPVCounters` (`search/classic/params.cc:360-368,585-586`).
    pub multi_pv: usize,
    pub per_pv_counters: bool,
    pub score_type: ScoreType,
    /// px0 `NodesPerSecondLimit` (`src/search/classic/params.cc:473-477,621`).
    pub nodes_per_second_limit: f32,
    /// px0 classic WDL calibration options (`params.cc:407-477,608-620`).
    pub uci_opponent: String,
    pub uci_rating_adv: f32,
    pub contempt: String,
    pub contempt_mode: ContemptMode,
    pub contempt_max_value: f32,
    pub wdl_calibration_elo: f32,
    pub wdl_contempt_attenuation: f32,
    pub wdl_max_s: f32,
    pub wdl_eval_objectivity: f32,
    pub wdl_draw_rate_target: f32,
    pub wdl_draw_rate_reference: f32,
    pub wdl_book_exit_bias: f32,
    /// px0 `NNCacheSize` (`src/neural/shared_params.cc:63-82`).
    pub nn_cache_size: usize,
    /// px0 `MoveOverheadMs` (`search/classic/stoppers/factory.cc:44-82`).
    pub move_overhead_ms: i64,
    /// px0 `Slowmover` (`search/classic/stoppers/factory.cc:68-82`).
    pub slowmover: f32,
    /// px0 `WeightsFile` (`src/neural/shared_params.cc:43-80`). This Rust
    /// port accepts the formal ONNX model rather than px0's protobuf weights.
    pub weights_file: String,
}

impl Default for UciOptions {
    fn default() -> Self {
        Self {
            show_wdl: false,
            show_eps: false,
            show_moves_left: false,
            multi_pv: 1,
            per_pv_counters: false,
            score_type: ScoreType::WdlMu,
            nodes_per_second_limit: 0.0,
            uci_opponent: String::new(),
            uci_rating_adv: 0.0,
            contempt: String::new(),
            contempt_mode: ContemptMode::Play,
            contempt_max_value: 420.0,
            wdl_calibration_elo: 0.0,
            wdl_contempt_attenuation: 1.0,
            wdl_max_s: 1.4,
            wdl_eval_objectivity: 1.0,
            wdl_draw_rate_target: 0.0,
            wdl_draw_rate_reference: 0.5,
            wdl_book_exit_bias: 0.65,
            nn_cache_size: DEFAULT_NN_CACHE_SIZE,
            move_overhead_ms: 200,
            slowmover: 1.0,
            weights_file: String::new(),
        }
    }
}

impl UciOptions {
    /// px0 `StringUciResponder::PopulateParams` (`uciloop.cc:263-268`)。
    pub fn populate_defaults() -> Self {
        Self::default()
    }

    /// px0 `OptionsParser::ListOptionsUci` 的已翻译 UCI options。
    pub fn list_options_uci(&self) -> Vec<String> {
        vec![
            format!("option name UCI_ShowWDL type check default {}", bool_uci(self.show_wdl)),
            format!("option name UCI_ShowEPS type check default {}", bool_uci(self.show_eps)),
            format!(
                "option name UCI_ShowMovesLeft type check default {}",
                bool_uci(self.show_moves_left)
            ),
            format!("option name MultiPV type spin default {} min 1 max 500", self.multi_pv),
            format!(
                "option name PerPVCounters type check default {}",
                bool_uci(self.per_pv_counters)
            ),
            format!(
                "option name ScoreType type combo default {} var centipawn var centipawn_with_drawscore var centipawn_2019 var centipawn_2018 var win_percentage var Q var W-L var WDL_mu",
                self.score_type.as_uci()
            ),
            // px0 `FloatOption::GetOptionString` advertises float values as
            // a UCI string (`src/utils/optionsparser.cc:473-475`).
            format!(
                "option name NodesPerSecondLimit type string default {}",
                self.nodes_per_second_limit
            ),
            format!("option name UCI_Opponent type string default {}", self.uci_opponent),
            format!("option name UCI_RatingAdv type string default {}", self.uci_rating_adv),
            format!("option name Contempt type string default {}", self.contempt),
            format!(
                "option name ContemptMode type combo default {} var play var white_side_analysis var black_side_analysis var disable",
                self.contempt_mode.as_uci()
            ),
            format!("option name ContemptMaxValue type string default {}", self.contempt_max_value),
            format!("option name WDLCalibrationElo type string default {}", self.wdl_calibration_elo),
            format!("option name WDLContemptAttenuation type string default {}", self.wdl_contempt_attenuation),
            format!("option name WDLMaxS type string default {}", self.wdl_max_s),
            format!(
                "option name WDLEvalObjectivity type string default {}",
                self.wdl_eval_objectivity
            ),
            format!("option name WDLDrawRateTarget type string default {}", self.wdl_draw_rate_target),
            format!("option name WDLDrawRateReference type string default {}", self.wdl_draw_rate_reference),
            format!("option name WDLBookExitBias type string default {}", self.wdl_book_exit_bias),
            format!(
                "option name NNCacheSize type spin default {} min 0 max 999999999",
                self.nn_cache_size
            ),
            format!("option name MoveOverheadMs type spin default {} min 0 max 100000000", self.move_overhead_ms),
            format!("option name Slowmover type string default {}", self.slowmover),
            format!("option name WeightsFile type string default {}", self.weights_file),
        ]
    }

    /// px0 `OptionsParser::SetUciOption` 的已翻译 option 子集。
    pub fn set_uci_option(&mut self, name: &str, value: &str) -> Result<(), EnginError> {
        match name {
            "UCI_ShowWDL" => self.show_wdl = parse_bool_option(value, "UCI_ShowWDL")?,
            "UCI_ShowEPS" => self.show_eps = parse_bool_option(value, "UCI_ShowEPS")?,
            "UCI_ShowMovesLeft" => self.show_moves_left = parse_bool_option(value, "UCI_ShowMovesLeft")?,
            "MultiPV" => self.multi_pv = parse_multi_pv(value)?,
            "PerPVCounters" => self.per_pv_counters = parse_bool_option(value, "PerPVCounters")?,
            "ScoreType" => self.score_type = parse_score_type(value)?,
            "NodesPerSecondLimit" => self.nodes_per_second_limit = parse_nps_limit(value)?,
            "UCI_Opponent" => self.uci_opponent = value.to_string(),
            "UCI_RatingAdv" => self.uci_rating_adv = parse_float_option(value, "UCI_RatingAdv", -10_000.0, 10_000.0)?,
            "Contempt" => self.contempt = value.to_string(),
            "ContemptMode" => self.contempt_mode = parse_contempt_mode(value)?,
            "ContemptMaxValue" => {
                self.contempt_max_value = parse_float_option(value, "ContemptMaxValue", 0.0, 10_000.0)?
            }
            "WDLCalibrationElo" => {
                self.wdl_calibration_elo = parse_float_option(value, "WDLCalibrationElo", 0.0, 10_000.0)?
            }
            "WDLContemptAttenuation" => {
                self.wdl_contempt_attenuation = parse_float_option(value, "WDLContemptAttenuation", -10.0, 10.0)?
            }
            "WDLMaxS" => self.wdl_max_s = parse_float_option(value, "WDLMaxS", 0.0, 10.0)?,
            "WDLEvalObjectivity" => {
                self.wdl_eval_objectivity = parse_float_option(value, "WDLEvalObjectivity", 0.0, 1.0)?
            }
            "WDLDrawRateTarget" => {
                self.wdl_draw_rate_target = parse_float_option(value, "WDLDrawRateTarget", 0.0, 0.999)?
            }
            "WDLDrawRateReference" => {
                self.wdl_draw_rate_reference = parse_float_option(value, "WDLDrawRateReference", 0.001, 0.999)?
            }
            "WDLBookExitBias" => self.wdl_book_exit_bias = parse_float_option(value, "WDLBookExitBias", -2.0, 2.0)?,
            "NNCacheSize" => self.nn_cache_size = parse_cache_size(value)?,
            "MoveOverheadMs" => self.move_overhead_ms = parse_move_overhead(value)?,
            "Slowmover" => self.slowmover = parse_slowmover(value)?,
            "WeightsFile" => self.weights_file = value.to_string(),
            _ => return Err(EnginError::Uci(format!("Unknown option: {name}"))),
        }
        Ok(())
    }
}

/// px0 `UciResponder` (`callbacks.h:143-148`)。
pub trait UciResponder {
    fn output_best_move(&mut self, info: &BestMoveInfo);
    fn output_thinking_info(&mut self, infos: &[ThinkingInfo]);
}

/// px0 `StringUciResponder` 发送边界（`uciloop.h:57-73`）。
pub trait StringUciResponder: UciResponder {
    fn send_raw_responses(&mut self, responses: &[String]);
    fn set_options(&mut self, options: UciOptions);

    /// px0 `StringUciResponder::SendRawResponse` (`uciloop.cc:270-272`)。
    fn send_raw_response(&mut self, response: &str) {
        self.send_raw_responses(&[response.to_string()]);
    }

    /// px0 `StringUciResponder::SendId` (`uciloop.cc:274-277`)。
    fn send_id(&mut self, version: &str) {
        self.send_raw_response(&format!("id name x7 v{version}"));
        self.send_raw_response("id author aaa");
    }
}

/// px0 `EngineControllerBase` (`uciloop.h:74-99`)。
pub trait EngineController {
    fn register_uci_responder(&mut self, responder: &mut dyn StringUciResponder);
    fn unregister_uci_responder(&mut self, responder: &mut dyn StringUciResponder);
    /// Rust ownership requires explicitly forwarding the global UCI options
    /// that px0's `Engine` reads from its shared `OptionsDict`.
    fn set_uci_options(&mut self, _options: &UciOptions) -> Result<(), EnginError> {
        Ok(())
    }
    fn ensure_ready(&mut self) -> Result<(), EnginError>;
    fn new_game(&mut self) -> Result<(), EnginError>;
    fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError>;
    fn go(&mut self, params: &GoParams, responder: &mut dyn StringUciResponder) -> Result<(), EnginError>;
    fn ponder_hit(&mut self) -> Result<(), EnginError>;
    fn wait(&mut self) -> Result<(), EnginError>;
    fn stop(&mut self, responder: &mut dyn StringUciResponder) -> Result<(), EnginError>;
}

/// px0 `UciLoop` (`uciloop.h:101-118`)。
pub struct UciLoop<'a> {
    pub responder: &'a mut dyn StringUciResponder,
    pub options: &'a mut UciOptions,
    pub engine: &'a mut dyn EngineController,
}

impl<'a> UciLoop<'a> {
    /// px0 `UciLoop::UciLoop` (`uciloop.cc:170-175`)。
    pub fn new(
        responder: &'a mut dyn StringUciResponder,
        options: &'a mut UciOptions,
        engine: &'a mut dyn EngineController,
    ) -> Self {
        responder.set_options(options.clone());
        engine.register_uci_responder(responder);
        Self {
            responder,
            options,
            engine,
        }
    }

    /// px0 `UciLoop::DispatchCommand` (`uciloop.cc:178-254`)。
    pub fn dispatch_command(
        &mut self,
        command: &str,
        params: &HashMap<String, String>,
        version: &str,
    ) -> Result<bool, EnginError> {
        match command {
            "uci" => {
                self.responder.send_id(version);
                for option in self.options.list_options_uci() {
                    self.responder.send_raw_response(&option);
                }
                self.responder.send_raw_response("uciok");
            }
            "isready" => {
                self.engine.ensure_ready()?;
                self.responder.send_raw_response("readyok");
            }
            "setoption" => {
                if get_or_empty(params, "name").is_empty() {
                    return Err(EnginError::Uci("setoption requires name".into()));
                }
                self.options
                    .set_uci_option(&get_or_empty(params, "name"), &get_or_empty(params, "value"))?;
                self.engine.set_uci_options(self.options)?;
                self.responder.set_options(self.options.clone());
            }
            "ucinewgame" => self.engine.new_game()?,
            "position" => {
                if contains_key(params, "fen") == contains_key(params, "startpos") {
                    return Err(EnginError::Uci("Position requires either fen or startpos".into()));
                }
                let moves = split_at_whitespace(&get_or_empty(params, "moves"));
                let fen = get_or_empty(params, "fen");
                self.engine
                    .set_position(if fen.is_empty() { STARTPOS_FEN } else { &fen }, &moves)?;
            }
            "go" => {
                let mut go_params = GoParams::default();
                // px0 only accepts `infinite` (`uciloop.cc:70,209-213`). `infinity`
                // is a local alias that sets the same `GoParams::infinite` flag.
                for flag in ["infinite", "infinity"] {
                    if !contains_key(params, flag) {
                        continue;
                    }
                    if !get_or_empty(params, flag).is_empty() {
                        return Err(EnginError::Uci(format!(
                            "Unexpected token {}",
                            get_or_empty(params, flag)
                        )));
                    }
                    go_params.infinite = true;
                }
                if contains_key(params, "searchmoves") {
                    go_params.searchmoves = split_at_whitespace(&get_or_empty(params, "searchmoves"));
                }
                if contains_key(params, "ponder") {
                    if !get_or_empty(params, "ponder").is_empty() {
                        return Err(EnginError::Uci(format!(
                            "Unexpected token {}",
                            get_or_empty(params, "ponder")
                        )));
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
                self.engine.go(&go_params, self.responder)?;
            }
            "wait" => self.engine.wait()?,
            "stop" => self.engine.stop(self.responder)?,
            "ponderhit" => self.engine.ponder_hit()?,
            "xyzzy" => self.responder.send_raw_response("Nothing happens."),
            "quit" => return Ok(false),
            _ => return Err(EnginError::Uci(format!("Unknown command: {command}"))),
        }
        Ok(true)
    }

    /// px0 `UciLoop::ProcessLine` (`uciloop.cc:256-261`)。
    pub fn process_line(&mut self, line: &str, version: &str) -> Result<bool, EnginError> {
        let (command, params) = parse_command(line)?;
        if command.is_empty() {
            return Ok(true);
        }
        self.dispatch_command(&command, &params, version)
    }
}

impl Drop for UciLoop<'_> {
    /// px0 `UciLoop::~UciLoop` (`uciloop.cc:176`).
    fn drop(&mut self) {
        self.engine.unregister_uci_responder(self.responder);
    }
}

/// px0 `ParseCommand` (`uciloop.cc:81-135`)。
pub fn parse_command(line: &str) -> Result<(String, HashMap<String, String>), EnginError> {
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

    if !known_commands().contains(command_token.as_str()) {
        return Err(EnginError::Uci(format!("Unknown command: {line}")));
    }

    if command_token == "setoption" {
        return parse_setoption(line).map(|params| (command_token, params));
    }

    let mut whitespace = "";
    while let Some(token) = {
        let t = read_token(&mut chars);
        if t.is_empty() {
            None
        } else {
            Some(t)
        }
    } {
        let command_keys = known_command_keys(command_token.as_str());
        if !command_keys.contains(token.as_str()) {
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

/// px0 `GetOrEmpty` (`uciloop.cc:137-143`)。
pub fn get_or_empty(params: &HashMap<String, String>, key: &str) -> String {
    params.get(key).cloned().unwrap_or_default()
}

/// px0 `GetNumeric` (`uciloop.cc:145-162`)。
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

/// px0 `ContainsKey` (`uciloop.cc:164-167`)。
pub fn contains_key(params: &HashMap<String, String>, key: &str) -> bool {
    params.contains_key(key)
}

/// px0 `StringUciResponder::OutputBestMove` (`uciloop.cc:279-287`)。
pub fn format_best_move(info: &BestMoveInfo) -> String {
    let mut res = format!("bestmove {}", info.bestmove);
    if !info.ponder.is_null() {
        res.push_str(&format!(" ponder {}", info.ponder));
    }
    if info.player != -1 {
        res.push_str(&format!(" player {}", info.player));
    }
    if info.game_id != -1 {
        res.push_str(&format!(" gameid {}", info.game_id));
    }
    if let Some(is_black) = info.is_black {
        res.push_str(if is_black { " side black" } else { " side white" });
    }
    res
}

/// px0 `StringUciResponder::OutputThinkingInfo` (`uciloop.cc:289-327`)。
pub fn format_thinking_info(info: &ThinkingInfo, options: &UciOptions) -> String {
    let mut res = String::from("info");
    if info.player != -1 {
        res.push_str(&format!(" player {}", info.player));
    }
    if info.game_id != -1 {
        res.push_str(&format!(" gameid {}", info.game_id));
    }
    if let Some(is_black) = info.is_black {
        res.push_str(if is_black { " side black" } else { " side white" });
    }
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
    if let Some(wdl) = info.wdl {
        if options.show_wdl {
            res.push_str(&format!(" wdl {} {} {}", wdl.w, wdl.d, wdl.l));
        }
    }
    if let Some(moves_left) = info.moves_left {
        if options.show_moves_left {
            res.push_str(&format!(" movesleft {moves_left}"));
        }
    }
    if info.hashfull >= 0 {
        res.push_str(&format!(" hashfull {}", info.hashfull));
    }
    if info.nps >= 0 {
        res.push_str(&format!(" nps {}", info.nps));
    }
    if info.eps >= 0 && options.show_eps {
        res.push_str(&format!(" eps {}", info.eps));
    }
    if info.tb_hits >= 0 {
        res.push_str(&format!(" tbhits {}", info.tb_hits));
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

/// 收集 stdout 响应用于 transcript 测试。
#[derive(Clone, Debug, Default)]
pub struct VecUciResponder {
    pub responses: Vec<String>,
    pub options: UciOptions,
}

/// px0 `StdoutUciResponder` (`uciloop.h:120-123`、`uciloop.cc:329-337`)。
#[derive(Clone, Debug, Default)]
pub struct StdoutUciResponder {
    pub options: UciOptions,
}

impl UciResponder for StdoutUciResponder {
    fn output_best_move(&mut self, info: &BestMoveInfo) {
        self.send_raw_response(&format_best_move(info));
    }

    fn output_thinking_info(&mut self, infos: &[ThinkingInfo]) {
        let lines: Vec<String> = infos
            .iter()
            .map(|info| format_thinking_info(info, &self.options))
            .collect();
        self.send_raw_responses(&lines);
    }
}

impl StringUciResponder for StdoutUciResponder {
    fn send_raw_responses(&mut self, responses: &[String]) {
        use std::io::{self, Write};
        let stdout = io::stdout();
        let mut lock = stdout.lock();
        for response in responses {
            let _ = writeln!(lock, "{response}");
        }
    }

    fn set_options(&mut self, options: UciOptions) {
        self.options = options;
    }
}

impl UciResponder for VecUciResponder {
    fn output_best_move(&mut self, info: &BestMoveInfo) {
        self.send_raw_response(&format_best_move(info));
    }

    fn output_thinking_info(&mut self, infos: &[ThinkingInfo]) {
        let lines: Vec<String> = infos
            .iter()
            .map(|info| format_thinking_info(info, &self.options))
            .collect();
        self.send_raw_responses(&lines);
    }
}

impl StringUciResponder for VecUciResponder {
    fn send_raw_responses(&mut self, responses: &[String]) {
        self.responses.extend_from_slice(responses);
    }

    fn set_options(&mut self, options: UciOptions) {
        self.options = options;
    }
}

fn known_commands() -> HashSet<&'static str> {
    HashSet::from([
        "uci",
        "isready",
        "setoption",
        "ucinewgame",
        "position",
        "go",
        "stop",
        "ponderhit",
        "quit",
        "xyzzy",
        "fen",
        "wait",
    ])
}

fn known_command_keys(command: &str) -> HashSet<&'static str> {
    match command {
        "setoption" => HashSet::from(["name", "value"]),
        "position" => HashSet::from(["fen", "startpos", "moves"]),
        "go" => HashSet::from([
            "infinite",
            "infinity",
            "wtime",
            "btime",
            "winc",
            "binc",
            "movestogo",
            "depth",
            "mate",
            "nodes",
            "movetime",
            "searchmoves",
            "ponder",
        ]),
        _ => HashSet::new(),
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

fn trim(value: &str) -> &str {
    value.trim()
}

fn bool_uci(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn parse_bool_option(value: &str, name: &str) -> Result<bool, EnginError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(EnginError::Uci(format!("Flag '{name}' must be either true or false"))),
    }
}

/// px0 `IntOption(kMultiPvId, 1, 500)` (`search/classic/params.cc:585`).
fn parse_multi_pv(value: &str) -> Result<usize, EnginError> {
    let value: usize = value
        .parse()
        .map_err(|_| EnginError::Uci("MultiPV must be an integer from 1 to 500".into()))?;
    if !(1..=500).contains(&value) {
        return Err(EnginError::Uci("MultiPV must be an integer from 1 to 500".into()));
    }
    Ok(value)
}

/// px0 `ChoiceOption(kScoreTypeId, ...)` (`search/classic/params.cc:587-595`).
fn parse_score_type(value: &str) -> Result<ScoreType, EnginError> {
    ScoreType::parse_uci(value).ok_or_else(|| EnginError::Uci(format!("Unknown ScoreType: {value}")))
}

/// px0 `ChoiceOption(kContemptModeId, ...)`
/// (`src/search/classic/params.h:117-123,params.cc:606-608`).
fn parse_contempt_mode(value: &str) -> Result<ContemptMode, EnginError> {
    ContemptMode::parse_uci(value).ok_or_else(|| EnginError::Uci(format!("Unknown ContemptMode: {value}")))
}

/// px0 `FloatOption(kNpsLimitId, 0.0f, 1e6f)`
/// (`src/search/classic/params.cc:473-477,621`).
fn parse_nps_limit(value: &str) -> Result<f32, EnginError> {
    let value: f32 = value
        .parse()
        .map_err(|_| EnginError::Uci("NodesPerSecondLimit must be a number from 0 to 1000000".into()))?;
    if !value.is_finite() || !(0.0..=1_000_000.0).contains(&value) {
        return Err(EnginError::Uci(
            "NodesPerSecondLimit must be a number from 0 to 1000000".into(),
        ));
    }
    Ok(value)
}

/// px0 `IntOption(kNNCacheSizeId, 0, 999999999)`
/// (`src/neural/shared_params.cc:63-82`).
fn parse_cache_size(value: &str) -> Result<usize, EnginError> {
    let value: usize = value
        .parse()
        .map_err(|_| EnginError::Uci("NNCacheSize must be an integer from 0 to 999999999".into()))?;
    if value > 999_999_999 {
        return Err(EnginError::Uci(
            "NNCacheSize must be an integer from 0 to 999999999".into(),
        ));
    }
    Ok(value)
}

/// px0 `IntOption(kMoveOverheadId, 0, 100000000)`
/// (`search/classic/stoppers/factory.cc:73-82`).
fn parse_move_overhead(value: &str) -> Result<i64, EnginError> {
    let value: i64 = value
        .parse()
        .map_err(|_| EnginError::Uci("MoveOverheadMs must be an integer from 0 to 100000000".into()))?;
    if !(0..=100_000_000).contains(&value) {
        return Err(EnginError::Uci(
            "MoveOverheadMs must be an integer from 0 to 100000000".into(),
        ));
    }
    Ok(value)
}

/// px0 `FloatOption(kSlowMoverId, 0.0, 100.0)`
/// (`search/classic/stoppers/factory.cc:80-82`).
fn parse_slowmover(value: &str) -> Result<f32, EnginError> {
    parse_float_option(value, "Slowmover", 0.0, 100.0)
}

fn parse_float_option(value: &str, name: &str, min: f32, max: f32) -> Result<f32, EnginError> {
    let value: f32 = value
        .parse()
        .map_err(|_| EnginError::Uci(format!("{name} must be a number from {min} to {max}")))?;
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(EnginError::Uci(format!("{name} must be a number from {min} to {max}")));
    }
    Ok(value)
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
    params.insert("name".to_string(), trim(name).to_string());
    if let Some(value) = value {
        params.insert("value".to_string(), trim(value).to_string());
    }
    Ok(params)
}

/// px0 `ParseCommand` scans `setoption` tokens left-to-right (`uciloop.cc:109-118`).
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

/// 测试用：记录 UCI 命令如何落到 `GameState`。
#[derive(Clone, Debug, Default)]
pub struct RecordingEngine {
    pub position: Option<GameState>,
    pub ready: bool,
    pub go_count: u32,
    pub last_go: Option<GoParams>,
    pub responder_registrations: u32,
}

impl EngineController for RecordingEngine {
    fn register_uci_responder(&mut self, _responder: &mut dyn StringUciResponder) {
        self.responder_registrations += 1;
    }

    fn unregister_uci_responder(&mut self, _responder: &mut dyn StringUciResponder) {
        self.responder_registrations -= 1;
    }

    fn ensure_ready(&mut self) -> Result<(), EnginError> {
        self.ready = true;
        Ok(())
    }

    fn new_game(&mut self) -> Result<(), EnginError> {
        self.set_position(STARTPOS_FEN, &[])
    }

    fn set_position(&mut self, fen: &str, moves: &[String]) -> Result<(), EnginError> {
        self.position = Some(GameState::from_fen_moves(fen, moves)?);
        Ok(())
    }

    fn go(&mut self, params: &GoParams, _responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        self.go_count += 1;
        self.last_go = Some(params.clone());
        Ok(())
    }

    fn ponder_hit(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn wait(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn stop(&mut self, _responder: &mut dyn StringUciResponder) -> Result<(), EnginError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callbacks::Wdl;
    use std::sync::Once;
    use xiangqi_core::{initialize_magic_bitboards, Move, Square};

    static INIT: Once = Once::new();

    fn ensure_init() {
        INIT.call_once(initialize_magic_bitboards);
    }

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
    fn go_infinity_aliases_infinite() {
        let (command, params) = parse_command("go infinity").expect("parse go infinity");
        assert_eq!(command, "go");
        assert!(contains_key(&params, "infinity"));
        assert!(!contains_key(&params, "infinite"));

        let mut options = UciOptions::populate_defaults();
        let mut engine = RecordingEngine::default();
        let mut responder = VecUciResponder::default();
        let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
        assert!(uci.process_line("go infinity", "0.0.0").expect("go infinity"));
        drop(uci);
        assert_eq!(engine.go_count, 1);
        assert_eq!(
            engine.last_go.expect("recorded go"),
            GoParams {
                infinite: true,
                ..GoParams::default()
            }
        );
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
        let mut options = UciOptions::populate_defaults();
        options
            .set_uci_option("WeightsFile", "data/x7.onnx")
            .expect("weights option");
        assert_eq!(options.weights_file, "data/x7.onnx");
        assert!(options
            .list_options_uci()
            .iter()
            .any(|line| line == "option name WeightsFile type string default data/x7.onnx"));
    }

    #[test]
    fn move_overhead_option_matches_px0_range_and_default() {
        let mut options = UciOptions::default();
        assert_eq!(options.move_overhead_ms, 200);
        options
            .set_uci_option("MoveOverheadMs", "17")
            .expect("valid px0 move overhead");
        assert_eq!(options.move_overhead_ms, 17);
        assert!(options.set_uci_option("MoveOverheadMs", "-1").is_err());
        assert!(options.set_uci_option("MoveOverheadMs", "100000001").is_err());
    }

    #[test]
    fn slowmover_option_matches_px0_range_and_default() {
        let mut options = UciOptions::default();
        assert_eq!(options.slowmover, 1.0);
        options.set_uci_option("Slowmover", "1.5").expect("valid px0 slowmover");
        assert_eq!(options.slowmover, 1.5);
        assert!(options.set_uci_option("Slowmover", "-0.1").is_err());
        assert!(options.set_uci_option("Slowmover", "100.1").is_err());
    }

    #[test]
    fn nodes_per_second_limit_matches_px0_float_range() {
        let mut options = UciOptions::populate_defaults();
        options
            .set_uci_option("NodesPerSecondLimit", "1234.5")
            .expect("px0 float option");
        assert_eq!(options.nodes_per_second_limit, 1234.5);
        assert!(options.set_uci_option("NodesPerSecondLimit", "-1").is_err());
        assert!(options.set_uci_option("NodesPerSecondLimit", "1000001").is_err());
    }

    /// px0 `ContemptMode` is a four-value ChoiceOption and
    /// `WDLEvalObjectivity` is clamped to `[0, 1]`
    /// (`src/search/classic/params.cc:606-620`).
    #[test]
    fn wdl_display_options_match_px0_choice_and_range() {
        let mut options = UciOptions::populate_defaults();
        options
            .set_uci_option("ContemptMode", "black_side_analysis")
            .expect("px0 contempt mode");
        options
            .set_uci_option("WDLEvalObjectivity", "0.25")
            .expect("px0 objectivity");
        assert_eq!(options.contempt_mode, ContemptMode::Black);
        assert_eq!(options.wdl_eval_objectivity, 0.25);
        assert!(options
            .list_options_uci()
            .iter()
            .any(|line| line.contains("option name ContemptMode type combo")));
        assert!(options
            .list_options_uci()
            .iter()
            .any(|line| line.contains("option name NNCacheSize type spin default 2000000")));
        options.set_uci_option("NNCacheSize", "0").expect("zero cache size");
        assert_eq!(options.nn_cache_size, 0);
        assert!(options.set_uci_option("NNCacheSize", "1000000000").is_err());
        assert!(options.set_uci_option("WDLEvalObjectivity", "1.1").is_err());
    }

    #[test]
    fn dispatch_position_keeps_full_history() {
        ensure_init();
        let mut options = UciOptions::populate_defaults();
        let mut engine = RecordingEngine::default();
        let mut responder = VecUciResponder::default();
        let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
        uci.process_line("position startpos moves h2h4 h9h7", "0.0.0")
            .expect("position");
        drop(uci);
        let state = engine.position.expect("position stored");
        assert_eq!(state.moves.len(), 2);
        assert_eq!(state.positions().len(), 3);
        assert_eq!(
            state.current_position().board().hash(),
            state.positions().last().unwrap().board().hash()
        );
    }

    #[test]
    fn uci_transcript() {
        let mut options = UciOptions::populate_defaults();
        let mut engine = RecordingEngine::default();
        let mut responder = VecUciResponder::default();
        let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
        assert!(uci.process_line("uci", "1.2.3").expect("uci"));
        assert!(uci.process_line("isready", "1.2.3").expect("isready"));
        assert!(uci
            .process_line("position startpos moves h2h4", "1.2.3")
            .expect("position"));
        assert!(uci.process_line("go depth 4", "1.2.3").expect("go"));
        drop(uci);
        assert_eq!(responder.responses[0], "id name x7 v1.2.3");
        assert_eq!(responder.responses.last().unwrap(), "readyok");
        assert_eq!(engine.go_count, 1);
        assert_eq!(engine.position.unwrap().moves.len(), 1);
    }

    #[test]
    fn responder_observes_updated_options_and_registration_lifecycle() {
        let mut options = UciOptions::populate_defaults();
        let mut engine = RecordingEngine::default();
        let mut responder = VecUciResponder::default();
        {
            let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
            uci.process_line("setoption name UCI_ShowWDL value true", "0.0.0")
                .expect("setoption");
        }
        assert!(responder.options.show_wdl);
        assert_eq!(engine.responder_registrations, 0);
    }

    #[test]
    fn default_thinking_info_matches_px0_sentinels() {
        assert_eq!(
            format_thinking_info(&ThinkingInfo::default(), &UciOptions::default()),
            "info"
        );
    }

    #[test]
    fn format_thinking_info_matches_px0_fields() {
        let options = UciOptions {
            show_wdl: true,
            show_eps: true,
            show_moves_left: true,
            ..UciOptions::default()
        };
        let info = ThinkingInfo {
            depth: 0,
            seldepth: 3,
            time: 12,
            nodes: 100,
            nps: 8000,
            eps: 42,
            hashfull: 7,
            score: Some(15),
            wdl: Some(Wdl { w: 100, d: 200, l: 300 }),
            moves_left: Some(9),
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
        assert!(formatted.contains("movesleft 9"));
        assert!(formatted.contains("eps 42"));
        assert!(formatted.contains(" pv h2h4"));
        assert!(formatted.contains(" string note"));
    }
}
