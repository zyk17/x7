//! px0 `src/chess/callbacks.h:42-102`。

use xiangqi_core::Move;

/// px0 `BestMoveInfo`。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BestMoveInfo {
    pub bestmove: Move,
    pub ponder: Move,
    pub player: i32,
    pub game_id: i32,
    pub is_black: Option<bool>,
}

impl BestMoveInfo {
    pub const fn new(bestmove: Move) -> Self {
        Self {
            bestmove,
            ponder: Move::NULL,
            player: -1,
            game_id: -1,
            is_black: None,
        }
    }
}

/// px0 `ThinkingInfo::WDL`。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Wdl {
    pub w: i32,
    pub d: i32,
    pub l: i32,
}

/// px0 `ThinkingInfo`。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThinkingInfo {
    pub depth: i32,
    pub seldepth: i32,
    pub time: i64,
    pub nodes: i64,
    pub nps: i32,
    pub eps: i32,
    pub hashfull: i32,
    pub mate: Option<i32>,
    pub score: Option<i32>,
    pub wdl: Option<Wdl>,
    pub tb_hits: i32,
    pub pv: Vec<Move>,
    pub multipv: i32,
    pub comment: String,
    pub player: i32,
    pub game_id: i32,
    pub is_black: Option<bool>,
    pub moves_left: Option<i32>,
}

impl Default for ThinkingInfo {
    /// px0 `ThinkingInfo` field defaults (`callbacks.h:58-101`).
    fn default() -> Self {
        Self {
            depth: -1,
            seldepth: -1,
            time: -1,
            nodes: -1,
            nps: -1,
            eps: -1,
            hashfull: -1,
            mate: None,
            score: None,
            wdl: None,
            tb_hits: -1,
            pv: Vec::new(),
            multipv: -1,
            comment: String::new(),
            player: -1,
            game_id: -1,
            is_black: None,
            moves_left: None,
        }
    }
}

/// Thread-safe output boundary owned by classic search.
///
/// px0 `SearchBase` stores a `UciResponder*` for the full search lifetime
/// (`src/search/search.h:45-99`), so the watchdog may emit `info` and
/// `bestmove` independently of the UCI command loop. Rust uses this separate
/// trait because the UCI loop still owns the concrete mutable responder.
pub trait SearchResponder: Send + Sync {
    fn output_best_move(&self, info: &BestMoveInfo);
    fn output_thinking_info(&self, infos: &[ThinkingInfo]);
}
