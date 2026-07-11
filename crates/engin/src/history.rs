//! px0 `PositionHistory`：完整局面链（`position.h` / `position.cc`）。
//!
//! NN 的 8 帧窗口不在此截断，仅在 [`crate::fen_tensor`] 编码时取最近
//! [`PX0_HISTORY_LEN`] 帧（px0 `encoder.cc:157-158` `history_planes=8`）。

use xiangqi_core::{is_under_check, them_chased, us_chased, GameResult, Move, Position, START_FEN};

pub const PX0_HISTORY_LEN: usize = 8;
const RULE60_DRAW: i32 = 120;

#[derive(Clone)]
pub struct HistoryEntry {
    pub position: Position,
    repetitions: i32,
    plies_since_prev_repetition: i32,
}

impl HistoryEntry {
    /// px0 `GetRepetitions() >= 1` → repetition plane（`encoder.cc:168-198`）。
    pub fn is_repeated(&self) -> bool {
        self.repetitions >= 1
    }
}

#[derive(Clone, Debug)]
pub struct HistoryDebugEntry {
    pub fen: String,
    pub repeated: bool,
    pub side_to_move: char,
    pub rule60: i32,
}

#[derive(Clone)]
pub struct PositionHistory {
    game_start: Position,
    game_moves: Vec<Move>,
    positions: Vec<HistoryEntry>,
}

impl PositionHistory {
    pub fn new_startpos() -> Self {
        let pos = Position::from_fen(START_FEN).expect("valid startpos");
        Self::from_position(pos)
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let pos = Position::from_fen(fen)?;
        Ok(Self::from_position(pos))
    }

    pub fn from_position(pos: Position) -> Self {
        let mut history = Self {
            game_start: pos.clone_for_search(),
            game_moves: Vec::new(),
            positions: Vec::new(),
        };
        history.reset(pos);
        history
    }

    pub fn from_positions(positions: Vec<Position>) -> Result<Self, String> {
        if positions.is_empty() {
            return Err("history 不能为空".into());
        }
        let mut history = Self {
            game_start: positions[0].clone_for_search(),
            game_moves: Vec::new(),
            positions: Vec::new(),
        };
        for pos in positions {
            history.append_position(pos.clone_for_search());
        }
        Ok(history)
    }

    pub fn game_start(&self) -> &Position {
        &self.game_start
    }

    pub fn game_moves(&self) -> &[Move] {
        &self.game_moves
    }

    pub fn game_start_key(&self) -> u64 {
        self.game_start.nn_input_key()
    }

    pub fn current(&self) -> &Position {
        &self.positions.last().expect("position history non-empty").position
    }

    /// px0 `GetPositions()`：完整链；NN 编码取 [`Self::nn_input_window`]。
    pub fn positions(&self) -> impl DoubleEndedIterator<Item = &Position> + ExactSizeIterator + '_ {
        self.positions.iter().map(|entry| &entry.position)
    }

    pub(crate) fn nn_input_window(&self) -> &[HistoryEntry] {
        let start = self.positions.len().saturating_sub(PX0_HISTORY_LEN);
        &self.positions[start..]
    }

    pub fn debug_entries(&self) -> Vec<HistoryDebugEntry> {
        self.positions
            .iter()
            .map(|entry| HistoryDebugEntry {
                fen: entry.position.fen(),
                repeated: entry.is_repeated(),
                side_to_move: if entry.position.side_to_move == xiangqi_core::types::Color::Black {
                    'b'
                } else {
                    'w'
                },
                rule60: entry.position.state.rule60,
            })
            .collect()
    }

    pub fn current_is_repeated(&self) -> bool {
        self.positions.last().is_some_and(|entry| entry.is_repeated())
    }

    pub fn repetitions(&self) -> i32 {
        self.positions.last().map(|entry| entry.repetitions).unwrap_or(0)
    }

    pub fn plies_since_prev_repetition(&self) -> i32 {
        self.positions
            .last()
            .map(|entry| entry.plies_since_prev_repetition)
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn same_input_window(&self, other: &Self) -> bool {
        let a = self.nn_input_window();
        let b = other.nn_input_window();
        if a.len() != b.len() {
            return false;
        }
        if self.current().state.rule60 != other.current().state.rule60 {
            return false;
        }
        a.iter().zip(b.iter()).all(|(lhs, rhs)| {
            lhs.is_repeated() == rhs.is_repeated() && lhs.position.nn_input_key() == rhs.position.nn_input_key()
        })
    }

    pub fn input_cache_key(&self) -> u64 {
        let mut key =
            0x9E37_79B9_7F4A_7C15u64 ^ ((self.positions.len() as u64) << 56) ^ self.current().state.rule60 as u64;
        for entry in self.nn_input_window() {
            key = key.rotate_left(9) ^ entry.position.nn_input_key().wrapping_mul(0x9E37_79B9_7F4A_7C15);
            if entry.is_repeated() {
                key ^= 0xA5A5_A5A5_A5A5_A5A5;
            }
        }
        key
    }

    pub fn reset_from_position(&mut self, pos: Position) {
        self.game_start = pos.clone_for_search();
        self.game_moves.clear();
        self.reset(pos);
    }

    pub fn push_move(&mut self, mv: Move) {
        self.game_moves.push(mv);
        self.append_move(mv);
    }

    pub fn push_position(&mut self, pos: Position) {
        self.append_position(pos.clone_for_search());
    }

    /// px0 搜索/worker 临时路径：`Append` 但不写入 `game_moves`。
    #[allow(dead_code)] // S2 will use this for lc0 SearchWorker path extension.
    pub(crate) fn push_search_position(&mut self, pos: Position) {
        self.append_position(pos);
    }

    pub fn append_move(&mut self, mv: Move) {
        let mut next = self.current().clone_for_search();
        next.do_move(mv);
        self.append_position(next);
    }

    /// px0 `PositionHistory::Pop`。
    pub fn pop_position(&mut self) {
        if self.positions.len() <= 1 {
            return;
        }
        self.positions.pop();
    }

    /// px0 `PositionHistory::Trim`（`position.h:120-122`）。
    pub fn trim(&mut self, size: usize) {
        if size < self.positions.len() {
            self.positions.truncate(size);
        }
    }

    /// px0 `ExtendNode`：`Trim(played_len)` + 路径 `Append`（`search.cc:1903-1906`）。
    #[allow(dead_code)] // S2 will use this for batched leaf history encoding.
    pub(crate) fn with_search_path(&self, path_moves: &[Move]) -> Self {
        let mut history = self.clone_for_search();
        let played_len = history.len();
        history.trim(played_len);
        for &mv in path_moves {
            history.append_move(mv);
        }
        history
    }

    pub fn rule60_draw(&self) -> bool {
        self.current().state.rule60 >= RULE60_DRAW
    }

    /// px0 `PositionHistory::RuleJudge`（position.cc:126-169）。
    pub fn rule_judge(&self) -> GameResult {
        let len = self.positions.len();
        assert!(len >= 3, "RuleJudge requires at least 3 positions");
        let last = &self.positions[len - 1];
        if last.position.state.rule60 < 4 {
            return GameResult::Undecided;
        }

        let mut check_them = is_under_check(&last.position);
        let mut check_us = is_under_check(&self.positions[len - 2].position);
        let mut chase_them = them_chased(&last.position) & !us_chased(&self.positions[len - 2].position);
        let mut chase_us =
            them_chased(&self.positions[len - 2].position) & !us_chased(&self.positions[len - 3].position);

        let last_key = last.position.nn_input_key();
        let mut idx = len as i32 - 3;
        while idx >= 0 {
            let pos = &self.positions[idx as usize];
            if is_under_check(&pos.position) {
                chase_them = 0;
                chase_us = 0;
            } else {
                check_them = false;
            }

            if pos.position.nn_input_key() == last_key && pos.repetitions == 0 {
                return if check_them || check_us {
                    if !check_us {
                        GameResult::BlackWon
                    } else if !check_them {
                        GameResult::WhiteWon
                    } else {
                        GameResult::Draw
                    }
                } else if chase_them != 0 || chase_us != 0 {
                    if chase_us == 0 {
                        GameResult::BlackWon
                    } else if chase_them == 0 {
                        GameResult::WhiteWon
                    } else {
                        GameResult::Draw
                    }
                } else {
                    GameResult::Draw
                };
            }

            if idx > 0 {
                if is_under_check(&self.positions[(idx - 1) as usize].position) {
                    chase_them = 0;
                    chase_us = 0;
                } else {
                    check_us = false;
                }
                chase_them &= them_chased(&pos.position) & !us_chased(&self.positions[(idx - 1) as usize].position);
                if idx - 2 >= 0 {
                    chase_us &= them_chased(&self.positions[(idx - 1) as usize].position)
                        & !us_chased(&self.positions[(idx - 2) as usize].position);
                }
            }
            idx -= 2;
        }

        panic!("Judging non-repetition move sequence");
    }

    /// px0 `ExtendNode` twofold 非和分支（search.cc:1940-1957）。
    pub fn twofold_forced_terminal(&self, result: GameResult) -> Option<i32> {
        if result == GameResult::Draw {
            return None;
        }
        let len = self.positions.len();
        if len < 2 {
            return None;
        }
        let last_key = self.positions[len - 1].position.nn_input_key();
        let mut idx = len - 1;
        let mut idx2 = idx as i32;
        loop {
            if idx2 <= 0 {
                return None;
            }
            idx2 -= 1;
            if self.positions[idx2 as usize].position.nn_input_key() == last_key {
                break;
            }
        }
        if idx2 > 0
            && self.positions[idx - 1].position.nn_input_key()
                == self.positions[(idx2 as usize) - 1].position.nn_input_key()
        {
            idx -= 1;
            idx2 += 1;
            while (idx2 as usize) != idx && self.positions[idx2 as usize].repetitions == 0 {
                idx2 += 1;
            }
            if idx2 as usize == idx && self.current().state.rule60 < RULE60_DRAW {
                return Some(self.plies_since_prev_repetition());
            }
        }
        None
    }

    pub fn clone_for_search(&self) -> Self {
        Self {
            game_start: self.game_start.clone_for_search(),
            game_moves: self.game_moves.clone(),
            positions: self.positions.clone(),
        }
    }

    fn reset(&mut self, pos: Position) {
        self.positions.clear();
        self.positions.push(HistoryEntry {
            position: pos,
            repetitions: 0,
            plies_since_prev_repetition: 0,
        });
    }

    fn append_position(&mut self, pos: Position) {
        let (repetitions, plies_since_prev_repetition) = compute_last_move_repetitions(&self.positions, &pos);
        self.positions.push(HistoryEntry {
            position: pos,
            repetitions,
            plies_since_prev_repetition,
        });
    }
}

impl Default for PositionHistory {
    fn default() -> Self {
        Self::new_startpos()
    }
}

/// px0 `PositionHistory::ComputeLastMoveRepetitions`（`position.cc:171-186`）。
fn compute_last_move_repetitions(existing: &[HistoryEntry], last: &Position) -> (i32, i32) {
    if last.state.rule60 < 4 {
        return (0, 0);
    }
    let size = existing.len() + 1;
    if size < 5 {
        return (0, 0);
    }
    let mut idx = (size - 5) as i32;
    while idx >= 0 {
        let pos = &existing[idx as usize].position;
        if pos.nn_input_key() == last.nn_input_key() {
            let cycle_length = (size - 1 - idx as usize) as i32;
            let repetitions = 1 + existing[idx as usize].repetitions;
            return (repetitions, cycle_length);
        }
        if pos.state.rule60 < 2 {
            return (0, 0);
        }
        idx -= 2;
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::{legal_moves_uci, uci_to_move};

    #[test]
    fn full_history_is_not_truncated() {
        let mut history = PositionHistory::new_startpos();
        for _ in 0..9 {
            let u = legal_moves_uci(history.current())
                .into_iter()
                .next()
                .expect("legal move uci");
            let mv = uci_to_move(history.current(), &u).expect("legal move");
            history.push_move(mv);
        }
        assert_eq!(history.len(), 10);
    }

    #[test]
    fn search_push_pop_restores_chain() {
        let mut history = PositionHistory::new_startpos();
        for _ in 0..7 {
            let u = legal_moves_uci(history.current())
                .into_iter()
                .next()
                .expect("legal move uci");
            let mv = uci_to_move(history.current(), &u).expect("legal move");
            history.push_move(mv);
        }
        let before: Vec<String> = history.positions().map(Position::fen).collect();
        let current_before = history.current().fen();

        let u = legal_moves_uci(history.current())
            .into_iter()
            .next()
            .expect("legal move uci");
        let mv = uci_to_move(history.current(), &u).expect("legal move");
        let mut next = history.current().clone_for_search();
        next.do_move(mv);
        history.push_search_position(next);
        history.pop_position();

        let after: Vec<String> = history.positions().map(Position::fen).collect();
        assert_eq!(after, before);
        assert_eq!(history.current().fen(), current_before);
    }

    #[test]
    fn same_input_window_detects_transposed_histories() {
        let mut a = PositionHistory::new_startpos();
        for u in ["h0g2", "h9g7", "b0c2", "b9c7"] {
            let mv = uci_to_move(a.current(), u).expect("legal move");
            a.push_move(mv);
        }

        let mut b = PositionHistory::new_startpos();
        for u in ["b0c2", "b9c7", "h0g2", "h9g7"] {
            let mv = uci_to_move(b.current(), u).expect("legal move");
            b.push_move(mv);
        }

        assert_eq!(a.current().fen(), b.current().fen());
        assert!(!a.same_input_window(&b));
        assert_ne!(a.input_cache_key(), b.input_cache_key());
    }

    #[test]
    fn opening_search_path_has_zero_repetitions() {
        let played = PositionHistory::new_startpos();
        let mut moves = Vec::new();
        let mut pos = played.current().clone_for_search();
        for u in ["e3e4", "c9e7", "b2e2", "h9g7", "b0c2", "b9c7", "a0b0", "a9b9", "b0b6"] {
            let mv = uci_to_move(&pos, u).expect("legal");
            moves.push(mv);
            pos.do_move(mv);
        }
        let history = played.with_search_path(&moves);
        assert_eq!(history.repetitions(), 0);
        assert_eq!(history.plies_since_prev_repetition(), 0);
    }

    #[test]
    fn with_search_path_matches_manual_extend() {
        let mut base = PositionHistory::new_startpos();
        for _ in 0..4 {
            let u = legal_moves_uci(base.current())
                .into_iter()
                .next()
                .expect("legal move uci");
            let mv = uci_to_move(base.current(), &u).expect("legal move");
            base.push_move(mv);
        }
        let u = legal_moves_uci(base.current())
            .into_iter()
            .next()
            .expect("legal move uci");
        let mv = uci_to_move(base.current(), &u).expect("legal move");

        let lean = base.with_search_path(std::slice::from_ref(&mv));
        let mut full = base.clone_for_search();
        full.append_move(mv);
        assert_eq!(lean.input_cache_key(), full.input_cache_key());
        assert!(lean.same_input_window(&full));
    }
}
