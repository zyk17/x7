//! Position / PositionHistory / GameResult。来源：px0 position。

use crate::board::board_to_fen;
use crate::hashcat::{hash_cat, hash_cat_u128s};
use crate::{ChessBoard, CoreError, Move};

/// 对局结果。枚举顺序使 `max()` 优先选更好的结果。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum GameResult {
    #[default]
    Undecided,
    BlackWon,
    Draw,
    WhiteWon,
}

impl GameResult {
    pub const fn negate(self) -> Self {
        match self {
            Self::Undecided => Self::Undecided,
            Self::BlackWon => Self::WhiteWon,
            Self::Draw => Self::Draw,
            Self::WhiteWon => Self::BlackWon,
        }
    }
}

/// 局面：棋盘始终以当前行棋方为 `ours` 视角保存。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    board: ChessBoard,
    rule60_ply: u32,
    us_check: u32,
    them_check: u32,
    repetitions: u32,
    cycle_length: u32,
    game_ply: u32,
}

impl Position {
    pub fn new(board: ChessBoard, rule60_ply: u32, game_ply: u32) -> Self {
        Self {
            board,
            rule60_ply,
            us_check: 0,
            them_check: 0,
            repetitions: 0,
            cycle_length: 0,
            game_ply,
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self, CoreError> {
        let (board, state) = ChessBoard::from_fen(fen)?;
        Ok(Self::new(board, state.rule60_ply, state.game_ply))
    }

    pub fn after(parent: &Self, mv: Move) -> Self {
        let mut position = Self {
            board: parent.board.clone(),
            rule60_ply: parent.rule60_ply,
            us_check: parent.them_check,
            them_check: parent.us_check,
            repetitions: 0,
            cycle_length: 0,
            game_ply: parent.game_ply + 1,
        };

        let is_zeroing = position.board.apply_move(mv);
        position.board.mirror();

        if !position.board.is_under_check() || {
            position.them_check += 1;
            position.them_check <= 10
        } {
            if position.us_check > 10 && parent.board.is_under_check() {
                position.us_check += 1;
            } else {
                position.rule60_ply += 1;
            }
        }

        if is_zeroing {
            position.rule60_ply = 0;
            position.us_check = 0;
            position.them_check = 0;
        }
        position
    }

    pub fn hash(&self) -> u64 {
        hash_cat_u128s(&[self.board.hash() as u128, self.repetitions as u128])
    }

    pub fn to_fen(&self) -> String {
        let mut result = board_to_fen(&self.board);
        result.push_str(" - - ");
        result.push_str(&self.rule60_ply.to_string());
        result.push(' ');
        result.push_str(&((self.game_ply + if self.is_black_to_move() { 1 } else { 2 }) / 2).to_string());
        result
    }

    pub fn debug_string(&self) -> String {
        format!("https://xiangqiai.com/#/{}", self.to_fen())
    }

    pub const fn board(&self) -> &ChessBoard {
        &self.board
    }
    pub const fn rule60_ply(&self) -> u32 {
        self.rule60_ply
    }
    pub const fn repetitions(&self) -> u32 {
        self.repetitions
    }
    pub const fn cycle_length(&self) -> u32 {
        self.cycle_length
    }
    pub const fn game_ply(&self) -> u32 {
        self.game_ply
    }
    pub const fn is_black_to_move(&self) -> bool {
        self.board.flipped()
    }

    pub fn set_repetitions(&mut self, repetitions: u32, cycle_length: u32) {
        self.repetitions = repetitions;
        self.cycle_length = cycle_length;
    }
}

/// 完整对局历史属于规则层，搜索只能借用或克隆它。
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PositionHistory {
    positions: Vec<Position>,
}

impl PositionHistory {
    pub fn from_positions(positions: Vec<Position>) -> Self {
        Self { positions }
    }

    pub fn starting(&self) -> &Position {
        self.positions.first().expect("PositionHistory is empty")
    }

    pub fn last(&self) -> &Position {
        self.positions.last().expect("PositionHistory is empty")
    }

    pub fn get(&self, index: usize) -> &Position {
        &self.positions[index]
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// Copies a root history without discarding this instance's reserved DFS
    /// capacity. px0 gives every search workspace a persistent
    /// `PositionHistory` and reuses it through `Trim`/`Append`
    pub fn copy_from_history(&mut self, source: &Self) {
        self.positions.clear();
        self.positions.extend_from_slice(source.positions());
    }

    pub fn reserve(&mut self, size: usize) {
        self.positions.reserve(size.saturating_sub(self.positions.len()));
    }

    pub fn reset(&mut self, board: ChessBoard, rule60_ply: u32, game_ply: u32) {
        self.positions.clear();
        self.positions.push(Position::new(board, rule60_ply, game_ply));
    }

    pub fn reset_position(&mut self, position: Position) {
        self.positions.clear();
        self.positions.push(position);
    }

    pub fn append(&mut self, mv: Move) {
        let next = Position::after(self.last(), mv);
        self.positions.push(next);
        let (repetitions, cycle_length) = self.compute_last_move_repetitions();
        self.positions
            .last_mut()
            .expect("position appended")
            .set_repetitions(repetitions, cycle_length);
    }

    pub fn pop(&mut self) {
        self.positions.pop();
    }

    pub fn trim(&mut self, size: usize) {
        self.positions.truncate(size);
    }

    pub fn compute_game_result(&self) -> GameResult {
        let last = self.last();
        if last.board.generate_legal_moves().is_empty() {
            return if self.is_black_to_move() {
                GameResult::WhiteWon
            } else {
                GameResult::BlackWon
            };
        }
        if last.repetitions >= 2 {
            let result = self.rule_judge();
            return if self.is_black_to_move() {
                result
            } else {
                result.negate()
            };
        }
        if !last.board.has_mating_material() || last.rule60_ply >= 120 {
            return GameResult::Draw;
        }
        GameResult::Undecided
    }

    /// 返回值按 px0 `MakeTerminal` 约定解释：`WhiteWon`→node `wl=+1`，
    /// `BlackWon`→`wl=-1`（incoming-edge 视角），**不是**绝对红黑胜负。
    /// 绝对结果见 [`Self::compute_game_result`]（白走时会取反）。
    pub fn rule_judge(&self) -> GameResult {
        let last = self.last();
        if last.rule60_ply < 4 {
            return GameResult::Undecided;
        }

        let len = self.positions.len();
        assert!(len >= 3, "RuleJudge requires a repetition history");

        let mut check_them = last.board.is_under_check();
        let mut check_us = self.positions[len - 2].board.is_under_check();
        let mut chase_them = last.board.them_chased() & !self.positions[len - 2].board.us_chased();
        let mut chase_us = self.positions[len - 2].board.them_chased() & !self.positions[len - 3].board.us_chased();

        let mut index = len - 3;
        loop {
            let position = &self.positions[index];
            if position.board.is_under_check() {
                chase_them = 0;
                chase_us = 0;
            } else {
                check_them = false;
            }

            if position.board == last.board && position.repetitions == 0 {
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

            if index >= 1 {
                if self.positions[index - 1].board.is_under_check() {
                    chase_them = 0;
                    chase_us = 0;
                } else {
                    check_us = false;
                }
                chase_them &= position.board.them_chased() & !self.positions[index - 1].board.us_chased();
                if index >= 2 {
                    chase_us &=
                        self.positions[index - 1].board.them_chased() & !self.positions[index - 2].board.us_chased();
                }
            }

            if index < 2 {
                break;
            }
            index -= 2;
        }

        panic!("px0 RuleJudge called without a repeat");
    }

    pub fn did_repeat_since_last_zeroing_move(&self) -> bool {
        for position in self.positions.iter().rev() {
            if position.repetitions > 0 {
                return true;
            }
            if position.rule60_ply == 0 {
                return false;
            }
        }
        false
    }

    pub fn hash_last(&self, positions: usize) -> u64 {
        let mut remaining = positions;
        let mut hash = positions as u64;
        for position in self.positions.iter().rev() {
            if remaining == 0 {
                break;
            }
            remaining -= 1;
            hash = hash_cat(hash, position.hash());
        }
        hash_cat(hash, self.last().rule60_ply as u64)
    }

    fn compute_last_move_repetitions(&self) -> (u32, u32) {
        let last = self.last();
        if last.rule60_ply < 4 {
            return (0, 0);
        }

        let mut index = self.positions.len() as isize - 5;
        while index >= 0 {
            let position = &self.positions[index as usize];
            if position.board == last.board {
                let cycle_length = self.positions.len() as u32 - 1 - index as u32;
                return (1 + position.repetitions, cycle_length);
            }
            if position.rule60_ply < 2 {
                return (0, 0);
            }
            index -= 2;
        }
        (0, 0)
    }

    pub fn is_black_to_move(&self) -> bool {
        self.last().is_black_to_move()
    }
}
