//! px0 `src/chess/gamestate.h:38-47` 与 `gamestate.cc:35-55`、`engine.cc:65-78`。

use crate::{CoreError, MoveList, Position};

/// px0 `GameState` (`gamestate.h:38-46`)。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameState {
    pub startpos: Position,
    pub moves: MoveList,
}

impl GameState {
    pub const fn new(startpos: Position, moves: MoveList) -> Self {
        Self { startpos, moves }
    }

    pub fn from_fen_moves(fen: &str, move_strs: &[impl AsRef<str>]) -> Result<Self, CoreError> {
        let startpos = Position::from_fen(fen)?;
        let mut board = startpos.board().clone();
        let mut moves = MoveList::with_capacity(move_strs.len());
        for move_str in move_strs {
            let mv = board.parse_move(move_str.as_ref())?;
            moves.push(mv);
            board.apply_move(mv);
            board.mirror();
        }
        Ok(Self { startpos, moves })
    }

    pub fn current_position(&self) -> Position {
        self.moves
            .iter()
            .fold(self.startpos.clone(), |pos, &mv| Position::after(&pos, mv))
    }

    pub fn positions(&self) -> Vec<Position> {
        let mut positions = Vec::with_capacity(self.moves.len() + 1);
        positions.push(self.startpos.clone());
        for &mv in &self.moves {
            let next = Position::after(positions.last().expect("startpos present"), mv);
            positions.push(next);
        }
        positions
    }
}
