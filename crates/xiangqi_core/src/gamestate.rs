//! px0 `src/chess/gamestate.h:38-47` 与 `gamestate.cc:35-55`、`engine.cc:65-78`。

use crate::{CoreError, MoveList, Position, PositionHistory};

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
        self.position_history().last().clone()
    }

    /// 从 UCI 的初始局面和完整 moves 重放规则历史。
    ///
    /// 不能只连续调用 `Position::after`：重复次数与 cycle length 只能由
    /// `PositionHistory::append` 在完整路径中计算。参考 px0
    /// `GameState` + `PositionHistory::Append`（`gamestate.cc:35-55`、
    /// `position.cc:113-124,171-186`）。
    pub fn position_history(&self) -> PositionHistory {
        let mut history = PositionHistory::default();
        history.reset_position(self.startpos.clone());
        for &mv in &self.moves {
            history.append(mv);
        }
        history
    }

    /// 包含初始局面与每一步后的完整规则 position。
    pub fn positions(&self) -> Vec<Position> {
        self.position_history().positions().to_vec()
    }
}
