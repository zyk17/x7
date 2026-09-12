//! UCI `GameState`：startpos + moves，供 `position` 命令构建完整历史。

use crate::{MoveList, Position, PositionHistory};

/// startpos 与后续着法。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameState {
    pub startpos: Position,
    pub moves: MoveList,
}

impl GameState {
    pub const fn new(startpos: Position, moves: MoveList) -> Self {
        Self { startpos, moves }
    }

    pub fn from_fen_moves(fen: &str, move_strs: &[impl AsRef<str>]) -> Result<Self, String> {
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
    /// 批量重放时仍须携带完整路径，才能正确计算重复次数。
    pub fn position_history(&self) -> PositionHistory {
        PositionHistory::from_position_and_moves(self.startpos.clone(), &self.moves)
    }

    /// 包含初始局面与每一步后的完整规则 position。
    pub fn positions(&self) -> Vec<Position> {
        self.position_history().positions().to_vec()
    }
}
