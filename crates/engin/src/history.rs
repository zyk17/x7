//! 引擎侧最小 history 容器：固定保留最近窗口，供 px0 classical 输入编码使用。

use std::collections::HashMap;

use xiangqi_core::{Position, START_FEN};

pub const PX0_HISTORY_LEN: usize = 8;

#[derive(Clone)]
pub(crate) struct HistoryEntry {
    pub position: Position,
    pub repeated: bool,
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
    game_moves: Vec<xiangqi_core::Move>,
    entries: Vec<HistoryEntry>,
    key_counts: HashMap<u64, usize>,
    transient_evicted: Vec<Option<HistoryEntry>>,
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
            entries: Vec::with_capacity(PX0_HISTORY_LEN),
            key_counts: HashMap::new(),
            transient_evicted: Vec::new(),
        };
        history.push_search_position(pos.clone_for_search());
        history
    }

    pub fn from_positions(positions: Vec<Position>) -> Result<Self, String> {
        if positions.is_empty() {
            return Err("history 不能为空".into());
        }
        let mut history = Self {
            game_start: positions[0].clone_for_search(),
            game_moves: Vec::new(),
            entries: Vec::with_capacity(PX0_HISTORY_LEN),
            key_counts: HashMap::new(),
            transient_evicted: Vec::new(),
        };
        for pos in positions {
            history.push_position(pos.clone_for_search());
        }
        Ok(history)
    }

    pub fn game_start(&self) -> &Position {
        &self.game_start
    }

    pub fn game_moves(&self) -> &[xiangqi_core::Move] {
        &self.game_moves
    }

    pub fn game_start_key(&self) -> u64 {
        self.game_start.nn_input_key()
    }

    pub fn current(&self) -> &Position {
        &self
            .entries
            .last()
            .expect("position history must always contain current position")
            .position
    }

    pub(crate) fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    pub fn positions(&self) -> impl DoubleEndedIterator<Item = &Position> + ExactSizeIterator + '_ {
        self.entries.iter().map(|entry| &entry.position)
    }

    pub fn debug_entries(&self) -> Vec<HistoryDebugEntry> {
        self.entries
            .iter()
            .map(|entry| HistoryDebugEntry {
                fen: entry.position.fen(),
                repeated: entry.repeated,
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
        self.entries.last().map(|entry| entry.repeated).unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn same_input_window(&self, other: &Self) -> bool {
        if self.entries.len() != other.entries.len() {
            return false;
        }
        if self.current().state.rule60 != other.current().state.rule60 {
            return false;
        }
        self.entries.iter().zip(other.entries.iter()).all(|(lhs, rhs)| {
            lhs.repeated == rhs.repeated && lhs.position.nn_input_key() == rhs.position.nn_input_key()
        })
    }

    pub fn input_cache_key(&self) -> u64 {
        let mut key =
            0x9E37_79B9_7F4A_7C15u64 ^ ((self.entries.len() as u64) << 56) ^ self.current().state.rule60 as u64;
        for entry in &self.entries {
            key = key.rotate_left(9) ^ entry.position.nn_input_key().wrapping_mul(0x9E37_79B9_7F4A_7C15);
            if entry.repeated {
                key ^= 0xA5A5_A5A5_A5A5_A5A5;
            }
        }
        key
    }

    pub fn reset_from_position(&mut self, pos: Position) {
        self.game_start = pos.clone_for_search();
        self.game_moves.clear();
        self.entries.clear();
        self.key_counts.clear();
        self.transient_evicted.clear();
        self.push_search_position(pos.clone_for_search());
    }

    pub fn push_move(&mut self, mv: xiangqi_core::Move) {
        self.game_moves.push(mv);
        let mut next = self.current().clone_for_search();
        next.do_move(mv);
        self.push_persistent_position(next);
    }

    pub fn push_position(&mut self, pos: Position) {
        self.push_persistent_position(pos.clone_for_search());
    }

    pub fn pop_position(&mut self) {
        if self.entries.len() <= 1 {
            return;
        }
        let entry = self.entries.pop().expect("history not empty");
        let key = entry.position.nn_input_key();
        let count = self.key_counts.get_mut(&key).expect("history key must exist");
        *count -= 1;
        if *count == 0 {
            self.key_counts.remove(&key);
        }
        if let Some(evicted) = self.transient_evicted.pop().flatten() {
            self.entries.insert(0, evicted);
        }
    }

    pub(crate) fn push_search_position(&mut self, pos: Position) {
        self.push_position_impl(pos, true);
    }

    fn push_persistent_position(&mut self, pos: Position) {
        self.push_position_impl(pos, false);
    }

    fn push_position_impl(&mut self, pos: Position, track_evicted: bool) {
        let key = pos.nn_input_key();
        let repeated = self.key_counts.contains_key(&key);
        *self.key_counts.entry(key).or_insert(0) += 1;
        self.entries.push(HistoryEntry {
            position: pos,
            repeated,
        });
        if self.entries.len() > PX0_HISTORY_LEN {
            let evicted = self.entries.remove(0);
            if track_evicted {
                self.transient_evicted.push(Some(evicted));
            }
        } else if track_evicted {
            self.transient_evicted.push(None);
        }
    }

    pub fn clone_for_search(&self) -> Self {
        Self {
            game_start: self.game_start.clone_for_search(),
            game_moves: self.game_moves.clone(),
            entries: self.entries.clone(),
            key_counts: self.key_counts.clone(),
            transient_evicted: Vec::new(),
        }
    }

    /// 仅复制 NN 编码所需状态；不复制 `game_moves` 等持久对局字段。
    pub(crate) fn clone_nn_input_state(&self) -> Self {
        Self {
            game_start: self.game_start.clone_for_search(),
            game_moves: Vec::new(),
            entries: self.entries.clone(),
            key_counts: self.key_counts.clone(),
            transient_evicted: Vec::new(),
        }
    }

    /// 在搜索根 history 上叠加 selection 路径，仅在 expand 时物化 NN 输入 history。
    pub(crate) fn extended_with_search_path(&self, suffix: &[Position]) -> Self {
        let mut history = self.clone_nn_input_state();
        for pos in suffix {
            history.push_search_position(pos.clone_for_search());
        }
        history
    }

    pub(crate) fn key_counts(&self) -> &HashMap<u64, usize> {
        &self.key_counts
    }

    /// selection 热路径：沿路径增量维护重复检测，避免每次 pick 复制整张 `key_counts`。
    pub(crate) fn push_search_path_position(
        base_key_counts: &HashMap<u64, usize>,
        path_key_counts: &mut HashMap<u64, usize>,
        pos: &Position,
    ) -> bool {
        let key = pos.nn_input_key();
        let repeated = base_key_counts.contains_key(&key) || path_key_counts.contains_key(&key);
        *path_key_counts.entry(key).or_insert(0) += 1;
        repeated
    }
}

impl Default for PositionHistory {
    fn default() -> Self {
        Self::new_startpos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::{legal_moves_uci, uci_to_move};

    #[test]
    fn history_window_is_capped() {
        let mut history = PositionHistory::new_startpos();
        for _ in 0..9 {
            let u = legal_moves_uci(history.current())
                .into_iter()
                .next()
                .expect("legal move uci");
            let mv = uci_to_move(history.current(), &u).expect("legal move");
            history.push_move(mv);
        }
        assert_eq!(history.len(), PX0_HISTORY_LEN);
    }

    #[test]
    fn search_push_pop_restores_window() {
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
    fn push_search_path_position_matches_full_clone_semantics() {
        let mut base = PositionHistory::new_startpos();
        for _ in 0..3 {
            let u = legal_moves_uci(base.current())
                .into_iter()
                .next()
                .expect("legal move uci");
            let mv = uci_to_move(base.current(), &u).expect("legal move");
            base.push_move(mv);
        }
        let mut cloned = base.clone_for_search();
        let mut overlay = HashMap::new();

        for _ in 0..2 {
            let u = legal_moves_uci(base.current())
                .into_iter()
                .next()
                .expect("legal move uci");
            let mv = uci_to_move(base.current(), &u).expect("legal move");
            let mut next = base.current().clone_for_search();
            next.do_move(mv);
            let from_clone = {
                cloned.push_search_position(next.clone_for_search());
                cloned.current_is_repeated()
            };
            let from_overlay =
                PositionHistory::push_search_path_position(base.key_counts(), &mut overlay, &next);
            assert_eq!(from_clone, from_overlay);
        }
    }

    #[test]
    fn extended_with_search_path_skips_game_moves_but_keeps_nn_input() {
        let mut base = PositionHistory::new_startpos();
        for _ in 0..4 {
            let u = legal_moves_uci(base.current())
                .into_iter()
                .next()
                .expect("legal move uci");
            let mv = uci_to_move(base.current(), &u).expect("legal move");
            base.push_move(mv);
        }
        assert!(!base.game_moves().is_empty());

        let u = legal_moves_uci(base.current())
            .into_iter()
            .next()
            .expect("legal move uci");
        let mv = uci_to_move(base.current(), &u).expect("legal move");
        let mut suffix_pos = base.current().clone_for_search();
        suffix_pos.do_move(mv);
        let suffix = vec![suffix_pos.clone_for_search()];

        let full = {
            let mut history = base.clone_for_search();
            for pos in &suffix {
                history.push_search_position(pos.clone_for_search());
            }
            history
        };
        let lean = base.extended_with_search_path(&suffix);

        assert!(lean.game_moves().is_empty());
        assert_eq!(lean.input_cache_key(), full.input_cache_key());
        assert!(lean.same_input_window(&full));
    }
}
