use std::sync::Once;

use xiangqi_core::{GameState, Move, Position, PositionHistory, STARTPOS_FEN, initialize_magic_bitboards};

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

fn parse_moves(start: &Position, move_strs: &[&str]) -> Vec<Move> {
    let mut position = start.clone();
    move_strs
        .iter()
        .map(|mv| {
            let parsed = position.board().parse_move(mv).expect("valid move");
            position = Position::after(&position, parsed);
            parsed
        })
        .collect()
}

#[test]
fn current_position_replays_moves() {
    ensure_init();
    let moves = ["h2h4", "h9h7", "h4h5", "h7h6"];
    let state = GameState::from_fen_moves(STARTPOS_FEN, &moves).expect("game state");
    let start = Position::from_fen(STARTPOS_FEN).expect("start position");
    let expected = PositionHistory::from_position_and_moves(start.clone(), &parse_moves(&start, &moves)).last().clone();
    assert_eq!(state.current_position().board(), expected.board(), "current board mismatch");
    assert_eq!(state.current_position().rule60_ply(), expected.rule60_ply(), "rule60 mismatch");
}

#[test]
fn positions_include_start_and_each_move() {
    ensure_init();
    let moves = ["h2h4", "h9h7"];
    let state = GameState::from_fen_moves(STARTPOS_FEN, &moves).expect("game state");
    let positions = state.positions();
    assert_eq!(positions.len(), moves.len() + 1);
    assert_eq!(positions[0].board(), state.startpos.board());

    let mut replay = state.startpos.clone();
    for (idx, _mv) in moves.iter().enumerate() {
        replay = Position::after(&replay, state.moves[idx]);
        assert_eq!(positions[idx + 1].board(), replay.board());
        assert_eq!(positions[idx + 1].rule60_ply(), replay.rule60_ply());
    }
}

#[test]
fn complex_fen_move_sequence_matches_history_boards() {
    ensure_init();
    let fen = "3k5/9/9/6c2/9/9/9/6R2/9/5K3 b - - 2 30";
    let moves = ["g6h6", "g2h2", "h6g6"];
    let state = GameState::from_fen_moves(fen, &moves).expect("game state");
    let start = Position::from_fen(fen).expect("start position");
    let history = PositionHistory::from_position_and_moves(start.clone(), &parse_moves(&start, &moves));
    for (game_state_pos, history_pos) in state.positions().iter().zip(history.positions()) {
        assert_eq!(game_state_pos.board(), history_pos.board());
    }
}
