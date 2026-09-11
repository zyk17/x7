use std::sync::Once;

use xiangqi_core::{ChessBoard, GameResult, Move, Position, PositionHistory, initialize_magic_bitboards};

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

fn history(fen: &str, moves: &[&str]) -> PositionHistory {
    let start = Position::from_fen(fen).expect("valid test FEN");
    let mut position = start.clone();
    let moves = moves
        .iter()
        .map(|text| {
            let mv: Move = position.board().parse_move(text).expect("test move");
            position = Position::after(&position, mv);
            mv
        })
        .collect::<Vec<_>>();
    PositionHistory::from_position_and_moves(start, &moves)
}

#[test]
fn set_fen_get_fen() {
    ensure_init();
    let source_fens = [
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 1 1",
        "1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w - - 0 1",
        "5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w - - 2 30",
        "CRN1k1b2/3ca4/4ba3/9/2nr5/9/9/4B4/4A4/4KA3 w - - 1 8",
        "R1N1k1b2/9/3aba3/9/2nr5/2B6/9/4B4/4A4/4KA3 w - - 0 10",
        "C1nNk4/9/9/9/9/9/n1pp5/B3C4/9/3A1K3 w - - 0 1",
        "4ka3/4a4/9/9/4N4/p8/9/4C3c/7n1/2BK5 w - - 0 1",
    ];
    for fen in source_fens {
        // FEN 提供 fullmove count，PositionHistory 保存 game ply。
        let (board, state) = ChessBoard::from_fen(fen).unwrap();
        let game_ply = 2 * state.game_ply - if board.flipped() { 1 } else { 2 };
        assert_eq!(Position::new(board, state.rule60_ply, game_ply).to_fen(), fen);
    }
}

#[test]
fn from_fen_converts_fullmove_to_game_ply() {
    ensure_init();
    let position = Position::from_fen("3k5/9/9/9/9/9/9/9/9/5K3 w - - 0 30").unwrap();
    assert_eq!(position.game_ply(), 30);
}

#[test]
fn compute_last_move_repetitions() {
    ensure_init();
    let once = history("3k5/9/9/6c2/9/9/9/6R2/9/5K3 b - - 2 30", &["g6h6", "g2h2", "h6g6", "h2g2"]);
    assert_eq!(once.last().repetitions(), 1);

    let twice = history(
        "3k5/9/9/6c2/9/9/9/6R2/9/5K3 b - - 2 30",
        &["g6h6", "g2h2", "h6g6", "h2g2", "g6h6", "g2h2", "h6g6", "h2g2"],
    );
    assert_eq!(twice.last().repetitions(), 2);
}

#[test]
fn indexed_bulk_replay_preserves_repetitions() {
    ensure_init();
    let start = Position::from_fen("3k5/9/9/6c2/9/9/9/6R2/9/5K3 b - - 2 30").expect("start");
    let mut position = start.clone();
    let mut moves = Vec::new();
    for text in ["g6h6", "g2h2", "h6g6", "h2g2", "g6h6", "g2h2", "h6g6", "h2g2"] {
        let mv = position.board().parse_move(text).expect("move");
        position = Position::after(&position, mv);
        moves.push(mv);
    }

    let bulk = PositionHistory::from_position_and_moves(start, &moves);
    let prefix = PositionHistory::from_positions(bulk.positions()[..5].to_vec());
    let extended = PositionHistory::from_prefix_and_moves(&prefix, &moves[4..]);

    assert_eq!(extended, bulk);
    assert_eq!(bulk.last().repetitions(), 2);
}

#[test]
fn detects_repetitions_since_last_zeroing_move() {
    ensure_init();

    let current = history("3k5/9/9/6rC1/9/9/9/6R2/9/5K3 b - - 2 30", &["g6h6", "g2h2", "h6g6", "h2g2", "g6h6"]);
    assert!(current.did_repeat_since_last_zeroing_move());

    let before = history("3k5/9/9/6rC1/9/9/9/5R3/9/5K3 b - - 2 30", &["g6h6", "f2h2", "h6g6", "h2g2", "g6h6", "g2h2"]);
    assert!(before.did_repeat_since_last_zeroing_move());

    let older = history(
        "3k5/9/9/6rC1/9/9/9/5R3/9/5K3 b - - 2 30",
        &["g6b6", "f2b2", "b6h6", "b2h2", "h6g6", "h2g2", "g6h6", "g2h2"],
    );
    assert!(older.did_repeat_since_last_zeroing_move());

    let before_zero =
        history("3k5/9/9/6rC1/9/9/9/6R2/9/5K3 b - - 2 30", &["g6f6", "g2f2", "f6g6", "f2g2", "g6h6", "g2h2"]);
    assert!(!before_zero.did_repeat_since_last_zeroing_move());

    let never = history("3k5/9/9/6rC1/9/9/9/6R2/9/5K3 b - - 2 30", &["g6c6", "g2f2"]);
    assert!(!never.did_repeat_since_last_zeroing_move());
}

#[test]
fn game_result_covers_repetition_cases() {
    ensure_init();

    let white_chase = history(
        "3k5/9/9/6c2/9/9/9/6R2/9/5K3 b - - 2 30",
        &["g6h6", "g2h2", "h6g6", "h2g2", "g6h6", "g2h2", "h6g6", "h2g2"],
    );
    assert_eq!(white_chase.compute_game_result(), GameResult::BlackWon);

    let black_chase = history(
        "3k5/9/7r1/9/9/9/9/6C2/9/5K3 b - - 2 30",
        &["h7g7", "g2h2", "g7h7", "h2g2", "h7g7", "g2h2", "g7h7", "h2g2"],
    );
    assert_eq!(black_chase.compute_game_result(), GameResult::WhiteWon);

    let white_check = history(
        "3k5/9/9/9/9/9/9/3R5/9/5K3 b - - 2 30",
        &["d9e9", "d2e2", "e9d9", "e2d2", "d9e9", "d2e2", "e9d9", "e2d2"],
    );
    assert_eq!(white_check.compute_game_result(), GameResult::BlackWon);

    let black_check = history(
        "3k5/9/4r4/9/9/9/9/9/9/5K3 b - - 2 30",
        &["e7f7", "f0e0", "f7e7", "e0f0", "e7f7", "f0e0", "f7e7", "e0f0"],
    );
    assert_eq!(black_check.compute_game_result(), GameResult::WhiteWon);

    for (fen, moves) in [
        ("3k5/9/6r2/9/9/9/9/9/6R2/5K3 b - - 2 30", ["g7h7", "g1h1", "h7g7", "h1g1"]),
        ("4c4/3k5/4b3b/9/9/2B4N1/4p4/3A5/2p1A4/5K3 w - - 2 30", ["h4g2", "e3f3", "g2h4", "f3e3"]),
        ("3k5/9/9/9/9/9/9/9/1r2ARn2/4K4 b", ["b1b0", "e1d0", "b0b1", "d0e1"]),
    ] {
        let moves = [moves, moves].concat();
        let history = history(fen, &moves);
        assert_eq!(history.compute_game_result(), GameResult::Draw, "fen={fen}");
    }
}
