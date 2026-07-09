use xiangqi_core::{legal_moves_uci, Position};

#[test]
fn knight_leg_block_is_respected() {
    let pos = Position::from_fen("4k4/9/9/9/9/9/9/9/1P7/1N2K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(
        !moves.iter().any(|mv| mv.starts_with("b0")),
        "blocked knight should have no legal moves: {moves:?}"
    );
}

#[test]
fn knight_leg_block_is_respected_on_cross_rank_move() {
    let pos = Position::from_fen("2bakab2/9/6n2/pnp1p1C1p/9/2P6/Pc4P1P/B8/9/1N1AKABN1 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(
        !moves.contains(&"h0f1".to_string()),
        "blocked knight leg on g0 should forbid h0f1: {moves:?}"
    );
}

#[test]
fn bishop_eye_block_is_respected() {
    let pos = Position::from_fen("4k4/9/9/9/9/9/9/4P4/9/2B1K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(moves.contains(&"c0a2".to_string()));
    assert!(
        !moves.contains(&"c0e2".to_string()),
        "blocked bishop eye should forbid c0e2: {moves:?}"
    );
}

#[test]
fn cannon_capture_requires_exactly_one_screen() {
    let pos = Position::from_fen("4k4/9/9/4r4/4P4/4C4/9/9/9/4K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(
        moves.contains(&"e4e6".to_string()),
        "cannon should capture rook with one screen: {moves:?}"
    );
}

#[test]
fn cannon_cannot_capture_without_screen() {
    let pos = Position::from_fen("4k4/9/9/4r4/9/4C4/9/9/9/4K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(
        !moves.contains(&"e4e6".to_string()),
        "cannon should not capture without screen: {moves:?}"
    );
}

#[test]
fn pawn_before_river_cannot_move_sideways() {
    let pos = Position::from_fen("4k4/9/9/9/9/9/9/4P4/9/4K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(moves.contains(&"e2e3".to_string()));
    assert!(!moves.contains(&"e2d2".to_string()));
    assert!(!moves.contains(&"e2f2".to_string()));
}

#[test]
fn pawn_after_river_can_move_sideways() {
    let pos = Position::from_fen("5k3/9/9/9/3P5/9/9/9/9/4K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(moves.contains(&"d5d6".to_string()));
    assert!(moves.contains(&"d5c5".to_string()));
    assert!(moves.contains(&"d5e5".to_string()));
}

#[test]
fn moving_blocker_that_exposes_flying_general_is_illegal() {
    let pos = Position::from_fen("4k4/9/9/9/4R4/9/9/9/9/4K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(
        !moves.contains(&"e5d5".to_string()),
        "rook may not move away and expose flying general: {moves:?}"
    );
    assert!(
        !moves.contains(&"e5f5".to_string()),
        "rook may not move away and expose flying general: {moves:?}"
    );
}

#[test]
fn when_in_check_unrelated_piece_moves_are_not_legal() {
    let pos = Position::from_fen("4k4/9/9/4r4/9/9/9/9/9/R3K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(
        !moves.contains(&"a0a1".to_string()),
        "side in check may not play unrelated rook move: {moves:?}"
    );
    assert!(
        !moves.contains(&"a0a9".to_string()),
        "side in check may not play unrelated rook move: {moves:?}"
    );
    assert!(!moves.is_empty(), "there should still be legal evasions: {moves:?}");
}

#[test]
fn cannon_check_also_forbids_unrelated_moves() {
    let pos = Position::from_fen("4k4/9/9/9/4c4/9/9/9/4P4/R3K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(
        !moves.contains(&"a0a1".to_string()),
        "side in cannon check may not play unrelated rook move: {moves:?}"
    );
    assert!(
        !moves.contains(&"a0a9".to_string()),
        "side in cannon check may not play unrelated rook move: {moves:?}"
    );
    assert!(!moves.is_empty(), "there should still be legal evasions: {moves:?}");
}

#[test]
fn only_forced_king_escape_remains_in_check_position() {
    let pos = Position::from_fen("R1cak4/4a4/4b1n2/2n5p/2P1p1b2/N4N3/8P/5C3/5r3/c1BA1KB2 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert_eq!(moves, vec!["f0e0".to_string()]);
}

#[test]
fn no_legal_moves_position_is_detected() {
    let pos = Position::from_fen("3Rkab2/4a4/2P1b4/p3C1c1p/9/4P4/2N5P/9/4A4/1RB1KAB2 b - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(moves.is_empty(), "expected no legal moves: {moves:?}");
}
