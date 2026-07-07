use xiangqi_core::{legal_moves_uci, Position};

#[test]
fn knight_leg_block_is_respected() {
    let pos = Position::from_fen("4k4/9/9/9/9/9/9/9/1P7/1N2K4 w - - 0 1").expect("fen");
    let moves = legal_moves_uci(&pos);
    assert!(!moves.iter().any(|mv| mv.starts_with("b0")), "blocked knight should have no legal moves: {moves:?}");
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
    assert!(!moves.contains(&"c0e2".to_string()), "blocked bishop eye should forbid c0e2: {moves:?}");
}
