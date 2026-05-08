//! Perft 与 do/undo 回归（与 pikafish-rust 测试一致）。

use xiangqi_core::movegen::{generate, ExtMove, GenType};
use xiangqi_core::types::*;
use xiangqi_core::Position;

const START_FEN: &str =
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

fn perft(pos: &Position, depth: i32) -> u64 {
    if depth <= 0 {
        return 1;
    }
    let mut moves = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; 256];
    let count = generate(pos, GenType::Legal, &mut moves);
    if depth == 1 {
        return count as u64;
    }
    let mut nodes = 0;
    for em in moves[..count].iter() {
        let mv = em.mv;
        assert!(
            mv.is_ok(),
            "Legal move gen produced invalid move raw={:x}",
            mv.raw()
        );
        let mut child = Position::new(pos.zobrist);
        child.set_fen(&pos.fen()).unwrap();
        child.do_move(mv);
        nodes += perft(&child, depth - 1);
    }
    nodes
}

#[test]
fn perft_startpos_depth1() {
    let pos = Position::from_fen(START_FEN).unwrap();
    assert_eq!(perft(&pos, 1), 44);
}

#[test]
fn perft_startpos_depth2() {
    let pos = Position::from_fen(START_FEN).unwrap();
    assert_eq!(perft(&pos, 2), 1926);
}

#[test]
fn perft_startpos_depth3() {
    let pos = Position::from_fen(START_FEN).unwrap();
    assert_eq!(perft(&pos, 3), 80069);
}

#[test]
fn perft_simple_position() {
    let pos =
        Position::from_fen("9/9/9/9/9/9/9/9/4k4/4K4 w - - 0 1").unwrap();
    let count = perft(&pos, 1);
    assert!(count > 0, "King should have at least one legal move");
}

#[test]
fn test_do_undo_move() {
    let mut pos = Position::from_fen(START_FEN).unwrap();
    let mut moves = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; 256];
    let count = generate(&pos, GenType::Legal, &mut moves);
    assert!(count > 0);

    for em in moves[..count].iter() {
        let fen0 = pos.fen();
        pos.do_move(em.mv);
        pos.undo_move(em.mv);
        assert_eq!(pos.fen(), fen0, "do/undo mismatch");
    }
}
