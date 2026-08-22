//! Pikafish《象棋程序竞赛规则》第三章第三节的循环棋例回归。
//! Source: <https://www.pikafish.com/rule.html> (2023-11-22).

use std::sync::Once;

use xiangqi_core::{GameResult, GameState, initialize_magic_bitboards};

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

#[test]
fn cycle_examples() {
    ensure_init();

    let cases = [
        (
            "例1",
            "4k4/9/2c2an2/4c4/6R2/9/9/4B4/4A4/3K1AB2 w",
            "g5e5 f7e8 e5g5 e8f7 g5e5 f7e8 e5g5 e8f7",
            GameResult::Draw,
        ),
        (
            "例2",
            "3ak1b2/4a4/4b3r/4C4/5R3/9/9/9/9/4KA3 w",
            "f5g5 e9f9 g5f5 f9e9 f5g5 e9f9 g5f5 f9e9",
            GameResult::Draw,
        ),
        (
            "例3",
            "2ba1k1r1/4a4/4b4/9/9/9/7c1/1R7/9/4K2R1 w",
            "b2b3 h3h2 b3b2 h2h3 b2b3 h3h2 b3b2 h2h3",
            GameResult::Draw,
        ),
        (
            "例4",
            "4k3c/9/4bn2n/8c/6R2/6P2/9/9/9/3K5 w",
            "g5i5 i6f6 i5g5 f6i6 g5i5 i6f6 i5g5 f6i6",
            GameResult::WhiteWon,
        ),
        (
            "例5",
            "5k3/9/9/p1CcC4/c8/9/9/9/9/4K4 w",
            "c6c5 d6d5 c5c6 d5d6 c6c5 d6d5 c5c6 d5d6",
            GameResult::Draw,
        ),
        (
            "例6",
            "5k3/9/9/9/9/3C5/9/4B4/3K5/2p6 w",
            "d1e1 c0d0 e1d1 d0c0 d1e1 c0d0 e1d1 d0c0",
            GameResult::BlackWon,
        ),
        (
            "例7",
            "3a1kb2/4a4/4r3b/7c1/9/9/9/8R/7CC/3K5 w",
            "i2h2 h6i6 h2i2 i6h6 i2h2 h6i6 h2i2 i6h6",
            GameResult::BlackWon,
        ),
        (
            "例8",
            "4kr3/4c4/7R1/4P4/9/9/4C4/9/9/4K4 w",
            "h7h8 e8e7 h8h7 e7e8 h7h8 e8e7 h8h7 e7e8",
            GameResult::BlackWon,
        ),
        (
            "例9",
            "2b1k4/9/4b4/4r3p/P5R1c/9/9/4C4/4K4/9 w",
            "g5g6 e6e5 g6g5 e5e6 g5g6 e6e5 g6g5 e5e6",
            GameResult::BlackWon,
        ),
        (
            "例10",
            "9/3rkr3/2c1ca3/9/9/9/9/9/4A4/3KC4 w",
            "e1d2 e7d7 d2e1 d7e7 e1d2 e7d7 d2e1 d7e7",
            GameResult::Draw,
        ),
        (
            "例11",
            "4k4/9/9/9/4C4/9/4r4/4C4/9/4K1B2 w",
            "e5f5 e3f3 f5e5 f3e3 e5f5 e3f3 f5e5 f3e3",
            GameResult::BlackWon,
        ),
        (
            "例12",
            "3k5/2R6/9/9/9/9/9/9/6r2/4K1N2 w",
            "c8c9 d9d8 c9c8 d8d9 c8c9 d9d8 c9c8 d8d9",
            GameResult::BlackWon,
        ),
        (
            "例13",
            "5k3/9/9/9/9/9/9/9/2p6/3KR1Bc1 w",
            "e0f0 f9e9 f0e0 e9f9 e0f0 f9e9 f0e0 e9f9",
            GameResult::BlackWon,
        ),
        (
            "例14",
            "3a1kb2/4a4/8b/9/4n4/2R6/9/4B4/9/4K4 w",
            "c4c5 e5d3 c5d5 d3b4 d5d4 b4c6 d4c4 c6e5 c4c5 e5d3 c5d5 d3b4 d5d4 b4c6 d4c4 c6e5",
            GameResult::Draw,
        ),
        (
            "例15",
            "3a1kb2/4a4/8b/9/4n4/2R6/9/4B4/9/4K4 w",
            "c4c5 e5d3 c5d5 d3b4 d5d4 b4c6 d4c4 c6e5 c4c5 e5d3 c5c3 d3e5 c3c5",
            GameResult::BlackWon,
        ),
        (
            "例16",
            "6R2/4k4/9/4r4/9/9/5p3/5A3/5K3/9 w",
            "g9g8 e8e9 g8f8 f3g3 f8g8 g3f3 g8g9 e9e8 g9g8 e8e9",
            GameResult::Draw,
        ),
        (
            "例17",
            "9/2N1k4/9/2r6/2b6/4C4/9/4B4/9/5K3 w",
            "e4c4 c5e7 c8a7 c6a6 c4e4 e7c5 a7c8 a6c6 e4c4 c5e7 c8a7 c6a6 c4e4 e7c5 a7c8 a6c6",
            GameResult::Draw,
        ),
        (
            "例18",
            "4k4/9/4n4/1c2N1c2/4N4/6P2/9/9/9/4K4 w",
            "e6g5 e7c6 g5e6 c6e7 e6g5 e7c6 g5e6 c6e7",
            GameResult::BlackWon,
        ),
        (
            "例19",
            "4k4/9/4n4/1c2N1r2/4N4/6P2/9/9/9/4K4 w",
            "e6g5 e7c6 g5e6 c6e7 e6g5 e7c6 g5e6 c6e7",
            GameResult::Draw,
        ),
        (
            "例20",
            "2R6/4k4/4b4/9/2b6/4r4/9/3K1A3/5C3/9 w",
            "f1e1 e8f8 e1f1 f8e8 f1e1 e8f8 e1f1 f8e8",
            GameResult::Draw,
        ),
        (
            "例21",
            "3k2b2/4a4/3ab4/9/9/9/9/3N5/7R1/c1c1K4 w",
            "e0e1 c0c1 e1e0 c1c0 e0e1 c0c1 e1e0 c1c0",
            GameResult::Draw,
        ),
        (
            "例22",
            "3k5/9/9/9/5r3/9/4n4/9/4AC3/3A1K3 w",
            "f1f2 e3d1 f2f1 d1e3 f1f2 e3d1 f2f1 d1e3",
            GameResult::WhiteWon,
        ),
        (
            "例23",
            "5k3/9/9/2N6/9/1N7/9/4BC3/9/cr1RK4 w",
            "b4c2 b0c0 c2b4 c0b0 b4c2 b0c0 c2b4 c0b0",
            GameResult::WhiteWon,
        ),
        (
            "例24",
            "5k3/9/9/9/9/4c4/3r5/3NB4/4A4/4K4 w",
            "e0d0 e4d4 d0e0 d4e4 e0d0 e4d4 d0e0 d4e4",
            GameResult::WhiteWon,
        ),
        (
            "例25",
            "3k5/9/9/9/9/9/9/9/1cr2Rn2/3AK4 w",
            "d0e1 c1c0 e1d0 c0c1 d0e1 c1c0 e1d0 c0c1",
            GameResult::Draw,
        ),
        (
            "例26",
            "3k5/9/9/9/9/9/9/9/1cr2Rn2/3AK4 w",
            "d0e1 c1c2 e1d0 c2c1 d0e1 c1c2 e1d0 c2c1",
            GameResult::Draw,
        ),
        (
            "例27",
            "3k5/9/9/9/9/9/9/9/1cr3Rn1/3AK4 w",
            "d0e1 c1c2 e1d0 c2c1 d0e1 c1c2 e1d0 c2c1",
            GameResult::Draw,
        ),
        (
            "例28",
            "3k5/9/3a5/2C6/2r6/2C6/2r6/5A3/9/5K3 w",
            "c4d4 c5d5 d4c4 d5c5 c4d4 c5d5 d4c4 d5c5",
            GameResult::Draw,
        ),
        (
            "例29",
            "3k5/9/3a5/2C6/2r6/2C6/2r6/5A3/9/5K3 w",
            "c4e4 c5e5 e4c4 e5c5 c4e4 c5e5 e4c4 e5c5",
            GameResult::BlackWon,
        ),
        (
            "例30",
            "3k5/9/3a5/2C6/2r6/2C6/2r6/2N2A3/9/5K3 w",
            "c4e4 c5e5 e4c4 e5c5 c4e4 c5e5 e4c4 e5c5",
            GameResult::Draw,
        ),
        (
            "例31",
            "3r5/4ck3/3R5/9/9/9/3cR4/9/4A4/4KA3 w",
            "e0d0 e8d8 d0e0 d8e8 e0d0 e8d8 d0e0 d8e8",
            GameResult::BlackWon,
        ),
        (
            "例32",
            "rCRak4/4a4/9/9/9/4p4/9/9/9/4K4 w",
            "e0d0 e4d4 d0e0 d4e4 e0d0 e4d4 d0e0 d4e4",
            GameResult::BlackWon,
        ),
        (
            "例33",
            "5k3/9/9/9/9/9/9/4p4/4A4/3K1ABc1 w",
            "d0d1 h0h1 d1d0 h1h0 d0d1 h0h1 d1d0 h1h0",
            GameResult::Draw,
        ),
        (
            "例34",
            "4ka3/4a4/5R3/2r6/2r6/9/3R5/9/9/2p2K3 w",
            "d3e3 e9d9 e3d3 d9e9 d3e3 e9d9 e3d3 d9e9",
            GameResult::Draw,
        ),
        (
            "例35",
            "1CRck4/4a4/9/9/9/9/9/9/3CA4/5K3 w",
            "e1d2 e8d7 d2e1 d7e8 e1d2 e8d7 d2e1 d7e8",
            GameResult::Draw,
        ),
        (
            "例36",
            "4k4/3P5/9/9/3c5/4C4/3c5/9/9/3K5 w",
            "e4d4 d5e5 d4e4 e5d5 e4d4 d5e5 d4e4 e5d5",
            GameResult::Draw,
        ),
        (
            "例37",
            "3a1a3/2R6/3k5/3r5/3N5/9/9/9/4A4/4KA3 w",
            "d5f4 d6f6 f4d5 f6d6 c8c7 d7d8 c7c8 d8d7",
            GameResult::Draw,
        ),
        (
            "例38",
            "4k4/9/9/9/9/9/7n1/4B4/3rAR3/4KA3 w",
            "f1f3 h3g1 f3f1 g1h3 f1h1 h3f2 h1f1 f2h3",
            GameResult::Draw,
        ),
        (
            "例39",
            "3ak4/c2R1R2c/4b4/9/9/9/9/9/9/4K4 w",
            "f8f6 d9e8 f6f8 e8d9 f8f6 d9e8 f6f8 e8d9",
            GameResult::Draw,
        ),
    ];

    for (name, fen, moves, expected) in cases {
        let moves: Vec<_> = moves.split_whitespace().collect();
        let history = GameState::from_fen_moves(fen, &moves)
            .unwrap_or_else(|error| panic!("{name}: invalid example: {error}"))
            .position_history();
        assert_eq!(
            history.last().repetitions(),
            2,
            "{name}: must end in a threefold repetition"
        );
        assert_eq!(history.compute_game_result(), expected, "{name}");
    }
}

/// 循环区间内只要存在将军，双方均不得以「捉」判负；此例中另一着虽形成捉，
/// 仍只能按长将判定。双方都并非每步将军，故判和。
#[test]
fn check_in_cycle_disables_chase_judgement() {
    ensure_init();

    let moves = ["i9i8", "h9h8", "i8i9", "h8h9"];
    let moves: Vec<_> = moves.into_iter().cycle().take(8).collect();
    let history = GameState::from_fen_moves("3ak1brC/4a4/4b4/p2Rp3p/9/2P3p2/P3P3P/2N6/3KA4/5Ac2 w - - 0 1", &moves)
        .expect("valid check/chase cycle")
        .position_history();

    assert_eq!(history.last().repetitions(), 2);
    assert!(
        history
            .positions()
            .iter()
            .any(|position| position.board().is_under_check())
    );
    assert_eq!(history.compute_game_result(), GameResult::Draw);
}
