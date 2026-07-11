use std::collections::BTreeSet;
use std::sync::Once;

use xiangqi_core::{initialize_magic_bitboards, startpos_board, ChessBoard, CoreError, STARTPOS_FEN};

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

fn perft(board: &ChessBoard, max_depth: u32, depth: u32) -> u64 {
    if depth == max_depth {
        return 1;
    }

    let moves = board.generate_pseudolegal_moves();
    let legal_moves = board.generate_legal_moves();
    let mut iter = legal_moves.iter();

    let mut total = 0u64;
    for mv in moves {
        if !board.is_legal_move(mv) {
            continue;
        }

        let legal = iter.next().expect("legal move ordering mismatch");
        assert_eq!(
            legal,
            &mv,
            "move order mismatch:\n{}\nlegal={} pseudo={}",
            board.debug_string(),
            legal,
            mv
        );

        let mut new_board = board.clone();
        new_board.apply_move(mv);
        new_board.mirror();
        total += perft(&new_board, max_depth, depth + 1);
    }
    assert!(iter.next().is_none(), "extra legal moves remain");
    total
}

#[test]
fn illegal_pawn_position() {
    ensure_init();
    let err = ChessBoard::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P2PP1P1P/1C5C1/9/RNBAKABNR w").unwrap_err();
    assert!(matches!(err, CoreError::InvalidFen(_)));
}

#[test]
fn pseudolegal_moves_starting_pos() {
    ensure_init();
    let (mut board, _) = ChessBoard::from_fen(STARTPOS_FEN).unwrap();
    board.mirror();
    assert_eq!(board.generate_pseudolegal_moves().len(), 44);
}

#[test]
fn partial_fen() {
    ensure_init();
    let (board, state) = ChessBoard::from_fen("rnbakabnr//1c5c1/p1p1p1p1p///P1P1P1P1P/1C2K2C1").unwrap();
    assert_eq!(board.generate_pseudolegal_moves().len(), 28);
    assert_eq!(state.rule60_ply, 0);
    assert_eq!(state.game_ply, 1);
}

#[test]
fn partial_fen_with_spaces() {
    ensure_init();
    let (board, state) = ChessBoard::from_fen("    rnbakabnr//1c5c1/p1p1p1p1p///P1P1P1P1P/1C2K2C1    w   ").unwrap();
    assert_eq!(board.generate_pseudolegal_moves().len(), 28);
    assert_eq!(state.rule60_ply, 0);
    assert_eq!(state.game_ply, 1);
}

#[test]
fn invalid_fen_cases() {
    ensure_init();
    let fens = [
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P2PP1P1P/1C5C1/9/RNBAKABNR w",
        "rrnbakabnr/9/1c5c1/p3p1p1p/3p5/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/6A2/RNBAK1BNR w",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/6B2/RNBAKA1NR w",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/6K2/RNBA1ABNR w",
    ];
    for fen in fens {
        assert!(ChessBoard::from_fen(fen).is_err(), "expected invalid fen: {fen}");
    }
}

#[test]
fn perft_starting_pos() {
    ensure_init();
    let (board, _) = ChessBoard::from_fen(STARTPOS_FEN).unwrap();
    assert_eq!(perft(&board, 1, 0), 44);
    assert_eq!(perft(&board, 2, 0), 1920);
    assert_eq!(perft(&board, 3, 0), 79666);
    assert_eq!(perft(&board, 4, 0), 3290240);
    assert_eq!(perft(&board, 5, 0), 133312995);
}

#[test]
fn perft_complex_positions() {
    ensure_init();
    let cases = [
        (
            "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w",
            [38u64, 1128, 43929, 1339047, 53112976],
        ),
        (
            "1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w",
            [7, 281, 8620, 326201, 10369923],
        ),
        (
            "5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w",
            [25, 424, 9850, 202884, 4739553],
        ),
        (
            "CRN1k1b2/3ca4/4ba3/9/2nr5/9/9/4B4/4A4/4KA3 w",
            [28, 516, 14808, 395483, 11842230],
        ),
        (
            "R1N1k1b2/9/3aba3/9/2nr5/2B6/9/4B4/4A4/4KA3 w",
            [21, 364, 7626, 162837, 3500505],
        ),
        ("C1nNk4/9/9/9/9/9/n1pp5/B3C4/9/3A1K3 w", [28, 222, 6241, 64971, 1914306]),
        (
            "4ka3/4a4/9/9/4N4/p8/9/4C3c/7n1/2BK5 w",
            [23, 345, 8124, 149272, 3513104],
        ),
        ("2b1ka3/9/b3N4/4n4/9/9/9/4C4/2p6/2BK5 w", [21, 195, 3883, 48060, 933096]),
        (
            "1C2ka3/9/C1Nab1n2/p3p3p/6p2/9/P3P3P/3AB4/3p2c2/c1BAK4 w",
            [30, 830, 22787, 649866, 17920736],
        ),
        (
            "CnN1k1b2/c3a4/4ba3/9/2nr5/9/9/4C4/4A4/4KA3 w",
            [19, 583, 11714, 376467, 8148177],
        ),
    ];

    for (fen, expected) in cases {
        let (board, _) = ChessBoard::from_fen(fen).unwrap();
        for (depth, &count) in expected.iter().enumerate() {
            assert_eq!(
                perft(&board, depth as u32 + 1, 0),
                count,
                "fen={fen} depth={}",
                depth + 1
            );
        }
    }
}

#[test]
fn has_mating_material_cases() {
    ensure_init();
    let (board, _) = ChessBoard::from_fen(STARTPOS_FEN).unwrap();
    assert!(board.has_mating_material());

    let (board, _) = ChessBoard::from_fen("3k5/9/9/9/9/9/9/9/9/5K3 w - - 0 1").unwrap();
    assert!(!board.has_mating_material());

    let (board, _) = ChessBoard::from_fen("3k5/4a4/9/9/9/9/9/9/4A4/3A1K3 w - - 0 1").unwrap();
    assert!(!board.has_mating_material());
    let (board, _) = ChessBoard::from_fen("3k5/4a4/9/9/9/9/9/5A3/4A4/2B2K3 w - - 0 1").unwrap();
    assert!(!board.has_mating_material());

    let (board, _) = ChessBoard::from_fen("3k5/4a4/9/9/9/9/9/5A3/R3A4/2B2K3 w - - 0 1").unwrap();
    assert!(board.has_mating_material());
    let (board, _) = ChessBoard::from_fen("3k5/4a4/8c/9/9/9/9/5A3/4A4/2B2K3 w - - 0 1").unwrap();
    assert!(board.has_mating_material());
    let (board, _) = ChessBoard::from_fen("3k5/4a4/9/9/9/9/9/N4A3/4A2N1/2B2K3 w - - 0 1").unwrap();
    assert!(board.has_mating_material());
}

#[test]
fn legal_move_set_starting_pos() {
    ensure_init();
    let (board, _) = ChessBoard::from_fen(STARTPOS_FEN).unwrap();
    let moves: BTreeSet<String> = board
        .generate_legal_moves()
        .into_iter()
        .map(|m| m.to_string())
        .collect();
    assert_eq!(moves.len(), 44);
    assert!(moves.contains("i3i4"));
    assert!(moves.contains("b0c2"));
}

#[test]
fn startpos_board_and_hash() {
    ensure_init();
    let (from_fen, _) = ChessBoard::from_fen(STARTPOS_FEN).unwrap();
    assert_eq!(*startpos_board(), from_fen);
    assert_ne!(from_fen.hash(), 0);
}
