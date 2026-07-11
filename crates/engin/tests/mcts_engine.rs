//! MCTS / UCI 回归护栏。

use engin::history::PositionHistory;
use engin::mcts::{MctsBudget, MctsConfig, MctsEngine, OnnxPolicyValueEval, SharedPolicy};
use std::io::Cursor;
use xiangqi_core::{legal_moves_uci, Position, START_FEN};

#[test]
fn tree_reuse_across_two_go_nodes() {
    let policy: SharedPolicy = None;
    let mut engine = MctsEngine::new(
        MctsConfig::default(),
        OnnxPolicyValueEval::new(policy, MctsConfig::default().nn_cache_size),
    );
    let history = PositionHistory::new_startpos();
    let budget = MctsBudget {
        max_playouts: Some(16),
        ..Default::default()
    };
    engine.search_root_history(&history, budget.clone()).expect("first");
    let nodes_after_first = engine.tree.len();
    assert!(nodes_after_first > 0);
    engine
        .search_root_history(&history, budget)
        .expect("second");
    assert!(engine.tree.len() >= nodes_after_first);
}

#[test]
fn go_wtime_via_uci_does_not_error() {
    let input = b"uci\nisready\nposition startpos\ngo wtime 60000 btime 60000 nodes 4\nquit\n";
    let mut out = Vec::new();
    engin::uci::run_uci_for_test(Cursor::new(&input[..]), &mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("bestmove"));
    assert!(!s.contains("未含可执行"));
}

#[test]
fn root_searchmoves_filter_limits_bestmove() {
    let pos0 = Position::from_fen(START_FEN).unwrap();
    let mut legals = legal_moves_uci(&pos0);
    legals.sort();
    let only = legals.first().expect("legal").clone();
    let mv = xiangqi_core::uci_to_move(&pos0, &only).expect("uci");

    let policy: SharedPolicy = None;
    let mut engine = MctsEngine::new(
        MctsConfig::default(),
        OnnxPolicyValueEval::new(policy, MctsConfig::default().nn_cache_size),
    );
    let history = PositionHistory::new_startpos();
    let result = engine
        .search_root_history_with_progress(
            &history,
            MctsBudget {
                max_nodes: Some(32),
                ..Default::default()
            },
            std::time::Duration::ZERO,
            Some(&[mv]),
            |_| {},
        )
        .expect("search");
    assert_eq!(result.best_move, Some(mv));
}

#[test]
fn multipv_engine_returns_two_pv_lines() {
    let policy: SharedPolicy = None;
    let mut config = MctsConfig::default();
    config.multi_pv = 2;
    let mut engine = MctsEngine::new(
        config,
        OnnxPolicyValueEval::new(policy, config.nn_cache_size),
    );
    let history = PositionHistory::new_startpos();
    let result = engine
        .search_root_history(
            &history,
            MctsBudget {
                max_nodes: Some(64),
                ..Default::default()
            },
        )
        .expect("search");
    assert_eq!(result.multi_pv, 2);
    assert_eq!(result.pv_lines.len(), 2);
    assert!(result.pv_lines[0].pv.first().is_some());
}

#[test]
fn go_nodes_movetime_ponder_stop_searchmoves_chain() {
    let pos0 = Position::from_fen(START_FEN).unwrap();
    let mv = legal_moves_uci(&pos0).into_iter().next().expect("mv");
    let input = format!(
        "uci\nisready\nsetoption name Threads value 1\nposition startpos moves {mv}\ngo ponder movetime 500 nodes 8\nstop\nposition startpos moves {mv}\ngo searchmoves {mv} nodes 8\nquit\n"
    );
    let mut out = Vec::new();
    engin::uci::run_uci_for_test(Cursor::new(input.as_bytes()), &mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("bestmove"));
    assert!(!s.ends_with("bestmove (none)"));
}

#[test]
fn fixed_fen_search_stats_in_expected_range() {
    let policy: SharedPolicy = None;
    let mut engine = MctsEngine::new(
        MctsConfig::default(),
        OnnxPolicyValueEval::new(policy, MctsConfig::default().nn_cache_size),
    );
    let history = PositionHistory::new_startpos();
    let result = engine
        .search_root_history(
            &history,
            MctsBudget {
                max_nodes: Some(32),
                ..Default::default()
            },
        )
        .expect("search");
    assert!(result.playouts > 0);
    assert!(result.nodes >= result.playouts as usize);
    assert!(result.depth >= 1);
    assert!(result.seldepth >= 1);
    assert!(!result.pv.is_empty());
}

#[test]
fn ponderhit_after_ponder_does_not_hang() {
    let pos0 = Position::from_fen(START_FEN).unwrap();
    let mv = legal_moves_uci(&pos0).into_iter().next().expect("mv");
    let input = format!(
        "uci\nisready\nposition startpos moves {mv}\ngo ponder nodes 4\nstop\nponderhit\ngo nodes 4\nquit\n"
    );
    let mut out = Vec::new();
    engin::uci::run_uci_for_test(Cursor::new(input.as_bytes()), &mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.matches("bestmove ").count() >= 1);
}
