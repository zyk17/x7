//! `xiangqi_core` 与 `engin` 的最小联调。

use engin::benchmark::{bench_one_json, default_benchmark_fen_strings, BenchJsonMeta, BenchSessionParams};
use engin::history::PositionHistory;
use engin::mcts::{MctsBudget, MctsConfig, MctsEngine, OnnxPolicyValueEval, SharedPolicy};
use engin::uci::{parse_position_history_uci, parse_position_uci};
use std::io::Cursor;
use xiangqi_core::{legal_moves_uci, uci_to_move, Position, START_FEN};

#[test]
fn parse_position_matches_stepwise_do_move() {
    let pos0 = Position::from_fen(START_FEN).unwrap();
    let first = legal_moves_uci(&pos0).into_iter().next().expect("至少一着");
    let line = format!("position startpos moves {first}");
    let pos_parsed = parse_position_uci(&line).expect("parse");

    let mut pos_step = pos0;
    let mv = uci_to_move(&pos_step, &first).expect("uci_to_move");
    assert!(pos_step.legal(mv));
    pos_step.do_move(mv);

    assert_eq!(pos_parsed.fen(), pos_step.fen());
}

#[test]
fn parse_position_history_matches_stepwise_do_move() {
    let pos0 = Position::from_fen(START_FEN).unwrap();
    let first = legal_moves_uci(&pos0).into_iter().next().expect("至少一着");
    let line = format!("position startpos moves {first}");
    let parsed = parse_position_history_uci(&line).expect("parse");

    let mut step = PositionHistory::new_startpos();
    let mv = uci_to_move(step.current(), &first).expect("uci_to_move");
    step.push_move(mv);

    assert_eq!(parsed.len(), step.len());
    assert_eq!(parsed.current().fen(), step.current().fen());
}

#[test]
fn mcts_returns_legal_bestmove_without_onnx() {
    let pos = Position::from_fen(START_FEN).unwrap();
    let policy: SharedPolicy = None;
    let mut engine = MctsEngine::new(MctsConfig::default(), OnnxPolicyValueEval::new(policy.clone()));
    let result = engine
        .search_root(
            &pos,
            MctsBudget {
                max_playouts: Some(64),
                max_nodes: None,
                max_depth: None,
                deadline: None,
                stop: None,
            },
        )
        .expect("mcts result");
    let best = result.best_move.expect("best move");
    let best_uci = xiangqi_core::move_to_uci(best);
    let legals = legal_moves_uci(&Position::from_fen(START_FEN).unwrap());
    assert!(legals.iter().any(|u| u == &best_uci));
}

#[test]
fn mcts_bestmove_resolves_check_without_onnx() {
    let fen = "k3r4/9/9/9/9/9/9/9/9/4K4 w - - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    assert!(pos.checkers() != 0, "test position must be in check");
    let policy: SharedPolicy = None;
    let mut engine = MctsEngine::new(MctsConfig::default(), OnnxPolicyValueEval::new(policy.clone()));
    let result = engine
        .search_root(
            &pos,
            MctsBudget {
                max_playouts: Some(32),
                max_nodes: None,
                max_depth: None,
                deadline: None,
                stop: None,
            },
        )
        .expect("mcts result");
    let best = result.best_move.expect("best move");
    assert!(pos.legal(best), "bestmove must be legal");
    let mut next = pos.clone_for_search();
    next.do_move(best);
    assert_eq!(next.checkers(), 0, "bestmove must resolve check");
}

#[test]
fn bench_json_has_expected_keys() {
    let policy: SharedPolicy = None;
    let meta = BenchJsonMeta::default();
    let session = BenchSessionParams {
        budget: MctsBudget {
            max_playouts: Some(32),
            max_nodes: None,
            max_depth: None,
            deadline: None,
            stop: None,
        },
        config: MctsConfig::default(),
        policy: &policy,
        meta: &meta,
        threads: 1,
    };
    let v = bench_one_json(START_FEN, &session);
    assert!(v.get("bestmove").is_some() || v.get("error").is_some());
    assert!(v.get("bench_config").is_some());
    assert!(v.get("mcts_config").is_some());
    assert!(v.get("playouts").is_some());
    assert!(v.get("root_visits").is_some());
    assert!(v.get("best_value").is_some());
}

#[test]
fn default_benchmark_fens_all_parse() {
    for fen in default_benchmark_fen_strings() {
        let p = Position::from_fen(fen).unwrap_or_else(|e| panic!("非法 FEN {fen}: {e}"));
        assert!(
            !legal_moves_uci(&p).is_empty() || engin::terminal_score(&p).is_some(),
            "基准局面应可走子或终局: {fen}"
        );
    }
}

#[test]
fn uci_setoption_then_go() {
    let input = b"uci\nisready\nsetoption name MctsPlayouts value 64\nsetoption name MctsCpuct value 1.5\nposition startpos\ngo nodes 2\nquit\n";
    let mut out = Vec::new();
    engin::uci::run_uci_for_test(Cursor::new(&input[..]), &mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("bestmove"));
    assert!(s.contains("MctsPlayouts"));
    assert!(s.contains("MctsCpuct"));
    assert!(s.contains("MctsFpuReduction"));
    assert!(s.contains("MctsBatchCap"));
    assert!(s.contains("MctsWorkers"));
}
