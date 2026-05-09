//! P3：`xiangqi_core` 走子/合法性与 `engin` UCI 解析、基准输出联调。

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use engin::benchmark::{
    bench_one_json, default_benchmark_fen_strings, BenchJsonMeta, BenchSessionParams,
};
use engin::eval::terminal_score;
use engin::uci::parse_position_uci;
use engin::TranspositionTable;
use engin::{root_search_iterative, NNLeafMode, NnEvalSession, RootSearchShared, SearchAblation, SearchLimits};
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
fn startpos_terminal_score_none_with_legal_moves() {
    let pos = Position::from_fen(START_FEN).unwrap();
    assert!(terminal_score(&pos).is_none());
    assert!(!legal_moves_uci(&pos).is_empty());
}

#[test]
fn root_search_with_ablation_policy_off_returns_legal_bestmove() {
    let mut pos = Position::from_fen(START_FEN).unwrap();
    let policy = Arc::new(Mutex::new(None));
    let vocab: HashMap<String, usize> = HashMap::new();
    let mut tt = TranspositionTable::new(4);
    let mut nn_eval = NnEvalSession::default();
    let mut shared = RootSearchShared {
        policy: &policy,
        vocab: &vocab,
        vocab_size: 0,
        tt: &mut tt,
        stop: None,
        ablation: SearchAblation {
            policy_ordering: false,
            nn_leaf_mode: NNLeafMode::MainLeafOnly,
        },
        nn_eval: &mut nn_eval,
    };
    let r = root_search_iterative(&mut pos, 2, &mut shared, SearchLimits::none()).expect("r");
    let legals = legal_moves_uci(&Position::from_fen(START_FEN).unwrap());
    assert!(legals.iter().any(|u| u == &r.best_uci));
}

#[test]
fn bench_json_has_expected_keys() {
    let policy = Arc::new(Mutex::new(None));
    let vocab: HashMap<String, usize> = HashMap::new();
    let meta = BenchJsonMeta::default();
    let session = BenchSessionParams {
        max_depth: 2,
        max_nodes: None,
        policy: &policy,
        vocab: &vocab,
        vocab_size: 0,
        ablation: SearchAblation {
            policy_ordering: true,
            nn_leaf_mode: NNLeafMode::MainLeafOnly,
        },
        hash_mb: 8,
        nn_eval_budget: 0,
        meta: &meta,
    };
    let v = bench_one_json(START_FEN, &session);
    assert!(v.get("bestmove").is_some() || v.get("error").is_some());
    assert!(v.get("bench_config").is_some());
    assert!(v.get("bench_profile").is_some());
    assert!(v.get("nn_eval_budget_used").is_some());
    assert!(v.get("nn_eval_main_leaf_calls").is_some());
    assert!(v.get("nn_eval_qsearch_calls").is_some());
    if let Some(bm) = v.get("bestmove").and_then(|x| x.as_str()) {
        assert!(bm.len() >= 4);
    }
}

#[test]
fn default_benchmark_fens_all_parse() {
    for fen in default_benchmark_fen_strings() {
        let p = Position::from_fen(fen).unwrap_or_else(|e| panic!("非法 FEN {fen}: {e}"));
        assert!(
            !legal_moves_uci(&p).is_empty() || engin::eval::terminal_score(&p).is_some(),
            "基准局面应可走子或终局: {fen}"
        );
    }
}

#[test]
fn uci_setoption_ablation_then_go() {
    let input = b"uci\nisready\nsetoption name UsePolicyOrdering value false\nsetoption name NNLeafMode value Off\nposition startpos\ngo depth 2\nquit\n";
    let mut out = Vec::new();
    engin::uci::run_uci_for_test(Cursor::new(&input[..]), &mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("bestmove"));
    assert!(s.contains("UsePolicyOrdering"));
    assert!(s.contains("NNLeafMode"));
}
