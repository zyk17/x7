//! P3 `go nodes` UCI transcript 验收。

use std::sync::Once;

use engin::{ClassicEngine, UciLoop, UciOptions, VecUciResponder};
use xiangqi_core::initialize_magic_bitboards;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

#[test]
fn classic_engine_go_nodes_emits_bestmove() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go nodes 8", "0.0.0").expect("go nodes");
    drop(uci);
    assert!(
        responder.responses.iter().any(|line| line.starts_with("bestmove ")),
        "expected bestmove, got {:?}",
        responder.responses
    );
    assert_eq!(engine.search().total_root_visits(), 8);
    let bestmove = responder
        .responses
        .iter()
        .find(|line| line.starts_with("bestmove "))
        .expect("bestmove line");
    assert!(bestmove.contains(" ponder a9a8"), "unexpected ponder: {bestmove}");
}

#[test]
fn classic_engine_movetime_runs_at_least_one_simulation() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go movetime 0", "0.0.0").expect("go movetime");
    drop(uci);

    assert!(engine.search().total_root_visits() >= 1);
    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}

#[test]
fn classic_engine_rejects_non_positive_node_budget() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    let error = uci
        .process_line("go nodes 0", "0.0.0")
        .expect_err("zero nodes must fail");
    assert_eq!(error.to_string(), "go nodes must be positive");
}
