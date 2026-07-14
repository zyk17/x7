//! P3 `go nodes` UCI transcript 验收。

use std::path::Path;
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

/// px0 `MultiPV` returns the independently ranked root children from
/// `Search::GetBestChildrenNoTemperature` (`search.cc:239-246,705-808`).
#[test]
fn classic_engine_multipv_emits_ranked_root_lines() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("setoption name MultiPV value 2", "0.0.0")
        .expect("multipv option");
    uci.process_line("setoption name ScoreType value Q", "0.0.0")
        .expect("score type option");
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go nodes 32", "0.0.0").expect("go nodes");
    drop(uci);

    assert!(
        responder.responses.iter().any(|line| line.contains(" multipv 1 pv ")),
        "responses: {:?}",
        responder.responses
    );
    assert!(
        responder.responses.iter().any(|line| line.contains(" multipv 2 pv ")),
        "responses: {:?}",
        responder.responses
    );
    assert!(
        responder.responses.iter().any(|line| line.contains(" score cp 0 ")),
        "responses: {:?}",
        responder.responses
    );
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

/// px0 `StringsToMovelist` filters root selection itself, not just the final
/// response (`src/search/classic/wrapper.cc:78-100`, `search.cc:1668-1740`).
#[test]
fn classic_engine_searchmoves_restricts_root_selection() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go nodes 8 searchmoves a0a1", "0.0.0")
        .expect("go searchmoves");
    drop(uci);

    assert!(
        responder.responses.iter().any(|line| line.starts_with("bestmove a0a1")),
        "responses: {:?}",
        responder.responses
    );
    for info in responder.responses.iter().filter(|line| line.starts_with("info ")) {
        assert!(info.contains(" pv a0a1"), "response escaped root filter: {info}");
    }
}

/// px0 throws when every requested `searchmoves` entry is illegal
/// (`src/search/classic/wrapper.cc:88-98`).
#[test]
fn classic_engine_rejects_all_illegal_searchmoves() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    let error = uci
        .process_line("go nodes 8 searchmoves a0a9", "0.0.0")
        .expect_err("illegal searchmoves must fail");
    assert_eq!(error.to_string(), "No legal searchmoves.");
}

#[test]
fn unavailable_engine_does_not_return_uniform_bestmove() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::unavailable();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go nodes 8", "0.0.0").expect("go nodes");
    drop(uci);
    assert!(responder
        .responses
        .iter()
        .any(|line| line.starts_with("info string cannot search:")));
    assert!(!responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}

/// px0 updates its backend configuration before accepting a new position
/// (`src/engine.cc:153-167,187-197`). The Rust `WeightsFile` subset accepts
/// the formal ONNX artifact instead of px0 protobuf weights.
#[test]
fn weights_file_enables_main_uci_onnx_search_if_local_x7_exists() {
    ensure_init();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/x7.onnx");
    if !path.is_file() {
        eprintln!("skip: {} is absent", path.display());
        return;
    }
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::unavailable();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line(&format!("setoption name WeightsFile value {}", path.display()), "0.0.0")
        .expect("configure weights");
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go nodes 1", "0.0.0").expect("go nodes");
    drop(uci);
    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}
