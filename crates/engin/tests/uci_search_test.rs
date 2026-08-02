//! Stream UCI lifecycle regressions.

use std::sync::Once;

use engin::{Engine, UciLoop, VecUciResponder};
use xiangqi_core::initialize_magic_bitboards;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

#[test]
fn go_nodes_reports_info_and_bestmove() {
    ensure_init();
    let mut engine = Engine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position startpos", "test").expect("position");
    uci.process_line("go nodes 8", "test").expect("go");
    uci.process_line("wait", "test").expect("wait");
    drop(uci);

    assert!(responder.responses.iter().any(|line| line.starts_with("info ")));
    assert!(
        responder
            .responses
            .iter()
            .any(|line| line.contains(" depth ") && line.contains(" seldepth "))
    );
    assert!(responder.responses.iter().any(|line| line.contains(" score cp ")));
    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}

#[test]
fn stop_emits_exactly_one_bestmove() {
    ensure_init();
    let mut engine = Engine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position startpos", "test").expect("position");
    uci.process_line("go infinite", "test").expect("go");
    uci.process_line("stop", "test").expect("stop");
    uci.process_line("wait", "test").expect("wait");
    drop(uci);

    assert_eq!(
        responder
            .responses
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count(),
        1
    );
}

#[test]
fn searchmoves_restricts_the_selected_move() {
    ensure_init();
    let mut engine = Engine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position startpos", "test").expect("position");
    uci.process_line("go nodes 8 searchmoves h2h3", "test").expect("go");
    uci.process_line("wait", "test").expect("wait");
    drop(uci);

    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove h2h3")));
}

#[test]
fn clock_go_uses_the_side_to_move_budget() {
    ensure_init();
    let mut engine = Engine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position startpos", "test").expect("position");
    uci.process_line("go wtime 1000 btime 1000 winc 0 binc 0 movestogo 20", "test")
        .expect("clock go");
    uci.process_line("wait", "test").expect("wait");
    drop(uci);

    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}

#[test]
fn proven_terminal_child_reports_uci_mate() {
    ensure_init();
    let mut engine = Engine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position fen 4k4/4PR3/3RC4/9/9/9/9/9/9/4K4 w - - 0 1", "test")
        .expect("position");
    uci.process_line("go nodes 8 searchmoves d7d8", "test").expect("go");
    uci.process_line("wait", "test").expect("wait");
    drop(uci);

    assert!(responder.responses.iter().any(|line| line.contains("score mate 1")));
    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove d7d8")));
}

#[test]
fn checkmated_root_reports_negative_uci_mate() {
    ensure_init();
    let mut engine = Engine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position fen 4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1", "test")
        .expect("position");
    uci.process_line("go nodes 1", "test").expect("go");
    uci.process_line("wait", "test").expect("wait");
    drop(uci);

    assert!(responder.responses.iter().any(|line| line.contains("score mate -1")));
    assert!(responder.responses.iter().any(|line| line == "bestmove a0a0"));
}

#[test]
fn missing_weights_never_falls_back_to_uniform_search() {
    ensure_init();
    let mut engine = Engine::new();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position startpos", "test").expect("position");
    uci.process_line("go nodes 8", "test").expect("go");
    drop(uci);

    assert!(
        responder
            .responses
            .iter()
            .any(|line| line.starts_with("info string cannot search:"))
    );
    assert!(!responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}

#[test]
fn unsupported_go_limits_are_rejected_without_stopping_the_current_search() {
    ensure_init();
    let mut engine = Engine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);
    uci.process_line("position startpos", "test").expect("position");
    uci.process_line("go infinite", "test").expect("infinite go");

    for command in [
        "go depth 1",
        "go mate 1",
        "go btime 1000",
        "go movetime 1000 wtime 1000 btime 1000",
        "go nodes 0",
    ] {
        assert!(uci.process_line(command, "test").is_err(), "{command}");
    }

    uci.process_line("stop", "test").expect("stop");
    uci.process_line("wait", "test").expect("wait");
    drop(uci);
    assert_eq!(
        responder
            .responses
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count(),
        1
    );
}
