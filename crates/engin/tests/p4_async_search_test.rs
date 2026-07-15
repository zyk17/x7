//! P4 异步搜索与 stopper 验收。

use std::sync::Once;

use engin::{ClassicEngine, UciLoop, UciOptions, VecUciResponder};
use xiangqi_core::initialize_magic_bitboards;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

#[test]
fn go_infinite_stop_emits_bestmove() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go infinite", "0.0.0").expect("go infinite");
    uci.process_line("stop", "0.0.0").expect("stop");
    drop(uci);
    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}

#[test]
fn untranslated_clock_manager_is_rejected() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    let error = uci
        .process_line("go wtime 1000 winc 0", "0.0.0")
        .expect_err("untranslated clock manager");
    drop(uci);
    assert!(error.to_string().contains("go wtime/btime time manager"));
    assert_eq!(engine.search().expect("uniform search").total_root_visits(), 0);
}

/// px0 `Engine::SetPosition` stops and joins an old search before replacing
/// its tree (`src/engine.cc:187-197`).  GUI callers can send this sequence
/// without a separate `stop`.
#[test]
fn position_replaces_an_infinite_search_without_racing_the_tree() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go infinite", "0.0.0").expect("go infinite");
    uci.process_line("position startpos moves h2e2", "0.0.0")
        .expect("replace position");
    uci.process_line("go nodes 8", "0.0.0").expect("go nodes");
    drop(uci);
    assert_eq!(engine.search().expect("uniform search").total_root_visits(), 8);
    assert_eq!(
        responder
            .responses
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count(),
        1,
        "aborted infinite search must not emit a stale bestmove"
    );
}
