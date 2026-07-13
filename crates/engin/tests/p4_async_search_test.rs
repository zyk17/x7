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
fn go_wtime_budget_emits_bestmove() {
    ensure_init();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);
    uci.process_line("position startpos", "0.0.0").expect("position");
    uci.process_line("go wtime 1000 winc 0", "0.0.0").expect("go wtime");
    drop(uci);
    assert!(engine.search().total_root_visits() >= 1);
    assert!(responder.responses.iter().any(|line| line.starts_with("bestmove ")));
}
