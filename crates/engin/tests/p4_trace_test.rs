//! P4 确定性 trace：UniformBackend 下 fixed-nodes stopper 可复现。

use std::sync::Once;

use engin::neural::backend::UniformBackend;
use engin::search::classic::ClassicSearch;
use engin::SearchBase;
use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

#[test]
fn startpos_trace_is_deterministic() {
    ensure_init();
    let run = |nodes: u32| -> (xiangqi_core::Move, u32) {
        let mut search = ClassicSearch::new(Box::new(UniformBackend::default()));
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        search.set_position(&state).expect("position");
        search.run_blocking_nodes(nodes)
    };
    let a = run(16);
    let b = run(16);
    assert_eq!(a.0, b.0);
    // px0 `VisitsStopper` stops after `UpdateCounters`, so a completed batch
    // may carry root visits past the target (`stoppers.cc:59-70`).
    assert!(a.1 >= 16);
    assert_eq!(a.1, b.1);
}

#[test]
fn eval_positions_file_entries_are_valid_fen() {
    ensure_init();
    let text = include_str!("../../../data/eval_positions.txt");
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        GameState::from_fen_moves(line, &[] as &[&str]).expect("valid fen");
    }
}
