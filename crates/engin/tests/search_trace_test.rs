//! P3 固定 nodes 搜索 trace 验收。

use std::sync::Once;

use engin::search::classic::{ClassicSearch, UniformBackend};
use engin::SearchBase;
use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

#[test]
fn fixed_nodes_increases_root_visits() {
    ensure_init();
    let mut search = ClassicSearch::new(Box::new(UniformBackend::default()));
    let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
    search.set_position(&state).expect("set position");
    let (best, visits) = search.run_blocking_nodes(16);
    assert_eq!(visits, 16);
    assert!(!best.is_null());
}

#[test]
fn fixed_nodes_search_returns_legal_move() {
    ensure_init();
    let mut search = ClassicSearch::new(Box::new(UniformBackend {
        wl: 0.2,
        d: 0.0,
        m: 0.0,
    }));
    let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
    search.set_position(&state).expect("set position");
    let (best, visits) = search.run_blocking_nodes(32);
    assert_eq!(visits, 32);
    assert!(!best.is_null());
}
