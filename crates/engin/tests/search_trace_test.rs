//! P3 固定 nodes 搜索 trace 验收。

use std::sync::Once;

use std::path::Path;

use engin::neural::backend::UniformBackend;
use engin::neural::onnx::OnnxBackend;
use engin::search::classic::ClassicSearch;
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
    let mut search = ClassicSearch::new(Box::new(UniformBackend::with_wdl(0.2, 0.0, 0.0)));
    let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
    search.set_position(&state).expect("set position");
    let (best, visits) = search.run_blocking_nodes(32);
    assert_eq!(visits, 32);
    assert!(!best.is_null());
}

/// P4 end-to-end smoke: px0 `SearchWorker` -> `NetworkAsBackendComputation`
/// (`src/search/classic/search.cc:1142-1231`, `src/neural/wrapper.cc:100-172`).
#[test]
fn local_x7_runs_mcts_with_cnn_if_present() {
    ensure_init();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/x7.onnx");
    if !path.is_file() {
        eprintln!("skip: {} is absent", path.display());
        return;
    }
    let backend = OnnxBackend::from_file(path).expect("load x7.onnx");
    let mut search = ClassicSearch::new(Box::new(backend));
    let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
    search.set_position(&state).expect("set position");
    let (best, visits) = search.run_blocking_nodes(4);
    assert_eq!(visits, 4);
    assert!(state.startpos.board().generate_legal_moves().contains(&best));
}
