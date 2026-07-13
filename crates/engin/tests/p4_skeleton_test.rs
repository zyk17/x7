//! P4 骨架边界与单线程 worker 回归。

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::Once;

use engin::neural::backend::{Backend, UniformBackend};
use engin::search::classic::{NodeToProcess, NodeTree, SearchParams, SearchWorker, WorkerSearchState};
use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

#[test]
fn node_to_process_matches_px0_visit_and_collision_shapes() {
    let visit = NodeToProcess::visit(3, 4);
    assert!(visit.is_extendable(false));
    assert!(!visit.is_extendable(true));
    assert!(!visit.is_collision);
    assert_eq!(visit.multivisit, 1);

    let collision = NodeToProcess::collision(3, 4, 2, 5);
    assert!(!collision.is_extendable(false));
    assert!(collision.is_collision);
    assert_eq!(collision.multivisit, 2);
    assert_eq!(collision.maxvisit, 5);
}

#[test]
fn uniform_backend_exposes_p4_computation() {
    let backend = UniformBackend::default();
    assert_eq!(backend.attributes().recommended_batch_size, 1);
    assert!(backend.create_computation().is_ok());
}

fn setup_startpos_tree() -> NodeTree {
    let mut tree = NodeTree::default();
    let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
    tree.reset_to_position(&state.startpos, &state.moves);
    tree
}

#[test]
fn worker_single_iteration_increases_root_visits() {
    ensure_init();
    let mut tree = setup_startpos_tree();
    let backend = UniformBackend::default();
    let params = SearchParams::default();
    let stop = Arc::new(AtomicBool::new(false));
    let search_state = WorkerSearchState::new(stop, i64::MAX);
    let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
    worker.execute_one_iteration().expect("one iteration");
    assert_eq!(tree.node(tree.current_head()).n(), 1);
}

#[test]
fn worker_matches_p3_root_visits_for_fixed_budget() {
    ensure_init();
    let mut tree = setup_startpos_tree();
    let backend = UniformBackend::default();
    let params = SearchParams::default();
    let stop = Arc::new(AtomicBool::new(false));
    let search_state = WorkerSearchState::new(stop, 16);
    let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
    worker.run_until_root_visits(16).expect("worker search");
    assert_eq!(tree.node(tree.current_head()).n(), 16);
}

#[test]
fn minibatch_visits_do_not_leak_root_in_flight() {
    ensure_init();
    let mut tree = setup_startpos_tree();
    let backend = UniformBackend::default();
    let params = SearchParams {
        minibatch_size: 4,
        ..SearchParams::default()
    };
    let stop = Arc::new(AtomicBool::new(false));
    let search_state = WorkerSearchState::new(stop, i64::MAX);
    let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);

    for _ in 0..8 {
        worker.execute_one_iteration().expect("batched iteration");
    }
    let root = tree.current_head();
    assert_eq!(tree.node(root).n_in_flight(), 0);
    assert_eq!(
        search_state.total_playouts.load(std::sync::atomic::Ordering::Acquire),
        tree.node(root).n() as u64
    );
}
