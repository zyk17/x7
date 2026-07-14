//! px0 `src/search/classic/search.h:49-260`、`search.cc:426-808,874-1055`、`wrapper.cc:53-141`。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use xiangqi_core::{GameState, Move};

use crate::callbacks::{BestMoveInfo, SearchResponder, ThinkingInfo, Wdl};
use crate::neural::backend::Backend;
use crate::search::SearchBase;
use crate::uci_loop::{GoParams, UciOptions};
use crate::EnginError;

use super::node::{Node, NodeTree};
use super::params::{
    accurate_wdl_rescale_params, get_contempt, simplified_wdl_rescale_params, ContemptMode, ScoreType, SearchParams,
};
use super::stoppers::timemgr::{IterationStats, StoppersHints};
use super::stoppers::{build_search_stoppers, ChainedSearchStopper, SearchStopper};
use super::worker::{SearchWorker, WorkerSearchState};
use crate::utils::fastmath::{fast_log, fast_logistic};

pub fn best_move(tree: &NodeTree, params: &SearchParams, root_move_filter: &[Move]) -> (Move, Move) {
    let root = tree.current_head();
    let root_is_black = tree.history().is_black_to_move();
    let best_edge = best_child_edge(tree, root, params, 0, root_move_filter);
    let best = best_edge.map(|idx| tree.node(root).edge(idx).mv).unwrap_or_else(|| {
        // px0's root filter is authoritative even before a child has an
        // evaluated visit (`wrapper.cc:78-100`, `search.cc:721-724`). A
        // terminal/early-stop fallback must therefore not escape it.
        root_move_filter.first().copied().unwrap_or_else(|| {
            tree.history()
                .last()
                .board()
                .generate_legal_moves()
                .first()
                .copied()
                .unwrap_or(Move::NULL)
        })
    });
    let ponder = best_edge
        .and_then(|idx| {
            let child = tree.node(root).child(idx)?;
            best_child_edge(tree, child, params, 1, &[]).map(|ponder_idx| tree.node(child).edge(ponder_idx).mv)
        })
        .unwrap_or(Move::NULL);
    (orient_move(best, root_is_black), orient_move(ponder, !root_is_black))
}

/// px0 `EdgeAndNode::GetMove` returns a default null `Move` when no edge
/// exists (`src/search/classic/node.h:356-404`). Do not mirror that sentinel:
/// `Move::Flip` only has board-move semantics and would turn null into `a9a9`.
fn orient_move(mv: Move, flip: bool) -> Move {
    if flip && !mv.is_null() {
        mv.flip()
    } else {
        mv
    }
}

/// px0 `Search::SendUciInfo` builds each principal variation by repeatedly
/// selecting the best no-temperature child (`src/search/classic/search.cc:343-350`).
/// `Move::NULL` is never appended: a dangling edge ends the PV exactly as it
/// does in px0.
fn principal_variation(tree: &NodeTree, params: &SearchParams, first_edge: usize) -> Vec<Move> {
    let mut pv = Vec::new();
    let mut parent = tree.current_head();
    let mut edge_idx = first_edge;
    let mut depth = 0;
    let mut flip = tree.history().is_black_to_move();

    loop {
        let mv = orient_move(tree.node(parent).edge(edge_idx).mv, flip);
        if mv.is_null() {
            break;
        }
        pv.push(mv);
        let Some(child) = tree.node(parent).child(edge_idx) else {
            break;
        };
        depth += 1;
        flip = !flip;
        let Some(next_edge) = best_child_edge(tree, child, params, depth, &[]) else {
            break;
        };
        parent = child;
        edge_idx = next_edge;
    }
    pv
}

/// px0 `Search::SendUciInfo` WDL integer conversion
/// (`src/search/classic/search.cc:324-336`).
fn wdl_from_wl_d(wl: f32, d: f32) -> Wdl {
    let mut w = (500.0 * (1.0 + wl - d)).round() as i32;
    let mut l = (500.0 * (1.0 - wl - d)).round() as i32;
    w = w.max(0);
    l = l.max(0);
    let mut draw = 1000 - w - l;
    if draw < 0 {
        w = (w + draw / 2).clamp(0, 1000);
        l = 1000 - w;
        draw = 0;
    }
    Wdl { w, d: draw, l }
}

/// px0 `WDLRescale` (`src/search/classic/search.cc:202-236`).
pub(crate) fn wdl_rescale(
    wl: &mut f32,
    d: &mut f32,
    mut ratio: f32,
    mut diff: f32,
    sign: f32,
    invert: bool,
    max_s: f32,
) -> f32 {
    if invert {
        diff = -diff;
        ratio = 1.0 / ratio;
    }
    let w = (1.0 + *wl - *d) / 2.0;
    let l = (1.0 - *wl - *d) / 2.0;
    const EPS: f32 = 0.0001;
    if !(EPS < w && w < 1.0 - EPS && EPS < *d && *d < 1.0 - EPS && EPS < l && l < 1.0 - EPS) {
        return 0.0;
    }
    let a = fast_log(1.0 / l - 1.0);
    let b = fast_log(1.0 / w - 1.0);
    let mut s = 2.0 / (a + b);
    if !invert {
        s = s.min(max_s);
    }
    let mu = (a - b) / (a + b);
    let mut s_new = s * ratio;
    if invert {
        std::mem::swap(&mut s, &mut s_new);
        s = s.min(max_s);
    }
    let mu_new = mu + sign * s * s * diff;
    let w_new = fast_logistic((-1.0 + mu_new) / s_new);
    let l_new = fast_logistic((-1.0 - mu_new) / s_new);
    *wl = w_new - l_new;
    *d = (1.0 - w_new - l_new).max(0.0);
    mu_new
}

/// px0 `Search::SendUciInfo` score branches (`search.cc:275-322`).
fn score_from_wdl(score_type: ScoreType, wl: &mut f32, d: &mut f32, q: f32, has_wdl: bool) -> i32 {
    match score_type {
        ScoreType::CentipawnWithDrawscore => (90.0 * (1.563_754_2 * q).tan()) as i32,
        ScoreType::Centipawn => (90.0 * (1.563_754_2 * *wl).tan()) as i32,
        ScoreType::Centipawn2019 => (295.0 * *wl / (1.0 - 0.976_953_15 * wl.powi(14))) as i32,
        ScoreType::Centipawn2018 => (290.680_63 * (1.548_090_8 * *wl).tan()) as i32,
        ScoreType::WinPercentage => (*wl * 5000.0 + 5000.0) as i32,
        ScoreType::Q => (q * 10_000.0) as i32,
        ScoreType::WinLoss => (*wl * 10_000.0) as i32,
        ScoreType::WdlMu => {
            let mu = wdl_rescale(wl, d, 1.0, 0.0, 1.0, true, 1.4);
            let centipawn_score = 45.0 * (1.567_280_8 * *wl).tan();
            if has_wdl
                && mu != 0.0
                && wl.abs() + *d < 0.996
                && (mu.abs() < 1.0 || centipawn_score.abs() < (100.0 * mu).abs())
            {
                (100.0 * mu) as i32
            } else {
                centipawn_score as i32
            }
        }
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum BestEdgeRank {
    TerminalLoss,
    TablebaseLoss,
    NonTerminal,
    TablebaseWin,
    TerminalWin,
}

fn draw_score(tree: &NodeTree, params: &SearchParams, is_odd_depth: bool) -> f32 {
    if is_odd_depth == tree.history().is_black_to_move() {
        params.draw_score
    } else {
        -params.draw_score
    }
}

/// px0 `Search::StartSearch` resolves `play` into a concrete side before
/// workers begin (`src/search/classic/search.cc:156-175`).
fn resolve_contempt_mode(configured: ContemptMode, infinite: bool, root_is_black: bool, ponder: bool) -> ContemptMode {
    match configured {
        ContemptMode::Play if infinite => ContemptMode::None,
        ContemptMode::Play if root_is_black != ponder => ContemptMode::Black,
        ContemptMode::Play => ContemptMode::White,
        mode => mode,
    }
}

fn best_edge_rank(node: Option<&Node>) -> BestEdgeRank {
    let Some(node) = node else {
        return BestEdgeRank::NonTerminal;
    };
    if node.n() == 0 || !node.is_terminal() || node.wl() == 0.0 {
        return BestEdgeRank::NonTerminal;
    }
    if node.is_tablebase_terminal() {
        if node.wl() < 0.0 {
            BestEdgeRank::TablebaseLoss
        } else {
            BestEdgeRank::TablebaseWin
        }
    } else if node.wl() < 0.0 {
        BestEdgeRank::TerminalLoss
    } else {
        BestEdgeRank::TerminalWin
    }
}

/// px0 `Search::GetBestChildrenNoTemperature` / `GetBestChildNoTemperature`
/// (`src/search/classic/search.cc:705-808`).
pub(super) fn best_child_edge(
    tree: &NodeTree,
    parent: usize,
    params: &SearchParams,
    depth: usize,
    root_move_filter: &[Move],
) -> Option<usize> {
    best_child_edges(tree, parent, params, 1, depth, root_move_filter)
        .into_iter()
        .next()
}

/// px0 `Search::GetBestChildrenNoTemperature` (`src/search/classic/search.cc:705-808`).
/// Bestmove and MultiPV use one ranking implementation so their ordering cannot drift.
pub(super) fn best_child_edges(
    tree: &NodeTree,
    parent: usize,
    params: &SearchParams,
    count: usize,
    depth: usize,
    root_move_filter: &[Move],
) -> Vec<usize> {
    if tree.node(parent).n() == 0 {
        return Vec::new();
    }
    let draw_score = draw_score(tree, params, depth % 2 == 1);
    let is_root = parent == tree.current_head();
    let mut edges: Vec<_> = (0..tree.node(parent).num_edges())
        .filter(|&edge_idx| {
            // px0 `Search::GetBestChildrenNoTemperature`
            // (`src/search/classic/search.cc:721-724`): `searchmoves` filters
            // only the current root, including MultiPV and final bestmove.
            !is_root || root_move_filter.is_empty() || root_move_filter.contains(&tree.node(parent).edge(edge_idx).mv)
        })
        .collect();
    edges.sort_unstable_by(|&left, &right| {
        if best_child_edge_is_better(tree, parent, left, right, draw_score) {
            std::cmp::Ordering::Less
        } else if best_child_edge_is_better(tree, parent, right, left, draw_score) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    edges.truncate(count);
    edges
}

fn best_child_edge_is_better(
    tree: &NodeTree,
    parent: usize,
    edge_idx: usize,
    best_idx: usize,
    draw_score: f32,
) -> bool {
    let candidate = tree.node(parent).child(edge_idx).map(|idx| tree.node(idx));
    let current = tree.node(parent).child(best_idx).map(|idx| tree.node(idx));
    let candidate_rank = best_edge_rank(candidate);
    let current_rank = best_edge_rank(current);
    if candidate_rank != current_rank {
        return candidate_rank > current_rank;
    }
    if candidate_rank == BestEdgeRank::NonTerminal
        && candidate.is_some_and(|node| node.n() != 0 && node.is_terminal())
        && current.is_some_and(|node| node.n() != 0 && node.is_terminal())
    {
        let candidate = candidate.expect("checked terminal candidate");
        let current = current.expect("checked terminal current");
        if candidate.is_tablebase_terminal() != current.is_tablebase_terminal() {
            return !candidate.is_tablebase_terminal();
        }
        return candidate.m() < current.m();
    }
    if candidate_rank == BestEdgeRank::NonTerminal {
        let candidate_n = candidate.map_or(0, Node::n);
        let current_n = current.map_or(0, Node::n);
        if candidate_n != current_n {
            return candidate_n > current_n;
        }
        let candidate_q = candidate.map_or(0.0, |node| node.q(draw_score));
        let current_q = current.map_or(0.0, |node| node.q(draw_score));
        if candidate_q != current_q {
            return candidate_q > current_q;
        }
        return tree.node(parent).edge(edge_idx).get_p() > tree.node(parent).edge(best_idx).get_p();
    }
    if candidate_rank > BestEdgeRank::NonTerminal {
        candidate.expect("winning candidate").m() < current.expect("winning current").m()
    } else {
        candidate.expect("losing candidate").m() > current.expect("losing current").m()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

    use super::{
        best_child_edge, best_move, orient_move, resolve_contempt_mode, score_from_wdl, wdl_from_wl_d, wdl_rescale,
        ContemptMode, ScoreType, SearchParams,
    };
    use crate::neural::backend::{
        Backend, BackendAttributes, BackendComputation, EvalPosition, EvalResult, UniformBackend,
    };
    use crate::search::classic::node::NodeTree;
    use crate::search::SearchBase;
    use crate::EnginError;

    static INIT: Once = Once::new();

    fn ensure_init() {
        INIT.call_once(initialize_magic_bitboards);
    }

    /// Deterministic backend used to exercise px0's multi-SearchWorker
    /// lifecycle without making the test depend on an installed ONNX runtime.
    #[derive(Clone, Debug, Default)]
    struct ParallelUniformBackend(UniformBackend);

    impl Backend for ParallelUniformBackend {
        fn evaluate(&self, history: &xiangqi_core::PositionHistory, legal_moves: &[xiangqi_core::Move]) -> EvalResult {
            self.0.evaluate(history, legal_moves)
        }

        fn attributes(&self) -> BackendAttributes {
            BackendAttributes {
                runs_on_cpu: false,
                suggested_num_search_threads: 2,
                recommended_batch_size: 4,
                maximum_batch_size: 4,
                ..BackendAttributes::default()
            }
        }

        fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
            self.0.create_computation()
        }

        fn cached_evaluation(&self, position: &EvalPosition) -> Option<EvalResult> {
            self.0.cached_evaluation(position)
        }
    }

    impl ParallelUniformBackend {
        /// Seeds px0's NN-cache equivalent so concurrent workers exercise the
        /// `AddInput -> FetchedImmediately -> OOO backup` path.
        fn seed_cache(&self, position: &EvalPosition) {
            let history = xiangqi_core::PositionHistory::from_positions(position.positions.clone());
            self.0
                .store_cache(position, self.0.evaluate(&history, &position.legal_moves));
        }
    }

    /// px0 `StartThreads` keeps several search workers alive while each only
    /// locks tree phases (`search.cc:1088-1140,1142-1211`).
    #[test]
    fn shared_tree_allows_two_search_workers() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut search = super::ClassicSearch::new(Box::new(ParallelUniformBackend::default()));
        search.set_position(&state).expect("position");
        // This test isolates the shared SearchWorker tree boundary. Task
        // workers have a separate lifecycle test in worker.rs.
        search
            .meta
            .lock()
            .expect("meta lock")
            .params
            .task_workers_per_search_worker = 0;

        let (best, visits) = search.run_blocking_nodes(32);

        assert!(!best.is_null());
        assert!(visits >= 32);
        assert_eq!(
            search
                .worker_state
                .thread_count
                .load(std::sync::atomic::Ordering::Acquire),
            3
        );
    }

    /// px0 allows the task-worker pipeline to run inside each SearchWorker
    /// while other workers overlap NN computation (`search.cc:1088-1211,
    /// 1322-1347,1494-1508,1828-1864`). Keep this bounded stress regression
    /// on the shared-tree phase boundary.
    #[test]
    fn shared_tree_combines_search_and_task_workers() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut search = super::ClassicSearch::new(Box::new(ParallelUniformBackend::default()));
        search.set_position(&state).expect("position");
        {
            let mut meta = search.meta.lock().expect("meta lock");
            meta.params.minibatch_size = 32;
            meta.params.task_workers_per_search_worker = 1;
            meta.params.max_collision_visits = 4;
            meta.params.max_collision_visits_scaling_start = 0;
            meta.params.max_collision_visits_scaling_end = 1;
            meta.params.minimum_work_size_for_processing = 2;
            meta.params.minimum_work_per_task_for_processing = 1;
            meta.params.solid_tree_threshold = 1;
        }

        let (best, visits) = search.run_blocking_nodes(64);

        assert!(!best.is_null());
        assert!(visits >= 64);
        let tree = search.tree.read().expect("tree lock");
        let root = tree.current_head();
        assert!(tree.node(root).has_solid_children());
        assert_eq!(tree.node(root).n_in_flight(), 0);
    }

    /// px0 publishes cache-hit out-of-order results during gather, while other
    /// SearchWorkers can continue their independent tree phases
    /// (`search.cc:1268-1419,1977-1987,2109-2173`). The shared collision list
    /// must be drained by backup and leave no root reservation behind.
    #[test]
    fn shared_workers_reconcile_out_of_order_cache_hits() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let backend = ParallelUniformBackend::default();
        let mut search = super::ClassicSearch::new(Box::new(backend.clone()));
        search.set_position(&state).expect("position");
        {
            let tree = search.tree.read().expect("tree lock");
            let history = tree.history();
            backend.seed_cache(&EvalPosition {
                positions: history.positions().to_vec(),
                legal_moves: history.last().board().generate_legal_moves(),
            });
        }
        {
            let mut meta = search.meta.lock().expect("meta lock");
            meta.params.minibatch_size = 4;
            meta.params.task_workers_per_search_worker = 0;
            meta.params.max_collision_visits = 4;
            meta.params.max_collision_visits_scaling_start = 0;
            meta.params.max_collision_visits_scaling_end = 1;
        }

        let (best, visits) = search.run_blocking_nodes(64);

        assert!(!best.is_null());
        assert!(visits >= 64);
        assert!(
            search
                .worker_state
                .total_batches
                .load(std::sync::atomic::Ordering::Acquire)
                > 0
        );
        assert!(
            search
                .worker_state
                .total_playouts
                .load(std::sync::atomic::Ordering::Acquire)
                >= u64::from(visits)
        );
        let tree = search.tree.read().expect("tree lock");
        assert_eq!(tree.node(tree.current_head()).n_in_flight(), 0);
        assert!(search
            .worker_state
            .shared_collisions
            .lock()
            .expect("collisions lock")
            .is_empty());
    }

    /// px0 `DoBackupUpdateSingleNode` may solidify root while other search
    /// workers continue selection (`search.cc:2211-2217`, `node.cc:245-288`).
    /// The Rust arena must keep every solid child slot selectable without
    /// producing an in-flight leak or a duplicate expansion.
    #[test]
    fn shared_workers_continue_through_root_solidification() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut search = super::ClassicSearch::new(Box::new(ParallelUniformBackend::default()));
        search.set_position(&state).expect("position");
        {
            let mut meta = search.meta.lock().expect("meta lock");
            meta.params.task_workers_per_search_worker = 0;
            meta.params.solid_tree_threshold = 1;
        }

        let (best, visits) = search.run_blocking_nodes(64);

        assert!(!best.is_null());
        assert!(visits >= 64);
        let tree = search.tree.read().expect("tree lock");
        let root = tree.current_head();
        assert!(tree.node(root).has_solid_children());
        assert_eq!(tree.node(root).n_in_flight(), 0);
    }

    #[test]
    fn best_child_uses_prior_until_a_child_has_more_visits() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut tree = NodeTree::default();
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let moves = state.startpos.board().generate_legal_moves();
        tree.node_mut(root).create_edges(&moves[..2].to_vec());
        tree.node_mut(root).edge_mut(0).set_p(0.2);
        tree.node_mut(root).edge_mut(1).set_p(0.8);
        assert!(tree.node_mut(root).try_start_score_update());
        tree.node_mut(root).finalize_score_update(0.0, 0.0, 0.0, 1);
        assert_eq!(best_child_edge(&tree, root, &SearchParams::default(), 0, &[]), Some(1));

        let child = tree.arena_mut().spawn_child(root, 0);
        assert!(tree.node_mut(child).try_start_score_update());
        tree.node_mut(child).finalize_score_update(0.0, 0.0, 0.0, 1);
        assert_eq!(best_child_edge(&tree, root, &SearchParams::default(), 0, &[]), Some(0));
    }

    #[test]
    fn root_filter_applies_before_the_root_has_visits() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut tree = NodeTree::default();
        tree.reset_to_position(&state.startpos, &state.moves);
        let allowed = state.startpos.board().parse_move("a0a1").expect("legal root move");

        assert_eq!(best_move(&tree, &SearchParams::default(), &[allowed]).0, allowed);
    }

    #[test]
    fn move_orientation_keeps_px0_null_ponder_null() {
        assert!(orient_move(xiangqi_core::Move::NULL, true).is_null());
    }

    #[test]
    fn wdl_integerization_matches_px0_send_uci_info() {
        assert_eq!(
            wdl_from_wl_d(0.1, 0.2),
            crate::callbacks::Wdl { w: 450, d: 200, l: 350 }
        );
    }

    /// px0 `WDLRescale` keeps a valid WDL distribution within its numerical
    /// guards (`src/search/classic/search.cc:202-236`).
    #[test]
    fn wdl_rescale_keeps_valid_distribution() {
        let mut wl = 0.2;
        let mut d = 0.4;
        let mu = wdl_rescale(&mut wl, &mut d, 1.0, 0.0, 1.0, true, 10.0);
        assert!(mu.is_finite());
        assert!(wl.is_finite() && d.is_finite());
        assert!((-1.0..=1.0).contains(&wl));
        assert!((0.0..=1.0).contains(&d));
    }

    #[test]
    fn play_contempt_resolves_like_px0_start_search() {
        assert_eq!(
            resolve_contempt_mode(ContemptMode::Play, true, false, false),
            ContemptMode::None
        );
        assert_eq!(
            resolve_contempt_mode(ContemptMode::Play, false, false, false),
            ContemptMode::White
        );
        assert_eq!(
            resolve_contempt_mode(ContemptMode::Play, false, true, false),
            ContemptMode::Black
        );
        assert_eq!(
            resolve_contempt_mode(ContemptMode::Play, false, false, true),
            ContemptMode::Black
        );
    }

    #[test]
    fn score_type_q_matches_px0_scaled_q() {
        let mut wl = 0.1;
        let mut d = 0.2;
        assert_eq!(score_from_wdl(ScoreType::Q, &mut wl, &mut d, 0.1234, true), 1234);
    }
}

#[derive(Clone, Debug)]
pub struct SearchOutput {
    pub bestmove: BestMoveInfo,
    pub infos: Vec<ThinkingInfo>,
}

struct SearchMeta {
    params: SearchParams,
    move_start: Instant,
    initial_visits: u32,
    stoppers_hints: StoppersHints,
    search_active: bool,
    /// px0 `Search::root_move_filter_` (`src/search/classic/search.h:168-171`).
    /// It is reconstructed from legal UCI `searchmoves` for every search.
    root_move_filter: Vec<Move>,
    /// px0 resolves `ContemptMode::PLAY` at search start
    /// (`src/search/classic/search.cc:156-175`).
    contempt_mode: ContemptMode,
}

/// px0 `last_outputted_info_edge_` and `last_outputted_uci_info_`
/// (`src/search/classic/search.h:175-176`).
#[derive(Default)]
struct LiveInfoState {
    edge: Option<usize>,
    depth: i32,
    seldepth: i32,
    time: i64,
}

/// px0 `Search::MaybeTriggerStop` 的 worker 可调用边界
/// (`src/search/classic/search.cc:596-620,2331-2334`).
///
/// px0 的 watchdog 与每个 `SearchWorker::UpdateCounters` 都触发同一个
/// stopper。Rust 端将该共享所有权显式化，避免只靠外层轮询而让一个缓存
/// batch 在 `go nodes` 后继续扩大。
pub(crate) struct SearchStopController {
    meta: Arc<Mutex<SearchMeta>>,
    stopper: Arc<Mutex<Option<ChainedSearchStopper>>>,
}

impl SearchStopController {
    fn populate_iteration_stats(meta: &SearchMeta, worker_state: &WorkerSearchState) -> IterationStats {
        let root_visits = worker_state.total_playouts.load(Ordering::Acquire) as i64;
        IterationStats {
            time_since_movestart: meta.move_start.elapsed().as_millis() as i64,
            time_since_first_batch: worker_state
                .first_batch
                .lock()
                .expect("first batch lock")
                .map_or(0, |first_batch| first_batch.elapsed().as_millis() as i64),
            total_nodes: root_visits,
            nodes_since_movestart: root_visits,
            batches_since_movestart: worker_state.total_batches.load(Ordering::Acquire) as i64,
            average_depth: {
                let playouts = worker_state.total_playouts.load(Ordering::Acquire);
                worker_state
                    .cum_depth
                    .load(Ordering::Acquire)
                    .checked_div(playouts)
                    .unwrap_or(0) as i32
            },
        }
    }

    pub(crate) fn maybe_trigger_stop(&self, worker_state: &WorkerSearchState) -> bool {
        let stats = {
            let meta = self.meta.lock().expect("meta lock");
            Self::populate_iteration_stats(&meta, worker_state)
        };
        let mut stopper_guard = self.stopper.lock().expect("stopper lock");
        let Some(stopper) = stopper_guard.as_mut() else {
            return false;
        };
        let mut meta = self.meta.lock().expect("meta lock");
        let mut hints = meta.stoppers_hints.clone();
        // px0 resets shared hints before every stopper pass; individual
        // stoppers then reduce the estimates (`search.cc:596-610`,
        // `stoppers/timemgr.cc:60-66`).
        hints.reset();
        let should_stop = stopper.should_stop(&stats, &mut hints);
        worker_state.set_remaining_playouts(hints.estimated_remaining_playouts());
        meta.stoppers_hints = hints;
        if should_stop {
            worker_state.stop.store(true, Ordering::Release);
        }
        should_stop
    }
}

pub struct ClassicSearch {
    tree: Arc<RwLock<NodeTree>>,
    worker_state: Arc<WorkerSearchState>,
    meta: Arc<Mutex<SearchMeta>>,
    backend: Arc<dyn Backend>,
    stop: Arc<AtomicBool>,
    stopper: Arc<Mutex<Option<ChainedSearchStopper>>>,
    stop_controller: Arc<SearchStopController>,
    /// px0 `SearchBase::uci_responder_` (`src/search/search.h:45-99`). The
    /// engine installs its forwarder before UCI starts a search; worker-side
    /// watchdog output is added in the following P4 translation point.
    responder: Option<Arc<dyn SearchResponder>>,
    /// px0 `ok_to_respond_bestmove_` / `bestmove_is_sent_`
    /// (`src/search/classic/search.h:146-151`).
    ok_to_respond_bestmove: Arc<AtomicBool>,
    bestmove_is_sent: Arc<AtomicBool>,
    live_info: Arc<Mutex<LiveInfoState>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    infinite: AtomicBool,
    pub outputs: Vec<SearchOutput>,
}

impl ClassicSearch {
    pub fn new(backend: Box<dyn Backend>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let meta = Arc::new(Mutex::new(SearchMeta {
            params: SearchParams::default(),
            move_start: Instant::now(),
            initial_visits: 0,
            stoppers_hints: StoppersHints::default(),
            search_active: false,
            root_move_filter: Vec::new(),
            contempt_mode: ContemptMode::None,
        }));
        let stopper = Arc::new(Mutex::new(None));
        Self {
            tree: Arc::new(RwLock::new(NodeTree::default())),
            worker_state: Arc::new(WorkerSearchState::new(Arc::clone(&stop), i64::MAX)),
            meta: Arc::clone(&meta),
            backend: Arc::from(backend),
            stop,
            stopper: Arc::clone(&stopper),
            stop_controller: Arc::new(SearchStopController { meta, stopper }),
            responder: None,
            ok_to_respond_bestmove: Arc::new(AtomicBool::new(true)),
            bestmove_is_sent: Arc::new(AtomicBool::new(false)),
            live_info: Arc::new(Mutex::new(LiveInfoState::default())),
            threads: Mutex::new(Vec::new()),
            infinite: AtomicBool::new(false),
            outputs: Vec::new(),
        }
    }

    pub fn total_root_visits(&self) -> u32 {
        let tree = self.tree.read().expect("tree lock");
        tree.node(tree.current_head()).n()
    }

    /// px0 `SearchBase` receives its responder during search construction
    /// (`src/search/search.h:45-55`). It is immutable while workers run.
    pub fn set_uci_responder(&mut self, responder: Arc<dyn SearchResponder>) {
        self.responder = Some(responder);
    }

    /// px0 `SearchBase::SetBackend` (`src/search/search.h:48-55`). Callers
    /// must stop search first; a worker holds a cloned backend for its full
    /// lifetime, exactly as px0 changes the backend only while stopped.
    pub fn set_backend(&mut self, backend: Box<dyn Backend>) -> Result<(), EnginError> {
        self.abort_search()?;
        self.backend = Arc::from(backend);
        Ok(())
    }

    /// px0 reads `MultiPV` and `PerPVCounters` from its immutable
    /// `BaseSearchParams` (`src/search/classic/params.h:101-103`). Rust keeps
    /// the already parsed UCI subset in `SearchParams` before StartSearch.
    pub fn set_uci_info_options(
        &mut self,
        multi_pv: usize,
        per_pv_counters: bool,
        score_type: ScoreType,
        nps_limit: f32,
    ) -> Result<(), EnginError> {
        self.abort_search()?;
        let mut meta = self.meta.lock().expect("meta lock");
        meta.params.multi_pv = multi_pv;
        meta.params.per_pv_counters = per_pv_counters;
        meta.params.score_type = score_type;
        meta.params.nps_limit = nps_limit;
        Ok(())
    }

    /// px0 `BaseSearchParams` freezes WDL calibration from OptionsDict before
    /// worker construction (`src/search/classic/params.cc:688-703`).
    pub fn set_wdl_options(&mut self, options: &UciOptions) -> Result<(), EnginError> {
        self.abort_search()?;
        let contempt =
            get_contempt(&options.uci_opponent, &options.contempt, options.uci_rating_adv).map_err(EnginError::Uci)?;
        let rescale = if options.wdl_calibration_elo == 0.0 {
            accurate_wdl_rescale_params(
                contempt,
                options.wdl_draw_rate_target,
                options.wdl_draw_rate_reference,
                options.wdl_book_exit_bias,
                options.contempt_max_value,
                options.wdl_contempt_attenuation,
            )
        } else {
            simplified_wdl_rescale_params(
                contempt,
                options.wdl_draw_rate_reference,
                options.wdl_calibration_elo,
                options.contempt_max_value,
                options.wdl_contempt_attenuation,
            )
        };
        let mut meta = self.meta.lock().expect("meta lock");
        meta.params.wdl_rescale_ratio = rescale.ratio;
        meta.params.wdl_rescale_diff = rescale.diff;
        meta.params.wdl_max_s = options.wdl_max_s;
        Ok(())
    }

    pub fn run_blocking_nodes(&mut self, nodes: u32) -> (Move, u32) {
        let params = GoParams {
            nodes: Some(nodes as i32),
            ..Default::default()
        };
        self.start_search(&params).expect("search");
        self.wait_search().expect("wait");
        let tree = self.tree.read().expect("tree lock");
        let meta = self.meta.lock().expect("meta lock");
        let (best, _) = best_move(&tree, &meta.params, &meta.root_move_filter);
        let visits = tree.node(tree.current_head()).n();
        (best, visits)
    }

    fn snapshot_output(
        tree: &NodeTree,
        meta: &SearchMeta,
        worker_state: &WorkerSearchState,
        has_wdl: bool,
    ) -> Option<SearchOutput> {
        let (best, ponder) = best_move(tree, &meta.params, &meta.root_move_filter);
        if best.is_null() {
            return None;
        }
        let elapsed = meta.move_start.elapsed();
        let total_playouts = worker_state.total_playouts.load(Ordering::Acquire);
        let first_batch = *worker_state.first_batch.lock().expect("first batch lock");
        let nps = first_batch.and_then(|started| {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            (elapsed_ms > 0).then(|| (total_playouts.saturating_mul(1000) / elapsed_ms) as i32)
        });
        let root = tree.current_head();
        let infos = best_child_edges(
            tree,
            root,
            &meta.params,
            meta.params.multi_pv,
            0,
            &meta.root_move_filter,
        )
        .into_iter()
        .enumerate()
        .map(|(index, edge_idx)| {
            let child = tree.node(root).child(edge_idx).map(|idx| tree.node(idx));
            let default_wl = -tree.node(root).wl();
            let default_d = tree.node(root).d();
            let mut wl = child.filter(|node| node.n() > 0).map_or(default_wl, Node::wl);
            let mut d = child.filter(|node| node.n() > 0).map_or(default_d, Node::d);
            let default_q = -tree.node(root).q(-draw_score(tree, &meta.params, false));
            let q = child
                .filter(|node| node.n() > 0)
                .map_or(default_q, |node| node.q(draw_score(tree, &meta.params, false)));
            let score = score_from_wdl(meta.params.score_type, &mut wl, &mut d, q, has_wdl);
            let mate = child.filter(|node| node.is_terminal() && wl != 0.0).map(|node| {
                let plies = node.m().round() as i32 / 2 + if node.is_tablebase_terminal() { 101 } else { 1 };
                if wl < 0.0 {
                    -plies
                } else {
                    plies
                }
            });
            ThinkingInfo {
                // px0 `Search::SendUciInfo` (`search.cc:249-336`).
                depth: (worker_state.cum_depth.load(Ordering::Acquire) / total_playouts.max(1)) as i32,
                seldepth: worker_state.max_depth.load(Ordering::Acquire) as i32,
                time: elapsed.as_millis() as i64,
                nodes: if meta.params.per_pv_counters {
                    child.map_or(0, Node::n) as i64
                } else {
                    total_playouts as i64 + meta.initial_visits as i64
                },
                nps: nps.unwrap_or(-1),
                eps: nps.map_or(-1, |_| {
                    let evaluations = worker_state.network_evaluations.load(Ordering::Acquire);
                    let elapsed_ms = first_batch.expect("nps needs first batch").elapsed().as_millis() as u64;
                    (evaluations.saturating_mul(1000) / elapsed_ms) as i32
                }),
                mate,
                score: mate.is_none().then_some(score),
                wdl: Some(wdl_from_wl_d(wl, d)),
                tb_hits: 0,
                pv: principal_variation(tree, &meta.params, edge_idx),
                multipv: if meta.params.multi_pv > 1 { index as i32 + 1 } else { -1 },
                ..ThinkingInfo::default()
            }
        })
        .collect();
        let mut bestmove = BestMoveInfo::new(best);
        bestmove.ponder = ponder;
        Some(SearchOutput { bestmove, infos })
    }

    fn start_threads(&mut self, how_many: usize) -> Result<(), EnginError> {
        let mut handles = self.threads.lock().expect("threads lock");
        if !handles.is_empty() {
            return Ok(());
        }
        let thread_count = if how_many == 0 {
            let attributes = self.backend.attributes();
            attributes.suggested_num_search_threads + usize::from(!attributes.runs_on_cpu)
        } else {
            how_many
        };
        self.worker_state.thread_count.store(thread_count, Ordering::Release);
        self.meta.lock().expect("meta lock").search_active = true;

        let tree = Arc::clone(&self.tree);
        let worker_state = Arc::clone(&self.worker_state);
        let meta = Arc::clone(&self.meta);
        let backend = Arc::clone(&self.backend);
        let stop = Arc::clone(&self.stop);
        let stop_controller = Arc::clone(&self.stop_controller);
        let responder = self.responder.clone();
        let ok_to_respond_bestmove = Arc::clone(&self.ok_to_respond_bestmove);
        let bestmove_is_sent = Arc::clone(&self.bestmove_is_sent);
        let watchdog_tree = Arc::clone(&tree);
        let watchdog_worker_state = Arc::clone(&worker_state);
        let watchdog_meta = Arc::clone(&meta);
        let watchdog_stop = Arc::clone(&stop);
        let watchdog_stop_controller = Arc::clone(&stop_controller);
        let live_info = Arc::clone(&self.live_info);
        let has_wdl = self.backend.attributes().has_wdl;
        // px0 starts a watchdog before search workers (`search.cc:874-896`).
        // It owns the terminal bestmove response; workers only advance tree
        // state and may request a stop through SearchStopController.
        handles.push(thread::spawn(move || loop {
            watchdog_stop_controller.maybe_trigger_stop(&watchdog_worker_state);
            if !watchdog_stop.load(Ordering::Acquire) {
                if let Some(responder) = &responder {
                    // Keep px0's tree-before-current-best lock order. Backup
                    // updates current_best_edge while holding the tree phase.
                    let tree = watchdog_tree.read().expect("tree lock");
                    let meta = watchdog_meta.lock().expect("meta lock");
                    let edge = *watchdog_worker_state.current_best_edge.lock().expect("best edge lock");
                    if edge.is_some() {
                        if let Some(output) =
                            ClassicSearch::snapshot_output(&tree, &meta, &watchdog_worker_state, has_wdl)
                        {
                            let info = output.infos.first().expect("root best edge has info");
                            let mut last = live_info.lock().expect("live info lock");
                            if edge != last.edge
                                || info.depth != last.depth
                                || info.seldepth != last.seldepth
                                || last.time + 5_000 < info.time
                            {
                                responder.output_thinking_info(&output.infos);
                                last.edge = edge;
                                last.depth = info.depth;
                                last.seldepth = info.seldepth;
                                last.time = info.time;
                            }
                        }
                    }
                }
            }
            if watchdog_stop.load(Ordering::Acquire)
                && ok_to_respond_bestmove.load(Ordering::Acquire)
                && watchdog_worker_state.total_playouts.load(Ordering::Acquire) > 0
                && bestmove_is_sent
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                if let Some(responder) = &responder {
                    let tree = watchdog_tree.read().expect("tree lock");
                    let meta = watchdog_meta.lock().expect("meta lock");
                    if let Some(output) = ClassicSearch::snapshot_output(&tree, &meta, &watchdog_worker_state, has_wdl)
                    {
                        responder.output_thinking_info(&output.infos);
                        responder.output_best_move(&output.bestmove);
                    }
                }
                break;
            }
            if bestmove_is_sent.load(Ordering::Acquire) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }));
        for _ in 0..thread_count {
            let tree = Arc::clone(&tree);
            let worker_state = Arc::clone(&worker_state);
            let meta = Arc::clone(&meta);
            let backend = Arc::clone(&backend);
            let stop = Arc::clone(&stop);
            let stop_controller = Arc::clone(&stop_controller);
            handles.push(thread::spawn(move || {
                let (mut params, root_move_filter, contempt_mode) = {
                    let meta = meta.lock().expect("meta lock");
                    (meta.params.clone(), meta.root_move_filter.clone(), meta.contempt_mode)
                };
                params.contempt_mode = contempt_mode;
                let mut worker = SearchWorker::new_shared_with_stop_controller_and_root_move_filter(
                    tree,
                    backend.as_ref(),
                    &params,
                    worker_state.as_ref(),
                    Some(Arc::clone(&stop_controller)),
                    &root_move_filter,
                );
                if worker.run_blocking().is_err() {
                    stop.store(true, Ordering::Release);
                }
            }));
        }
        Ok(())
    }
}

impl SearchBase for ClassicSearch {
    fn new_game(&mut self) -> Result<(), EnginError> {
        self.wait_search()?;
        *self.tree.write().expect("tree lock") = NodeTree::default();
        self.worker_state = Arc::new(WorkerSearchState::new(Arc::clone(&self.stop), i64::MAX));
        self.outputs.clear();
        Ok(())
    }

    fn set_position(&mut self, state: &GameState) -> Result<(), EnginError> {
        self.tree
            .write()
            .expect("tree lock")
            .reset_to_position(&state.startpos, &state.moves);
        Ok(())
    }

    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError> {
        self.outputs.clear();
        self.stop.store(false, Ordering::Release);
        self.infinite.store(params.infinite, Ordering::Release);
        self.ok_to_respond_bestmove.store(!params.infinite, Ordering::Release);
        self.bestmove_is_sent.store(false, Ordering::Release);
        *self.live_info.lock().expect("live info lock") = LiveInfoState::default();

        let nodes = params.nodes.filter(|&n| n > 0);
        if params.nodes.is_some() && nodes.is_none() {
            return Err(EnginError::Uci("go nodes must be positive".into()));
        }
        if let Some(movetime) = params.movetime {
            if movetime < 0 {
                return Err(EnginError::Uci("go movetime must be non-negative".into()));
            }
        }

        let has_budget = nodes.is_some()
            || params.movetime.is_some()
            || params.infinite
            || params.wtime.is_some()
            || params.btime.is_some();
        if !has_budget {
            return Err(EnginError::PortIncomplete("time-based search stopper"));
        }

        {
            let tree = self.tree.read().expect("tree lock");
            // px0 `StringsToMovelist` (`src/search/classic/wrapper.cc:78-100`)
            // parses at the root, retains legal requests only, and rejects a
            // non-empty `searchmoves` list if none of its moves are legal.
            let board = tree.history().last().board();
            let legal_moves = board.generate_legal_moves();
            let mut root_move_filter = Vec::with_capacity(params.searchmoves.len());
            for move_text in &params.searchmoves {
                if let Ok(mv) = board.parse_move(move_text) {
                    if legal_moves.contains(&mv) {
                        root_move_filter.push(mv);
                    }
                }
            }
            if !params.searchmoves.is_empty() && root_move_filter.is_empty() {
                return Err(EnginError::Uci("No legal searchmoves.".into()));
            }
            let mut meta = self.meta.lock().expect("meta lock");
            meta.move_start = Instant::now();
            meta.initial_visits = tree.node(tree.current_head()).n();
            meta.root_move_filter = root_move_filter;
            meta.contempt_mode = resolve_contempt_mode(
                meta.params.contempt_mode,
                params.infinite,
                tree.history().is_black_to_move(),
                params.ponder,
            );
            *self.worker_state.first_batch.lock().expect("first batch lock") = None;
            meta.stoppers_hints.reset();
            self.worker_state.total_playouts.store(0, Ordering::Release);
            self.worker_state.total_batches.store(0, Ordering::Release);
            self.worker_state.network_evaluations.store(0, Ordering::Release);
            self.worker_state.cum_depth.store(0, Ordering::Release);
            self.worker_state.max_depth.store(0, Ordering::Release);
            self.worker_state
                .set_max_concurrent_searchers(meta.params.max_concurrent_searchers);
            self.worker_state
                .shared_collisions
                .lock()
                .expect("collisions lock")
                .clear();
            // px0 creates a fresh `Search` object for every `StartSearch`
            // (`src/search/classic/wrapper.cc:126-150`), so its cached root
            // edge never crosses a UCI search boundary.
            *self.worker_state.current_best_edge.lock().expect("best edge lock") = None;
            if let Some(limit) = nodes {
                meta.stoppers_hints.update_estimated_remaining_playouts(limit as i64);
                self.worker_state.set_remaining_playouts(limit as i64);
            }
            let chain = build_search_stoppers(params, tree.history(), false);
            *self.stopper.lock().expect("stopper lock") = Some(chain);
            if nodes.is_none() {
                if let Some(ms) = params.movetime {
                    meta.stoppers_hints.update_estimated_remaining_time_ms(ms);
                }
            }
        }

        self.start_threads(0)?;
        Ok(())
    }

    fn start_clock(&mut self) -> Result<(), EnginError> {
        self.meta.lock().expect("meta lock").move_start = Instant::now();
        Ok(())
    }

    fn wait_search(&mut self) -> Result<(), EnginError> {
        // px0 `Search::Wait` only joins threads (`search.cc:1035-1041`). The
        // watchdog, not this caller, owns stopping and response timing.
        let handles: Vec<_> = self.threads.lock().expect("threads lock").drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
        self.meta.lock().expect("meta lock").search_active = false;
        Ok(())
    }

    fn stop_search(&mut self) -> Result<(), EnginError> {
        // px0 `Search::Stop` is non-blocking and enables the final response
        // for an otherwise silent infinite search (`search.cc:1019-1025`).
        self.ok_to_respond_bestmove.store(true, Ordering::Release);
        self.stop.store(true, Ordering::Release);
        Ok(())
    }

    fn abort_search(&mut self) -> Result<(), EnginError> {
        // px0 `Search::Abort` suppresses bestmove before stopping/joining
        // (`search.cc:1027-1033`).
        self.bestmove_is_sent.store(true, Ordering::Release);
        self.stop.store(true, Ordering::Release);
        self.wait_search()
    }
}
