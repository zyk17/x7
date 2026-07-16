//! px0 `src/search/classic/search.h:49-260`、`search.cc:426-808,874-1055`、`wrapper.cc:53-141`。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use parking_lot::RwLock;
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
use super::stoppers::{build_search_stoppers, ChainedSearchStopper, LegacyTimeManager, SearchStopper};
use super::worker::{cancel_shared_collisions, SearchWorker, WorkerSearchState};
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

/// px0 `Search::SendUciInfo` display-side WDL dependencies
/// (`src/search/classic/search.cc:275-291`).
struct WdlDisplayContext<'a> {
    params: &'a SearchParams,
    contempt_mode: ContemptMode,
    root_is_black: bool,
}

/// px0 `Search::SendUciInfo` score branches (`search.cc:275-322`).
fn score_from_wdl(
    score_type: ScoreType,
    wl: &mut f32,
    d: &mut f32,
    q: f32,
    has_wdl: bool,
    display: WdlDisplayContext<'_>,
) -> i32 {
    let mut mu_uci = 0.0;
    if score_type == ScoreType::WdlMu
        || (display.params.wdl_rescale_diff != 0.0 && display.contempt_mode != ContemptMode::None)
    {
        let sign = if (display.contempt_mode == ContemptMode::Black) == display.root_is_black {
            1.0
        } else {
            -1.0
        };
        mu_uci = wdl_rescale(
            wl,
            d,
            display.params.wdl_rescale_ratio,
            if display.contempt_mode == ContemptMode::None {
                0.0
            } else {
                display.params.wdl_rescale_diff * display.params.wdl_eval_objectivity
            },
            sign,
            true,
            display.params.wdl_max_s,
        );
    }
    match score_type {
        ScoreType::CentipawnWithDrawscore => (90.0 * (1.563_754_2 * q).tan()) as i32,
        ScoreType::Centipawn => (90.0 * (1.563_754_2 * *wl).tan()) as i32,
        ScoreType::Centipawn2019 => (295.0 * *wl / (1.0 - 0.976_953_15 * wl.powi(14))) as i32,
        ScoreType::Centipawn2018 => (290.680_63 * (1.548_090_8 * *wl).tan()) as i32,
        ScoreType::WinPercentage => (*wl * 5000.0 + 5000.0) as i32,
        ScoreType::Q => (q * 10_000.0) as i32,
        ScoreType::WinLoss => (*wl * 10_000.0) as i32,
        ScoreType::WdlMu => {
            let centipawn_score = 45.0 * (1.567_280_8 * *wl).tan();
            if has_wdl
                && mu_uci != 0.0
                && wl.abs() + *d < 0.996
                && (mu_uci.abs() < 1.0 || centipawn_score.abs() < (100.0 * mu_uci).abs())
            {
                (100.0 * mu_uci) as i32
            } else {
                centipawn_score as i32
            }
        }
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum BestEdgeRank {
    NonTerminal,
    TerminalWin,
    TerminalLoss,
    TablebaseWin,
    TablebaseLoss,
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

#[derive(Clone, Debug)]
struct SearchOutput {
    pub bestmove: BestMoveInfo,
    pub infos: Vec<ThinkingInfo>,
}

struct SearchMeta {
    params: SearchParams,
    move_start: Instant,
    /// px0 `Search::nps_start_time_` (`src/search/classic/search.h:185-186`).
    /// The watchdog initializes it after the first completed playout, rather
    /// than at worker or backend creation time.
    nps_start_time: Option<Instant>,
    initial_visits: u32,
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
    /// px0 wakes its watchdog immediately after `FireStopInternal`
    /// (`src/search/classic/search.cc:981-1025`).
    watchdog_cv: Arc<Condvar>,
}

impl SearchStopController {
    fn populate_iteration_stats(meta: &mut SearchMeta, worker_state: &WorkerSearchState) -> IterationStats {
        let total_playouts = worker_state.total_playouts.load(Ordering::Acquire) as i64;
        let time_since_first_batch = meta
            .nps_start_time
            .map_or(0, |started| started.elapsed().as_millis() as i64);
        // px0 initializes `nps_start_time_` only after assembling the current
        // stats, so this first observation still reports zero elapsed time
        // (`src/search/classic/search.cc:908-918`).
        if meta.nps_start_time.is_none() && total_playouts > 0 {
            meta.nps_start_time = Some(Instant::now());
        }
        IterationStats {
            time_since_movestart: meta.move_start.elapsed().as_millis() as i64,
            time_since_first_batch,
            // px0 keeps the tree-reuse visits in `total_nodes`, while the
            // current-search budget sees only newly completed playouts
            // (`src/search/classic/search.cc:908-922`).
            total_nodes: total_playouts + i64::from(meta.initial_visits),
            nodes_since_movestart: total_playouts,
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

    pub(crate) fn maybe_trigger_stop(&self, worker_state: &WorkerSearchState, hints: &mut StoppersHints) -> bool {
        let stats = {
            let mut meta = self.meta.lock().expect("meta lock");
            Self::populate_iteration_stats(&mut meta, worker_state)
        };
        // px0 refuses to invoke stoppers before the root has completed its
        // first visit (`src/search/classic/search.cc:596-610`). In
        // particular, a zero-time request must not turn an unexpanded root
        // into a terminal search result.
        if stats.total_nodes == 0 {
            return false;
        }
        let mut stopper_guard = self.stopper.lock().expect("stopper lock");
        let Some(stopper) = stopper_guard.as_mut() else {
            return false;
        };
        // px0 resets the caller-owned hints before every stopper pass. A
        // SearchWorker retains its own hints for gathering, while the
        // watchdog owns a separate copy for its wait deadline
        // (`search.h:368-369`, `search.cc:596-610,981-1017`).
        hints.reset();
        let should_stop = stopper.should_stop(&stats, hints);
        if should_stop {
            worker_state.stop.store(true, Ordering::Release);
            self.watchdog_cv.notify_all();
        }
        should_stop
    }

    /// px0 `Search::MaybeTriggerStop` calls `OnSearchDone` exactly once after
    /// claiming the terminal bestmove response (`search.cc:596-620`).
    pub(crate) fn on_search_done(&self, worker_state: &WorkerSearchState) {
        let stats = {
            let mut meta = self.meta.lock().expect("meta lock");
            Self::populate_iteration_stats(&mut meta, worker_state)
        };
        if let Some(stopper) = self.stopper.lock().expect("stopper lock").as_mut() {
            stopper.on_search_done(&stats);
        }
    }

    /// px0's `GetTimeSinceFirstBatch` with the `GetTimeSinceStart` fallback
    /// used by `NodesPerSecondLimit` (`src/search/classic/search.cc:393-398,
    /// 1213-1231`).
    pub(crate) fn nps_elapsed_or_move_start_ms(&self) -> i64 {
        let meta = self.meta.lock().expect("meta lock");
        meta.nps_start_time.map_or_else(
            || meta.move_start.elapsed().as_millis() as i64,
            |started| started.elapsed().as_millis() as i64,
        )
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
    /// px0 `Search::watchdog_cv_` (`src/search/classic/search.h:132-151`).
    /// It shares `meta`'s mutex, the Rust equivalent of px0's
    /// `counters_mutex_` pairing in `WatchdogThread`.
    watchdog_cv: Arc<Condvar>,
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
    /// px0 factory default `legacy` manager (`stoppers/factory.cc:73-114`).
    time_manager: LegacyTimeManager,
    /// Pending `MoveOverheadMs`; px0 only reconstructs the manager for a new
    /// game or a position outside the retained tree (`wrapper.cc:100-112`).
    move_overhead_ms: i64,
    /// Same pending lifecycle for px0 `Slowmover` (`stoppers/factory.cc:73-114`).
    slowmover: f32,
}

impl ClassicSearch {
    pub fn new(backend: Box<dyn Backend>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let meta = Arc::new(Mutex::new(SearchMeta {
            params: SearchParams::default(),
            move_start: Instant::now(),
            nps_start_time: None,
            initial_visits: 0,
            root_move_filter: Vec::new(),
            contempt_mode: ContemptMode::None,
        }));
        let stopper = Arc::new(Mutex::new(None));
        let watchdog_cv = Arc::new(Condvar::new());
        Self {
            tree: Arc::new(RwLock::new(NodeTree::default())),
            worker_state: Arc::new(WorkerSearchState::new(Arc::clone(&stop))),
            meta: Arc::clone(&meta),
            backend: Arc::from(backend),
            stop,
            stopper: Arc::clone(&stopper),
            stop_controller: Arc::new(SearchStopController {
                meta,
                stopper,
                watchdog_cv: Arc::clone(&watchdog_cv),
            }),
            watchdog_cv,
            responder: None,
            ok_to_respond_bestmove: Arc::new(AtomicBool::new(true)),
            bestmove_is_sent: Arc::new(AtomicBool::new(false)),
            live_info: Arc::new(Mutex::new(LiveInfoState::default())),
            threads: Mutex::new(Vec::new()),
            infinite: AtomicBool::new(false),
            time_manager: LegacyTimeManager::new(200, 1.0),
            move_overhead_ms: 200,
            slowmover: 1.0,
        }
    }

    pub fn total_root_visits(&self) -> u32 {
        let tree = self.tree.read();
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
        meta.params.wdl_eval_objectivity = options.wdl_eval_objectivity;
        meta.params.contempt_mode = options.contempt_mode;
        Ok(())
    }

    /// px0 applies `NNCacheSize` through `Engine::UpdateBackendConfig`
    /// (`src/engine.cc:153-167`) before each new search parameter snapshot.
    pub fn set_nn_cache_size(&mut self, size: usize) {
        self.backend.set_cache_size(size);
    }

    /// px0 `PopulateTimeManagementOptions` / `MakeTimeManager`
    /// (`stoppers/factory.cc:73-114`). The active manager is deliberately
    /// preserved until px0 would create a new game manager.
    pub fn set_time_management_options(&mut self, move_overhead_ms: i64, slowmover: f32) {
        self.move_overhead_ms = move_overhead_ms;
        self.slowmover = slowmover;
    }

    pub fn run_blocking_nodes(&mut self, nodes: u32) -> (Move, u32) {
        let params = GoParams {
            nodes: Some(nodes as i32),
            ..Default::default()
        };
        self.start_search(&params).expect("search");
        self.wait_search().expect("wait");
        let tree = self.tree.read();
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
        let nps = meta.nps_start_time.and_then(|started| {
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
            let score = score_from_wdl(
                meta.params.score_type,
                &mut wl,
                &mut d,
                q,
                has_wdl,
                WdlDisplayContext {
                    params: &meta.params,
                    contempt_mode: meta.contempt_mode,
                    root_is_black: tree.history().is_black_to_move(),
                },
            );
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
                    let elapsed_ms = meta
                        .nps_start_time
                        .expect("nps needs nps start time")
                        .elapsed()
                        .as_millis() as u64;
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
        let watchdog_cv = Arc::clone(&self.watchdog_cv);
        let live_info = Arc::clone(&self.live_info);
        let has_wdl = self.backend.attributes().has_wdl;
        // px0 starts a watchdog before search workers (`search.cc:874-896`).
        // It owns the terminal bestmove response; workers only advance tree
        // state and may request a stop through SearchStopController.
        handles.push(thread::spawn(move || {
            // px0 `Search::WatchdogThread` owns a separate hints instance
            // from every SearchWorker (`search.cc:981-1017`).
            let mut watchdog_hints = StoppersHints::default();
            loop {
                watchdog_stop_controller.maybe_trigger_stop(&watchdog_worker_state, &mut watchdog_hints);
                if let Some(responder) = &responder {
                    // px0 calls `MaybeOutputInfo` after every stopper pass,
                    // including the pass that first sets `stop_`
                    // (`src/search/classic/search.cc:981-1017`). Keep its
                    // tree-before-counters order while outputting the final
                    // info and the infinite/ponder limit warning.
                    let tree = watchdog_tree.read();
                    let meta = watchdog_meta.lock().expect("meta lock");
                    let edge = *watchdog_worker_state.current_best_edge.lock().expect("best edge lock");
                    if !bestmove_is_sent.load(Ordering::Acquire) && edge.is_some() {
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
                                if watchdog_stop.load(Ordering::Acquire)
                                    && !ok_to_respond_bestmove.load(Ordering::Acquire)
                                {
                                    responder.output_thinking_info(&[ThinkingInfo {
                                        comment: "WARNING: Search has reached limit and does not make any progress."
                                            .into(),
                                        ..ThinkingInfo::default()
                                    }]);
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
                    watchdog_stop_controller.on_search_done(&watchdog_worker_state);
                    if let Some(responder) = &responder {
                        let tree = watchdog_tree.read();
                        let meta = watchdog_meta.lock().expect("meta lock");
                        if let Some(output) =
                            ClassicSearch::snapshot_output(&tree, &meta, &watchdog_worker_state, has_wdl)
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
                // px0 `Search::WatchdogThread` waits on `watchdog_cv_` using
                // `counters_mutex_`, with a 1..=100 ms timeout
                // (`src/search/classic/search.cc:981-1017`). This avoids a
                // fixed polling loop while preserving an upper bound for
                // wakeups when an external event is missed.
                let meta = watchdog_meta.lock().expect("meta lock");
                let remaining_ms = watchdog_hints.estimated_remaining_time_ms().clamp(1, 100) as u64;
                let _ = watchdog_cv
                    .wait_timeout_while(meta, Duration::from_millis(remaining_ms), |_| {
                        !watchdog_stop.load(Ordering::Acquire)
                    })
                    .expect("meta lock");
            }
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

    /// Rust's non-owning UCI responder must remain valid until every worker
    /// has stopped. px0 keeps its UCI loop alive for the engine lifetime; on
    /// this ownership boundary, finite searches are allowed to finish while an
    /// unbounded search receives the same `Stop` then `Wait` sequence
    /// (`src/search/classic/search.cc:1019-1041`).
    pub fn finish_for_responder_drop(&mut self) -> Result<(), EnginError> {
        if self.infinite.load(Ordering::Acquire) {
            self.stop_search()?;
        }
        self.wait_search()
    }
}

impl SearchBase for ClassicSearch {
    fn new_game(&mut self) -> Result<(), EnginError> {
        self.wait_search()?;
        // px0 `Engine::NewGame` clears the `CachingBackend` before rebuilding
        // search state (`src/engine.cc:199-203`). The backend owns this cache;
        // a non-caching test backend intentionally treats it as a no-op.
        self.backend.clear_cache();
        *self.tree.write() = NodeTree::default();
        self.time_manager = LegacyTimeManager::new(self.move_overhead_ms, self.slowmover);
        self.worker_state = Arc::new(WorkerSearchState::new(Arc::clone(&self.stop)));
        Ok(())
    }

    fn set_position(&mut self, state: &GameState) -> Result<(), EnginError> {
        let same_game = self.tree.write().reset_to_position(&state.startpos, &state.moves);
        if !same_game {
            self.time_manager = LegacyTimeManager::new(self.move_overhead_ms, self.slowmover);
        }
        Ok(())
    }

    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError> {
        // px0 translates these limits through DepthStopper/MateStopper
        // (`src/search/classic/stoppers/common.cc:118-160`). This port has not
        // translated their complete IterationStats dependencies, so reject the
        // UCI request consistently rather than silently ignoring it when a
        // nodes or movetime budget is also present.
        if params.depth.is_some() || params.mate.is_some() {
            return Err(EnginError::PortIncomplete("go depth/go mate stopper"));
        }
        self.stop.store(false, Ordering::Release);
        self.infinite.store(params.infinite, Ordering::Release);
        // px0 initializes this as `!infinite && !ponder` in `Search::Search`
        // (`src/search/classic/search.cc:138-141`). UCI ponder remains
        // unsupported, but the internal search contract must not diverge.
        self.ok_to_respond_bestmove
            .store(!params.infinite && !params.ponder, Ordering::Release);
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

        let has_clock_budget = {
            let tree = self.tree.read();
            if tree.history().is_black_to_move() {
                params.btime.is_some()
            } else {
                params.wtime.is_some()
            }
        };
        let has_budget = nodes.is_some() || params.movetime.is_some() || has_clock_budget || params.infinite;
        if !has_budget {
            return Err(EnginError::PortIncomplete("time-based search stopper"));
        }

        let infinite_play_contempt_warning = {
            let tree = self.tree.read();
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
            // px0 `Search::Search` warns while constructing the search when
            // `play` contempt is disabled for an infinite search
            // (`src/search/classic/search.cc:156-170`).
            let warn = params.infinite
                && meta.params.contempt_mode == ContemptMode::Play
                && meta.params.wdl_rescale_diff != 0.0;
            meta.nps_start_time = None;
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
            let time_manager_stopper = self.time_manager.get_stopper(params, tree.history().last());
            let chain = build_search_stoppers(
                params,
                false,
                self.time_manager.move_overhead_ms(),
                time_manager_stopper,
            );
            *self.stopper.lock().expect("stopper lock") = Some(chain);
            warn
        };

        if infinite_play_contempt_warning {
            if let Some(responder) = &self.responder {
                responder.output_thinking_info(&[ThinkingInfo {
                    comment: "WARNING: Contempt mode set to 'disable' as 'play' not supported for infinite search."
                        .into(),
                    ..ThinkingInfo::default()
                }]);
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
        // px0 `Search::Wait` joins all threads (`search.cc:1035-1041`), and
        // its per-go `Search` destructor then clears shared collision virtual
        // visits (`search.cc:1044-1064`). This long-lived wrapper performs
        // both operations at the equivalent post-join boundary.
        let handles: Vec<_> = self.threads.lock().expect("threads lock").drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
        {
            let mut tree = self.tree.write();
            cancel_shared_collisions(&self.worker_state, &mut tree);
        }
        Ok(())
    }

    fn stop_search(&mut self) -> Result<(), EnginError> {
        // px0 `Search::Stop` is non-blocking and enables the final response
        // for an otherwise silent infinite search (`search.cc:1019-1025`).
        self.ok_to_respond_bestmove.store(true, Ordering::Release);
        self.stop.store(true, Ordering::Release);
        self.watchdog_cv.notify_all();
        Ok(())
    }

    fn abort_search(&mut self) -> Result<(), EnginError> {
        // px0 `Search::Abort` suppresses bestmove before stopping/joining
        // (`search.cc:1027-1033`).
        self.bestmove_is_sent.store(true, Ordering::Release);
        self.stop.store(true, Ordering::Release);
        self.watchdog_cv.notify_all();
        self.wait_search()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, Once};

    use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

    use super::{
        best_child_edge, best_move, orient_move, resolve_contempt_mode, score_from_wdl, wdl_from_wl_d, wdl_rescale,
        ContemptMode, ScoreType, SearchParams, WdlDisplayContext,
    };
    use crate::callbacks::{BestMoveInfo, SearchResponder, ThinkingInfo};
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

    #[derive(Default)]
    struct RecordingSearchResponder {
        infos: Mutex<Vec<ThinkingInfo>>,
    }

    impl SearchResponder for RecordingSearchResponder {
        fn output_best_move(&self, _: &BestMoveInfo) {}

        fn output_thinking_info(&self, infos: &[ThinkingInfo]) {
            self.infos.lock().expect("infos lock").extend_from_slice(infos);
        }
    }

    /// Deterministic backend used to exercise px0's multi-SearchWorker
    /// lifecycle without making the test depend on an installed ONNX runtime.
    #[derive(Clone, Debug, Default)]
    struct ParallelUniformBackend(UniformBackend);

    impl Backend for ParallelUniformBackend {
        fn evaluate(
            &self,
            history: &xiangqi_core::PositionHistory,
            legal_moves: &[xiangqi_core::Move],
        ) -> Arc<EvalResult> {
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

        fn cached_evaluation(&self, position: &EvalPosition) -> Option<Arc<EvalResult>> {
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

    /// A configured px0 `TaskWorkers` value starts owned processing tasks
    /// without breaking the shared-tree search lifecycle. This verifies the
    /// safe task/result boundary across multiple SearchWorkers
    /// (`search.h:205-244`, `search.cc:1088-1140,1268-1508`).
    #[test]
    fn shared_search_workers_complete_with_owned_task_workers() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut search = super::ClassicSearch::new(Box::new(ParallelUniformBackend::default()));
        search.set_position(&state).expect("position");
        {
            let mut meta = search.meta.lock().expect("meta lock");
            meta.params.minibatch_size = 8;
            meta.params.task_workers_per_search_worker = 1;
            meta.params.minimum_work_size_for_processing = 2;
            meta.params.minimum_work_per_task_for_processing = 1;
            meta.params.max_collision_visits = 8;
            meta.params.max_collision_visits_scaling_start = 0;
            meta.params.max_collision_visits_scaling_end = 1;
            meta.params.out_of_order_eval = false;
        }

        let (best, visits) = search.run_blocking_nodes(96);

        assert!(!best.is_null());
        assert!(visits >= 96);
        assert_eq!(
            search
                .worker_state
                .thread_count
                .load(std::sync::atomic::Ordering::Acquire),
            3
        );
        let tree = search.tree.read();
        assert!(!tree.has_in_flight_visits());
        assert!(search
            .worker_state
            .shared_collisions
            .lock()
            .expect("collisions lock")
            .is_empty());
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
            let tree = search.tree.read();
            let history = tree.history();
            backend.seed_cache(&EvalPosition {
                positions: history.positions().to_vec(),
                legal_moves: history.last().board().generate_legal_moves(),
            });
        }
        {
            let mut meta = search.meta.lock().expect("meta lock");
            meta.params.minibatch_size = 4;
            meta.params.task_workers_per_search_worker = 1;
            meta.params.minimum_work_size_for_processing = 2;
            meta.params.minimum_work_per_task_for_processing = 1;
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
        let tree = search.tree.read();
        assert!(!tree.has_in_flight_visits());
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
        let tree = search.tree.read();
        let root = tree.current_head();
        assert!(tree.node(root).has_solid_children());
        assert!(!tree.has_in_flight_visits());
    }

    /// px0 `Abort -> Wait` joins every search worker and cancels collision
    /// reservations before the next UCI position/search boundary
    /// (`src/search/classic/search.cc:1027-1064`).
    #[test]
    fn abort_wait_leaves_no_in_flight_visits_anywhere_in_tree() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut search = super::ClassicSearch::new(Box::new(ParallelUniformBackend::default()));
        search.set_position(&state).expect("position");
        search
            .start_search(&crate::uci_loop::GoParams {
                infinite: true,
                ..Default::default()
            })
            .expect("start infinite search");
        std::thread::sleep(std::time::Duration::from_millis(10));
        search.abort_search().expect("abort and wait");

        let tree = search.tree.read();
        assert!(!tree.has_in_flight_visits());
        assert!(search
            .worker_state
            .shared_collisions
            .lock()
            .expect("collisions lock")
            .is_empty());
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

    /// px0 `PopulateCommonIterationStats` includes visits retained by tree
    /// reuse in `total_nodes`, but not in `nodes_since_movestart`
    /// (`src/search/classic/search.cc:908-922`).
    #[test]
    fn iteration_stats_include_reused_root_visits() {
        let mut meta = super::SearchMeta {
            params: SearchParams::default(),
            move_start: std::time::Instant::now(),
            nps_start_time: None,
            initial_visits: 17,
            root_move_filter: Vec::new(),
            contempt_mode: ContemptMode::Play,
        };
        let worker_state = super::WorkerSearchState::default();
        worker_state
            .total_playouts
            .store(5, std::sync::atomic::Ordering::Release);

        let stats = super::SearchStopController::populate_iteration_stats(&mut meta, &worker_state);

        assert_eq!(stats.total_nodes, 22);
        assert_eq!(stats.nodes_since_movestart, 5);
        assert!(meta.nps_start_time.is_some());
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
    fn infinite_play_contempt_emits_px0_warning_before_workers_start() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut search = super::ClassicSearch::new(Box::new(UniformBackend::default()));
        search.set_position(&state).expect("position");
        search.meta.lock().expect("meta lock").params.wdl_rescale_diff = 0.1;

        let responder = Arc::new(RecordingSearchResponder::default());
        search.set_uci_responder(Arc::clone(&responder) as Arc<dyn SearchResponder>);
        search
            .start_search(&crate::uci_loop::GoParams {
                infinite: true,
                ..Default::default()
            })
            .expect("infinite search");

        let infos = responder.infos.lock().expect("infos lock");
        assert!(infos.iter().any(|info| {
            info.comment == "WARNING: Contempt mode set to 'disable' as 'play' not supported for infinite search."
        }));
        drop(infos);
        search.abort_search().expect("abort");
    }

    #[test]
    fn ponder_suppresses_bestmove_like_px0_search_constructor() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut search = super::ClassicSearch::new(Box::new(UniformBackend::default()));
        search.set_position(&state).expect("position");
        search
            .start_search(&crate::uci_loop::GoParams {
                nodes: Some(1),
                ponder: true,
                ..Default::default()
            })
            .expect("internal ponder search");

        assert!(!search.ok_to_respond_bestmove.load(std::sync::atomic::Ordering::Acquire));
        search.abort_search().expect("abort");
    }

    #[test]
    fn score_type_q_matches_px0_scaled_q() {
        let mut wl = 0.1;
        let mut d = 0.2;
        assert_eq!(
            score_from_wdl(
                ScoreType::Q,
                &mut wl,
                &mut d,
                0.1234,
                true,
                WdlDisplayContext {
                    params: &SearchParams::default(),
                    contempt_mode: ContemptMode::None,
                    root_is_black: false,
                },
            ),
            1234
        );
    }

    /// px0 applies `WDLEvalObjectivity` only to the display-side contempt
    /// diff, after the search-side WDL was already backed up
    /// (`src/search/classic/search.cc:279-291`).
    #[test]
    fn wdl_eval_objectivity_controls_display_contempt_only() {
        let mut params = SearchParams {
            wdl_rescale_diff: 0.2,
            ..SearchParams::default()
        };
        let mut subjective_wl = 0.1;
        let mut subjective_d = 0.4;
        score_from_wdl(
            ScoreType::Q,
            &mut subjective_wl,
            &mut subjective_d,
            0.0,
            true,
            WdlDisplayContext {
                params: &params,
                contempt_mode: ContemptMode::White,
                root_is_black: false,
            },
        );

        params.wdl_eval_objectivity = 0.0;
        let mut objective_wl = 0.1;
        let mut objective_d = 0.4;
        score_from_wdl(
            ScoreType::Q,
            &mut objective_wl,
            &mut objective_d,
            0.0,
            true,
            WdlDisplayContext {
                params: &params,
                contempt_mode: ContemptMode::White,
                root_is_black: false,
            },
        );

        assert_ne!(subjective_wl, objective_wl);
        assert_ne!(subjective_d, objective_d);
    }
}
