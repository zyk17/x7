//! px0 `src/search/classic/search.h:49-260`、`search.cc:426-808,874-1055`、`wrapper.cc:53-141`。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use xiangqi_core::{GameState, Move};

use crate::callbacks::{BestMoveInfo, ThinkingInfo, Wdl};
use crate::neural::backend::Backend;
use crate::search::SearchBase;
use crate::uci_loop::GoParams;
use crate::EnginError;

use super::node::{Node, NodeTree};
use super::params::SearchParams;
use super::stoppers::timemgr::{IterationStats, StoppersHints};
use super::stoppers::{build_search_stoppers, ChainedSearchStopper, SearchStopper};
use super::worker::{SearchWorker, WorkerSearchState};

pub fn best_move(tree: &NodeTree, params: &SearchParams) -> (Move, Move) {
    let root = tree.current_head();
    let root_is_black = tree.history().is_black_to_move();
    let best_edge = best_child_edge(tree, root, params, 0);
    let best = best_edge.map(|idx| tree.node(root).edge(idx).mv).unwrap_or_else(|| {
        tree.history()
            .last()
            .board()
            .generate_legal_moves()
            .first()
            .copied()
            .unwrap_or(Move::NULL)
    });
    let ponder = best_edge
        .and_then(|idx| {
            let child = tree.node(root).child(idx)?;
            best_child_edge(tree, child, params, 1).map(|ponder_idx| tree.node(child).edge(ponder_idx).mv)
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
        let Some(next_edge) = best_child_edge(tree, child, params, depth) else {
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
pub(super) fn best_child_edge(tree: &NodeTree, parent: usize, params: &SearchParams, depth: usize) -> Option<usize> {
    if tree.node(parent).n() == 0 {
        return None;
    }
    let draw_score = draw_score(tree, params, depth % 2 == 1);
    let mut best = None;
    for edge_idx in 0..tree.node(parent).num_edges() {
        let candidate = tree.node(parent).child(edge_idx).map(|idx| tree.node(idx));
        let candidate_rank = best_edge_rank(candidate);
        let replace = match best {
            None => true,
            Some(best_idx) => {
                let current = tree.node(parent).child(best_idx).map(|idx| tree.node(idx));
                let current_rank = best_edge_rank(current);
                if candidate_rank != current_rank {
                    candidate_rank > current_rank
                } else if candidate_rank == BestEdgeRank::NonTerminal
                    && candidate.is_some_and(|node| node.n() != 0 && node.is_terminal())
                    && current.is_some_and(|node| node.n() != 0 && node.is_terminal())
                {
                    let candidate = candidate.expect("checked terminal candidate");
                    let current = current.expect("checked terminal current");
                    if candidate.is_tablebase_terminal() != current.is_tablebase_terminal() {
                        !candidate.is_tablebase_terminal()
                    } else {
                        candidate.m() < current.m()
                    }
                } else if candidate_rank == BestEdgeRank::NonTerminal {
                    let candidate_n = candidate.map_or(0, Node::n);
                    let current_n = current.map_or(0, Node::n);
                    if candidate_n != current_n {
                        candidate_n > current_n
                    } else {
                        let candidate_q = candidate.map_or(0.0, |node| node.q(draw_score));
                        let current_q = current.map_or(0.0, |node| node.q(draw_score));
                        if candidate_q != current_q {
                            candidate_q > current_q
                        } else {
                            tree.node(parent).edge(edge_idx).get_p() > tree.node(parent).edge(best_idx).get_p()
                        }
                    }
                } else if candidate_rank > BestEdgeRank::NonTerminal {
                    candidate.expect("winning candidate").m() < current.expect("winning current").m()
                } else {
                    candidate.expect("losing candidate").m() > current.expect("losing current").m()
                }
            }
        };
        if replace {
            best = Some(edge_idx);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use xiangqi_core::{initialize_magic_bitboards, GameState, STARTPOS_FEN};

    use super::{best_child_edge, orient_move, wdl_from_wl_d, SearchParams};
    use crate::search::classic::node::NodeTree;

    static INIT: Once = Once::new();

    fn ensure_init() {
        INIT.call_once(initialize_magic_bitboards);
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
        assert_eq!(best_child_edge(&tree, root, &SearchParams::default(), 0), Some(1));

        let child = tree.arena_mut().spawn_child(root, 0);
        assert!(tree.node_mut(child).try_start_score_update());
        tree.node_mut(child).finalize_score_update(0.0, 0.0, 0.0, 1);
        assert_eq!(best_child_edge(&tree, root, &SearchParams::default(), 0), Some(0));
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
}

#[derive(Clone, Debug)]
pub struct SearchOutput {
    pub bestmove: BestMoveInfo,
    pub info: ThinkingInfo,
}

struct SearchMeta {
    params: SearchParams,
    move_start: Instant,
    initial_visits: u32,
    stoppers_hints: StoppersHints,
    search_active: bool,
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
    tree: Arc<Mutex<NodeTree>>,
    worker_state: Arc<WorkerSearchState>,
    meta: Arc<Mutex<SearchMeta>>,
    backend: Arc<dyn Backend>,
    stop: Arc<AtomicBool>,
    stopper: Arc<Mutex<Option<ChainedSearchStopper>>>,
    stop_controller: Arc<SearchStopController>,
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
        }));
        let stopper = Arc::new(Mutex::new(None));
        Self {
            tree: Arc::new(Mutex::new(NodeTree::default())),
            worker_state: Arc::new(WorkerSearchState::new(Arc::clone(&stop), i64::MAX)),
            meta: Arc::clone(&meta),
            backend: Arc::from(backend),
            stop,
            stopper: Arc::clone(&stopper),
            stop_controller: Arc::new(SearchStopController { meta, stopper }),
            threads: Mutex::new(Vec::new()),
            infinite: AtomicBool::new(false),
            outputs: Vec::new(),
        }
    }

    pub fn total_root_visits(&self) -> u32 {
        let tree = self.tree.lock().expect("tree lock");
        tree.node(tree.current_head()).n()
    }

    /// px0 `SearchBase::SetBackend` (`src/search/search.h:48-55`). Callers
    /// must stop search first; a worker holds a cloned backend for its full
    /// lifetime, exactly as px0 changes the backend only while stopped.
    pub fn set_backend(&mut self, backend: Box<dyn Backend>) -> Result<(), EnginError> {
        self.abort_search()?;
        self.backend = Arc::from(backend);
        Ok(())
    }

    pub fn run_blocking_nodes(&mut self, nodes: u32) -> (Move, u32) {
        let params = GoParams {
            nodes: Some(nodes as i32),
            ..Default::default()
        };
        self.start_search(&params).expect("search");
        self.wait_search().expect("wait");
        let tree = self.tree.lock().expect("tree lock");
        let meta = self.meta.lock().expect("meta lock");
        let (best, _) = best_move(&tree, &meta.params);
        let visits = tree.node(tree.current_head()).n();
        (best, visits)
    }

    fn maybe_trigger_stop(&mut self) -> Result<bool, EnginError> {
        Ok(self.stop_controller.maybe_trigger_stop(&self.worker_state))
    }

    fn emit_outputs(&mut self) -> Result<(), EnginError> {
        let tree = self.tree.lock().expect("tree lock");
        let meta = self.meta.lock().expect("meta lock");
        let (best, ponder) = best_move(&tree, &meta.params);
        if best.is_null() {
            return Ok(());
        }
        let elapsed = meta.move_start.elapsed();
        let total_playouts = self.worker_state.total_playouts.load(Ordering::Acquire);
        let first_batch = *self.worker_state.first_batch.lock().expect("first batch lock");
        let nps = first_batch.and_then(|started| {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            (elapsed_ms > 0).then(|| (total_playouts.saturating_mul(1000) / elapsed_ms) as i32)
        });
        let root = tree.current_head();
        let info = best_child_edge(&tree, root, &meta.params, 0).map(|edge_idx| {
            let child = tree.node(root).child(edge_idx).map(|idx| tree.node(idx));
            let default_wl = -tree.node(root).wl();
            let default_d = tree.node(root).d();
            let wl = child.filter(|node| node.n() > 0).map_or(default_wl, Node::wl);
            let d = child.filter(|node| node.n() > 0).map_or(default_d, Node::d);
            ThinkingInfo {
                // `Search::SendUciInfo` (`search.cc:249-270`). ScoreType and
                // its WDL rescaling are not translated until the px0 options
                // layer exists, so no synthetic cp score is emitted here.
                depth: (self.worker_state.cum_depth.load(Ordering::Acquire) / total_playouts.max(1)) as i32,
                seldepth: self.worker_state.max_depth.load(Ordering::Acquire) as i32,
                time: elapsed.as_millis() as i64,
                nodes: total_playouts as i64 + meta.initial_visits as i64,
                nps: nps.unwrap_or(-1),
                eps: nps.map_or(-1, |_| {
                    let evaluations = self.worker_state.network_evaluations.load(Ordering::Acquire);
                    let elapsed_ms = first_batch.expect("nps needs first batch").elapsed().as_millis() as u64;
                    (evaluations.saturating_mul(1000) / elapsed_ms) as i32
                }),
                wdl: Some(wdl_from_wl_d(wl, d)),
                tb_hits: 0,
                pv: principal_variation(&tree, &meta.params, edge_idx),
                multipv: 1,
                ..ThinkingInfo::default()
            }
        });
        let mut bestmove = BestMoveInfo::new(best);
        bestmove.ponder = ponder;
        self.outputs.push(SearchOutput {
            bestmove,
            info: info.unwrap_or_default(),
        });
        Ok(())
    }

    fn start_threads(&mut self, how_many: usize) -> Result<(), EnginError> {
        let mut handles = self.threads.lock().expect("threads lock");
        if !handles.is_empty() {
            return Ok(());
        }
        let thread_count = if how_many == 0 {
            self.backend.attributes().suggested_num_search_threads.max(1)
        } else {
            how_many
        };
        // px0 `Search::StartThreads` (`search.cc:1088-1140`) can run several
        // workers because `SearchWorker` only holds `nodes_mutex_` around tree
        // mutation. This port still owns `NodeTree` for a whole iteration, so
        // accepting more than one worker would only advertise fake parallelism.
        if thread_count != 1 {
            return Err(EnginError::PortIncomplete("P4 parallel SearchWorker"));
        }
        self.worker_state.thread_count.store(thread_count, Ordering::Release);
        self.meta.lock().expect("meta lock").search_active = true;

        let tree = Arc::clone(&self.tree);
        let worker_state = Arc::clone(&self.worker_state);
        let meta = Arc::clone(&self.meta);
        let backend = Arc::clone(&self.backend);
        let stop = Arc::clone(&self.stop);
        let stop_controller = Arc::clone(&self.stop_controller);
        for _ in 0..thread_count {
            let tree = Arc::clone(&tree);
            let worker_state = Arc::clone(&worker_state);
            let meta = Arc::clone(&meta);
            let backend = Arc::clone(&backend);
            let stop = Arc::clone(&stop);
            let stop_controller = Arc::clone(&stop_controller);
            handles.push(thread::spawn(move || {
                let params = meta.lock().expect("meta lock").params.clone();
                let mut tree = tree.lock().expect("tree lock");
                let mut worker = SearchWorker::new_with_stop_controller(
                    &mut tree,
                    backend.as_ref(),
                    &params,
                    worker_state.as_ref(),
                    Some(Arc::clone(&stop_controller)),
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
        *self.tree.lock().expect("tree lock") = NodeTree::default();
        self.worker_state = Arc::new(WorkerSearchState::new(Arc::clone(&self.stop), i64::MAX));
        self.outputs.clear();
        Ok(())
    }

    fn set_position(&mut self, state: &GameState) -> Result<(), EnginError> {
        self.tree
            .lock()
            .expect("tree lock")
            .reset_to_position(&state.startpos, &state.moves);
        Ok(())
    }

    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError> {
        self.outputs.clear();
        self.stop.store(false, Ordering::Release);
        self.infinite.store(params.infinite, Ordering::Release);

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
            let tree = self.tree.lock().expect("tree lock");
            let mut meta = self.meta.lock().expect("meta lock");
            meta.move_start = Instant::now();
            meta.initial_visits = tree.node(tree.current_head()).n();
            *self.worker_state.first_batch.lock().expect("first batch lock") = None;
            meta.stoppers_hints.reset();
            self.worker_state.total_playouts.store(0, Ordering::Release);
            self.worker_state.total_batches.store(0, Ordering::Release);
            self.worker_state.network_evaluations.store(0, Ordering::Release);
            self.worker_state.cum_depth.store(0, Ordering::Release);
            self.worker_state.max_depth.store(0, Ordering::Release);
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

        if !params.infinite {
            self.wait_search()?;
            self.emit_outputs()?;
        }
        Ok(())
    }

    fn start_clock(&mut self) -> Result<(), EnginError> {
        self.meta.lock().expect("meta lock").move_start = Instant::now();
        Ok(())
    }

    fn wait_search(&mut self) -> Result<(), EnginError> {
        loop {
            // px0 `Engine::EnsureSearchStopped()` is also called before the
            // first position exists (`engine.cc:146-151,187-197`).  Do not
            // ask a stopper for root statistics unless a worker is active.
            if self.stop.load(Ordering::Acquire) || !self.meta.lock().expect("meta lock").search_active {
                break;
            }
            if self.maybe_trigger_stop()? {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        self.stop.store(true, Ordering::Release);
        self.meta.lock().expect("meta lock").search_active = false;
        let handles: Vec<_> = self.threads.lock().expect("threads lock").drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
        Ok(())
    }

    fn stop_search(&mut self) -> Result<(), EnginError> {
        self.stop.store(true, Ordering::Release);
        self.wait_search()?;
        if self.infinite.load(Ordering::Acquire) {
            self.emit_outputs()?;
        }
        Ok(())
    }

    fn abort_search(&mut self) -> Result<(), EnginError> {
        self.stop.store(true, Ordering::Release);
        self.wait_search()
    }
}
