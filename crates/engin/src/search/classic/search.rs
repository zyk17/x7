//! px0 `src/search/classic/search.h:49-260`、`search.cc:426-808,874-1055`、`wrapper.cc:53-141`。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use xiangqi_core::{GameState, Move};

use crate::callbacks::{BestMoveInfo, ThinkingInfo};
use crate::neural::backend::Backend;
use crate::search::SearchBase;
use crate::uci_loop::GoParams;
use crate::EnginError;

use super::node::NodeTree;
use super::params::SearchParams;
use super::stoppers::timemgr::{IterationStats, StoppersHints};
use super::stoppers::{build_search_stoppers, ChainedSearchStopper, SearchStopper};
use super::worker::{SearchWorker, WorkerSearchState};

pub fn best_move(tree: &NodeTree, params: &SearchParams) -> (Move, Move) {
    let root = tree.current_head();
    let root_is_black = tree.history().is_black_to_move();
    let best_edge = best_child_edge(tree, root, params);
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
            best_child_edge(tree, child, params).map(|ponder_idx| tree.node(child).edge(ponder_idx).mv)
        })
        .unwrap_or(Move::NULL);
    (
        if root_is_black { best.flip() } else { best },
        if root_is_black { ponder } else { ponder.flip() },
    )
}

fn best_child_edge(tree: &NodeTree, parent: usize, params: &SearchParams) -> Option<usize> {
    let mut best_idx = None;
    let mut best_n = 0;
    let mut best_q = f32::NEG_INFINITY;
    let mut best_p = f32::NEG_INFINITY;
    for edge_idx in 0..tree.node(parent).num_edges() {
        let n = tree
            .node(parent)
            .child(edge_idx)
            .map(|child| tree.node(child).n())
            .unwrap_or(0);
        let q = tree
            .node(parent)
            .child(edge_idx)
            .map(|child| tree.node(child).q(params.draw_score))
            .unwrap_or(0.0);
        let p = tree.node(parent).edge(edge_idx).get_p();
        let better = n > best_n || (n == best_n && n > 0 && q > best_q) || (n == best_n && n == 0 && p > best_p);
        if better {
            best_idx = Some(edge_idx);
            best_n = n;
            best_q = q;
            best_p = p;
        }
    }
    best_idx
}

#[derive(Clone, Debug)]
pub struct SearchOutput {
    pub bestmove: BestMoveInfo,
    pub info: ThinkingInfo,
}

struct SearchMeta {
    params: SearchParams,
    move_start: Instant,
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
        let root_n = tree.node(tree.current_head()).n();
        if best.is_null() {
            return Ok(());
        }
        let elapsed = meta.move_start.elapsed();
        let mut bestmove = BestMoveInfo::new(best);
        bestmove.ponder = ponder;
        self.outputs.push(SearchOutput {
            bestmove,
            info: ThinkingInfo {
                depth: self.worker_state.max_depth.load(Ordering::Acquire) as i32,
                nodes: root_n as i64,
                time: elapsed.as_millis() as i64,
                multipv: 1,
                ..ThinkingInfo::default()
            },
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
