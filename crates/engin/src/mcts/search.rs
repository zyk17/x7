use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use crate::history::PositionHistory;

use super::backend::BackendComputation;
use super::coordinator::{
    acquire_searcher_slot, ensure_tree_quiescent, init_pending_searchers, release_searcher_slot,
    SharedCollisions, SharedMctsTree,
};
use super::worker::{
    apply_minibatch, budget_exhausted, gather_minibatch, progress_from_tree, result_from_tree,
    total_in_flight_in_tree, worker_batch_limit, GatherParams, PendingKind,
};
use super::{
    MctsBudget, MctsConfig, MctsNodeId, MctsSearchProgress, MctsSearchResult, MctsTree, OnnxPolicyValueEval,
    PolicyValueEval, SearchStats,
};

const WATCHDOG_MIN_WAIT: Duration = Duration::from_millis(1);
const WATCHDOG_MAX_WAIT: Duration = Duration::from_millis(100);

pub(crate) struct SearchSession<'a, E> {
    pub tree: &'a mut MctsTree,
    pub config: MctsConfig,
    pub root_id: MctsNodeId,
    pub root_history: PositionHistory,
    pub budget: MctsBudget,
    pub stats: Arc<SearchStats>,
    pub eval: &'a mut E,
}

impl<'a, E> SearchSession<'a, E>
where
    E: PolicyValueEval,
{
    pub fn run_with_progress<F>(
        &mut self,
        info_interval: Duration,
        mut on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let batch_limit = self.config.search_batch_size.max(1);
        let mut next_report_at = if info_interval.is_zero() {
            None
        } else {
            Some(std::time::Instant::now() + info_interval)
        };

        while !budget_exhausted(
            &self.budget,
            self.stats.total_playouts(),
            0,
            self.stats.initial_visits(),
            Some(self.stats.as_ref()),
        ) {
            if !execute_one_iteration(self, batch_limit)? {
                break;
            }
            if let Some(deadline) = next_report_at {
                let now = std::time::Instant::now();
                if now >= deadline && self.stats.total_playouts() > 0 {
                    on_progress(&progress_from_tree(self.tree, self.root_id, self.stats.as_ref()));
                    next_report_at = Some(now + info_interval);
                }
            }
        }

        debug_assert_eq!(total_in_flight_in_tree(self.tree), 0);
        Ok(result_from_tree(self.tree, self.root_id, self.stats.as_ref()))
    }
}

/// px0 `SearchWorker::ExecuteOneIteration` 七步。
pub(crate) fn execute_one_iteration<E>(
    session: &mut SearchSession<'_, E>,
    batch_limit: usize,
) -> Result<bool, E::Error>
where
    E: PolicyValueEval,
{
    let mut backend = BackendComputation::new(session.eval);
    let gather_params = GatherParams {
        config: session.config,
        budget: &session.budget,
        base_playouts: session.stats.total_playouts(),
        in_flight_playouts: 0,
        initial_visits: session.stats.initial_visits(),
        batch_limit,
        stats: Some(session.stats.as_ref()),
        root_id: session.root_id,
        root_visits: session
            .tree
            .get(session.root_id)
            .map(|root| root.visits)
            .unwrap_or(0),
        thread_count: 1,
        backend_waiting: 0,
    };

    let iteration = gather_minibatch(session.tree, &session.root_history, &gather_params);
    if iteration.playouts == 0 {
        return Ok(false);
    }

    for pending in &iteration.pending {
        if let PendingKind::Expand { task } = &pending.kind {
            backend.add_input(task.as_ref());
        }
    }

    let outputs = match backend.compute_blocking() {
        Ok(outputs) => outputs,
        Err(err) => {
            super::worker::cancel_minibatch(session.tree, iteration);
            return Err(err);
        }
    };

    session.stats.mark_first_batch();
    session.stats.add_minibatch(&iteration);
    let shared = SharedCollisions::default();
    shared.collect(&iteration);
    apply_minibatch(session.tree, iteration, &outputs, Some(&shared));
    Ok(true)
}

pub(crate) fn run_parallel_with_progress<F>(
    tree: &mut MctsTree,
    config: MctsConfig,
    evaluator: &mut OnnxPolicyValueEval,
    history: &PositionHistory,
    root_id: MctsNodeId,
    budget: MctsBudget,
    threads: usize,
    info_interval: Duration,
    mut on_progress: F,
) -> Result<MctsSearchResult, String>
where
    F: FnMut(&MctsSearchProgress) + Send,
{
    let root_history = history.clone_for_search();
    let shared_policy = evaluator.policy.clone();
    let shared_cache = evaluator.shared_cache();
    let stop = budget
        .stop
        .clone()
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let initial_visits = tree.get(root_id).map(|root| root.visits).unwrap_or(0);
    let stats = Arc::new(SearchStats::new(initial_visits));
    let active_workers = Arc::new(std::sync::atomic::AtomicUsize::new(threads));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let shared_tree: SharedMctsTree = Arc::new(RwLock::new(std::mem::take(tree)));
    let shared_collisions = Arc::new(SharedCollisions::default());
    let pending_searchers = init_pending_searchers(config);
    let backend_waiting = Arc::new(AtomicI32::new(0));
    let wait = if info_interval.is_zero() {
        WATCHDOG_MAX_WAIT
    } else {
        info_interval.clamp(WATCHDOG_MIN_WAIT, WATCHDOG_MAX_WAIT)
    };

    thread::scope(|scope| {
        let watchdog_stop = Arc::clone(&stop);
        let watchdog_workers = Arc::clone(&active_workers);
        let watchdog_tree = Arc::clone(&shared_tree);
        let watchdog_stats = Arc::clone(&stats);
        scope.spawn(move || {
            while watchdog_workers.load(Ordering::Relaxed) > 0 {
                thread::sleep(wait);
                if watchdog_stop.load(Ordering::SeqCst) {
                    continue;
                }
                if let Ok(tree_guard) = watchdog_tree.read() {
                    if watchdog_stats.total_playouts() > 0 {
                        on_progress(&progress_from_tree(
                            &*tree_guard,
                            root_id,
                            watchdog_stats.as_ref(),
                        ));
                    }
                }
            }
        });

        for _ in 0..threads {
            let stop = Arc::clone(&stop);
            let stats = Arc::clone(&stats);
            let active_workers = Arc::clone(&active_workers);
            let first_error = Arc::clone(&first_error);
            let root_history = root_history.clone_for_search();
            let budget = budget.clone();
            let shared_policy = shared_policy.clone();
            let shared_cache = shared_cache.clone();
            let shared_tree = Arc::clone(&shared_tree);
            let shared_collisions = Arc::clone(&shared_collisions);
            let backend_waiting = Arc::clone(&backend_waiting);
            let pending_searchers = pending_searchers.clone();
            scope.spawn(move || {
                let mut eval = OnnxPolicyValueEval::with_shared_cache(shared_policy, shared_cache);
                loop {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }

                    if let Some(pending) = pending_searchers.as_ref() {
                        acquire_searcher_slot(pending);
                    }

                    let batch_limit = worker_batch_limit(config, threads);
                    let iteration = {
                        let mut tree_guard = shared_tree.write().unwrap_or_else(|e| e.into_inner());
                        if budget_exhausted(
                            &budget,
                            stats.total_playouts(),
                            0,
                            stats.initial_visits(),
                            Some(stats.as_ref()),
                        ) {
                            None
                        } else {
                            let gather_params = GatherParams {
                                config,
                                budget: &budget,
                                base_playouts: stats.total_playouts(),
                                in_flight_playouts: 0,
                                initial_visits: stats.initial_visits(),
                                batch_limit,
                                stats: Some(stats.as_ref()),
                                root_id,
                                root_visits: tree_guard.get(root_id).map(|root| root.visits).unwrap_or(0),
                                thread_count: threads,
                                backend_waiting: backend_waiting.load(Ordering::Relaxed),
                            };
                            let iteration = gather_minibatch(&mut tree_guard, &root_history, &gather_params);
                            if iteration.playouts == 0 {
                                None
                            } else {
                                shared_collisions.collect(&iteration);
                                Some(iteration)
                            }
                        }
                    };

                    if let Some(pending) = pending_searchers.as_ref() {
                        release_searcher_slot(pending);
                    }

                    let Some(iteration) = iteration else {
                        break;
                    };

                    if !stats.try_add_minibatch(&budget, &iteration) {
                        let mut tree_guard = shared_tree.write().unwrap_or_else(|e| e.into_inner());
                        super::worker::cancel_minibatch(&mut tree_guard, iteration);
                        break;
                    }

                    let mut backend = BackendComputation::new(&mut eval);
                    let mut needs_eval = false;
                    for pending in &iteration.pending {
                        if let PendingKind::Expand { task } = &pending.kind {
                            backend.add_input(task.as_ref());
                            needs_eval = true;
                        }
                    }

                    backend_waiting.fetch_add(1, Ordering::Relaxed);
                    let outputs = if needs_eval {
                        match backend.compute_blocking() {
                            Ok(outputs) => {
                                stats.mark_first_batch();
                                outputs
                            }
                            Err(err) => {
                                backend_waiting.fetch_sub(1, Ordering::Relaxed);
                                let mut tree_guard = shared_tree.write().unwrap_or_else(|e| e.into_inner());
                                stats.rollback_minibatch(&iteration);
                                super::worker::cancel_minibatch(&mut tree_guard, iteration);
                                *first_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(err);
                                stop.store(true, Ordering::SeqCst);
                                break;
                            }
                        }
                    } else {
                        stats.mark_first_batch();
                        Vec::new()
                    };
                    backend_waiting.fetch_sub(1, Ordering::Relaxed);

                    {
                        let mut tree_guard = shared_tree.write().unwrap_or_else(|e| e.into_inner());
                        apply_minibatch(
                            &mut tree_guard,
                            iteration,
                            &outputs,
                            Some(shared_collisions.as_ref()),
                        );
                    }
                }
                active_workers.fetch_sub(1, Ordering::Relaxed);
            });
        }
    });

    if let Some(err) = first_error.lock().unwrap_or_else(|e| e.into_inner()).take() {
        return Err(err);
    }

    let shared_tree =
        Arc::try_unwrap(shared_tree).map_err(|_| "parallel search tree still shared".to_string())?;
    let restored = shared_tree
        .into_inner()
        .map_err(|_| "parallel search tree lock poisoned".to_string())?;
    ensure_tree_quiescent(&restored)?;
    *tree = restored;
    Ok(result_from_tree(tree, root_id, stats.as_ref()))
}
