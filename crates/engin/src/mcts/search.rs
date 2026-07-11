use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::history::PositionHistory;

use super::backend::BackendComputation;
use super::worker::{
    acquire_searcher_slot, budget_exhausted, do_backup_update, ensure_tree_quiescent,
    fetch_minibatch_results, gather_has_work, gather_minibatch, init_pending_searchers,
    maybe_prefetch_into_cache, progress_from_tree, release_searcher_slot, result_from_tree,
    total_in_flight_in_tree, worker_batch_limit, GatherParams, SelectionScratch, SharedCollisions,
    SharedMctsTree,
};
use super::{
    MctsBudget, MctsConfig, MctsNodeId, MctsSearchProgress, MctsSearchResult, MctsTree, OnnxPolicyValueEval,
    PolicyValueEval, SearchStats,
};

const WATCHDOG_MIN_WAIT: Duration = Duration::from_millis(1);
const WATCHDOG_MAX_WAIT: Duration = Duration::from_millis(100);
const RETRY_SLEEP: Duration = Duration::from_millis(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmptyIterationAction {
    Break,
    Continue,
    Yield,
    Sleep,
}

#[inline]
fn is_unbounded_search(budget: &MctsBudget) -> bool {
    budget.max_playouts.is_none()
        && budget.max_nodes.is_none()
        && budget.max_depth.is_none()
        && budget.deadline.is_none()
}

#[inline]
fn should_yield_retry(retry_without_playout: u32, retry_yield_interval: u32) -> bool {
    retry_without_playout >= retry_yield_interval.max(1)
}

#[inline]
fn should_sleep_retry(config: MctsConfig, budget: &MctsBudget, retry_without_playout: u32) -> bool {
    is_unbounded_search(budget) && retry_without_playout >= config.retry_sleep_interval.max(1)
}

fn empty_iteration_action(
    budget: &MctsBudget,
    config: MctsConfig,
    stats: &SearchStats,
    retry_without_playout: u32,
) -> EmptyIterationAction {
    if budget_exhausted(
        budget,
        stats.total_playouts(),
        0,
        stats.initial_visits(),
        Some(stats),
    ) {
        return EmptyIterationAction::Break;
    }
    if should_sleep_retry(config, budget, retry_without_playout) {
        return EmptyIterationAction::Sleep;
    }
    if should_yield_retry(retry_without_playout, config.retry_yield_interval) {
        return EmptyIterationAction::Yield;
    }
    EmptyIterationAction::Continue
}

fn apply_empty_iteration(
    budget: &MctsBudget,
    config: MctsConfig,
    stats: &SearchStats,
    retry_without_playout: &mut u32,
) -> EmptyIterationAction {
    let action = empty_iteration_action(budget, config, stats, *retry_without_playout);
    if matches!(action, EmptyIterationAction::Break) {
        return action;
    }
    stats.add_retry_without_playout();
    *retry_without_playout = retry_without_playout.saturating_add(1);
    let action = empty_iteration_action(budget, config, stats, *retry_without_playout);
    if matches!(action, EmptyIterationAction::Yield | EmptyIterationAction::Sleep) {
        *retry_without_playout = 0;
    }
    action
}

fn search_stopped(budget: &MctsBudget) -> bool {
    budget
        .stop
        .as_ref()
        .is_some_and(|stop| stop.load(Ordering::SeqCst))
}

fn apply_nps_limit(config: MctsConfig, stats: &SearchStats, budget: &MctsBudget) {
    if config.nps_limit == 0 {
        return;
    }
    while !budget_exhausted(budget, stats.total_playouts(), 0, stats.initial_visits(), Some(stats)) {
        let elapsed = stats.nps_elapsed_ms();
        if elapsed == 0 {
            break;
        }
        let nps = SearchStats::playouts_per_second(stats.total_playouts(), elapsed);
        if nps <= config.nps_limit {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

pub(crate) struct SearchSession<'a, E> {
    pub tree: &'a mut MctsTree,
    pub config: MctsConfig,
    pub batch_limit: usize,
    pub root_id: MctsNodeId,
    pub root_history: PositionHistory,
    pub budget: MctsBudget,
    pub stats: Arc<SearchStats>,
    pub eval: &'a mut E,
    pub selection_scratch: SelectionScratch,
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
        let batch_limit = self.batch_limit;
        let mut next_report_at = if info_interval.is_zero() {
            None
        } else {
            Some(std::time::Instant::now() + info_interval)
        };
        let mut retry_without_playout = 0u32;

        let shared_collisions = SharedCollisions::default();
        while !budget_exhausted(
            &self.budget,
            self.stats.total_playouts(),
            0,
            self.stats.initial_visits(),
            Some(self.stats.as_ref()),
        ) {
            if execute_one_iteration(self, batch_limit, &shared_collisions)? {
                retry_without_playout = 0;
            } else {
                match apply_empty_iteration(
                    &self.budget,
                    self.config,
                    self.stats.as_ref(),
                    &mut retry_without_playout,
                ) {
                    EmptyIterationAction::Break => break,
                    EmptyIterationAction::Continue => continue,
                    EmptyIterationAction::Yield => {
                        thread::yield_now();
                        continue;
                    }
                    EmptyIterationAction::Sleep => {
                        thread::sleep(RETRY_SLEEP);
                        continue;
                    }
                }
            }
            if let Some(deadline) = next_report_at {
                let now = std::time::Instant::now();
                if now >= deadline && self.stats.total_playouts() > 0 {
                    on_progress(&progress_from_tree(
                        self.tree,
                        self.root_id,
                        self.stats.as_ref(),
                        self.config,
                    ));
                    next_report_at = Some(now + info_interval);
                }
            }
        }

        shared_collisions.cancel_all(self.tree);
        debug_assert_eq!(total_in_flight_in_tree(self.tree), 0);
        Ok(result_from_tree(
            self.tree,
            self.root_id,
            self.stats.as_ref(),
            self.config,
        ))
    }
}

/// lc0 `SearchWorker::ExecuteOneIteration`（search.cc:1209-1230, 1507-1573, 2018-2377）。
pub(crate) fn execute_one_iteration<E>(
    session: &mut SearchSession<'_, E>,
    batch_limit: usize,
    shared_collisions: &SharedCollisions,
) -> Result<bool, E::Error>
where
    E: PolicyValueEval,
{
    let stop = search_stopped(&session.budget);
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

    let mut iteration = gather_minibatch(
        session.tree,
        &session.root_history,
        &gather_params,
        &mut session.selection_scratch,
        Some(&mut backend),
        stop,
    );
    // 1 gather (search.cc:1303-1439)
    if iteration.pending.is_empty() {
        return Ok(false);
    }
    if !gather_has_work(&iteration, backend.used_batch_size()) {
        super::worker::cancel_minibatch(session.tree, iteration);
        shared_collisions.cancel_all(session.tree);
        return Ok(false);
    }

    // 2 collect collisions (search.cc:1507+)
    shared_collisions.collect(&iteration.pending);

    // 3 prefetch (search.cc:2018-2050)
    maybe_prefetch_into_cache(
        session.tree,
        session.root_id,
        &session.root_history,
        session.config,
        &mut backend,
        stop,
    );

    // 4 NN compute
    let outputs = if backend.used_batch_size() > 0 {
        match backend.compute_blocking() {
            Ok(outputs) => {
                session.stats.mark_first_batch();
                outputs
            }
            Err(err) => {
                super::worker::cancel_minibatch(session.tree, iteration);
                shared_collisions.cancel_all(session.tree);
                return Err(err);
            }
        }
    } else {
        Vec::new()
    };

    // 5 fetch (search.cc:2151+)
    fetch_minibatch_results(
        session.tree,
        session.eval,
        &mut iteration,
        &outputs,
        session.config,
        session.root_id,
    );
    // 6 backup + 7 counters (search.cc:2217-2377)
    session.stats.commit_minibatch(&iteration);
    do_backup_update(
        session.tree,
        &iteration,
        Some(session.stats.as_ref()),
        Some(shared_collisions),
    );
    apply_nps_limit(session.config, session.stats.as_ref(), &session.budget);
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
    let backend_attrs = shared_policy.as_ref().map(|pool| pool.backend_attributes());
    let batch_limit = worker_batch_limit(config, backend_attrs);
    let shared_cache = evaluator.shared_cache();
    let stop = budget
        .stop
        .clone()
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let initial_visits = tree.get(root_id).map(|root| root.visits).unwrap_or(0);
    let stats = Arc::new(SearchStats::new(initial_visits));
    let active_workers = Arc::new(std::sync::atomic::AtomicUsize::new(threads));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let shared_tree: SharedMctsTree = Arc::new(Mutex::new(std::mem::take(tree)));
    let pending_searchers = init_pending_searchers(config);
    let shared_collisions = Arc::new(SharedCollisions::default());
    let backend_waiting = Arc::new(AtomicI32::new(0));
    let wait = if info_interval.is_zero() {
        WATCHDOG_MAX_WAIT
    } else {
        info_interval.clamp(WATCHDOG_MIN_WAIT, WATCHDOG_MAX_WAIT)
    };

    let worker_sessions = match shared_policy.as_ref() {
        Some(pool) => Some(pool.resize_sessions(threads).map_err(|e| e.to_string())?),
        None => None,
    };
    thread::scope(|scope| {
        scope.spawn({
            let stop = Arc::clone(&stop);
            let active_workers = Arc::clone(&active_workers);
            let shared_tree = Arc::clone(&shared_tree);
            let stats = Arc::clone(&stats);
            move || {
                while active_workers.load(Ordering::Relaxed) > 0 {
                    thread::sleep(wait);
                    if stop.load(Ordering::SeqCst) {
                        continue;
                    }
                    if let Ok(tree_guard) = shared_tree.lock() {
                        if stats.total_playouts() > 0 {
                            on_progress(&progress_from_tree(
                                &*tree_guard,
                                root_id,
                                stats.as_ref(),
                                config,
                            ));
                        }
                    }
                }
            }
        });

        for worker_idx in 0..threads {
            let stop = Arc::clone(&stop);
            let stats = Arc::clone(&stats);
            let active_workers = Arc::clone(&active_workers);
            let first_error = Arc::clone(&first_error);
            let root_history = root_history.clone_for_search();
            let budget = budget.clone();
            let shared_policy = shared_policy.clone();
            let shared_cache = shared_cache.clone();
            let shared_tree = Arc::clone(&shared_tree);
            let backend_waiting = Arc::clone(&backend_waiting);
            let pending_searchers = pending_searchers.clone();
            let shared_collisions = Arc::clone(&shared_collisions);
            let worker_session = worker_sessions
                .as_ref()
                .and_then(|sessions| sessions.get(worker_idx).cloned());
            scope.spawn(move || {
                let mut eval = OnnxPolicyValueEval::with_dedicated_session(
                    shared_policy,
                    worker_session,
                    shared_cache,
                );
                let mut retry_without_playout = 0u32;
                let mut selection_scratch = SelectionScratch::default();
                loop {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }

                    if let Some(pending) = pending_searchers.as_ref() {
                        acquire_searcher_slot(pending, config.search_spin_backoff);
                    }

                let mut backend = BackendComputation::new(&mut eval);
                let iteration = {
                    let mut tree_guard = shared_tree.lock().unwrap_or_else(|e| e.into_inner());
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
                        let iteration = gather_minibatch(
                            &mut tree_guard,
                            &root_history,
                            &gather_params,
                            &mut selection_scratch,
                            Some(&mut backend),
                            stop.load(Ordering::SeqCst),
                        );
                        if iteration.pending.is_empty() {
                            None
                        } else if !gather_has_work(&iteration, backend.used_batch_size()) {
                            super::worker::cancel_minibatch(&mut tree_guard, iteration);
                            shared_collisions.cancel_all(&mut tree_guard);
                            None
                        } else {
                            shared_collisions.collect(&iteration.pending);
                            Some(iteration)
                        }
                    }
                };

                if let Some(pending) = pending_searchers.as_ref() {
                    release_searcher_slot(pending);
                }

                let Some(mut iteration) = iteration else {
                    match apply_empty_iteration(&budget, config, stats.as_ref(), &mut retry_without_playout) {
                        EmptyIterationAction::Break => break,
                        EmptyIterationAction::Continue => continue,
                        EmptyIterationAction::Yield => {
                            thread::yield_now();
                            continue;
                        }
                        EmptyIterationAction::Sleep => {
                            thread::sleep(RETRY_SLEEP);
                            continue;
                        }
                    }
                };
                retry_without_playout = 0;

                {
                    let tree_guard = shared_tree.lock().unwrap_or_else(|e| e.into_inner());
                    maybe_prefetch_into_cache(
                        &*tree_guard,
                        root_id,
                        &root_history,
                        config,
                        &mut backend,
                        stop.load(Ordering::SeqCst),
                    );
                }
                backend_waiting.fetch_add(1, Ordering::Relaxed);
                let outputs = if backend.used_batch_size() > 0 {
                    match backend.compute_blocking() {
                        Ok(outputs) => {
                            stats.mark_first_batch();
                            outputs
                        }
                        Err(err) => {
                            backend_waiting.fetch_sub(1, Ordering::Relaxed);
                            let mut tree_guard = shared_tree.lock().unwrap_or_else(|e| e.into_inner());
                            super::worker::cancel_minibatch(&mut tree_guard, iteration);
                            shared_collisions.cancel_all(&mut tree_guard);
                            *first_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(err);
                            stop.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
                } else {
                    Vec::new()
                };
                backend_waiting.fetch_sub(1, Ordering::Relaxed);

                let mut tree_guard = shared_tree.lock().unwrap_or_else(|e| e.into_inner());
                fetch_minibatch_results(
                    &mut tree_guard,
                    &mut eval,
                    &mut iteration,
                    &outputs,
                    config,
                    root_id,
                );
                if !stats.try_commit_minibatch(&budget, &iteration) {
                    super::worker::cancel_minibatch(&mut tree_guard, iteration);
                    shared_collisions.cancel_all(&mut tree_guard);
                    break;
                }
                do_backup_update(
                    &mut tree_guard,
                    &iteration,
                    Some(stats.as_ref()),
                    Some(&shared_collisions),
                );
                apply_nps_limit(config, stats.as_ref(), &budget);
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
    let mut restored = shared_tree
        .into_inner()
        .map_err(|_| "parallel search tree lock poisoned".to_string())?;
    shared_collisions.cancel_all(&mut restored);
    ensure_tree_quiescent(&restored)?;
    *tree = restored;
    Ok(result_from_tree(tree, root_id, stats.as_ref(), config))
}

#[cfg(test)]
mod tests {
    use super::{
        apply_empty_iteration, empty_iteration_action, should_sleep_retry, should_yield_retry,
        EmptyIterationAction, MctsBudget,
    };
    use crate::mcts::{MctsConfig, SearchStats};
    use std::time::Duration;

    #[test]
    fn retry_yield_honors_min_interval() {
        assert!(!should_yield_retry(0, 0));
        assert!(should_yield_retry(1, 0));
        assert!(!should_yield_retry(3, 4));
        assert!(should_yield_retry(4, 4));
    }

    #[test]
    fn sleep_retry_only_for_unbounded_budget() {
        let config = MctsConfig {
            retry_sleep_interval: 8,
            ..MctsConfig::default()
        };
        let bounded = MctsBudget {
            max_playouts: Some(100),
            ..MctsBudget::default()
        };
        let unbounded = MctsBudget::default();
        assert!(!should_sleep_retry(config, &bounded, 8));
        assert!(should_sleep_retry(config, &unbounded, 8));
    }

    #[test]
    fn empty_iteration_breaks_when_budget_done() {
        let stats = SearchStats::new(0);
        let budget = MctsBudget {
            deadline: Some(std::time::Instant::now() - Duration::from_millis(1)),
            ..MctsBudget::default()
        };
        assert_eq!(
            empty_iteration_action(&budget, MctsConfig::default(), &stats, 1),
            EmptyIterationAction::Break
        );
    }

    #[test]
    fn empty_iteration_continues_when_budget_remains() {
        let stats = SearchStats::new(0);
        let budget = MctsBudget {
            max_playouts: Some(100),
            ..MctsBudget::default()
        };
        assert_eq!(
            empty_iteration_action(&budget, MctsConfig::default(), &stats, 1),
            EmptyIterationAction::Continue
        );
    }

    #[test]
    fn apply_empty_iteration_does_not_count_break_on_exit() {
        let stats = SearchStats::new(0);
        let budget = MctsBudget {
            deadline: Some(std::time::Instant::now() - Duration::from_millis(1)),
            ..MctsBudget::default()
        };
        let mut local = 0u32;
        assert_eq!(
            apply_empty_iteration(&budget, MctsConfig::default(), &stats, &mut local),
            EmptyIterationAction::Break
        );
        assert_eq!(stats.retry_without_playout(), 0);
    }

    #[test]
    fn apply_empty_iteration_counts_real_retries() {
        let stats = SearchStats::new(0);
        let budget = MctsBudget {
            max_playouts: Some(100),
            ..MctsBudget::default()
        };
        let mut local = 0u32;
        assert_eq!(
            apply_empty_iteration(&budget, MctsConfig::default(), &stats, &mut local),
            EmptyIterationAction::Continue
        );
        assert_eq!(stats.retry_without_playout(), 1);
        assert_eq!(local, 1);
    }
}
