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
    total_in_flight_in_tree, worker_batch_limit, GatherParams, PendingKind, SelectionScratch,
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
fn should_yield_backend_pressure(threads: usize, backend_waiting: i32, threshold: i32) -> bool {
    threads > 1 && backend_waiting > threshold
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

pub(crate) struct SearchSession<'a, E> {
    pub tree: &'a mut MctsTree,
    pub config: MctsConfig,
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
        let batch_limit = self.config.search_batch_size.max(1);
        let mut next_report_at = if info_interval.is_zero() {
            None
        } else {
            Some(std::time::Instant::now() + info_interval)
        };
        let mut retry_without_playout = 0u32;

        while !budget_exhausted(
            &self.budget,
            self.stats.total_playouts(),
            0,
            self.stats.initial_visits(),
            Some(self.stats.as_ref()),
        ) {
            if execute_one_iteration(self, batch_limit)? {
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

    let iteration = gather_minibatch(
        session.tree,
        &session.root_history,
        &gather_params,
        &mut session.selection_scratch,
    );
    if iteration.playouts == 0 {
        return Ok(false);
    }

    let needs_eval = iteration
        .pending
        .iter()
        .any(|pending| matches!(pending.kind, PendingKind::Expand { .. }));

    let outputs = if needs_eval {
        let mut backend = BackendComputation::new(session.eval);
        for pending in &iteration.pending {
            if let PendingKind::Expand { task } = &pending.kind {
                backend.add_input(task);
            }
        }
        match backend.compute_blocking() {
            Ok(outputs) => {
                session.stats.mark_first_batch();
                outputs
            }
            Err(err) => {
                super::worker::cancel_minibatch(session.tree, iteration);
                return Err(err);
            }
        }
    } else {
        Vec::new()
    };

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
                let mut retry_without_playout = 0u32;
                let mut selection_scratch = SelectionScratch::default();
                loop {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }

                    if should_yield_backend_pressure(
                        threads,
                        backend_waiting.load(Ordering::Relaxed),
                        config.thread_idling_threshold,
                    ) {
                        thread::yield_now();
                        continue;
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
                            let iteration = gather_minibatch(
                                &mut tree_guard,
                                &root_history,
                                &gather_params,
                                &mut selection_scratch,
                            );
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

                    if !stats.try_add_minibatch(&budget, &iteration) {
                        let mut tree_guard = shared_tree.write().unwrap_or_else(|e| e.into_inner());
                        super::worker::cancel_minibatch(&mut tree_guard, iteration);
                        break;
                    }

                    let mut backend = BackendComputation::new(&mut eval);
                    let mut needs_eval = false;
                    for pending in &iteration.pending {
                        if let PendingKind::Expand { task } = &pending.kind {
                            backend.add_input(task);
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

#[cfg(test)]
mod tests {
    use super::{
        apply_empty_iteration, empty_iteration_action, should_sleep_retry, should_yield_backend_pressure,
        should_yield_retry, EmptyIterationAction, MctsBudget,
    };
    use crate::mcts::{MctsConfig, SearchStats};
    use std::time::Duration;

    #[test]
    fn backend_pressure_yield_only_for_parallel() {
        assert!(!should_yield_backend_pressure(1, 10, 1));
        assert!(!should_yield_backend_pressure(4, 1, 1));
        assert!(should_yield_backend_pressure(4, 2, 1));
    }

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
