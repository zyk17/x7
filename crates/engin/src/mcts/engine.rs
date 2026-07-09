use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Move, MAX_MOVES};
use xiangqi_core::{generate, Position};

use crate::history::PositionHistory;

use super::{
    MctsBudget, MctsConfig, MctsNode, MctsNodeId, MctsTree, OnnxPolicyValueEval, PolicyValueEval, PolicyValueInput,
    PolicyValueOutput, PolicyValueTask,
};

const SEARCH_BATCH_SIZE_CAP: usize = 8192;
const SEARCH_COLLISION_PLAYOUT_FACTOR: usize = 4;
const EVAL_GATHER_WAIT: Duration = Duration::from_micros(500);
const MAX_EVAL_CONSUMERS: usize = 8;
const MAX_WORKER_PIPELINE_DEPTH: usize = 4;
const MAX_PARALLEL_GATHER_CHUNK: usize = 128;
const STOPPED_EVAL: &str = "__stopped__";

/// MCTS 单次搜索结果。
#[derive(Clone, Debug)]
pub struct MctsMoveStat {
    pub mv: Move,
    pub prior: f32,
    pub visits: u32,
    pub q: f32,
}

/// MCTS 单次搜索结果。
#[derive(Clone, Debug, Default)]
pub struct MctsSearchResult {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    pub playouts: u32,
    pub root_visits: u32,
    pub nodes: usize,
    pub depth: u32,
    pub seldepth: u32,
    pub root_value: f32,
    pub best_value: f32,
    pub best_mate: Option<i32>,
    pub moves: Vec<MctsMoveStat>,
}

#[derive(Clone, Debug, Default)]
pub struct MctsSearchProgress {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    pub playouts: u32,
    pub root_visits: u32,
    pub nodes: usize,
    pub depth: u32,
    pub seldepth: u32,
    pub root_value: f32,
    pub best_value: f32,
    pub best_mate: Option<i32>,
    pub moves: Vec<MctsMoveStat>,
}

/// MCTS 引擎最小实现。
///
/// 当前版本提供：
/// - PUCT 选边
/// - 单线程 iteration + minibatch 搜索流水线
/// - in-flight / collision / multivisit
/// - value 回传
/// - 以 visit 为主的根着法选择
pub struct MctsEngine<E> {
    pub config: MctsConfig,
    pub evaluator: E,
    pub tree: MctsTree,
    root_id: Option<MctsNodeId>,
    root_history: Option<PositionHistory>,
}

impl<E> MctsEngine<E> {
    pub fn new(config: MctsConfig, evaluator: E) -> Self {
        Self {
            config,
            evaluator,
            tree: MctsTree::new(),
            root_id: None,
            root_history: None,
        }
    }
}

impl<E> MctsEngine<E>
where
    E: PolicyValueEval,
{
    pub fn search_root(&mut self, pos: &Position, budget: MctsBudget) -> Result<MctsSearchResult, E::Error> {
        let history = PositionHistory::from_position(pos.clone_for_search());
        self.search_root_history(&history, budget)
    }

    pub fn search_root_with_progress<F>(
        &mut self,
        pos: &Position,
        budget: MctsBudget,
        info_interval: Duration,
        on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let history = PositionHistory::from_position(pos.clone_for_search());
        self.search_root_history_with_progress(&history, budget, info_interval, on_progress)
    }

    pub fn search_root_history(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
    ) -> Result<MctsSearchResult, E::Error> {
        self.search_root_history_with_progress(history, budget, Duration::ZERO, |_| {})
    }

    pub fn search_root_history_with_progress<F>(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
        info_interval: Duration,
        mut on_progress: F,
    ) -> Result<MctsSearchResult, E::Error>
    where
        F: FnMut(&MctsSearchProgress),
    {
        let Some(root_id) = self.prepare_root(history)? else {
            return Ok(MctsSearchResult::default());
        };

        let root_history = history.clone_for_search();
        let mut playouts = 0u32;
        let mut seldepth = 0u32;
        let mut next_report_at = if info_interval.is_zero() {
            None
        } else {
            Some(Instant::now() + info_interval)
        };

        while !budget_exhausted(&budget, playouts, self.tree.len()) {
            let raw_limit = remaining_batch_capacity(&budget, playouts, self.tree.len(), self.config.search_batch_size)
                .unwrap_or(self.config.search_batch_size)
                .max(1);
            let batch_limit = self.gather_batch_limit(root_id, raw_limit);
            let iteration = self.gather_iteration(root_id, &root_history, &budget, playouts, batch_limit);
            if iteration.playouts == 0 {
                break;
            }

            let iter_playouts = iteration.playouts;
            seldepth = seldepth.max(iteration.seldepth);
            self.apply_iteration(iteration)?;
            debug_assert_eq!(self.total_in_flight(), 0);

            playouts = playouts.saturating_add(iter_playouts);

            if let Some(deadline) = next_report_at {
                let now = Instant::now();
                if now >= deadline {
                    on_progress(&self.progress_from_root(root_id, playouts, seldepth));
                    next_report_at = Some(now + info_interval);
                }
            }
        }

        Ok(self.result_from_root(root_id, playouts, seldepth))
    }

    fn prepare_root(&mut self, history: &PositionHistory) -> Result<Option<MctsNodeId>, E::Error> {
        if self.try_reuse_exact_root(history) {
            self.root_history = Some(history.clone_for_search());
            return Ok(self.root_id);
        }
        if self.try_reuse_appended_path(history) {
            return Ok(self.root_id);
        }
        self.tree.clear();
        self.root_id = None;
        self.root_history = None;
        let root_id = self.initialize_root(history)?;
        if root_id.is_some() {
            self.root_id = root_id;
            self.root_history = Some(history.clone_for_search());
        }
        Ok(root_id)
    }

    fn try_reuse_exact_root(&mut self, history: &PositionHistory) -> bool {
        matches!(
            (self.root_id, self.root_history.as_ref()),
            (Some(_), Some(old_history))
                if old_history.same_input_window(history)
        )
    }

    fn try_reuse_appended_path(&mut self, history: &PositionHistory) -> bool {
        let (Some(root_id), Some(old_history)) = (self.root_id, self.root_history.as_ref()) else {
            return false;
        };
        let Some(path) = appended_history_moves(old_history, history) else {
            return false;
        };
        if path.is_empty() {
            return false;
        }
        let Some(child_id) = self.follow_child_path(root_id, &path) else {
            return false;
        };
        let (new_tree, new_root_id) = self.tree.copy_subtree(child_id);
        self.tree = new_tree;
        self.root_id = Some(new_root_id);
        self.root_history = Some(history.clone_for_search());
        true
    }

    fn follow_child_path(&self, root_id: MctsNodeId, path: &[Move]) -> Option<MctsNodeId> {
        let mut node_id = root_id;
        for mv in path {
            let node = self.tree.get(node_id)?;
            node_id = node
                .children
                .iter()
                .find(|edge| edge.mv == *mv)
                .and_then(|edge| edge.child)?;
        }
        Some(node_id)
    }

    fn initialize_root(&mut self, history: &PositionHistory) -> Result<Option<MctsNodeId>, E::Error> {
        let pos = history.current();
        let mut buf = [ExtMove {
            mv: Move::none(),
            value: 0,
        }; MAX_MOVES];
        let n = generate(pos, GenType::Legal, &mut buf);
        if n == 0 {
            return Ok(None);
        }

        let legal: Vec<Move> = buf[..n].iter().map(|e| e.mv).collect();
        let out = self.evaluator.evaluate(PolicyValueInput {
            position: pos,
            history,
            legal_moves: &legal,
        })?;

        let mut root = MctsNode {
            state_key: pos.key(),
            visits: 0,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: Vec::with_capacity(legal.len()),
        };
        for (i, mv) in legal.iter().copied().enumerate() {
            root.children.push(super::EdgeStats {
                mv,
                prior: out.priors.get(i).copied().unwrap_or(0.0),
                visits: 0,
                in_flight: 0,
                expanding: false,
                value_sum: 0.0,
                child: None,
            });
        }
        Ok(Some(self.tree.add_node(root)))
    }

    fn gather_batch_limit(&self, root_id: MctsNodeId, configured_limit: usize) -> usize {
        let Some(root) = self.tree.get(root_id) else {
            return configured_limit.max(1);
        };
        gather_batch_limit_for_root_visits(root.visits, configured_limit)
    }

}

fn gather_batch_limit_for_root_visits(root_visits: u32, configured_limit: usize) -> usize {
    let ramp_limit = if root_visits < 8 {
        1
    } else if root_visits < 32 {
        4
    } else if root_visits < 128 {
        8
    } else if root_visits < 512 {
        16
    } else if root_visits < 2_048 {
        32
    } else if root_visits < 8_192 {
        64
    } else if root_visits < 32_768 {
        128
    } else if root_visits < 131_072 {
        256
    } else if root_visits < 524_288 {
        512
    } else if root_visits < 2_097_152 {
        1024
    } else {
        configured_limit
    };
    configured_limit.min(ramp_limit).max(1)
}

fn parallel_gather_batch_limit_for_root_visits(root_visits: u32, configured_limit: usize, threads: usize) -> usize {
    let base = gather_batch_limit_for_root_visits(root_visits, configured_limit);
    if threads <= 1 {
        return base;
    }
    let worker_floor = threads.clamp(1, 16) * 4;
    if root_visits < 32 {
        configured_limit.min(worker_floor).max(base)
    } else if root_visits < 128 {
        configured_limit.min(worker_floor * 2).max(base)
    } else {
        base
    }
}

impl<E> MctsEngine<E>
where
    E: PolicyValueEval,
{
    fn gather_iteration(
        &mut self,
        root_id: MctsNodeId,
        root_history: &PositionHistory,
        budget: &MctsBudget,
        base_playouts: u32,
        batch_limit: usize,
    ) -> SearchIteration {
        gather_iteration_with(
            &mut self.tree,
            self.config,
            root_id,
            root_history,
            budget,
            base_playouts,
            batch_limit,
        )
    }

    fn apply_iteration(&mut self, iteration: SearchIteration) -> Result<(), E::Error> {
        let outputs = Self::evaluate_iteration_with(&mut self.evaluator, &iteration)?;
        self.apply_iteration_outputs(iteration, &outputs);
        Ok(())
    }

    fn evaluate_iteration_with<V>(eval: &mut V, iteration: &SearchIteration) -> Result<Vec<PolicyValueOutput>, V::Error>
    where
        V: PolicyValueEval,
    {
        let mut tasks = Vec::new();
        for pending in &iteration.pending {
            if let PendingKind::Expand { task } = &pending.kind {
                tasks.push(task.as_ref().clone());
            }
        }
        eval.evaluate_many(&tasks)
    }

    fn apply_iteration_outputs(&mut self, iteration: SearchIteration, outputs: &[PolicyValueOutput]) {
        apply_iteration_outputs_with(&mut self.tree, iteration, outputs)
    }

    fn progress_from_root(&self, root_id: MctsNodeId, playouts: u32, seldepth: u32) -> MctsSearchProgress {
        progress_from_tree(&self.tree, root_id, playouts, seldepth)
    }

    fn result_from_root(&self, root_id: MctsNodeId, playouts: u32, seldepth: u32) -> MctsSearchResult {
        result_from_tree(&self.tree, root_id, playouts, seldepth)
    }

    #[cfg(test)]
    fn extract_pv(&self, root_id: MctsNodeId) -> Vec<Move> {
        extract_pv_from_tree(&self.tree, root_id)
    }

    #[cfg(test)]
    fn pv_summary(&self, root_id: MctsNodeId) -> PvSummary {
        pv_summary_from_tree(&self.tree, root_id)
    }

    fn total_in_flight(&self) -> u32 {
        total_in_flight_in_tree(&self.tree)
    }
}

impl MctsEngine<OnnxPolicyValueEval> {
    pub fn search_root_history_parallel_with_progress<F>(
        &mut self,
        history: &PositionHistory,
        budget: MctsBudget,
        threads: usize,
        info_interval: Duration,
        mut on_progress: F,
    ) -> Result<MctsSearchResult, String>
    where
        F: FnMut(&MctsSearchProgress),
    {
        if threads <= 1 {
            return self.search_root_history_with_progress(history, budget, info_interval, on_progress);
        }

        let Some(root_id) = self.prepare_root(history)? else {
            return Ok(MctsSearchResult::default());
        };
        let root_history = history.clone_for_search();
        let shared_policy = self.evaluator.policy.clone();
        let shared_cache = self.evaluator.shared_cache();
        let search_batch_size = self.config.search_batch_size.max(1);
        let stop = budget.stop.clone().unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let playouts = Arc::new(AtomicU32::new(0));
        let reserved_playouts = Arc::new(AtomicU32::new(0));
        let seldepth = Arc::new(AtomicU32::new(0));
        let active_workers = Arc::new(AtomicUsize::new(threads));
        let first_error = Arc::new(Mutex::new(None::<String>));
        let tree = Mutex::new(&mut self.tree);
        let config = self.config;
        let coordinator = EvalCoordinator::new();
        let thread_count = threads.max(1);
        let worker_pipeline_depth = worker_pipeline_depth(search_batch_size, thread_count);
        let eval_uses_gpu = shared_policy
            .as_ref()
            .map(|policy_pool| policy_uses_gpu(policy_pool))
            .unwrap_or(false);
        let eval_thread_count = if let Some(policy_pool) = shared_policy.as_ref() {
            if policy_uses_gpu(policy_pool) {
                1
            } else {
                thread_count.min(MAX_EVAL_CONSUMERS).max(1)
            }
        } else {
            1
        };
        let eval_sessions = if let Some(policy_pool) = shared_policy.as_ref() {
            policy_pool
                .resize_sessions(eval_thread_count)
                .map_err(|e| e.to_string())?
        } else {
            Vec::new()
        };
        let max_eval_requests = thread_count.saturating_mul(worker_pipeline_depth).max(1);
        let max_eval_tasks = (search_batch_size * max_eval_requests).min(262_144);
        let eval_chunk_cap = eval_chunk_limit(search_batch_size, thread_count, eval_uses_gpu);

        thread::scope(|scope| {
            let eval_consumers = if eval_sessions.is_empty() {
                vec![None]
            } else {
                eval_sessions.into_iter().map(Some).collect::<Vec<_>>()
            };
            for session in eval_consumers {
                let coordinator = Arc::clone(&coordinator);
                let stop = Arc::clone(&stop);
                let first_error = Arc::clone(&first_error);
                let shared_policy = shared_policy.clone();
                let shared_cache = shared_cache.clone();
                scope.spawn(move || {
                    let mut shared_eval = match session {
                        Some(session) => OnnxPolicyValueEval::with_session(shared_policy, Some(session), shared_cache),
                        None => OnnxPolicyValueEval::with_shared_cache(shared_policy, shared_cache),
                    };
                    while let Some(requests) = coordinator.drain_batch(
                        max_eval_requests,
                        dynamic_eval_target(search_batch_size, thread_count),
                        max_eval_tasks,
                    ) {
                        let total_tasks = requests.iter().map(|req| req.tasks.len()).sum::<usize>();
                        if total_tasks == 0 {
                            for request in requests {
                                request.complete(Ok(Vec::new()));
                            }
                            continue;
                        }

                        let mut flat_tasks = Vec::with_capacity(total_tasks);
                        let mut request_lens = Vec::with_capacity(requests.len());
                        for request in &requests {
                            flat_tasks.extend(request.tasks.iter().cloned());
                            request_lens.push(request.tasks.len());
                        }

                        match evaluate_flat_tasks_chunked(&mut shared_eval, &flat_tasks, eval_chunk_cap, &stop) {
                            Ok(outputs) => {
                                debug_assert_eq!(outputs.len(), request_lens.iter().sum::<usize>());
                                let mut outputs = outputs.into_iter();
                                for (request, len) in requests.into_iter().zip(request_lens.into_iter()) {
                                    request.complete(Ok(outputs.by_ref().take(len).collect()));
                                }
                            }
                            Err(err) => {
                                if err != STOPPED_EVAL {
                                    *first_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(err.clone());
                                    stop.store(true, Ordering::SeqCst);
                                }
                                for request in requests {
                                    request.complete(Err(err.clone()));
                                }
                                break;
                            }
                        }
                    }
                });
            }

            for _ in 0..thread_count {
                let stop = Arc::clone(&stop);
                let playouts = Arc::clone(&playouts);
                let reserved_playouts = Arc::clone(&reserved_playouts);
                let seldepth = Arc::clone(&seldepth);
                let active_workers = Arc::clone(&active_workers);
                let first_error = Arc::clone(&first_error);
                let root_history = root_history.clone_for_search();
                let budget = budget.clone();
                let coordinator = Arc::clone(&coordinator);
                let tree = &tree;
                scope.spawn(move || {
                    let mut pending = VecDeque::<PendingEval>::new();
                    loop {
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                        if pending.len() < worker_pipeline_depth && !stop.load(Ordering::SeqCst) {
                            let maybe_iteration = {
                                let started_playouts = playouts
                                    .load(Ordering::Relaxed)
                                    .saturating_add(reserved_playouts.load(Ordering::Relaxed));
                                let mut guard = tree.lock().unwrap_or_else(|e| e.into_inner());
                                let tree_ref: &mut MctsTree = &mut **guard;
                                if budget_exhausted(&budget, started_playouts, tree_ref.len()) {
                                    None
                                } else {
                                    let raw_limit = remaining_batch_capacity(
                                        &budget,
                                        started_playouts,
                                        tree_ref.len(),
                                        config.search_batch_size,
                                    )
                                    .unwrap_or(config.search_batch_size)
                                    .max(1);
                                    let local_chunk_limit = parallel_gather_limit(
                                        parallel_gather_batch_limit_for_root_visits(
                                            tree_ref.get(root_id).map(|root| root.visits).unwrap_or(0),
                                            raw_limit,
                                            thread_count,
                                        ),
                                        thread_count,
                                    );
                                    drop(guard);
                                    let iteration = gather_iteration_incremental_with(
                                        tree,
                                        config,
                                        root_id,
                                        &root_history,
                                        &budget,
                                        started_playouts,
                                        local_chunk_limit,
                                    );
                                    (iteration.playouts > 0).then_some(iteration)
                                }
                            };
                            if let Some(iteration) = maybe_iteration {
                                let iter_playouts = iteration.playouts;
                                seldepth.fetch_max(iteration.seldepth, Ordering::Relaxed);
                                let tasks = iteration
                                    .pending
                                    .iter()
                                    .filter_map(|entry| match &entry.kind {
                                        PendingKind::Expand { task } => Some(Arc::clone(task)),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>();
                                let request = if tasks.is_empty() {
                                    let request = EvalRequest::new(Vec::new());
                                    request.complete(Ok(Vec::new()));
                                    request
                                } else {
                                    let request = EvalRequest::new(tasks);
                                    coordinator.submit(Arc::clone(&request));
                                    request
                                };
                                reserved_playouts.fetch_add(iter_playouts, Ordering::Relaxed);
                                pending.push_back(PendingEval {
                                    iteration,
                                    request,
                                    playouts: iter_playouts,
                                });
                                if pending.len() < worker_pipeline_depth && !stop.load(Ordering::SeqCst) {
                                    continue;
                                }
                            }
                        }

                        let mut completed = collect_ready_evals(&mut pending);
                        if completed.is_empty() {
                            let Some(next_eval) = pending.pop_front() else {
                                break;
                            };
                            let Some(result) = next_eval.request.wait_result_until_stop(&stop) else {
                                pending.push_front(next_eval);
                                cancel_pending_evals(&mut pending, tree, &reserved_playouts);
                                break;
                            };
                            completed.push((next_eval, result));
                            completed.extend(collect_ready_evals(&mut pending));
                        }

                        if !apply_completed_batch(
                            completed,
                            &first_error,
                            &stop,
                            tree,
                            &reserved_playouts,
                            &playouts,
                        ) {
                            break;
                        }
                    }
                    if !pending.is_empty() {
                        if stop.load(Ordering::SeqCst) {
                            cancel_pending_evals(&mut pending, tree, &reserved_playouts);
                        } else {
                            let mut completed = Vec::with_capacity(pending.len());
                            while let Some(next_eval) = pending.pop_front() {
                                let result = next_eval.request.wait_result();
                                completed.push((next_eval, result));
                            }
                            let _ = apply_completed_batch(
                                completed,
                                &first_error,
                                &stop,
                                tree,
                                &reserved_playouts,
                                &playouts,
                            );
                        }
                    }
                    active_workers.fetch_sub(1, Ordering::SeqCst);
                });
            }

            let mut next_report_at = if info_interval.is_zero() {
                None
            } else {
                Some(Instant::now() + info_interval)
            };
            while active_workers.load(Ordering::SeqCst) > 0 {
                if let Some(deadline) = next_report_at {
                    let now = Instant::now();
                    if now < deadline {
                        thread::sleep(deadline - now);
                    }
                    let snapshot = {
                        let guard = tree.lock().unwrap_or_else(|e| e.into_inner());
                        let tree_ref: &MctsTree = &guard;
                        let current_playouts = playouts.load(Ordering::Relaxed);
                        let current_seldepth = seldepth.load(Ordering::Relaxed);
                        (current_playouts > 0)
                            .then(|| progress_from_tree(tree_ref, root_id, current_playouts, current_seldepth))
                    };
                    if let Some(snapshot) = snapshot {
                        on_progress(&snapshot);
                    }
                    next_report_at = Some(Instant::now() + info_interval);
                } else {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            coordinator.shutdown();
        });

        if let Some(err) = first_error.lock().unwrap_or_else(|e| e.into_inner()).take() {
            return Err(err);
        }
        debug_assert_eq!(self.total_in_flight(), 0);
        let elapsed_playouts = playouts.load(Ordering::Relaxed);
        let elapsed_seldepth = seldepth.load(Ordering::Relaxed);
        Ok(result_from_tree(
            &self.tree,
            root_id,
            elapsed_playouts,
            elapsed_seldepth,
        ))
    }
}

fn policy_uses_gpu(policy_pool: &crate::policy_onnx::PolicySessionPool) -> bool {
    let chain = policy_pool.provider_chain();
    chain.contains("CUDA") || chain.contains("DirectML")
}

fn gather_iteration_with(
    tree: &mut MctsTree,
    config: MctsConfig,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    budget: &MctsBudget,
    base_playouts: u32,
    batch_limit: usize,
) -> SearchIteration {
    let mut iteration = SearchIteration::default();
    let mut slots = HashMap::<PendingKey, usize>::new();
    let mut stalled_retries = 0usize;
    let playout_limit = batch_limit
        .saturating_mul(SEARCH_COLLISION_PLAYOUT_FACTOR)
        .max(batch_limit);
    while iteration.pending.len() < batch_limit && iteration.playouts < playout_limit as u32 {
        if budget_exhausted(
            budget,
            base_playouts.saturating_add(iteration.playouts),
            tree.len().saturating_add(iteration.pending_new_nodes()),
        ) {
            break;
        }
        let Some(pending) = select_pending_with(tree, config, root_id, root_history) else {
            stalled_retries = stalled_retries.saturating_add(1);
            if stalled_retries >= playout_limit {
                break;
            }
            continue;
        };
        stalled_retries = 0;
        iteration.playouts = iteration.playouts.saturating_add(1);

        if let Some(&slot) = slots.get(&pending.key) {
            iteration.pending[slot].multivisit = iteration.pending[slot].multivisit.saturating_add(1);
            continue;
        }

        let slot = iteration.pending.len();
        slots.insert(pending.key.clone(), slot);
        iteration.seldepth = iteration.seldepth.max(pending.path.len() as u32);
        iteration.pending.push(pending);
    }

    iteration
}

fn gather_iteration_incremental_with(
    tree: &Mutex<&mut MctsTree>,
    config: MctsConfig,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    budget: &MctsBudget,
    base_playouts: u32,
    batch_limit: usize,
) -> SearchIteration {
    let mut iteration = SearchIteration::default();
    let mut slots = HashMap::<PendingKey, usize>::new();
    let mut stalled_retries = 0usize;
    let playout_limit = batch_limit
        .saturating_mul(SEARCH_COLLISION_PLAYOUT_FACTOR)
        .max(batch_limit);

    while iteration.pending.len() < batch_limit && iteration.playouts < playout_limit as u32 {
        let pending = {
            let mut guard = tree.lock().unwrap_or_else(|e| e.into_inner());
            let tree_ref: &mut MctsTree = &mut **guard;
            if budget_exhausted(
                budget,
                base_playouts.saturating_add(iteration.playouts),
                tree_ref.len().saturating_add(iteration.pending_new_nodes()),
            ) {
                return iteration;
            }
            select_pending_with(tree_ref, config, root_id, root_history)
        };

        let Some(pending) = pending else {
            stalled_retries = stalled_retries.saturating_add(1);
            if stalled_retries >= playout_limit {
                break;
            }
            continue;
        };

        stalled_retries = 0;
        iteration.playouts = iteration.playouts.saturating_add(1);

        if let Some(&slot) = slots.get(&pending.key) {
            iteration.pending[slot].multivisit = iteration.pending[slot].multivisit.saturating_add(1);
            continue;
        }

        let slot = iteration.pending.len();
        slots.insert(pending.key.clone(), slot);
        iteration.seldepth = iteration.seldepth.max(pending.path.len() as u32);
        iteration.pending.push(pending);
    }

    iteration
}

fn select_pending_with(
    tree: &mut MctsTree,
    config: MctsConfig,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
) -> Option<PendingNode> {
    let mut path = Vec::<PathStep>::new();
    let mut node_id = root_id;
    let mut pos = root_history.current().clone_for_search();
    let mut history = root_history.clone_for_search();

    loop {
        let node = tree.get(node_id).expect("selected node must exist");
        if let Some(value) = node.terminal_value {
            return Some(PendingNode {
                key: PendingKey::ExistingLeaf(node_id),
                path,
                kind: PendingKind::ExistingTerminal {
                    leaf_id: node_id,
                    value,
                },
                multivisit: 1,
            });
        }

        let edge_idx = select_edge(node, config, path.is_empty());
        let mv = node.children[edge_idx].mv;
        let child_id = node.children[edge_idx].child;

        let parent_id = node_id;
        let edge = &mut tree.get_mut(parent_id).expect("selected node must exist").children[edge_idx];
        if child_id.is_none() && edge.expanding {
            return None;
        }
        edge.in_flight = edge.in_flight.saturating_add(1);
        if child_id.is_none() {
            edge.expanding = true;
        }

        pos.do_move(mv);
        history.push_search_position(pos.clone_for_search());
        path.push(PathStep {
            node_id: parent_id,
            edge_idx,
        });

        if history.current_is_repeated() {
            return Some(PendingNode {
                key: PendingKey::NewEdge(parent_id, edge_idx),
                path,
                kind: PendingKind::NewTerminal {
                    state_key: pos.key(),
                    value: 0.0,
                },
                multivisit: 1,
            });
        }

        if let Some(child_id) = child_id {
            node_id = child_id;
            continue;
        }

        let mut buf = [ExtMove {
            mv: Move::none(),
            value: 0,
        }; MAX_MOVES];
        let n = generate(&pos, GenType::Legal, &mut buf);
        if n == 0 {
            return Some(PendingNode {
                key: PendingKey::NewEdge(parent_id, edge_idx),
                path,
                kind: PendingKind::NewTerminal {
                    state_key: pos.key(),
                    value: -1.0,
                },
                multivisit: 1,
            });
        }

        let legal_moves = buf[..n].iter().map(|e| e.mv).collect::<Vec<_>>();
        return Some(PendingNode {
            key: PendingKey::NewEdge(parent_id, edge_idx),
            path,
            kind: PendingKind::Expand {
                task: Arc::new(PolicyValueTask {
                    position: pos,
                    history,
                    legal_moves,
                }),
            },
            multivisit: 1,
        });
    }
}

fn apply_iteration_outputs_with(tree: &mut MctsTree, iteration: SearchIteration, outputs: &[PolicyValueOutput]) {
    let mut eval_cursor = 0usize;
    for pending in iteration.pending {
        match pending.kind {
            PendingKind::ExistingTerminal { leaf_id, value } => {
                add_leaf_visit_with(tree, leaf_id, value, pending.multivisit);
                backup_path_with(tree, &pending.path, value, pending.multivisit);
            }
            PendingKind::NewTerminal { state_key, value } => {
                let parent = pending.path.last().expect("new terminal must have parent");
                add_terminal_child_with(
                    tree,
                    parent.node_id,
                    parent.edge_idx,
                    state_key,
                    value,
                    pending.multivisit,
                );
                backup_path_with(tree, &pending.path, value, pending.multivisit);
            }
            PendingKind::Expand { task } => {
                let out = outputs.get(eval_cursor).expect("batched eval must match task count");
                eval_cursor += 1;
                let parent = pending.path.last().expect("expanded leaf must have parent");
                add_expanded_child_with(
                    tree,
                    parent.node_id,
                    parent.edge_idx,
                    task.as_ref(),
                    out,
                    pending.multivisit,
                );
                backup_path_with(tree, &pending.path, out.value, pending.multivisit);
            }
        }
    }
}

fn cancel_iteration_with(tree: &mut MctsTree, iteration: SearchIteration) {
    for pending in iteration.pending {
        cancel_pending_with(tree, &pending);
    }
}

fn cancel_pending_with(tree: &mut MctsTree, pending: &PendingNode) {
    for step in pending.path.iter().rev() {
        let node = tree.get_mut(step.node_id).expect("path node must exist");
        let edge = &mut node.children[step.edge_idx];
        edge.in_flight = edge.in_flight.saturating_sub(pending.multivisit);
    }
    if matches!(pending.kind, PendingKind::NewTerminal { .. } | PendingKind::Expand { .. }) {
        if let Some(parent) = pending.path.last() {
            let node = tree.get_mut(parent.node_id).expect("path node must exist");
            let edge = &mut node.children[parent.edge_idx];
            if edge.child.is_none() {
                edge.expanding = false;
            }
        }
    }
}

fn add_expanded_child_with(
    tree: &mut MctsTree,
    parent_id: MctsNodeId,
    edge_idx: usize,
    task: &PolicyValueTask,
    out: &PolicyValueOutput,
    multivisit: u32,
) -> MctsNodeId {
    let mut node = MctsNode {
        state_key: task.position.key(),
        visits: multivisit,
        value_sum: out.value * multivisit as f32,
        expanded: true,
        terminal_value: None,
        children: Vec::with_capacity(task.legal_moves.len()),
    };
    for (i, mv) in task.legal_moves.iter().copied().enumerate() {
        node.children.push(super::EdgeStats {
            mv,
            prior: out.priors.get(i).copied().unwrap_or(0.0),
            visits: 0,
            in_flight: 0,
            expanding: false,
            value_sum: 0.0,
            child: None,
        });
    }
    let child_id = tree.add_node(node);
    let edge = &mut tree.get_mut(parent_id).expect("parent node must exist").children[edge_idx];
    edge.child = Some(child_id);
    edge.expanding = false;
    child_id
}

fn add_terminal_child_with(
    tree: &mut MctsTree,
    parent_id: MctsNodeId,
    edge_idx: usize,
    state_key: u64,
    value: f32,
    multivisit: u32,
) -> MctsNodeId {
    let child_id = tree.add_node(MctsNode {
        state_key,
        visits: multivisit,
        value_sum: value * multivisit as f32,
        expanded: true,
        terminal_value: Some(value),
        children: Vec::new(),
    });
    let edge = &mut tree.get_mut(parent_id).expect("parent node must exist").children[edge_idx];
    edge.child = Some(child_id);
    edge.expanding = false;
    child_id
}

fn add_leaf_visit_with(tree: &mut MctsTree, leaf_id: MctsNodeId, value: f32, multivisit: u32) {
    if multivisit == 0 {
        return;
    }
    let leaf = tree.get_mut(leaf_id).expect("leaf node must exist");
    leaf.visits = leaf.visits.saturating_add(multivisit);
    leaf.value_sum += value * multivisit as f32;
}

fn backup_path_with(tree: &mut MctsTree, path: &[PathStep], mut value: f32, multivisit: u32) {
    for step in path.iter().rev() {
        value = -value;
        let signed_delta = value * multivisit as f32;
        let node = tree.get_mut(step.node_id).expect("path node must exist");
        node.visits = node.visits.saturating_add(multivisit);
        node.value_sum += signed_delta;
        let edge = &mut node.children[step.edge_idx];
        edge.visits = edge.visits.saturating_add(multivisit);
        edge.value_sum += signed_delta;
        edge.in_flight = edge.in_flight.saturating_sub(multivisit);
    }
}

fn progress_from_tree(tree: &MctsTree, root_id: MctsNodeId, playouts: u32, seldepth: u32) -> MctsSearchProgress {
    let root = tree.get(root_id).expect("root must exist");
    let summary = pv_summary_from_tree(tree, root_id);
    MctsSearchProgress {
        best_move: summary.best_move,
        pv: summary.pv,
        playouts,
        root_visits: root.visits,
        nodes: tree.len(),
        depth: depth_from_tree(tree, root_id),
        seldepth,
        root_value: root.mean_value(),
        best_value: summary.best_value,
        best_mate: summary.best_mate,
        moves: root
            .children
            .iter()
            .map(|edge| MctsMoveStat {
                mv: edge.mv,
                prior: edge.prior,
                visits: edge.visits,
                q: edge.mean_q(),
            })
            .collect(),
    }
}

fn result_from_tree(tree: &MctsTree, root_id: MctsNodeId, playouts: u32, seldepth: u32) -> MctsSearchResult {
    let root = tree.get(root_id).expect("root must exist");
    let progress = progress_from_tree(tree, root_id, playouts, seldepth);
    MctsSearchResult {
        best_move: progress.best_move,
        pv: progress.pv,
        playouts: progress.playouts,
        root_visits: progress.root_visits,
        nodes: progress.nodes,
        depth: progress.depth,
        seldepth: progress.seldepth,
        root_value: progress.root_value,
        best_value: progress.best_value,
        best_mate: progress.best_mate,
        moves: root
            .children
            .iter()
            .map(|edge| MctsMoveStat {
                mv: edge.mv,
                prior: edge.prior,
                visits: edge.visits,
                q: edge.mean_q(),
            })
            .collect(),
    }
}

fn extract_pv_from_tree(tree: &MctsTree, root_id: MctsNodeId) -> Vec<Move> {
    pv_summary_from_tree(tree, root_id).pv
}

fn pv_summary_from_tree(tree: &MctsTree, root_id: MctsNodeId) -> PvSummary {
    let mut pv = Vec::new();
    let mut node_id = root_id;
    let mut best_value = tree.get(root_id).map(MctsNode::mean_value).unwrap_or(0.0);
    let mut best_mate = None;
    let mut ply = 0usize;
    while let Some(node) = tree.get(node_id) {
        let Some(edge) = node.children.iter().max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then_with(|| a.mean_q().partial_cmp(&b.mean_q()).unwrap_or(std::cmp::Ordering::Equal))
        }) else {
            break;
        };
        if edge.visits == 0 {
            break;
        }
        if pv.is_empty() {
            best_value = edge.mean_q();
        }
        pv.push(edge.mv);
        let Some(child_id) = edge.child else {
            break;
        };
        ply += 1;
        if let Some(terminal_value) = tree.get(child_id).and_then(|child| child.terminal_value) {
            let root_q = if ply % 2 == 0 { terminal_value } else { -terminal_value };
            if best_value.abs() >= 0.999 && root_q.signum() == best_value.signum() {
                let mate_moves = ((ply as i32) + 1) / 2;
                best_mate = Some(if root_q > 0.0 { mate_moves } else { -mate_moves });
            }
            break;
        }
        node_id = child_id;
    }
    PvSummary {
        best_move: pv.first().copied(),
        pv,
        best_value,
        best_mate,
    }
}

fn total_in_flight_in_tree(tree: &MctsTree) -> u32 {
    let mut total = 0u32;
    for idx in 0..tree.len() {
        let Some(node) = tree.get(MctsNodeId(idx)) else {
            continue;
        };
        total = total.saturating_add(node.children.iter().map(|edge| edge.in_flight).sum::<u32>());
    }
    total
}

fn depth_from_tree(tree: &MctsTree, root_id: MctsNodeId) -> u32 {
    extract_pv_from_tree(tree, root_id).len() as u32
}

fn budget_exhausted(budget: &MctsBudget, playouts: u32, nodes: usize) -> bool {
    if let Some(target_playouts) = budget.max_playouts {
        if playouts >= target_playouts.max(1) {
            return true;
        }
    }
    if let Some(target_nodes) = budget.max_nodes {
        if nodes >= target_nodes as usize {
            return true;
        }
    }
    if let Some(deadline) = budget.deadline {
        if Instant::now() >= deadline {
            return true;
        }
    }
    if let Some(stop) = budget.stop.as_ref() {
        if stop.load(std::sync::atomic::Ordering::SeqCst) {
            return true;
        }
    }
    false
}

fn remaining_batch_capacity(budget: &MctsBudget, playouts: u32, nodes: usize, batch_size: usize) -> Option<usize> {
    let remaining_playouts = budget
        .max_playouts
        .map(|limit| limit.saturating_sub(playouts) as usize)
        .unwrap_or(batch_size);
    let remaining_nodes = budget
        .max_nodes
        .map(|limit| limit.saturating_sub(nodes as u32) as usize)
        .unwrap_or(batch_size);
    let cap = remaining_playouts
        .min(remaining_nodes)
        .min(batch_size.min(SEARCH_BATCH_SIZE_CAP));
    if cap == 0 {
        None
    } else {
        Some(cap)
    }
}

fn appended_history_moves(old_history: &PositionHistory, new_history: &PositionHistory) -> Option<Vec<Move>> {
    let old_positions = old_history.positions().collect::<Vec<_>>();
    let new_positions = new_history.positions().collect::<Vec<_>>();
    let old_fens = old_positions.iter().map(|pos| pos.fen()).collect::<Vec<_>>();
    let new_fens = new_positions.iter().map(|pos| pos.fen()).collect::<Vec<_>>();
    if old_positions.is_empty() || new_positions.len() < 2 {
        return None;
    }

    let max_overlap = old_positions.len().min(new_positions.len());
    let overlap = (1..=max_overlap).rev().find(|&len| {
        old_fens[old_fens.len() - len..]
            .iter()
            .zip(new_fens[..len].iter())
            .all(|(a, b)| a == b)
    })?;
    if overlap == new_positions.len() {
        return Some(Vec::new());
    }

    let mut path = Vec::with_capacity(new_positions.len() - overlap);
    for idx in (overlap - 1)..(new_positions.len() - 1) {
        let mv = transition_move(new_positions[idx], new_positions[idx + 1])?;
        path.push(mv);
    }
    Some(path)
}

fn transition_move(from: &Position, to: &Position) -> Option<Move> {
    let target_fen = to.fen();
    let mut buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(from, GenType::Legal, &mut buf);
    for edge in &buf[..n] {
        let mv = edge.mv;
        let mut next = from.clone_for_search();
        next.do_move(mv);
        if next.fen() == target_fen {
            return Some(mv);
        }
    }
    None
}

fn select_edge(node: &MctsNode, config: MctsConfig, is_root: bool) -> usize {
    let parent_effective_visits = node
        .children
        .iter()
        .map(|edge| edge.visits.saturating_add(edge.in_flight))
        .sum::<u32>()
        .max(1) as f32;
    let sqrt_parent = parent_effective_visits.sqrt();
    let parent_q = node.mean_value();
    let cpuct = config.cpuct_for(is_root, node.visits);
    let visited_policy = node
        .children
        .iter()
        .filter(|edge| edge.visits.saturating_add(edge.in_flight) > 0)
        .map(|edge| edge.prior)
        .sum::<f32>()
        .clamp(0.0, 1.0);
    let fpu_reduction = config.fpu_for(is_root) * visited_policy.sqrt();
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let has_selectable_idle_edge = node.children.iter().any(|edge| edge.child.is_some() || !edge.expanding);

    for (idx, edge) in node.children.iter().enumerate() {
        if has_selectable_idle_edge && edge.child.is_none() && edge.expanding {
            continue;
        }
        let q = if edge.visits == 0 {
            (parent_q - fpu_reduction).clamp(-1.0, 1.0)
        } else {
            edge.mean_q()
        };
        let effective_visits = edge.visits.saturating_add(edge.in_flight) as f32;
        let u = cpuct * edge.prior * sqrt_parent / (1.0 + effective_visits);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    best_idx
}

#[derive(Clone, Copy, Debug)]
struct PathStep {
    node_id: MctsNodeId,
    edge_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PendingKey {
    ExistingLeaf(MctsNodeId),
    NewEdge(MctsNodeId, usize),
}

#[derive(Clone)]
enum PendingKind {
    ExistingTerminal { leaf_id: MctsNodeId, value: f32 },
    NewTerminal { state_key: u64, value: f32 },
    Expand { task: Arc<PolicyValueTask> },
}

#[derive(Clone)]
struct PendingNode {
    key: PendingKey,
    path: Vec<PathStep>,
    kind: PendingKind,
    multivisit: u32,
}

#[derive(Default)]
struct SearchIteration {
    pending: Vec<PendingNode>,
    playouts: u32,
    seldepth: u32,
}

#[derive(Default)]
struct PvSummary {
    best_move: Option<Move>,
    pv: Vec<Move>,
    best_value: f32,
    best_mate: Option<i32>,
}

impl SearchIteration {
    fn pending_new_nodes(&self) -> usize {
        self.pending
            .iter()
            .filter(|pending| {
                matches!(
                    pending.kind,
                    PendingKind::NewTerminal { .. } | PendingKind::Expand { .. }
                )
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiangqi_core::uci_to_move;
    use xiangqi_core::Position;

    #[derive(Default)]
    struct StubEval;

    impl PolicyValueEval for StubEval {
        type Error = String;

        fn evaluate(&mut self, input: PolicyValueInput<'_>) -> Result<super::super::PolicyValueOutput, Self::Error> {
            Ok(super::super::PolicyValueOutput {
                priors: vec![1.0 / input.legal_moves.len() as f32; input.legal_moves.len()],
                value: 0.0,
            })
        }
    }

    #[test]
    fn gather_iteration_spreads_root_edges() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let batch_size = engine.config.search_batch_size;
        let iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, batch_size);
        assert!(!iteration.pending.is_empty());
        assert!(iteration.pending.len() <= batch_size);
        assert!(iteration.playouts >= iteration.pending.len() as u32);

        let mut unique = std::collections::HashSet::new();
        for pending in &iteration.pending {
            unique.insert(pending.path.first().expect("root edge").edge_idx);
        }
        assert!(unique.len() > 1, "in-flight should spread root selections");
    }

    #[test]
    fn gather_batch_limit_ramps_up_with_root_visits() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        assert_eq!(engine.gather_batch_limit(root_id, 32), 1);
        engine.tree.get_mut(root_id).expect("root").visits = 8;
        assert_eq!(engine.gather_batch_limit(root_id, 32), 4);
        engine.tree.get_mut(root_id).expect("root").visits = 32;
        assert_eq!(engine.gather_batch_limit(root_id, 32), 8);
        engine.tree.get_mut(root_id).expect("root").visits = 128;
        assert_eq!(engine.gather_batch_limit(root_id, 32), 16);
        engine.tree.get_mut(root_id).expect("root").visits = 512;
        assert_eq!(engine.gather_batch_limit(root_id, 32), 32);
        engine.tree.get_mut(root_id).expect("root").visits = 8_192;
        assert_eq!(engine.gather_batch_limit(root_id, 2_048), 128);
        engine.tree.get_mut(root_id).expect("root").visits = 131_072;
        assert_eq!(engine.gather_batch_limit(root_id, 2_048), 512);
    }

    #[test]
    fn parallel_gather_batch_limit_opens_faster_for_workers() {
        assert_eq!(parallel_gather_batch_limit_for_root_visits(0, 512, 8), 32);
        assert_eq!(parallel_gather_batch_limit_for_root_visits(16, 512, 8), 32);
        assert_eq!(parallel_gather_batch_limit_for_root_visits(64, 512, 8), 64);
        assert_eq!(parallel_gather_batch_limit_for_root_visits(256, 512, 8), 16);
        assert_eq!(parallel_gather_batch_limit_for_root_visits(8_192, 2_048, 8), 128);
    }

    #[test]
    fn dynamic_eval_target_prefers_partial_batches() {
        assert_eq!(dynamic_eval_target(64, 8), 64);
        assert_eq!(dynamic_eval_target(256, 8), 256);
        assert_eq!(dynamic_eval_target(2_048, 8), 2_048);
        assert_eq!(dynamic_eval_target(2_048, 4), 1_024);
        assert_eq!(dynamic_eval_target(8_192, 8), 2_048);
        assert_eq!(dynamic_eval_target(8_192, 16), 4_096);
    }

    #[test]
    fn worker_pipeline_depth_scales_with_batch_cap() {
        assert_eq!(worker_pipeline_depth(64, 8), 1);
        assert_eq!(worker_pipeline_depth(256, 1), 1);
        assert_eq!(worker_pipeline_depth(256, 8), 2);
        assert_eq!(worker_pipeline_depth(512, 8), 4);
        assert_eq!(worker_pipeline_depth(2_048, 16), 4);
    }

    #[test]
    fn parallel_gather_limit_keeps_worker_chunks_small() {
        assert_eq!(parallel_gather_limit(32, 8), 4);
        assert_eq!(parallel_gather_limit(256, 8), 32);
        assert_eq!(parallel_gather_limit(512, 8), 64);
        assert_eq!(parallel_gather_limit(2_048, 8), 128);
        assert_eq!(parallel_gather_limit(2_048, 1), 2_048);
    }

    #[test]
    fn eval_chunk_limit_caps_gpu_batches_but_not_cpu_batches() {
        assert_eq!(eval_chunk_limit(64, 8, true), 32);
        assert_eq!(eval_chunk_limit(512, 8, true), 32);
        assert_eq!(eval_chunk_limit(2_048, 8, true), 32);
        assert_eq!(eval_chunk_limit(512, 8, false), dynamic_eval_target(512, 8));
    }

    #[test]
    fn select_edge_avoids_expanding_new_edge_when_idle_alternative_exists() {
        let mut node = MctsNode {
            state_key: 1,
            visits: 32,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: Vec::new(),
        };
        node.children.push(super::super::EdgeStats {
            mv: Move::make(xiangqi_core::types::Square::SQ_A0, xiangqi_core::types::Square::SQ_A1),
            prior: 0.9,
            visits: 0,
            in_flight: 1,
            expanding: true,
            value_sum: 0.0,
            child: None,
        });
        node.children.push(super::super::EdgeStats {
            mv: Move::make(xiangqi_core::types::Square::SQ_B0, xiangqi_core::types::Square::SQ_B1),
            prior: 0.1,
            visits: 0,
            in_flight: 0,
            expanding: false,
            value_sum: 0.0,
            child: None,
        });
        assert_eq!(select_edge(&node, MctsConfig::default(), true), 1);
    }

    #[test]
    fn apply_iteration_clears_inflight_and_updates_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        engine.apply_iteration(iteration).expect("apply ok");

        let root = engine.tree.get(root_id).expect("root");
        assert_eq!(root.visits, 4);
        assert!(root.children.iter().all(|edge| edge.in_flight == 0));
    }

    #[test]
    fn cancel_iteration_clears_inflight_and_expanding_flags() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert!(iteration.pending.iter().any(|pending| {
            matches!(pending.kind, PendingKind::NewTerminal { .. } | PendingKind::Expand { .. })
        }));

        cancel_iteration_with(&mut engine.tree, iteration);

        let root = engine.tree.get(root_id).expect("root");
        assert!(root.children.iter().all(|edge| edge.in_flight == 0));
        assert!(root.children.iter().all(|edge| !edge.expanding));
    }

    #[test]
    fn collision_merges_existing_terminal_leaf_multivisit() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let mv = uci_to_move(pos, "h2e2").expect("legal move");

        engine.tree.clear();
        let leaf_id = engine.tree.add_node(MctsNode {
            state_key: 42,
            visits: 0,
            value_sum: 0.0,
            expanded: true,
            terminal_value: Some(1.0),
            children: Vec::new(),
        });
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 0,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv,
                prior: 1.0,
                visits: 0,
                in_flight: 0,
                expanding: false,
                value_sum: 0.0,
                child: Some(leaf_id),
            }],
        });

        let iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert_eq!(iteration.pending.len(), 1);
        assert_eq!(
            iteration.pending[0].multivisit,
            (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32
        );

        engine.apply_iteration(iteration).expect("apply ok");
        let root = engine.tree.get(root_id).expect("root");
        let leaf = engine.tree.get(leaf_id).expect("leaf");
        assert_eq!(root.visits, (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32);
        assert_eq!(root.children[0].visits, (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32);
        assert_eq!(root.children[0].in_flight, 0);
        assert_eq!(leaf.visits, (4 * SEARCH_COLLISION_PLAYOUT_FACTOR) as u32);
    }

    #[test]
    fn pv_ignores_inflight_only_edges() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let _iteration = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert!(engine.extract_pv(root_id).is_empty());
    }

    #[test]
    fn expanding_new_edge_is_not_queued_twice_before_backup() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let mv = uci_to_move(pos, "h2e2").expect("legal move");
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 0,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv,
                prior: 1.0,
                visits: 0,
                in_flight: 0,
                expanding: false,
                value_sum: 0.0,
                child: None,
            }],
        });

        let first = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert_eq!(first.pending.len(), 1);
        let second = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert_eq!(second.pending.len(), 0);
        assert_eq!(second.playouts, 0);

        engine.apply_iteration(first).expect("apply ok");
        let third = engine.gather_iteration(root_id, &history, &MctsBudget::default(), 0, 4);
        assert!(third.playouts > 0);
    }

    #[test]
    fn pv_summary_uses_best_edge_q_and_detects_terminal_mate() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let best_mv = uci_to_move(pos, "h2e2").expect("legal move");
        let alt_mv = uci_to_move(pos, "b2e2").expect("legal move");

        let mate_leaf = engine.tree.add_node(MctsNode {
            state_key: 1,
            visits: 1,
            value_sum: -1.0,
            expanded: true,
            terminal_value: Some(-1.0),
            children: Vec::new(),
        });
        let alt_leaf = engine.tree.add_node(MctsNode {
            state_key: 2,
            visits: 10,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: Vec::new(),
        });
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 11,
            value_sum: -0.1,
            expanded: true,
            terminal_value: None,
            children: vec![
                super::super::EdgeStats {
                    mv: best_mv,
                    prior: 0.5,
                    visits: 1,
                    in_flight: 0,
                    expanding: false,
                    value_sum: 1.0,
                    child: Some(mate_leaf),
                },
                super::super::EdgeStats {
                    mv: alt_mv,
                    prior: 0.5,
                    visits: 10,
                    in_flight: 0,
                    expanding: false,
                    value_sum: 0.0,
                    child: Some(alt_leaf),
                },
            ],
        });

        let summary = engine.pv_summary(root_id);
        assert_eq!(summary.best_move, Some(alt_mv));
        assert_eq!(summary.best_value, 0.0);
        assert_eq!(summary.best_mate, None);

        let root = engine.tree.get_mut(root_id).expect("root");
        root.children[0].visits = 12;
        root.children[0].value_sum = 12.0;
        let summary = engine.pv_summary(root_id);
        assert_eq!(summary.best_move, Some(best_mv));
        assert_eq!(summary.best_value, 1.0);
        assert_eq!(summary.best_mate, Some(1));
    }

    #[test]
    fn pv_summary_does_not_report_mate_for_unsolved_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let pos = history.current();
        let best_mv = uci_to_move(pos, "h2e2").expect("legal move");
        let reply_mv = uci_to_move(pos, "b2e2").expect("legal move");

        let terminal_leaf = engine.tree.add_node(MctsNode {
            state_key: 11,
            visits: 1,
            value_sum: -1.0,
            expanded: true,
            terminal_value: Some(-1.0),
            children: Vec::new(),
        });
        let reply_node = engine.tree.add_node(MctsNode {
            state_key: 12,
            visits: 8,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv: reply_mv,
                prior: 1.0,
                visits: 8,
                in_flight: 0,
                expanding: false,
                value_sum: 0.0,
                child: Some(terminal_leaf),
            }],
        });
        let root_id = engine.tree.add_node(MctsNode {
            state_key: pos.key(),
            visits: 16,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv: best_mv,
                prior: 1.0,
                visits: 16,
                in_flight: 0,
                expanding: false,
                value_sum: 3.2,
                child: Some(reply_node),
            }],
        });

        let summary = engine.pv_summary(root_id);
        assert_eq!(summary.best_move, Some(best_mv));
        assert!((summary.best_value - 0.2).abs() < 1e-6);
        assert_eq!(summary.best_mate, None);
    }

    #[test]
    fn search_smoke_returns_legal_move() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let pos = Position::from_fen(xiangqi_core::START_FEN).expect("start");
        let result = engine
            .search_root(
                &pos,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("search ok");
        assert!(result.best_move.is_some());
    }

    #[test]
    fn gather_iteration_respects_tight_playout_budget() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let root_id = engine.initialize_root(&history).expect("init ok").expect("root");

        let budget = MctsBudget {
            max_playouts: Some(3),
            max_nodes: None,
            deadline: None,
            stop: None,
        };
        let iteration = engine.gather_iteration(root_id, &history, &budget, 0, engine.config.search_batch_size);
        assert_eq!(iteration.playouts, 3);
    }

    #[test]
    fn repeated_search_reuses_same_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        assert_eq!(first.root_visits, 16);

        let second = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(8),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        assert_eq!(second.playouts, 8);
        assert!(
            second.root_visits >= 24,
            "root visits should continue from prior search"
        );
    }

    #[test]
    fn child_position_reuses_subtree_as_new_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(32),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        let best = first.best_move.expect("best move");

        let mut next_history = history.clone_for_search();
        next_history.push_move(best);
        let second = engine
            .search_root_history(
                &next_history,
                MctsBudget {
                    max_playouts: Some(4),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        assert!(
            second.root_visits > second.playouts,
            "new root should inherit child visits"
        );
    }

    #[test]
    fn appended_two_move_history_reuses_deeper_subtree_as_new_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        let mv1 = first.best_move.expect("best move");

        let mut history1 = history.clone_for_search();
        history1.push_move(mv1);
        let second = engine
            .search_root_history(
                &history1,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        let mv2 = second.best_move.expect("reply");

        let mut history2 = history.clone_for_search();
        history2.push_move(mv1);
        history2.push_move(mv2);
        let third = engine
            .search_root_history(
                &history2,
                MctsBudget {
                    max_playouts: Some(1),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("third");
        assert!(
            third.root_visits > third.playouts,
            "deeper appended path should retain subtree visits"
        );
    }

    #[test]
    fn transposed_history_does_not_reuse_exact_root() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);

        let mut history_a = PositionHistory::new_startpos();
        for u in ["h0g2", "h9g7", "b0c2", "b9c7"] {
            let mv = uci_to_move(history_a.current(), u).expect("legal move");
            history_a.push_move(mv);
        }
        let first = engine
            .search_root_history(
                &history_a,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        assert_eq!(first.root_visits, 16);

        let mut history_b = PositionHistory::new_startpos();
        for u in ["b0c2", "b9c7", "h0g2", "h9g7"] {
            let mv = uci_to_move(history_b.current(), u).expect("legal move");
            history_b.push_move(mv);
        }
        assert_eq!(history_a.current().fen(), history_b.current().fen());
        assert!(!history_a.same_input_window(&history_b));

        let second = engine
            .search_root_history(
                &history_b,
                MctsBudget {
                    max_playouts: Some(1),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("second");
        assert_eq!(second.root_visits, 1, "different history window must rebuild root");
    }

    #[test]
    fn repeated_leaf_is_scored_as_draw() {
        let mut history = PositionHistory::new_startpos();
        for u in ["h0g2", "h9g7", "g2h0", "g7h9"] {
            let mv = uci_to_move(history.current(), u).expect("legal move");
            history.push_move(mv);
        }
        assert!(
            history.current_is_repeated(),
            "history should already contain repeated root"
        );

        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let result = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("search");
        assert!(result.best_move.is_some());
        assert!(result.moves.iter().all(|mv| mv.q.abs() <= 1.0));
    }

    #[test]
    fn reused_tree_does_not_inflate_reported_seldepth() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let root_id = engine.tree.add_node(MctsNode {
            state_key: 1,
            visits: 32,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv: Move::make(xiangqi_core::types::Square::SQ_A0, xiangqi_core::types::Square::SQ_A1),
                prior: 1.0,
                visits: 32,
                in_flight: 0,
                expanding: false,
                value_sum: 0.0,
                child: None,
            }],
        });
        let reused_child = engine.tree.add_node(MctsNode {
            state_key: 2,
            visits: 32,
            value_sum: 0.0,
            expanded: true,
            terminal_value: None,
            children: vec![super::super::EdgeStats {
                mv: Move::make(xiangqi_core::types::Square::SQ_A1, xiangqi_core::types::Square::SQ_A2),
                prior: 1.0,
                visits: 32,
                in_flight: 0,
                expanding: false,
                value_sum: 0.0,
                child: None,
            }],
        });
        let reused_leaf = engine.tree.add_node(MctsNode {
            state_key: 3,
            visits: 32,
            value_sum: 0.0,
            expanded: true,
            terminal_value: Some(0.0),
            children: Vec::new(),
        });
        engine.tree.get_mut(root_id).expect("root").children[0].child = Some(reused_child);
        engine.tree.get_mut(reused_child).expect("child").children[0].child = Some(reused_leaf);

        let progress = engine.progress_from_root(root_id, 1, 1);
        let result = engine.result_from_root(root_id, 1, 1);
        assert_eq!(progress.seldepth, 1);
        assert_eq!(result.seldepth, 1);
        assert_eq!(progress.depth, 2, "pv depth may still reflect reused subtree");
    }

    #[test]
    fn fresh_search_root_visits_match_playouts() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let result = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(24),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("search");
        assert_eq!(result.playouts, 24);
        assert_eq!(result.root_visits, 24);
        assert!(result.nodes >= 1);
    }

    #[test]
    fn non_appended_history_does_not_reuse_root_subtree() {
        let mut engine = MctsEngine::new(MctsConfig::default(), StubEval);
        let history = PositionHistory::new_startpos();
        let first = engine
            .search_root_history(
                &history,
                MctsBudget {
                    max_playouts: Some(16),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("first");
        let mv1 = first.best_move.expect("best move");

        let mut history_a = history.clone_for_search();
        history_a.push_move(mv1);
        let _ = engine
            .search_root_history(
                &history_a,
                MctsBudget {
                    max_playouts: Some(8),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("second");

        let mut history_b = PositionHistory::new_startpos();
        let alt = uci_to_move(history_b.current(), "c3c4").expect("alt legal");
        history_b.push_move(alt);
        let third = engine
            .search_root_history(
                &history_b,
                MctsBudget {
                    max_playouts: Some(4),
                    max_nodes: None,
                    deadline: None,
                    stop: None,
                },
            )
            .expect("third");
        assert_eq!(third.root_visits, 4, "non-appended history should rebuild root");
    }
}

struct EvalRequest {
    tasks: Vec<Arc<PolicyValueTask>>,
    result: Mutex<Option<Result<Vec<PolicyValueOutput>, String>>>,
    ready: Condvar,
}

impl EvalRequest {
    fn new(tasks: Vec<Arc<PolicyValueTask>>) -> Arc<Self> {
        Arc::new(Self {
            tasks,
            result: Mutex::new(None),
            ready: Condvar::new(),
        })
    }

    fn complete(&self, result: Result<Vec<PolicyValueOutput>, String>) {
        let mut slot = self.result.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some(result);
        self.ready.notify_one();
    }

    fn wait_result(&self) -> Result<Vec<PolicyValueOutput>, String> {
        let mut slot = self.result.lock().unwrap_or_else(|e| e.into_inner());
        while slot.is_none() {
            slot = self.ready.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
        slot.take().expect("eval result must be populated")
    }

    fn wait_result_until_stop(&self, stop: &AtomicBool) -> Option<Result<Vec<PolicyValueOutput>, String>> {
        let mut slot = self.result.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(result) = slot.take() {
                return Some(result);
            }
            if stop.load(Ordering::SeqCst) {
                return None;
            }
            let (next_slot, _) = self
                .ready
                .wait_timeout(slot, Duration::from_millis(1))
                .unwrap_or_else(|e| e.into_inner());
            slot = next_slot;
        }
    }

    fn take_ready_result(&self) -> Option<Result<Vec<PolicyValueOutput>, String>> {
        let mut slot = self.result.lock().unwrap_or_else(|e| e.into_inner());
        slot.take()
    }
}

#[derive(Default)]
struct EvalCoordinatorState {
    queue: VecDeque<Arc<EvalRequest>>,
    shutdown: bool,
}

struct EvalCoordinator {
    state: Mutex<EvalCoordinatorState>,
    wake: Condvar,
}

struct PendingEval {
    iteration: SearchIteration,
    request: Arc<EvalRequest>,
    playouts: u32,
}

fn collect_ready_evals(
    pending: &mut VecDeque<PendingEval>,
) -> Vec<(PendingEval, Result<Vec<PolicyValueOutput>, String>)> {
    let mut completed = Vec::new();
    let mut idx = 0usize;
    while idx < pending.len() {
        if let Some(result) = pending[idx].request.take_ready_result() {
            if let Some(done) = pending.remove(idx) {
                completed.push((done, result));
            }
        } else {
            idx += 1;
        }
    }
    completed
}

fn cancel_pending_evals(
    pending: &mut VecDeque<PendingEval>,
    tree: &Mutex<&mut MctsTree>,
    reserved_playouts: &Arc<AtomicU32>,
) {
    if pending.is_empty() {
        return;
    }
    let canceled_playouts = pending.iter().map(|entry| entry.playouts).sum::<u32>();
    reserved_playouts.fetch_sub(canceled_playouts, Ordering::Relaxed);
    let mut guard = tree.lock().unwrap_or_else(|e| e.into_inner());
    let tree_ref: &mut MctsTree = &mut **guard;
    while let Some(next_eval) = pending.pop_front() {
        cancel_iteration_with(tree_ref, next_eval.iteration);
    }
}

fn evaluate_flat_tasks_chunked(
    eval: &mut OnnxPolicyValueEval,
    tasks: &[Arc<PolicyValueTask>],
    chunk_cap: usize,
    stop: &AtomicBool,
) -> Result<Vec<PolicyValueOutput>, String> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_cap = chunk_cap.max(1);
    let mut outputs = Vec::with_capacity(tasks.len());
    for chunk in tasks.chunks(chunk_cap) {
        if stop.load(Ordering::SeqCst) {
            return Err(STOPPED_EVAL.to_string());
        }
        outputs.extend(eval.evaluate_many_shared(chunk)?);
    }
    Ok(outputs)
}

fn apply_completed_batch(
    completed: Vec<(PendingEval, Result<Vec<PolicyValueOutput>, String>)>,
    first_error: &Arc<Mutex<Option<String>>>,
    stop: &Arc<AtomicBool>,
    tree: &Mutex<&mut MctsTree>,
    reserved_playouts: &Arc<AtomicU32>,
    playouts: &Arc<AtomicU32>,
) -> bool {
    if completed.is_empty() {
        return true;
    }

    let applied_playouts = completed
        .iter()
        .map(|(pending_eval, _)| pending_eval.playouts)
        .sum::<u32>();
    for (pending_eval, result) in &completed {
        if let Err(err) = result {
            reserved_playouts.fetch_sub(pending_eval.playouts, Ordering::Relaxed);
            if err == STOPPED_EVAL {
                let mut guard = tree.lock().unwrap_or_else(|e| e.into_inner());
                let tree_ref: &mut MctsTree = &mut **guard;
                for (pending_eval, _) in completed {
                    cancel_iteration_with(tree_ref, pending_eval.iteration);
                }
                return false;
            }
            *first_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(err.clone());
            stop.store(true, Ordering::SeqCst);
            return false;
        }
    }

    reserved_playouts.fetch_sub(applied_playouts, Ordering::Relaxed);
    playouts.fetch_add(applied_playouts, Ordering::Relaxed);

    for (pending_eval, result) in completed {
        let outputs = result.as_ref().expect("errors handled above");
        let mut guard = tree.lock().unwrap_or_else(|e| e.into_inner());
        let tree_ref: &mut MctsTree = &mut **guard;
        apply_iteration_outputs_with(tree_ref, pending_eval.iteration, outputs);
    }
    true
}

impl EvalCoordinator {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(EvalCoordinatorState::default()),
            wake: Condvar::new(),
        })
    }

    fn submit(&self, request: Arc<EvalRequest>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.queue.push_back(request);
        self.wake.notify_one();
    }

    fn drain_batch(&self, max_requests: usize, target_tasks: usize, max_tasks: usize) -> Option<Vec<Arc<EvalRequest>>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.queue.is_empty() && !state.shutdown {
            state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        if state.queue.is_empty() && state.shutdown {
            return None;
        }

        let mut drained = Vec::new();
        let mut task_count = 0usize;
        let target_tasks = target_tasks.clamp(1, max_tasks.max(1));
        let gather_deadline = Instant::now() + EVAL_GATHER_WAIT;
        loop {
            while let Some(front) = state.queue.front() {
                if !drained.is_empty()
                    && (drained.len() >= max_requests || task_count.saturating_add(front.tasks.len()) > max_tasks)
                {
                    return Some(drained);
                }
                let request = state.queue.pop_front().expect("front exists");
                task_count = task_count.saturating_add(request.tasks.len());
                drained.push(request);
                if task_count >= target_tasks || drained.len() >= max_requests {
                    return Some(drained);
                }
            }
            if state.shutdown
                || drained.is_empty()
                || drained.len() >= max_requests
                || task_count >= max_tasks
                || Instant::now() >= gather_deadline
            {
                return Some(drained);
            }
            let timeout = gather_deadline.saturating_duration_since(Instant::now());
            let (next_state, _) = self
                .wake
                .wait_timeout(state, timeout)
                .unwrap_or_else(|e| e.into_inner());
            state = next_state;
        }
    }

    fn shutdown(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.shutdown = true;
        self.wake.notify_all();
    }
}

fn dynamic_eval_target(batch_cap: usize, threads: usize) -> usize {
    let per_wave = threads.clamp(1, 16);
    let target = if batch_cap <= 64 {
        batch_cap
    } else if batch_cap <= 256 {
        batch_cap.min(64 * per_wave)
    } else if batch_cap <= 1_024 {
        batch_cap.min(128 * per_wave)
    } else {
        batch_cap.min(256 * per_wave)
    };
    target.clamp(1, batch_cap.max(1))
}

fn worker_pipeline_depth(batch_cap: usize, threads: usize) -> usize {
    if threads <= 1 || batch_cap <= 64 {
        1
    } else if batch_cap <= 256 {
        2
    } else {
        MAX_WORKER_PIPELINE_DEPTH
    }
}

fn parallel_gather_limit(batch_limit: usize, threads: usize) -> usize {
    if threads <= 1 {
        return batch_limit.max(1);
    }
    let per_worker = batch_limit.div_ceil(threads.clamp(1, 16));
    per_worker.clamp(1, MAX_PARALLEL_GATHER_CHUNK).min(batch_limit.max(1))
}

fn eval_chunk_limit(batch_cap: usize, threads: usize, uses_gpu: bool) -> usize {
    let target = dynamic_eval_target(batch_cap, threads);
    if uses_gpu {
        target.min(32).max(1)
    } else {
        target.max(1)
    }
}
