use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Move, MAX_MOVES};
use xiangqi_core::{generate, Position};

use crate::history::PositionHistory;
use crate::policy_onnx::BackendAttributes;

use super::backend::{BackendComputation, SharedBackendComputation};
use super::node::{cancel_score_update, terminal_wdl, MctsNode, TerminalKind};
use super::task_workers::{
    plan_processing_task_ranges, PickTaskEnqueue, PickingDispatch, ProcessingDispatch,
    TaskWorkerGatherCtx,
};
use super::{
    EdgeStats, MctsBudget, MctsConfig, MctsMoveStat, MctsNodeId, MctsSearchProgress, MctsSearchResult,
    MctsTree, OnnxPolicyValueEval, PolicyValueEval, PolicyValueOutput, PolicyValueTask, PvLineInfo, SearchStats,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PathStep {
    pub node_id: MctsNodeId,
    pub edge_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PendingKey {
    ExistingLeaf(MctsNodeId),
    NewEdge(MctsNodeId, usize),
}

#[derive(Clone)]
pub(crate) enum PendingKind {
    ExistingTerminal {
        leaf_id: MctsNodeId,
        wl: f32,
        d: f32,
        m: f32,
    },
    NewTerminal {
        state_key: u64,
        wl: f32,
        d: f32,
        m: f32,
        terminal_kind: TerminalKind,
    },
    ExpandInPlace {
        node_id: MctsNodeId,
    },
    Collision {
        max_count: u32,
    },
}

#[derive(Clone)]
pub(crate) struct PendingNode {
    #[allow(dead_code)]
    pub key: PendingKey,
    pub path: Vec<PathStep>,
    pub kind: PendingKind,
    pub multivisit: u32,
    #[allow(dead_code)]
    pub collision_upsize: u32,
    pub nn_queried: bool,
    pub is_cache_hit: bool,
    pub ooo_completed: bool,
    pub eval: Option<PolicyValueOutput>,
    pub task: Option<Arc<PolicyValueTask>>,
}

impl PendingNode {
    pub(crate) fn new(
        key: PendingKey,
        path: Vec<PathStep>,
        kind: PendingKind,
        multivisit: u32,
        collision_upsize: u32,
    ) -> Self {
        Self {
            nn_queried: false,
            is_cache_hit: false,
            ooo_completed: false,
            eval: None,
            task: None,
            key,
            path,
            kind,
            multivisit,
            collision_upsize,
        }
    }

    fn can_eval_out_of_order(&self, tree: &MctsTree) -> bool {
        if self.ooo_completed {
            return false;
        }
        if self.is_cache_hit {
            return true;
        }
        match self.kind {
            PendingKind::ExistingTerminal { .. } | PendingKind::NewTerminal { .. } => true,
            PendingKind::ExpandInPlace { node_id } => tree
                .get(node_id)
                .is_some_and(|node| node.is_terminal()),
            PendingKind::Collision { .. } => false,
        }
    }
}

#[derive(Default)]
pub(crate) struct SearchIteration {
    pub pending: Vec<PendingNode>,
    /// lc0 `minibatch_size`：非碰撞 pending 条目数（不含 multivisit 权重）。
    pub minibatch_size: usize,
    pub seldepth: u32,
    pub number_out_of_order: u32,
}

pub(crate) struct GatherParams<'a> {
    pub config: MctsConfig,
    pub budget: &'a MctsBudget,
    pub base_playouts: u32,
    pub in_flight_playouts: u32,
    pub initial_visits: u32,
    pub batch_limit: usize,
    pub stats: Option<&'a SearchStats>,
    pub root_id: MctsNodeId,
    pub root_visits: u32,
    pub thread_count: usize,
    pub backend_waiting: i32,
    /// lc0 `TaskWorkersPerSearchWorker` 解析值；CPU / auto-on-CPU 时为 0。
    pub task_workers: usize,
    /// GPU 并行路径专用；单线程 search 为 `None`。
    pub onnx_task_ctx: Option<Arc<TaskWorkerGatherCtx<OnnxPolicyValueEval>>>,
    /// 并行 gather 时与 `onnx_task_ctx` 同用；供 picking/processing wait 前释放树锁。
    pub tree_shared: Option<Arc<Mutex<MctsTree>>>,
    pub stats_arc: Option<Arc<SearchStats>>,
}

/// gather 阶段树访问：单线程 `&mut` 或并行 `Arc<Mutex<_>>`。
pub(crate) enum TreeGatherAccess<'a> {
    Local(&'a mut MctsTree),
    Shared(&'a Arc<Mutex<MctsTree>>),
}

impl<'a> TreeGatherAccess<'a> {
    pub(crate) fn with_tree<R>(&mut self, f: impl FnOnce(&mut MctsTree) -> R) -> R {
        match self {
            Self::Local(tree) => f(tree),
            Self::Shared(arc) => f(&mut arc.lock().unwrap_or_else(|e| e.into_inner())),
        }
    }
}

#[derive(Default)]
pub(crate) struct PvSummary {
    pub best_move: Option<Move>,
    pub pv: Vec<Move>,
    pub best_value: f32,
    pub best_mate: Option<i32>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeRank {
    TerminalLoss,
    NonTerminal,
    TerminalWin,
}

pub(crate) fn is_collision_kind(kind: &PendingKind) -> bool {
    matches!(kind, PendingKind::Collision { .. })
}

/// gather 后是否仍有 NN 或 backup 工作（lc0 1295 / 2170）。
pub(crate) fn gather_has_work(iteration: &SearchIteration, used_batch_size: usize) -> bool {
    if used_batch_size > 0 {
        return true;
    }
    iteration
        .pending
        .iter()
        .any(|pending| !is_collision_kind(&pending.kind))
}

/// pending 中非碰撞 multivisit 总和（lc0 `total_playouts_ += multivisit` 口径）。
pub(crate) fn pending_playouts(iteration: &SearchIteration) -> u32 {
    iteration
        .pending
        .iter()
        .filter(|pending| !is_collision_kind(&pending.kind))
        .map(|pending| pending.multivisit)
        .sum()
}

pub(crate) fn playout_depth(pending: &PendingNode) -> u32 {
    let path_len = pending.path.len() as u32;
    match pending.kind {
        PendingKind::ExistingTerminal { .. }
        | PendingKind::NewTerminal { .. }
        | PendingKind::Collision { .. } => path_len,
        PendingKind::ExpandInPlace { .. } => path_len.saturating_add(1),
    }
}

pub(crate) fn worker_batch_limit(config: MctsConfig, attrs: Option<BackendAttributes>) -> usize {
    config.effective_minibatch_size(attrs.as_ref())
}

pub(crate) fn remaining_playout_budget(
    budget: &MctsBudget,
    session_playouts: u32,
    in_flight_playouts: u32,
    initial_visits: u32,
    want: u32,
) -> u32 {
    let mut cap = want;
    let scheduled = session_playouts.saturating_add(in_flight_playouts);
    if let Some(max_playouts) = budget.max_playouts {
        cap = cap.min(max_playouts.saturating_sub(scheduled));
    }
    if let Some(max_nodes) = budget.max_nodes {
        let total_nodes = scheduled.saturating_add(initial_visits);
        if total_nodes >= max_nodes {
            return 0;
        }
        cap = cap.min(max_nodes.saturating_sub(total_nodes));
    }
    cap
}

pub(crate) fn should_break_gather_for_thread_idling(
    thread_count: usize,
    backend_waiting: i32,
    config: MctsConfig,
    used_batch_size: usize,
) -> bool {
    thread_count > 1
        && used_batch_size > config.idling_minimum_work as usize
        && (thread_count as i32 - backend_waiting) > config.thread_idling_threshold
}

pub(crate) enum ProcessingBackend<'b, E> {
    Local(&'b mut BackendComputation<E>),
    Shared(&'b SharedBackendComputation<E>),
}

impl<'b, E> ProcessingBackend<'b, E>
where
    E: PolicyValueEval,
{
    pub(crate) fn used_batch_size(&self) -> usize {
        match self {
            Self::Local(backend) => backend.used_batch_size(),
            Self::Shared(backend) => backend.used_batch_size(),
        }
    }

    fn add_input(&mut self, task: &Arc<PolicyValueTask>) -> bool {
        match self {
            Self::Local(backend) => backend.add_input(task),
            Self::Shared(backend) => backend.add_input(task),
        }
    }

    pub(crate) fn add_prefetch_input(&mut self, task: &Arc<PolicyValueTask>) -> bool {
        match self {
            Self::Local(backend) => backend.add_prefetch_input(task),
            Self::Shared(backend) => backend.add_prefetch_input(task),
        }
    }

    fn with_eval_mut<R>(&mut self, f: impl FnOnce(&mut E) -> R) -> R {
        match self {
            Self::Local(backend) => f(backend.eval_mut()),
            Self::Shared(backend) => backend.with_eval_mut(f),
        }
    }

    pub(crate) fn compute_blocking(&mut self) -> Result<Vec<PolicyValueOutput>, E::Error> {
        match self {
            Self::Local(backend) => backend.compute_blocking(),
            Self::Shared(backend) => backend.compute_blocking(),
        }
    }
}

/// lc0 `GatherMinibatch`（search.cc:1268-1421）。
pub(crate) fn gather_minibatch<'b, E>(
    tree: &mut TreeGatherAccess<'_>,
    root_history: &PositionHistory,
    params: &GatherParams<'_>,
    scratch: &mut SelectionScratch,
    processing: &mut ProcessingBackend<'b, E>,
    stop: bool,
) -> SearchIteration
where
    E: super::PolicyValueEval,
{
    let mut iteration = SearchIteration::default();
    iteration.number_out_of_order = 0;
    let max_ooo = params.config.max_out_of_order(params.batch_limit);
    let mut pick_workspace = PickWorkspace::default();
    let mut picked = Vec::<PendingNode>::new();
    let mut minibatch_size = 0usize;

    let session_playouts = params
        .stats
        .map(SearchStats::total_playouts)
        .unwrap_or(params.base_playouts);
    let remaining_n = remaining_playout_budget(
        params.budget,
        session_playouts,
        params.in_flight_playouts,
        params.initial_visits,
        u32::MAX,
    ) as i64;
    let mut collisions_left =
        calculate_collisions_left(i64::from(params.root_visits).min(remaining_n), params.config);

    while minibatch_size < params.batch_limit && (iteration.number_out_of_order as usize) < max_ooo {
        let used_batch_size = processing.used_batch_size();
        let scheduled_playouts = session_playouts
            .saturating_add(pending_playouts(&iteration))
            .saturating_add(params.in_flight_playouts);

        if minibatch_size > 0 && used_batch_size == 0
        {
            break;
        }

        if should_break_gather_for_thread_idling(
            params.thread_count,
            params.backend_waiting,
            params.config,
            used_batch_size,
        ) && minibatch_size > 0
        {
            break;
        }

        if budget_exhausted(
            params.budget,
            scheduled_playouts,
            0,
            params.initial_visits,
            params.stats,
        ) {
            break;
        }

        let remaining = remaining_playout_budget(
            params.budget,
            scheduled_playouts,
            0,
            params.initial_visits,
            u32::MAX,
        );
        let room = params.batch_limit.saturating_sub(minibatch_size);
        let remaining_pick = remaining.min(i32::MAX as u32) as i32;
        let ooo_room = max_ooo.saturating_sub(iteration.number_out_of_order as usize) as i32;
        let pick_limit = collisions_left
            .min(remaining_pick)
            .min(room as i32)
            .min(ooo_room);
        if pick_limit <= 0 {
            break;
        }

        picked.clear();
        let new_start = iteration.pending.len();
        let completed = if params.onnx_task_ctx.is_some() {
            pick_nodes_to_extend_parallel(
                params,
                params.root_id,
                root_history,
                scheduled_playouts,
                pick_limit,
                &mut picked,
                scratch,
                &mut pick_workspace,
            )
        } else {
            tree.with_tree(|tree| {
                pick_nodes_to_extend_task(
                    tree,
                    params.root_id,
                    root_history,
                    params.root_id,
                    0,
                    &[],
                    params.config,
                    params.stats,
                    params.budget,
                    scheduled_playouts,
                    pick_limit,
                    &mut picked,
                    scratch,
                    &mut pick_workspace,
                    None::<&dyn PickTaskEnqueue>,
                    pick_limit,
                    &mut 0,
                )
            })
        };
        if completed <= 0 && picked.is_empty() {
            break;
        }

        iteration.pending.extend(picked.iter().cloned());
        for pending in &picked {
            if !is_collision_kind(&pending.kind) {
                minibatch_size += 1;
            }
        }

        let minibatch_len = iteration.pending.len();
        let non_collisions_picked = picked
            .iter()
            .filter(|pending| !is_collision_kind(&pending.kind))
            .count();
        let needs_process_wait = run_process_picked_phase(
            tree,
            root_history,
            params,
            params.root_id,
            &mut iteration,
            new_start,
            minibatch_len,
            non_collisions_picked,
            processing,
        );
        if needs_process_wait {
            if let Some(ctx) = params.onnx_task_ctx.as_ref() {
                ctx.pool.wait_for_tasks();
                ctx.pool.clear_processing_dispatch();
            }
        }

        let mut some_ooo = false;
        for pending in iteration.pending.iter().skip(new_start) {
            if pending.ooo_completed {
                some_ooo = true;
                break;
            }
        }
        if some_ooo {
            tree.with_tree(|tree| {
                apply_out_of_order_backups(
                    tree,
                    &mut iteration,
                    params.stats,
                    new_start,
                    &mut minibatch_size,
                );
            });
        }

        for idx in new_start..iteration.pending.len() {
            let pending = &iteration.pending[idx];
            if !is_collision_kind(&pending.kind) {
                continue;
            }
            let mut multivisit = pending.multivisit;
            if let PendingKind::Collision { max_count } = pending.kind {
                if max_count > 0 && collisions_left > multivisit as i32 {
                    let extra = (max_count.min(collisions_left as u32)).saturating_sub(multivisit);
                    if extra > 0 {
                        multivisit = multivisit.saturating_add(extra);
                        tree.with_tree(|tree| {
                            increment_collision_ancestors(tree, &pending.path, extra);
                        });
                        iteration.pending[idx].multivisit = multivisit;
                    }
                }
            }
            collisions_left = collisions_left.saturating_sub(multivisit as i32);
            if collisions_left <= 0 {
                iteration.minibatch_size = minibatch_size;
                return iteration;
            }
            if stop {
                iteration.minibatch_size = minibatch_size;
                return iteration;
            }
        }
    }

    iteration.minibatch_size = minibatch_size;
    iteration
}

pub(crate) fn gather_minibatch_with_local<E>(
    tree: &mut MctsTree,
    root_history: &PositionHistory,
    params: &GatherParams<'_>,
    scratch: &mut SelectionScratch,
    backend: &mut BackendComputation<E>,
    stop: bool,
) -> (SearchIteration, usize)
where
    E: super::PolicyValueEval,
{
    let mut access = TreeGatherAccess::Local(tree);
    let mut processing = ProcessingBackend::Local(backend);
    let iteration = gather_minibatch(
        &mut access,
        root_history,
        params,
        scratch,
        &mut processing,
        stop,
    );
    (iteration, processing.used_batch_size())
}

pub(crate) fn gather_minibatch_with_shared<'b, E>(
    tree: &Arc<Mutex<MctsTree>>,
    root_history: &PositionHistory,
    params: &GatherParams<'_>,
    scratch: &mut SelectionScratch,
    backend: &'b SharedBackendComputation<E>,
    stop: bool,
) -> (SearchIteration, usize)
where
    E: super::PolicyValueEval,
{
    let mut access = TreeGatherAccess::Shared(tree);
    let mut processing = ProcessingBackend::Shared(backend);
    let iteration = gather_minibatch(
        &mut access,
        root_history,
        params,
        scratch,
        &mut processing,
        stop,
    );
    (iteration, processing.used_batch_size())
}

/// lc0 gather 内 OOO 收尾：碰撞撤销 + 提前 backup。
pub(crate) fn apply_out_of_order_backups(
    tree: &mut MctsTree,
    iteration: &mut SearchIteration,
    stats: Option<&SearchStats>,
    from: usize,
    minibatch_size: &mut usize,
) {
    let mut i = iteration.pending.len();
    while i > from {
        i -= 1;
        let is_collision = is_collision_kind(&iteration.pending[i].kind);
        let is_ooo = iteration.pending[i].ooo_completed;
        if is_collision {
            let pending = iteration.pending.remove(i);
            cancel_collision_path(tree, &pending.path, pending.multivisit);
        } else if is_ooo {
            let pending = iteration.pending.remove(i);
            do_backup_single(tree, &pending, stats);
            iteration.number_out_of_order = iteration.number_out_of_order.saturating_add(1);
            *minibatch_size = minibatch_size.saturating_sub(1);
        }
    }
}

/// lc0 `ProcessPickedTask` 分发（search.cc:1353-1384,1445-1484）。
/// 返回 `true` 表示已入队 worker tasks，调用方需在释放树锁后 `wait_for_tasks`。
fn run_process_picked_phase<E>(
    tree: &mut TreeGatherAccess<'_>,
    root_history: &PositionHistory,
    params: &GatherParams<'_>,
    root_id: MctsNodeId,
    iteration: &mut SearchIteration,
    new_start: usize,
    pending_len: usize,
    _non_collisions: usize,
    processing: &mut ProcessingBackend<'_, E>,
) -> bool
where
    E: super::PolicyValueEval,
{
    let (worker_ranges, main_start) = plan_processing_task_ranges(
        new_start,
        pending_len,
        &iteration.pending,
        params.config,
        params.task_workers,
    );

    if worker_ranges.is_empty() {
        tree.with_tree(|tree| {
            process_picked_range(
                tree,
                root_history,
                params.config,
                root_id,
                iteration,
                new_start,
                pending_len,
                processing,
            );
        });
        return false;
    }

    let Some(ctx) = params.onnx_task_ctx.as_deref() else {
        tree.with_tree(|tree| {
            for range in &worker_ranges {
                process_picked_range(
                    tree,
                    root_history,
                    params.config,
                    root_id,
                    iteration,
                    range.start,
                    range.end,
                    processing,
                );
            }
            process_picked_range(
                tree,
                root_history,
                params.config,
                root_id,
                iteration,
                main_start,
                pending_len,
                processing,
            );
        });
        return false;
    };

    let Some(tree_shared) = params.tree_shared.as_ref() else {
        tree.with_tree(|tree| {
            for range in &worker_ranges {
                process_picked_range(
                    tree,
                    root_history,
                    params.config,
                    root_id,
                    iteration,
                    range.start,
                    range.end,
                    processing,
                );
            }
            process_picked_range(
                tree,
                root_history,
                params.config,
                root_id,
                iteration,
                main_start,
                pending_len,
                processing,
            );
        });
        return false;
    };

    // lc0: ResetTasks → enqueue worker ranges → main ProcessPickedTask → WaitForTasks
    let iteration_arc = Arc::new(Mutex::new(std::mem::take(iteration)));
    ctx.pool.reset_tasks();
    for range in &worker_ranges {
        ctx.pool.enqueue_processing(range.start, range.end);
    }
    ctx.pool.set_processing_dispatch(ProcessingDispatch {
        tree: Arc::clone(tree_shared),
        iteration: Arc::clone(&iteration_arc),
        root_history: root_history.clone_for_search(),
        config: params.config,
        root_id,
        backend: Arc::clone(&ctx.backend_shared),
    });
    ctx.pool.wake_workers();
    tree.with_tree(|tree| {
        let mut iteration_guard = iteration_arc.lock().unwrap_or_else(|e| e.into_inner());
        process_picked_range(
            tree,
            root_history,
            params.config,
            root_id,
            &mut iteration_guard,
            main_start,
            pending_len,
            processing,
        );
    });
    *iteration = match Arc::try_unwrap(iteration_arc) {
        Ok(mutex) => mutex.into_inner().unwrap_or_else(|e| e.into_inner()),
        Err(arc) => {
            let mut guard = arc.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        }
    };
    true
}

fn pick_nodes_to_extend_parallel(
    params: &GatherParams<'_>,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    session_playouts: u32,
    collision_limit: i32,
    receiver: &mut Vec<PendingNode>,
    scratch: &mut SelectionScratch,
    workspace: &mut PickWorkspace,
) -> i32 {
    let Some(ctx) = params.onnx_task_ctx.as_ref() else {
        return 0;
    };
    let Some(tree_shared) = params.tree_shared.as_ref() else {
        return 0;
    };
    ctx.pool.reset_tasks();
    ctx.pool.set_picking_dispatch(PickingDispatch {
        tree: Arc::clone(tree_shared),
        root_id,
        root_history: root_history.clone_for_search(),
        config: params.config,
        budget: params.budget.clone(),
        session_playouts,
        stats: params.stats_arc.clone(),
    });
    ctx.pool.wake_workers();
    workspace.reset();
    scratch.reset();
    let mut passed_off = 0i32;
    let completed = {
        let mut tree = tree_shared.lock().unwrap_or_else(|e| e.into_inner());
        pick_nodes_to_extend_task(
            &mut tree,
            root_id,
            root_history,
            root_id,
            0,
            &[],
            params.config,
            params.stats,
            params.budget,
            session_playouts,
            collision_limit,
            receiver,
            scratch,
            workspace,
            Some(ctx.pool.inner.as_ref() as &dyn PickTaskEnqueue),
            collision_limit,
            &mut passed_off,
        )
    };
    ctx.pool.wait_for_tasks();
    ctx.pool.merge_gathering_results(receiver);
    ctx.pool.clear_picking_dispatch();
    completed
}

/// task worker 线程入口（lc0 `ProcessPickedTask`）。
pub(crate) fn process_picked_range_shared<E>(
    dispatch: &ProcessingDispatch<E>,
    start: usize,
    end: usize,
) where
    E: super::PolicyValueEval + Send + Sync + 'static,
{
    let mut tree = dispatch.tree.lock().unwrap_or_else(|e| e.into_inner());
    let mut iteration = dispatch.iteration.lock().unwrap_or_else(|e| e.into_inner());
    let mut backend = ProcessingBackend::Shared(&*dispatch.backend);
    process_picked_range(
        &mut tree,
        &dispatch.root_history,
        dispatch.config,
        dispatch.root_id,
        &mut iteration,
        start,
        end,
        &mut backend,
    );
}

fn process_picked_range<E>(
    tree: &mut MctsTree,
    root_history: &PositionHistory,
    config: MctsConfig,
    root_id: MctsNodeId,
    iteration: &mut SearchIteration,
    start: usize,
    end: usize,
    backend: &mut ProcessingBackend<'_, E>,
) where
    E: super::PolicyValueEval,
{
    for pending in &mut iteration.pending[start..end] {
        if is_collision_kind(&pending.kind) {
            continue;
        }
        iteration.seldepth = iteration.seldepth.max(playout_depth(pending));

        match &mut pending.kind {
            PendingKind::ExpandInPlace { .. } => {
                if let Some(task) = extend_node(tree, root_history, config, pending) {
                    pending.nn_queried = true;
                    pending.task = Some(Arc::clone(&task));
                    pending.is_cache_hit = backend.add_input(&task);
                }
            }
            _ => {}
        }

        if config.out_of_order_eval && pending.can_eval_out_of_order(tree) {
            backend.with_eval_mut(|eval| {
                fetch_single_node(tree, eval, config, root_id, pending);
            });
            pending.ooo_completed = true;
        }
    }
}

fn extend_node(
    tree: &mut MctsTree,
    root_history: &PositionHistory,
    _config: MctsConfig,
    pending: &PendingNode,
) -> Option<Arc<PolicyValueTask>> {
    let PendingKind::ExpandInPlace { node_id } = pending.kind else {
        return None;
    };
    let node = tree.get(node_id)?;
    if node.expanded || !node.children.is_empty() || node.is_terminal() {
        return None;
    }
    let pos = position_for_pending(tree, root_history, pending);
    let mut buf = vec![ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(&pos, GenType::Legal, &mut buf);
    if n == 0 {
        if let Some(node) = tree.get_mut(node_id) {
            node.expanded = true;
            node.terminal_kind = TerminalKind::Generic;
            node.terminal_value = Some(-1.0);
        }
        return None;
    }
    let legal_moves = buf[..n].iter().map(|e| e.mv).collect::<Vec<_>>();
    if let Some(node) = tree.get_mut(node_id) {
        if node.children.is_empty() {
            node.children.reserve(legal_moves.len());
            for mv in &legal_moves {
                node.children.push(EdgeStats {
                    mv: *mv,
                    prior: 0.0,
                    visits: 0,
                    in_flight: 0,
                    wl: 0.0,
                    d: 0.0,
                    m: 0.0,
                    child: None,
                });
            }
        }
    }
    let history = history_for_pending(tree, root_history, pending);
    Some(Arc::new(PolicyValueTask {
        position: pos,
        history,
        legal_moves,
    }))
}

fn position_for_pending(
    tree: &MctsTree,
    root_history: &PositionHistory,
    pending: &PendingNode,
) -> Position {
    let mut pos = root_history.current().clone_for_search();
    for step in &pending.path {
        let Some(node) = tree.get(step.node_id) else {
            break;
        };
        let Some(edge) = node.children.get(step.edge_idx) else {
            break;
        };
        pos.do_move(edge.mv);
    }
    pos
}

fn history_for_pending(
    tree: &MctsTree,
    root_history: &PositionHistory,
    pending: &PendingNode,
) -> PositionHistory {
    let mut positions = Vec::with_capacity(pending.path.len());
    let mut pos = root_history.current().clone_for_search();
    for step in &pending.path {
        let Some(node) = tree.get(step.node_id) else {
            break;
        };
        let Some(edge) = node.children.get(step.edge_idx) else {
            break;
        };
        pos.do_move(edge.mv);
        positions.push(pos.clone_for_search());
    }
    root_history.extended_with_search_path(&positions)
}

fn apply_dirichlet_noise(priors: &mut [f32], config: MctsConfig) {
    if priors.is_empty() || config.root_dirichlet_epsilon <= 0.0 {
        return;
    }
    // lc0 `ApplyDirichletNoise`（search.cc:202-218）：`GetGamma(alpha, 1.0)` 后归一化混合。
    let alpha = config.root_dirichlet_alpha.max(1e-6) as f64;
    let eps = config.root_dirichlet_epsilon.clamp(0.0, 1.0) as f64;
    let mut noise: Vec<f32> = priors
        .iter()
        .map(|_| random_gamma(alpha, 1.0) as f32)
        .collect();
    let sum: f32 = noise.iter().sum();
    if sum <= f32::MIN {
        return;
    }
    for n in &mut noise {
        *n /= sum;
    }
    for (p, n) in priors.iter_mut().zip(noise) {
        *p = ((1.0 - eps) * *p as f64 + eps * n as f64) as f32;
    }
}

fn random_gamma(shape: f64, scale: f64) -> f64 {
    if !(shape.is_finite() && scale.is_finite() && shape > 0.0 && scale > 0.0) {
        return 0.0;
    }
    if shape >= 1.0 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = random_normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = random_u01();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v * scale;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }
    let u = random_u01();
    random_gamma(shape + 1.0, scale) * u.powf(1.0 / shape)
}

fn random_normal() -> f64 {
    loop {
        let u1 = random_u01();
        let u2 = random_u01();
        if u1 > f64::MIN {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            return r * theta.cos();
        }
    }
}

fn random_u01() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static RNG_STATE: Cell<u64> = Cell::new(0);
    }
    RNG_STATE.with(|state| {
        let mut x = state.get();
        if x == 0 {
            x = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E3779B97F4A7C15);
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        (x >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    })
}

/// lc0 `FetchSingleNodeResult`。
pub(crate) fn fetch_single_node<E: super::PolicyValueEval>(
    tree: &mut MctsTree,
    eval: &mut E,
    config: MctsConfig,
    root_id: MctsNodeId,
    pending: &mut PendingNode,
) {
    if is_collision_kind(&pending.kind) || pending.eval.is_some() {
        return;
    }
    match pending.kind {
        PendingKind::ExistingTerminal { leaf_id, wl, d, m } => {
            pending.eval = Some(PolicyValueOutput {
                wl,
                d,
                m,
                value: wl,
                priors: Vec::new(),
            });
            let _ = leaf_id;
        }
        PendingKind::NewTerminal { wl, d, m, .. } => {
            pending.eval = Some(PolicyValueOutput {
                wl,
                d,
                m,
                value: wl,
                priors: Vec::new(),
            });
        }
        PendingKind::ExpandInPlace { node_id } => {
            if !pending.nn_queried {
                if let Some(node) = tree.get(node_id) {
                    pending.eval = Some(PolicyValueOutput {
                        wl: node.wl,
                        d: node.d,
                        m: node.m,
                        value: node.wl,
                        priors: Vec::new(),
                    });
                }
                return;
            }
            let Some(task) = pending.task.as_ref() else {
                return;
            };
            let out = if pending.is_cache_hit {
                eval.evaluate_cached(task.as_input())
            } else {
                None
            };
            let Some(mut out) = out else {
                return;
            };
            out.wl = -out.wl;
            if config.root_dirichlet_epsilon > 0.0 && node_id == root_id {
                apply_dirichlet_noise(&mut out.priors, config);
            }
            expand_node_in_place(tree, node_id, task.as_ref(), &out);
            pending.eval = Some(out);
        }
        PendingKind::Collision { .. } => {}
    }
}

/// lc0 `FetchMinibatchResults`。
pub(crate) fn fetch_minibatch_results<E: super::PolicyValueEval>(
    tree: &mut MctsTree,
    eval: &mut E,
    iteration: &mut SearchIteration,
    outputs: &[PolicyValueOutput],
    config: MctsConfig,
    root_id: MctsNodeId,
) {
    let mut eval_cursor = 0usize;
    for pending in &mut iteration.pending {
        if pending.ooo_completed || is_collision_kind(&pending.kind) {
            continue;
        }
        if let PendingKind::ExpandInPlace { node_id } = pending.kind {
            if pending.nn_queried && pending.eval.is_none() {
                let Some(task) = pending.task.as_ref() else {
                    continue;
                };
                let mut out = outputs
                    .get(eval_cursor)
                    .cloned()
                    .expect("batched eval must match nn_queried count");
                eval_cursor += 1;
                out.wl = -out.wl;
                if config.root_dirichlet_epsilon > 0.0 && node_id == root_id {
                    apply_dirichlet_noise(&mut out.priors, config);
                }
                expand_node_in_place(tree, node_id, task.as_ref(), &out);
                pending.eval = Some(out);
            } else if pending.eval.is_none() {
                fetch_single_node(tree, eval, config, root_id, pending);
            }
        } else if pending.eval.is_none() {
            fetch_single_node(tree, eval, config, root_id, pending);
        }
    }
}

/// lc0 `DoBackupUpdateSingleNode`。
pub(crate) fn do_backup_single(
    tree: &mut MctsTree,
    pending: &PendingNode,
    stats: Option<&SearchStats>,
) {
    if is_collision_kind(&pending.kind) {
        return;
    }
    let (wl, d, m) = match &pending.eval {
        Some(eval) => (eval.wl, eval.d, eval.m),
        None => match &pending.kind {
            PendingKind::ExistingTerminal { wl, d, m, .. } => (*wl, *d, *m),
            PendingKind::NewTerminal { wl, d, m, .. } => (*wl, *d, *m),
            _ => return,
        },
    };
    match pending.kind {
        PendingKind::ExistingTerminal { leaf_id, .. } => {
            do_backup_from_leaf(tree, leaf_id, &pending.path, wl, d, m, pending.multivisit);
        }
        PendingKind::NewTerminal {
            state_key,
            terminal_kind,
            ..
        } => {
            let parent = pending.path.last().expect("new terminal must have parent");
            let child_id = add_terminal_child(
                tree,
                parent.node_id,
                parent.edge_idx,
                state_key,
                wl,
                d,
                m,
                terminal_kind,
            );
            do_backup_from_leaf(tree, child_id, &pending.path, wl, d, m, pending.multivisit);
        }
        PendingKind::ExpandInPlace { node_id } => {
            do_backup_from_leaf(tree, node_id, &pending.path, wl, d, m, pending.multivisit);
        }
        PendingKind::Collision { .. } => {}
    }
    if let Some(stats) = stats {
        stats.record_backup(pending);
    }
}

/// lc0 `DoBackupUpdate`。
pub(crate) fn do_backup_update(
    tree: &mut MctsTree,
    iteration: &SearchIteration,
    stats: Option<&SearchStats>,
    shared_collisions: Option<&SharedCollisions>,
) {
    let mut had_work = iteration.number_out_of_order > 0;
    for pending in &iteration.pending {
        if is_collision_kind(&pending.kind) {
            continue;
        }
        do_backup_single(tree, pending, stats);
        had_work = true;
    }
    if had_work {
        if let Some(shared) = shared_collisions {
            shared.cancel_all(tree);
        }
    }
}

pub(crate) struct SelectionScratch {
    pub path_positions: Vec<xiangqi_core::Position>,
    pub path_key_counts: HashMap<u64, usize>,
}

impl Default for SelectionScratch {
    fn default() -> Self {
        Self {
            path_positions: Vec::with_capacity(16),
            path_key_counts: HashMap::new(),
        }
    }
}

impl SelectionScratch {
    pub fn reset(&mut self) {
        self.path_positions.clear();
        self.path_key_counts.clear();
    }
}

/// lc0 edge `GetN`：子节点 `n_started()`，否则边 `n_started()`。
pub fn edge_get_n(tree: &MctsTree, edge: &EdgeStats) -> u32 {
    if let Some(child_id) = edge.child {
        if let Some(child) = tree.get(child_id) {
            return child.n_started();
        }
    }
    edge.n_started()
}

pub(crate) fn do_backup_from_leaf(
    tree: &mut MctsTree,
    leaf_id: MctsNodeId,
    path: &[PathStep],
    mut wl: f32,
    d: f32,
    mut m: f32,
    multivisit: u32,
) {
    if let Some(leaf) = tree.get(leaf_id) {
        if leaf.is_terminal() {
            wl = leaf.wl;
            m = leaf.m;
        }
    }
    if let Some(leaf) = tree.get_mut(leaf_id) {
        leaf.finalize_score_update(wl, d, m, multivisit);
    }
    for step in path.iter().rev() {
        wl = -wl;
        m += 1.0;
        let node = tree.get_mut(step.node_id).expect("path node must exist");
        node.finalize_score_update(wl, d, m, multivisit);
        let edge = &mut node.children[step.edge_idx];
        edge.finalize_score_update(wl, d, m, multivisit);
    }
}

pub(crate) fn expand_node_in_place(
    tree: &mut MctsTree,
    node_id: MctsNodeId,
    task: &PolicyValueTask,
    out: &PolicyValueOutput,
) {
    let node = tree.get_mut(node_id).expect("expand node must exist");
    node.expanded = true;
    if node.children.is_empty() {
        node.children.reserve(task.legal_moves.len());
        for (idx, mv) in task.legal_moves.iter().copied().enumerate() {
            node.children.push(EdgeStats {
                mv,
                prior: out.priors.get(idx).copied().unwrap_or(0.0),
                visits: 0,
                in_flight: 0,
                wl: 0.0,
                d: 0.0,
                m: 0.0,
                child: None,
            });
        }
        return;
    }
    for (idx, mv) in task.legal_moves.iter().copied().enumerate() {
        let prior = out.priors.get(idx).copied().unwrap_or(0.0);
        if let Some(edge) = node.children.iter_mut().find(|edge| edge.mv == mv) {
            edge.prior = prior;
        }
    }
}

pub(crate) fn add_terminal_child(
    tree: &mut MctsTree,
    parent_id: MctsNodeId,
    edge_idx: usize,
    state_key: u64,
    wl: f32,
    d: f32,
    m: f32,
    terminal_kind: TerminalKind,
) -> MctsNodeId {
    let tv = if wl > 0.0 {
        1.0
    } else if wl < 0.0 {
        -1.0
    } else {
        0.0
    };
    let child_id = tree.add_node(MctsNode {
        state_key,
        visits: 0,
        in_flight: 0,
        wl,
        d,
        m,
        expanded: true,
        terminal_kind,
        terminal_value: Some(tv),
        children: Vec::new(),
    });
    let edge = &mut tree
        .get_mut(parent_id)
        .expect("parent node must exist")
        .children[edge_idx];
    edge.child = Some(child_id);
    child_id
}

pub(crate) fn cancel_collision_path(tree: &mut MctsTree, path: &[PathStep], multivisit: u32) {
    cancel_collision_ancestors(tree, path, multivisit);
}

pub(crate) fn cancel_pending(tree: &mut MctsTree, pending: &PendingNode) {
    if is_collision_kind(&pending.kind) {
        cancel_collision_path(tree, &pending.path, pending.multivisit);
        return;
    }
    match &pending.kind {
        PendingKind::ExpandInPlace { node_id, .. } => {
            if let Some(node) = tree.get_mut(*node_id) {
                cancel_score_update(&mut node.in_flight, pending.multivisit);
            }
        }
        PendingKind::ExistingTerminal { leaf_id, .. } => {
            if let Some(node) = tree.get_mut(*leaf_id) {
                cancel_score_update(&mut node.in_flight, pending.multivisit);
            }
        }
        PendingKind::NewTerminal { .. } => {
            if let Some(step) = pending.path.last() {
                if let Some(node) = tree.get_mut(step.node_id) {
                    if let Some(edge) = node.children.get_mut(step.edge_idx) {
                        cancel_score_update(&mut edge.in_flight, pending.multivisit);
                    }
                }
            }
        }
        PendingKind::Collision { .. } => {}
    }
    for step in pending.path.iter().rev() {
        if let Some(node) = tree.get_mut(step.node_id) {
            cancel_score_update(&mut node.in_flight, pending.multivisit);
            if let Some(edge) = node.children.get_mut(step.edge_idx) {
                cancel_score_update(&mut edge.in_flight, pending.multivisit);
            }
        }
    }
}

pub(crate) fn cancel_minibatch(tree: &mut MctsTree, iteration: SearchIteration) {
    for pending in iteration.pending {
        cancel_pending(tree, &pending);
    }
}

pub(crate) fn progress_from_tree(
    tree: &MctsTree,
    root_id: MctsNodeId,
    stats: &SearchStats,
    config: MctsConfig,
) -> MctsSearchProgress {
    let root = tree.get(root_id).expect("root must exist");
    let summary = pv_summary_from_tree(tree, root_id, config);
    let multi_pv = config.multi_pv.clamp(1, 500);
    let pv_lines = if multi_pv > 1 {
        multi_pv_lines_from_tree(tree, root_id, config)
    } else {
        Vec::new()
    };
    MctsSearchProgress {
        best_move: summary.best_move,
        pv: summary.pv,
        pv_lines,
        multi_pv,
        playouts: stats.total_playouts(),
        root_visits: root.visits,
        nodes: stats.uci_nodes(),
        tree_nodes: tree.reachable_len(),
        depth: stats.depth(),
        seldepth: stats.max_depth(),
        root_value: root.mean_value(),
        best_value: summary.best_value,
        best_mate: summary.best_mate,
        nps_elapsed_ms: stats.nps_elapsed_ms(),
        retry_without_playout: stats.retry_without_playout(),
        moves: root
            .children
            .iter()
            .map(|edge| MctsMoveStat {
                mv: edge.mv,
                prior: edge.prior,
                visits: edge_get_n(tree, edge),
                q: edge.mean_q(),
            })
            .collect(),
        per_pv_counters: config.per_pv_counters,
    }
}

pub(crate) fn result_from_tree(
    tree: &MctsTree,
    root_id: MctsNodeId,
    stats: &SearchStats,
    config: MctsConfig,
) -> MctsSearchResult {
    let progress = progress_from_tree(tree, root_id, stats, config);
    MctsSearchResult {
        best_move: progress.best_move,
        pv: progress.pv,
        pv_lines: progress.pv_lines,
        multi_pv: progress.multi_pv,
        playouts: progress.playouts,
        root_visits: progress.root_visits,
        nodes: progress.nodes,
        tree_nodes: progress.tree_nodes,
        depth: progress.depth,
        seldepth: progress.seldepth,
        root_value: progress.root_value,
        best_value: progress.best_value,
        best_mate: progress.best_mate,
        nps_elapsed_ms: progress.nps_elapsed_ms,
        retry_without_playout: progress.retry_without_playout,
        moves: progress.moves,
        per_pv_counters: progress.per_pv_counters,
    }
}

pub(crate) fn ensure_node_twofold_correct_for_depth(
    tree: &mut MctsTree,
    stats: &SearchStats,
    child_id: MctsNodeId,
    path: &[PathStep],
) {
    let depth = path.len() as u32;
    let child = tree.get(child_id).expect("child");
    if !child.is_twofold_terminal() || depth >= child.m as u32 {
        return;
    }
    let wl = child.wl;
    let d = child.d;
    let m = child.m;
    let terminal_visits = child.visits;
    if let Some(child) = tree.get_mut(child_id) {
        child.revert_terminal_visits(wl, d, m, terminal_visits);
        child.make_not_terminal();
    }
    stats.subtract_initial_visits(terminal_visits);
    let mut depth_counter = 0u32;
    for step in path.iter().rev() {
        depth_counter += 1;
        if depth_counter > depth {
            break;
        }
        let node = tree.get_mut(step.node_id).expect("ancestor");
        node.revert_terminal_visits(wl, d, m + depth_counter as f32, terminal_visits);
        let edge = &mut node.children[step.edge_idx];
        edge.revert_terminal_visits(wl, d, m + depth_counter as f32, terminal_visits);
    }
}

fn edge_rank(edge: &EdgeStats, child: Option<&MctsNode>) -> EdgeRank {
    if edge.visits == 0 {
        return EdgeRank::NonTerminal;
    }
    let Some(child) = child else {
        return EdgeRank::NonTerminal;
    };
    let Some(tv) = child.terminal_value else {
        return EdgeRank::NonTerminal;
    };
    if tv > 0.0 {
        EdgeRank::TerminalWin
    } else if tv < 0.0 {
        EdgeRank::TerminalLoss
    } else {
        EdgeRank::NonTerminal
    }
}

fn edge_visits_for_cmp(tree: &MctsTree, edge: &EdgeStats) -> u32 {
    edge_get_n(tree, edge)
}

fn edge_cmp(
    tree: &MctsTree,
    a: &EdgeStats,
    a_child: Option<&MctsNode>,
    b: &EdgeStats,
    b_child: Option<&MctsNode>,
) -> std::cmp::Ordering {
    let a_rank = edge_rank(a, a_child);
    let b_rank = edge_rank(b, b_child);
    if a_rank != b_rank {
        return a_rank.cmp(&b_rank);
    }
    let a_n = edge_visits_for_cmp(tree, a);
    let b_n = edge_visits_for_cmp(tree, b);
    if a_n != b_n {
        return a_n.cmp(&b_n);
    }
    if a_n == 0 {
        return a
            .prior
            .partial_cmp(&b.prior)
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if a_rank == EdgeRank::NonTerminal {
        return a
            .mean_q()
            .partial_cmp(&b.mean_q())
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if a_rank == EdgeRank::TerminalWin {
        return a.m.partial_cmp(&b.m).unwrap_or(std::cmp::Ordering::Equal);
    }
    b.m.partial_cmp(&a.m).unwrap_or(std::cmp::Ordering::Equal)
}

fn get_best_children_no_temperature(
    tree: &MctsTree,
    parent_id: MctsNodeId,
    count: usize,
    root_id: MctsNodeId,
    root_move_filter: Option<&[Move]>,
) -> Vec<(usize, MctsNodeId, Move)> {
    // lc0 `GetBestChildrenNoTemperature` (search.cc:727-824)
    let Some(parent) = tree.get(parent_id) else {
        return Vec::new();
    };
    if parent.n_started() == 0 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..parent.children.len()).collect();
    if parent_id == root_id {
        if let Some(filter) = root_move_filter {
            indices.retain(|&idx| filter.contains(&parent.children[idx].mv));
        }
    }
    if indices.is_empty() {
        return Vec::new();
    }

    let take = count.min(indices.len());
    indices.sort_unstable_by(|&a, &b| {
        let a_edge = &parent.children[a];
        let b_edge = &parent.children[b];
        edge_cmp(
            tree,
            a_edge,
            a_edge.child.and_then(|id| tree.get(id)),
            b_edge,
            b_edge.child.and_then(|id| tree.get(id)),
        )
        .reverse()
    });
    indices.truncate(take);

    indices
        .into_iter()
        .map(|idx| {
            let edge = &parent.children[idx];
            let child_id = edge.child.unwrap_or(parent_id);
            (idx, child_id, edge.mv)
        })
        .collect()
}

fn get_best_children(tree: &MctsTree, parent_id: MctsNodeId) -> Vec<(usize, MctsNodeId, Move)> {
    get_best_children_no_temperature(tree, parent_id, 1, parent_id, None)
}

fn score_for_root_edge(tree: &MctsTree, root_id: MctsNodeId, edge_idx: usize) -> (f32, Option<i32>) {
    let Some(root) = tree.get(root_id) else {
        return (0.0, None);
    };
    let edge = &root.children[edge_idx];
    let mut best_value = edge.mean_q();
    let mut best_mate = None;
    if edge_get_n(tree, edge) > 0 {
        if let Some(child_id) = edge.child {
            if let Some(child) = tree.get(child_id) {
                if child.is_terminal() {
                    best_value = edge.mean_q();
                    if edge.wl.abs() > f32::EPSILON {
                        let mate = (edge.get_m(0.0).round() as i32) / 2 + 1;
                        best_mate = Some(if edge.wl > 0.0 { mate } else { -mate });
                    }
                }
            }
        }
    }
    (best_value, best_mate)
}

fn pv_line_from_root_edge(
    tree: &MctsTree,
    root_id: MctsNodeId,
    edge_idx: usize,
) -> (Vec<Move>, f32, Option<i32>) {
    // lc0 `SendUciInfo` PV walk (search.cc:367-371)
    let Some(root) = tree.get(root_id) else {
        return (Vec::new(), 0.0, None);
    };
    let edge = &root.children[edge_idx];
    let (best_value, best_mate) = score_for_root_edge(tree, root_id, edge_idx);
    let mut pv = vec![edge.mv];
    let mut node_id = edge.child.unwrap_or(root_id);

    loop {
        if tree.get(node_id).is_some_and(|child| child.is_terminal()) {
            break;
        }
        let children = get_best_children_no_temperature(tree, node_id, 1, root_id, None);
        if children.is_empty() {
            break;
        }
        let (_, child_id, mv) = children[0];
        pv.push(mv);
        if child_id == node_id {
            break;
        }
        node_id = child_id;
    }

    (pv, best_value, best_mate)
}

fn multi_pv_lines_from_tree(tree: &MctsTree, root_id: MctsNodeId, config: MctsConfig) -> Vec<PvLineInfo> {
    let max_pv = config.multi_pv.clamp(1, 500) as usize;
    let root_edges = get_best_children_no_temperature(tree, root_id, max_pv, root_id, None);
    root_edges
        .into_iter()
        .enumerate()
        .map(|(idx, (edge_idx, _, _))| {
            let visits = tree
                .get(root_id)
                .map(|root| pick_edge_visits(tree, &root.children[edge_idx]))
                .unwrap_or(0);
            let (pv, best_value, best_mate) = pv_line_from_root_edge(tree, root_id, edge_idx);
            PvLineInfo {
                multipv: (idx + 1) as u32,
                best_value,
                best_mate,
                visits,
                pv,
            }
        })
        .collect()
}

pub(crate) fn pv_summary_from_tree(tree: &MctsTree, root_id: MctsNodeId, _config: MctsConfig) -> PvSummary {
    let mut pv = Vec::new();
    let mut node_id = root_id;
    let mut best_value = tree.get(root_id).map(MctsNode::mean_value).unwrap_or(0.0);
    let mut best_mate = None;

    loop {
        let Some(node) = tree.get(node_id) else {
            break;
        };
        if node.n_started() == 0 {
            break;
        }
        let children = get_best_children(tree, node_id);
        if children.is_empty() {
            break;
        }
        let (edge_idx, child_id, mv) = children[0];
        let edge = &node.children[edge_idx];
        if edge_get_n(tree, edge) == 0 && edge.child.is_none() {
            break;
        }
        if pv.is_empty() && edge_get_n(tree, edge) > 0 {
            if let Some(child_id) = edge.child {
                if let Some(child) = tree.get(child_id) {
                    if child.is_terminal() {
                        best_value = edge.mean_q();
                        if edge.wl.abs() > f32::EPSILON {
                            let mate = (edge.get_m(0.0).round() as i32) / 2 + 1;
                            best_mate = Some(if edge.wl > 0.0 { mate } else { -mate });
                        }
                    } else {
                        best_value = edge.mean_q();
                    }
                } else {
                    best_value = edge.mean_q();
                }
            } else {
                best_value = edge.mean_q();
            }
        }
        pv.push(mv);
        if tree.get(child_id).is_some_and(|child| child.is_terminal()) {
            break;
        }
        if child_id == node_id {
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

pub(crate) fn total_in_flight_in_tree(tree: &MctsTree) -> u32 {
    let mut total = 0u32;
    tree.for_each_reachable(|node_id| {
        let Some(node) = tree.get(node_id) else {
            return;
        };
        total = total.saturating_add(node.in_flight);
        total = total.saturating_add(node.children.iter().map(|edge| edge.in_flight).sum::<u32>());
    });
    total
}

pub(crate) fn budget_exhausted(
    budget: &MctsBudget,
    session_playouts: u32,
    in_flight_playouts: u32,
    initial_visits: u32,
    stats: Option<&SearchStats>,
) -> bool {
    if let Some(target_depth) = budget.max_depth {
        if let Some(stats) = stats {
            if stats.total_playouts() > 0 && stats.max_depth() >= target_depth {
                return true;
            }
        }
    }
    if let Some(target_playouts) = budget.max_playouts {
        if session_playouts.saturating_add(in_flight_playouts) >= target_playouts.max(1) {
            return true;
        }
    }
    if let Some(target_nodes) = budget.max_nodes {
        if session_playouts
            .saturating_add(in_flight_playouts)
            .saturating_add(initial_visits)
            >= target_nodes
        {
            return true;
        }
    }
    if let Some(deadline) = budget.deadline {
        if Instant::now() >= deadline {
            return true;
        }
    }
    if let Some(stop) = budget.stop.as_ref() {
        if stop.load(Ordering::SeqCst) {
            return true;
        }
    }
    false
}


pub(crate) fn ponder_move_from_tree(
    tree: &MctsTree,
    root_id: MctsNodeId,
    best_move: Move,
) -> Option<Move> {
    let root = tree.get(root_id)?;
    let edge = root.children.iter().find(|e| e.mv == best_move)?;
    let child_id = edge.child?;
    let child = tree.get(child_id)?;
    if child.children.is_empty() {
        return None;
    }
    child
        .children
        .iter()
        .max_by_key(|e| e.visits.saturating_add(e.in_flight))
        .map(|e| e.mv)
}

/// lc0 风格：搜索树单写者互斥。
pub(crate) type SharedMctsTree = Arc<Mutex<MctsTree>>;

#[derive(Clone)]
pub(crate) struct SharedCollisionEntry {
    pub path: Vec<PathStep>,
    pub multivisit: u32,
}

/// lc0 `Search::shared_collisions_`：跨 worker 记录 collision，有 backup 时统一 cancel。
#[derive(Default)]
pub(crate) struct SharedCollisions {
    entries: Mutex<Vec<SharedCollisionEntry>>,
}

impl SharedCollisions {
    pub fn collect(&self, pending: &[PendingNode]) {
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        for item in pending {
            if !is_collision_kind(&item.kind) {
                continue;
            }
            guard.push(SharedCollisionEntry {
                path: item.path.clone(),
                multivisit: item.multivisit,
            });
        }
    }

    /// lc0 `CancelSharedCollisions`：从 collision 节点 parent 链向上 cancel。
    pub fn cancel_all(&self, tree: &mut MctsTree) {
        let entries = {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        };
        for entry in entries {
            cancel_collision_ancestors(tree, &entry.path, entry.multivisit);
        }
    }
}

/// lc0 upsize/cancel：沿 collision 节点祖先（`path` 中 node_id）调整 in-flight。
pub(crate) fn increment_collision_ancestors(tree: &mut MctsTree, path: &[PathStep], multivisit: u32) {
    if multivisit == 0 {
        return;
    }
    for step in path.iter().rev() {
        if let Some(node) = tree.get_mut(step.node_id) {
            node.increment_n_in_flight(multivisit);
        }
    }
}

pub(crate) fn cancel_collision_ancestors(tree: &mut MctsTree, path: &[PathStep], multivisit: u32) {
    if multivisit == 0 {
        return;
    }
    for step in path.iter().rev() {
        if let Some(node) = tree.get_mut(step.node_id) {
            cancel_score_update(&mut node.in_flight, multivisit);
        }
    }
}

pub(crate) fn calculate_collisions_left(nodes: i64, config: MctsConfig) -> i32 {
    let end = i64::from(config.max_collision_visits_scaling_end.max(1));
    let start = i64::from(config.max_collision_visits_scaling_start.max(1));
    if nodes >= end {
        return config.max_collision_visits;
    }
    if nodes <= start {
        return 1;
    }
    let ratio = ((nodes - start) as f32 / (end - start) as f32)
        .powf(config.max_collision_visits_scaling_power.max(0.01));
    mix(config.max_collision_visits, 1, ratio)
}

fn mix(high: i32, low: i32, ratio: f32) -> i32 {
    (low as f32 + (high - low) as f32 * ratio).round() as i32
}

pub(crate) fn init_pending_searchers(config: MctsConfig) -> Option<Arc<AtomicI32>> {
    if config.max_concurrent_searchers == 0 {
        return None;
    }
    Some(Arc::new(AtomicI32::new(config.max_concurrent_searchers)))
}

pub(crate) fn acquire_searcher_slot(pending: &AtomicI32, spin_backoff: bool) {
    let mut backoff = 1u32;
    loop {
        let available = pending.load(Ordering::Acquire);
        if available == 0 {
            if spin_backoff {
                std::thread::sleep(Duration::from_micros(backoff.into()));
                backoff = (backoff * 2).min(1024);
            } else {
                std::thread::yield_now();
            }
            continue;
        }
        if pending
            .compare_exchange_weak(available, available - 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
        if spin_backoff {
            std::thread::sleep(Duration::from_micros(backoff.into()));
            backoff = (backoff * 2).min(1024);
        }
    }
}

pub(crate) fn release_searcher_slot(pending: &AtomicI32) {
    pending.fetch_add(1, Ordering::AcqRel);
}

pub(crate) fn ensure_tree_quiescent(tree: &MctsTree) -> Result<(), String> {
    let in_flight = total_in_flight_in_tree(tree);
    if in_flight != 0 {
        return Err(format!("search ended with {in_flight} in-flight updates"));
    }
    Ok(())
}

/// 外部 stop / 提前 join 时清掉残留 virtual loss，避免 `ensure_tree_quiescent` 误杀 partial bestmove。
pub(crate) fn clear_in_flight_in_tree(tree: &mut MctsTree) {
    let mut ids = Vec::new();
    tree.for_each_reachable(|id| ids.push(id));
    for id in ids {
        let Some(node) = tree.get_mut(id) else {
            continue;
        };
        node.in_flight = 0;
        for edge in &mut node.children {
            edge.in_flight = 0;
        }
    }
}

// --- lc0 MaybePrefetchIntoCache (was prefetch.rs) ---

/// batch 未满时递归预取未访问叶到 NN cache（lc0 `SharedLock` 读树）。
pub(crate) fn maybe_prefetch_into_cache<E>(
    tree: &MctsTree,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    config: MctsConfig,
    backend: &mut BackendComputation<E>,
    stop: bool,
) where
    E: PolicyValueEval,
{
    if stop {
        return;
    }
    let used = backend.used_batch_size();
    if used == 0 || used >= config.max_prefetch as usize {
        return;
    }
    let budget = config.max_prefetch as usize - used;
    if budget == 0 {
        return;
    }
    let positions = Vec::new();
    let node_path = vec![root_id];
    let _ = prefetch_into_cache_local(
        tree,
        root_id,
        root_history,
        &positions,
        &node_path,
        config,
        backend,
        budget,
        false,
        stop,
    );
}

pub(crate) fn maybe_prefetch_processing_backend<E>(
    tree: &MctsTree,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    config: MctsConfig,
    backend: &mut ProcessingBackend<'_, E>,
    stop: bool,
) where
    E: PolicyValueEval,
{
    if stop {
        return;
    }
    let used = backend.used_batch_size();
    if used == 0 || used >= config.max_prefetch as usize {
        return;
    }
    let budget = config.max_prefetch as usize - used;
    if budget == 0 {
        return;
    }
    let positions = Vec::new();
    let node_path = vec![root_id];
    let _ = prefetch_into_cache(
        tree,
        root_id,
        root_history,
        &positions,
        &node_path,
        config,
        backend,
        budget,
        false,
        stop,
    );
}

trait PrefetchBackend<E: PolicyValueEval> {
    fn add_prefetch_input(&mut self, task: &Arc<PolicyValueTask>) -> bool;
}

impl<E: PolicyValueEval> PrefetchBackend<E> for BackendComputation<E> {
    fn add_prefetch_input(&mut self, task: &Arc<PolicyValueTask>) -> bool {
        BackendComputation::add_prefetch_input(self, task)
    }
}

impl<'b, E: PolicyValueEval> PrefetchBackend<E> for ProcessingBackend<'b, E> {
    fn add_prefetch_input(&mut self, task: &Arc<PolicyValueTask>) -> bool {
        ProcessingBackend::add_prefetch_input(self, task)
    }
}

fn prefetch_into_cache_local<E>(
    tree: &MctsTree,
    node_id: MctsNodeId,
    root_history: &PositionHistory,
    path_positions: &[Position],
    path_nodes: &[MctsNodeId],
    config: MctsConfig,
    backend: &mut BackendComputation<E>,
    budget: usize,
    is_odd_depth: bool,
    stop: bool,
) -> usize
where
    E: PolicyValueEval,
{
    prefetch_into_cache(
        tree,
        node_id,
        root_history,
        path_positions,
        path_nodes,
        config,
        backend,
        budget,
        is_odd_depth,
        stop,
    )
}

fn prefetch_into_cache<E, B>(
    tree: &MctsTree,
    node_id: MctsNodeId,
    root_history: &PositionHistory,
    path_positions: &[Position],
    path_nodes: &[MctsNodeId],
    config: MctsConfig,
    backend: &mut B,
    mut budget: usize,
    is_odd_depth: bool,
    stop: bool,
) -> usize
where
    E: PolicyValueEval,
    B: PrefetchBackend<E>,
{
    if budget == 0 || stop {
        return 0;
    }
    let Some(node) = tree.get(node_id) else {
        return 0;
    };
    if node.n_started() == 0 {
        return prefetch_leaf(root_history, path_positions, backend, budget, stop);
    }
    if node.is_terminal() || node.children.is_empty() {
        return 0;
    }

    let draw_score = if is_odd_depth {
        -config.draw_score
    } else {
        config.draw_score
    };
    let is_root = path_positions.is_empty();
    let cpuct = config.cpuct_for(is_root, node.visits);
    let puct_mult = cpuct * (node.children_visits().max(1) as f32).sqrt();
    let parent_q = node.mean_value_with_draw(draw_score);
    let visited_policy = node
        .children
        .iter()
        .filter(|e| edge_get_n(tree, e) > 0)
        .map(|e| e.prior)
        .sum::<f32>()
        .clamp(0.0, 1.0);
    let fpu_q = config.get_fpu(is_root, parent_q, visited_policy);

    let mut scored = Vec::new();
    for (idx, edge) in node.children.iter().enumerate() {
        if edge.prior == 0.0 {
            continue;
        }
        let n = edge_get_n(tree, edge);
        let q = if n == 0 { fpu_q } else { edge.mean_q() };
        let u = edge.prior * puct_mult / (1.0 + n as f32);
        scored.push((-(u + q), idx));
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut spent = 0usize;
    for (_, edge_idx) in scored {
        if budget == 0 || stop {
            break;
        }
        let edge = &node.children[edge_idx];
        if edge_get_n(tree, edge) > 0 {
            if let Some(child_id) = edge.child {
                if path_nodes.contains(&child_id) {
                    continue;
                }
                let mut pos = path_positions
                    .last()
                    .map(|p| p.clone_for_search())
                    .unwrap_or_else(|| root_history.current().clone_for_search());
                pos.do_move(edge.mv);
                let mut child_positions = path_positions.to_vec();
                child_positions.push(pos.clone_for_search());
                let mut child_nodes = path_nodes.to_vec();
                child_nodes.push(child_id);
                let used = prefetch_into_cache(
                    tree,
                    child_id,
                    root_history,
                    &child_positions,
                    &child_nodes,
                    config,
                    backend,
                    budget.saturating_sub(spent),
                    !is_odd_depth,
                    stop,
                );
                spent += used;
            }
            continue;
        }
        if let Some(child_id) = edge.child {
            if path_nodes.contains(&child_id) {
                continue;
            }
            if let Some(child) = tree.get(child_id) {
                if child.n_started() == 0
                    && !child.expanded
                    && !child.is_terminal()
                    && child.children.is_empty()
                {
                    let mut pos = path_positions
                    .last()
                    .map(|p| p.clone_for_search())
                    .unwrap_or_else(|| root_history.current().clone_for_search());
                    pos.do_move(edge.mv);
                    let mut child_positions = path_positions.to_vec();
                    child_positions.push(pos.clone_for_search());
                    let used = prefetch_leaf(root_history, &child_positions, backend, 1, stop);
                    spent += used;
                    budget = budget.saturating_sub(used);
                }
            }
        }
    }
    spent
}

fn prefetch_leaf<E, B>(
    root_history: &PositionHistory,
    path_positions: &[Position],
    backend: &mut B,
    budget: usize,
    stop: bool,
) -> usize
where
    E: PolicyValueEval,
    B: PrefetchBackend<E>,
{
    if budget == 0 || stop {
        return 0;
    }
    let pos = path_positions
        .last()
        .unwrap_or_else(|| root_history.current());
    let mut buf = vec![ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut buf);
    if n == 0 {
        return 0;
    }
    let legal_moves = buf[..n].iter().map(|e| e.mv).collect::<Vec<_>>();
    let history = root_history.extended_with_search_path(path_positions);
    let task = Arc::new(PolicyValueTask {
        position: pos.clone_for_search(),
        history,
        legal_moves,
    });
    if backend.add_prefetch_input(&task) {
        1
    } else {
        0
    }
}

const MAX_VTP_EDGES: usize = 256;

pub(crate) struct PickWorkspace {
    pub visits_to_perform: Vec<Vec<i32>>,
    pub vtp_last_filled: Vec<i32>,
    pub vtp_buffer: Vec<Vec<i32>>,
    pub current_path: Vec<i32>,
}

impl PickWorkspace {
    pub(crate) fn reset(&mut self) {
        self.visits_to_perform.clear();
        self.vtp_last_filled.clear();
        self.vtp_buffer.clear();
        self.current_path.clear();
    }
}

impl Default for PickWorkspace {
    fn default() -> Self {
        Self {
            visits_to_perform: Vec::new(),
            vtp_last_filled: Vec::new(),
            vtp_buffer: Vec::new(),
            current_path: Vec::new(),
        }
    }
}

/// lc0 `SearchWorker::PickNodesToExtendTask`（search.cc:1573-1919）。
pub(crate) fn pick_nodes_to_extend_task(
    tree: &mut MctsTree,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    start_node_id: MctsNodeId,
    base_depth: usize,
    moves_to_base: &[PathStep],
    config: MctsConfig,
    stats: Option<&SearchStats>,
    budget: &MctsBudget,
    session_playouts: u32,
    collision_limit: i32,
    receiver: &mut Vec<PendingNode>,
    scratch: &mut SelectionScratch,
    workspace: &mut PickWorkspace,
    enqueue: Option<&dyn PickTaskEnqueue>,
    top_collision_limit: i32,
    passed_off: &mut i32,
) -> i32 {
    if collision_limit <= 0 {
        return 0;
    }
    workspace.visits_to_perform.clear();
    workspace.vtp_last_filled.clear();
    workspace.current_path.clear();
    if moves_to_base.is_empty() {
        scratch.reset();
    } else {
        scratch.path_positions.clear();
    }

    let remaining_playouts = remaining_playout_budget(
        budget,
        session_playouts,
        0,
        stats.map(|s| s.initial_visits()).unwrap_or(0),
        u32::MAX,
    );
    let best_root_n = tree
        .get(root_id)
        .map(|root| {
            root.children
                .iter()
                .map(|edge| pick_edge_visits(tree, edge))
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);

    let mut node_id = start_node_id;
    let mut path: Vec<PathStep> = moves_to_base.to_vec();
    let base_key_counts = root_history.key_counts();
    let mut pos = if moves_to_base.is_empty() {
        root_history.current().clone_for_search()
    } else {
        let mut pos = root_history.current().clone_for_search();
        for step in moves_to_base {
            let Some(mv) = tree
                .get(step.node_id)
                .and_then(|n| n.children.get(step.edge_idx).map(|e| e.mv))
            else {
                break;
            };
            pos.do_move(mv);
            scratch.path_positions.push(pos.clone_for_search());
        }
        pos
    };
    let mut completed_visits = 0i32;
    let mut max_limit = i32::MAX;

    let mut best_root_idx = 0usize;
    let mut is_root_node = node_id == root_id && path.is_empty();
    workspace.current_path.push(-1);
    while let Some(&path_tail) = workspace.current_path.last() {
        if path_tail == -1 {
            let mut cur_limit = if workspace.current_path.len() > 1 {
                let edge_idx = workspace.current_path[workspace.current_path.len() - 2] as usize;
                workspace
                    .visits_to_perform
                    .last()
                    .and_then(|vtp| vtp.get(edge_idx).copied())
                    .unwrap_or(0)
            } else {
                collision_limit
            };

            let Some(node_snapshot) = tree.get(node_id) else {
                break;
            };
            let node_visits = node_snapshot.visits;
            let node_terminal = node_snapshot.is_terminal();
            let children_empty = node_snapshot.children.is_empty();

            if node_terminal {
                if let Some(mut pending) = pending_from_node(
                    tree,
                    root_id,
                    root_history,
                    scratch,
                    &path,
                    &pos,
                    base_key_counts,
                    node_id,
                ) {
                    pending.multivisit = cur_limit as u32;
                    receiver.push(pending);
                    completed_visits += cur_limit;
                } else if cur_limit > 0 {
                    receiver.push(PendingNode::new(
                        pending_key_for_node(node_id, &path),
                        path.clone(),
                        PendingKind::Collision {
                            max_count: collision_max_count(
                                cur_limit,
                                collision_limit,
                                max_limit,
                                is_root_node,
                            ),
                        },
                        cur_limit as u32,
                        0,
                    ));
                    completed_visits += cur_limit;
                }
                ascend_after_n0_shortcut(
                    &mut node_id,
                    root_id,
                    &mut path,
                    &mut pos,
                    root_history,
                    scratch,
                    workspace,
                );
                continue;
            }

            if children_empty {
                if is_root_node {
                    if let Some(node) = tree.get_mut(node_id) {
                        if node.try_start_score_update() {
                            cur_limit -= 1;
                            if let Some(mut pending) = pending_from_node(
                                tree,
                                root_id,
                                root_history,
                                scratch,
                                &path,
                                &pos,
                                base_key_counts,
                                node_id,
                            ) {
                                pending.multivisit = 1;
                                receiver.push(pending);
                                completed_visits += 1;
                            }
                        }
                    }
                } else {
                    let started = tree
                        .get_mut(node_id)
                        .map(|node| node.try_start_score_update())
                        .unwrap_or(false);
                    if started {
                        if let Some(mut pending) = pending_from_node(
                            tree,
                            root_id,
                            root_history,
                            scratch,
                            &path,
                            &pos,
                            base_key_counts,
                            node_id,
                        ) {
                            pending.multivisit = cur_limit as u32;
                            receiver.push(pending);
                            completed_visits += cur_limit;
                            cur_limit = 0;
                        } else if let Some(node) = tree.get_mut(node_id) {
                            cancel_score_update(&mut node.in_flight, 1);
                        }
                    }
                }
                if cur_limit > 0 {
                    receiver.push(PendingNode::new(
                        pending_key_for_node(node_id, &path),
                        path.clone(),
                        PendingKind::Collision {
                            max_count: collision_max_count(
                                cur_limit,
                                collision_limit,
                                max_limit,
                                is_root_node,
                            ),
                        },
                        cur_limit as u32,
                        0,
                    ));
                    completed_visits += cur_limit;
                }
                ascend_after_n0_shortcut(
                    &mut node_id,
                    root_id,
                    &mut path,
                    &mut pos,
                    root_history,
                    scratch,
                    workspace,
                );
                continue;
            }

            if is_root_node {
                if let Some(root) = tree.get_mut(root_id) {
                    root.increment_n_in_flight(cur_limit as u32);
                }
                best_root_idx = tree
                    .get(node_id)
                    .map(|n| {
                        n.children
                            .iter()
                            .enumerate()
                            .max_by_key(|(_, edge)| pick_edge_visits(tree, edge))
                            .map(|(idx, _)| idx)
                            .unwrap_or(0)
                    })
                    .unwrap_or(0);
            }

            let num_edges = tree
                .get(node_id)
                .map(|n| n.children.len().min(MAX_VTP_EDGES))
                .unwrap_or(0);
            if num_edges == 0 {
                backtrack_pick_tree(
                    &mut node_id,
                    root_id,
                    &mut path,
                    &mut pos,
                    root_history,
                    scratch,
                    workspace,
                );
                continue;
            }
            let edge_priors: Vec<f32> = tree
                .get(node_id)
                .map(|n| n.children.iter().take(num_edges).map(|e| e.prior).collect())
                .unwrap_or_default();
            let edge_moves: Vec<Move> = tree
                .get(node_id)
                .map(|n| n.children.iter().take(num_edges).map(|e| e.mv).collect())
                .unwrap_or_default();
            let children_visits = tree
                .get(node_id)
                .map(|n| n.children_visits())
                .unwrap_or(0);

            if let Some(mut vtp) = workspace.vtp_buffer.pop() {
                vtp.resize(MAX_VTP_EDGES, 0);
                workspace.visits_to_perform.push(vtp);
            } else {
                workspace.visits_to_perform.push(vec![0i32; MAX_VTP_EDGES]);
            }
            workspace.vtp_last_filled.push(-1);

            let depth_from_root = path.len() as u32;
            let draw_score = if depth_from_root % 2 == 0 {
                config.draw_score
            } else {
                -config.draw_score
            };
            let cpuct = config.cpuct_for(is_root_node, node_visits);
            let sqrt_parent = (children_visits.max(1)) as f32;
            let puct_mult = cpuct * sqrt_parent.sqrt();
            let parent_q = tree
                .get(node_id)
                .map(|n| n.mean_value_with_draw(draw_score))
                .unwrap_or(0.0);
            let visited_policy = tree
                .get(node_id)
                .map(|n| {
                    n.children
                        .iter()
                        .take(num_edges)
                        .filter(|edge| edge_n_started(tree, edge, is_root_node, config) > 0.0)
                        .map(|edge| edge.prior)
                        .sum::<f32>()
                })
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let default_q = config
                .get_fpu(is_root_node, parent_q, visited_policy)
                .clamp(-1.0, 1.0);

            let mut current_util = vec![f32::NEG_INFINITY; num_edges];
            let mut current_score = vec![0.0f32; num_edges];
            let mut current_nstarted = vec![0i32; num_edges];

            for idx in 0..num_edges {
                let edge = tree.get(node_id).and_then(|n| n.children.get(idx));
                let Some(edge) = edge else { continue };
                let q = edge_q(tree, edge, default_q, draw_score);
                current_util[idx] = q;
                current_nstarted[idx] = edge_n_started(tree, edge, is_root_node, config) as i32;
                current_score[idx] =
                    edge_priors[idx] * puct_mult / (1.0 + current_nstarted[idx] as f32) + q;
            }

            let level = workspace.visits_to_perform.len() - 1;
            while cur_limit > 0 {
                let mut best = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                let mut best_without_u = f32::NEG_INFINITY;
                let mut second_best = f32::NEG_INFINITY;
                let mut can_exit = false;

                for idx in 0..num_edges {
                    if is_root_node && config.smart_pruning_factor > 0.0 && best_root_n > 0 && idx != best_root_idx {
                        let edge = tree.get(node_id).and_then(|n| n.children.get(idx));
                        if let Some(edge) = edge {
                            let n = pick_edge_visits(tree, edge);
                            if n < best_root_n
                                && remaining_playouts < best_root_n.saturating_sub(n)
                            {
                                continue;
                            }
                        }
                    }
                    let score = current_score[idx];
                    let util = current_util[idx];
                    let nstarted = current_nstarted[idx];
                    if score > best {
                        second_best = best;
                        best = score;
                        best_idx = idx;
                        best_without_u = util;
                    } else if score > second_best {
                        second_best = score;
                    }
                    if can_exit {
                        break;
                    }
                    if nstarted == 0 {
                        can_exit = true;
                    }
                }

                if best <= f32::NEG_INFINITY / 2.0 || best_idx >= num_edges {
                    if is_root_node {
                        best_idx = best_root_idx.min(num_edges.saturating_sub(1));
                        best = current_score[best_idx];
                        best_without_u = current_util[best_idx];
                    }
                    if best <= f32::NEG_INFINITY / 2.0 {
                        break;
                    }
                }

                let mut new_visits;
                if second_best > f32::NEG_INFINITY / 2.0 {
                    let mut estimated = i32::MAX;
                    if best_without_u < second_best {
                        let n1 = current_nstarted[best_idx] + 1;
                        estimated = (edge_priors[best_idx] * puct_mult / (second_best - best_without_u)
                            - n1 as f32
                            + 1.0)
                            .max(1.0)
                            .min(1e9) as i32;
                    }
                    max_limit = max_limit.min(estimated);
                    new_visits = cur_limit.min(estimated);
                } else {
                    new_visits = cur_limit;
                }

                let (edge_visits, edge_in_flight, existing_child) = tree
                    .get(node_id)
                    .map(|n| {
                        let edge = &n.children[best_idx];
                        (edge.visits, edge.in_flight, edge.child)
                    })
                    .unwrap_or((0, 0, None));
                // 边级 in-flight（尚无子节点）：与 lc0 子节点 TryStart 失败后的 collision 等价。
                if existing_child.is_none() && edge_visits == 0 && edge_in_flight > 0 {
                    receiver.push(PendingNode::new(
                        PendingKey::NewEdge(node_id, best_idx),
                        path.clone(),
                        PendingKind::Collision { max_count: 0 },
                        new_visits as u32,
                        0,
                    ));
                    cur_limit -= new_visits;
                    continue;
                }

                {
                    let vtp = &mut workspace.visits_to_perform[level];
                    if best_idx as i32 > workspace.vtp_last_filled[level] {
                        let start = (workspace.vtp_last_filled[level] + 1).max(0) as usize;
                        for slot in &mut vtp[start..=best_idx] {
                            *slot = 0;
                        }
                    }
                    vtp[best_idx] += new_visits;
                }
                cur_limit -= new_visits;

                let mv = edge_moves[best_idx];
                let mut next_pos = pos.clone_for_search();
                next_pos.do_move(mv);
                let child_id = tree.get_or_spawn_child(node_id, best_idx, next_pos.key());
                let Some(child_id) = child_id else {
                    break;
                };
                if let Some(stats) = stats {
                    ensure_node_twofold_correct_for_depth(tree, stats, child_id, &path);
                }

                if let Some(child) = tree.get_mut(child_id) {
                    let mut decremented = false;
                    if child.try_start_score_update() {
                        new_visits -= 1;
                        decremented = true;
                        if child.visits > 0 && !child.is_terminal() {
                            child.increment_n_in_flight(new_visits as u32);
                            current_nstarted[best_idx] += new_visits;
                        }
                        current_score[best_idx] = edge_priors[best_idx] * puct_mult
                            / (1.0 + current_nstarted[best_idx] as f32)
                            + current_util[best_idx];
                    }

                    let child_visits = child.visits;
                    let child_terminal = child.is_terminal();
                    if decremented && (child_visits == 0 || child_terminal) {
                        let vtp = &mut workspace.visits_to_perform[level];
                        vtp[best_idx] = vtp[best_idx].saturating_sub(1);
                        queue_immediate_child_visit(
                            tree,
                            root_id,
                            root_history,
                            receiver,
                            &path,
                            scratch,
                            &pos,
                            base_key_counts,
                            node_id,
                            best_idx,
                            mv,
                            child_id,
                            &mut completed_visits,
                        );
                    }
                }

                if best_idx as i32 > workspace.vtp_last_filled[level]
                    && workspace.visits_to_perform[level][best_idx] > 0
                {
                    workspace.vtp_last_filled[level] = best_idx as i32;
                }
            }

            is_root_node = false;

            if let Some(enqueue) = enqueue {
                let level = workspace.visits_to_perform.len().saturating_sub(1);
                if level < workspace.visits_to_perform.len() {
                    let vtp_last = workspace.vtp_last_filled[level];
                    let num_edges = tree
                        .get(node_id)
                        .map(|n| n.children.len().min(MAX_VTP_EDGES))
                        .unwrap_or(0);
                    for idx in 0..=vtp_last as usize {
                        if idx >= num_edges {
                            break;
                        }
                        let child_limit = workspace.visits_to_perform[level][idx];
                        if child_limit <= config.minimum_work_size_for_picking {
                            continue;
                        }
                        let remaining =
                            top_collision_limit - *passed_off - completed_visits;
                        if child_limit >= (remaining * 2 / 3) {
                            continue;
                        }
                        if child_limit + *passed_off + completed_visits
                            >= top_collision_limit
                                - config.minimum_remaining_work_size_for_picking
                        {
                            continue;
                        }
                        let child_id = {
                            let edge_mv = tree
                                .get(node_id)
                                .and_then(|n| n.children.get(idx).map(|e| e.mv));
                            let existing = tree
                                .get(node_id)
                                .and_then(|n| n.children.get(idx).and_then(|e| e.child));
                            if let Some(id) = existing {
                                Some(id)
                            } else if let Some(mv) = edge_mv {
                                let mut next_pos = pos.clone_for_search();
                                next_pos.do_move(mv);
                                tree.get_or_spawn_child(node_id, idx, next_pos.key())
                            } else {
                                None
                            }
                        };
                        let Some(child_id) = child_id else {
                            continue;
                        };
                        let Some(child) = tree.get(child_id) else {
                            continue;
                        };
                        if child.visits == 0 || child.is_terminal() {
                            continue;
                        }
                        let mut path_to_child = path.clone();
                        path_to_child.push(PathStep {
                            node_id,
                            edge_idx: idx,
                        });
                        let child_base_depth = path.len() + base_depth;
                        if enqueue.try_enqueue_gathering(
                            child_id,
                            child_base_depth,
                            path_to_child,
                            child_limit,
                        ) {
                            workspace.visits_to_perform[level][idx] = 0;
                            *passed_off += child_limit;
                        }
                    }
                }
            }
            // lc0：UCT 分配完成后 current_path.back() 仍为 -1，进入子节点选择。
        }
        let min_idx = workspace.current_path.last().copied().unwrap_or(-1);
        let mut found_child = false;
        if let (Some(vtp), Some(&vtp_last)) = (
            workspace.visits_to_perform.last(),
            workspace.vtp_last_filled.last(),
        ) {
            if vtp_last > min_idx {
                let node = tree.get(node_id).expect("pick node must exist");
                for (idx, edge) in node.children.iter().enumerate().take(MAX_VTP_EDGES) {
                    if (idx as i32) <= min_idx {
                        continue;
                    }
                    if vtp.get(idx).copied().unwrap_or(0) <= 0 {
                        if idx as i32 >= vtp_last {
                            break;
                        }
                        continue;
                    }
                    let mv = edge.mv;
                    pos.do_move(mv);
                    scratch.path_positions.push(pos.clone_for_search());
                    path.push(PathStep {
                        node_id,
                        edge_idx: idx,
                    });
                    let child_key = pos.key();
                    let child_id = tree
                        .get_or_spawn_child(node_id, idx, child_key)
                        .expect("spawn child");
                    if let Some(stats) = stats {
                        ensure_node_twofold_correct_for_depth(tree, stats, child_id, &path);
                    }
                    node_id = child_id;
                    if let Some(tail) = workspace.current_path.last_mut() {
                        if *tail == -1 {
                            *tail = idx as i32;
                        } else {
                            workspace.current_path.push(idx as i32);
                        }
                    } else {
                        workspace.current_path.push(idx as i32);
                    }
                    workspace.current_path.push(-1);
                    found_child = true;
                    break;
                }
            }
        }
        if !found_child {
            backtrack_pick_tree(
                &mut node_id,
                root_id,
                &mut path,
                &mut pos,
                root_history,
                scratch,
                workspace,
            );
        }
    }

    completed_visits
}

/// lc0：`GetParent()` + `current_path.pop_back()`；不 pop `visits_to_perform`。
fn ascend_after_n0_shortcut(
    node_id: &mut MctsNodeId,
    root_id: MctsNodeId,
    path: &mut Vec<PathStep>,
    pos: &mut Position,
    root_history: &PositionHistory,
    scratch: &mut SelectionScratch,
    workspace: &mut PickWorkspace,
) {
    if let Some(step) = path.pop() {
        if !scratch.path_positions.is_empty() {
            scratch.path_positions.pop();
        }
        *node_id = step.node_id;
        *pos = rebuild_position(root_history, scratch);
    } else {
        *node_id = root_id;
        *pos = root_history.current().clone_for_search();
    }
    workspace.current_path.pop();
}

/// lc0：`!found_child` 回溯，vtp 移入 buffer 复用。
fn backtrack_pick_tree(
    node_id: &mut MctsNodeId,
    root_id: MctsNodeId,
    path: &mut Vec<PathStep>,
    pos: &mut Position,
    root_history: &PositionHistory,
    scratch: &mut SelectionScratch,
    workspace: &mut PickWorkspace,
) {
    if let Some(step) = path.pop() {
        if !scratch.path_positions.is_empty() {
            scratch.path_positions.pop();
        }
        *node_id = step.node_id;
        *pos = rebuild_position(root_history, scratch);
    } else {
        *node_id = root_id;
        *pos = root_history.current().clone_for_search();
    }
    workspace.current_path.pop();
    if let Some(vtp) = workspace.visits_to_perform.pop() {
        workspace.vtp_buffer.push(vtp);
    }
    if !workspace.vtp_last_filled.is_empty() {
        workspace.vtp_last_filled.pop();
    }
}

fn queue_immediate_child_visit(
    tree: &MctsTree,
    root_id: MctsNodeId,
    root_history: &PositionHistory,
    receiver: &mut Vec<PendingNode>,
    path: &[PathStep],
    scratch: &SelectionScratch,
    parent_pos: &Position,
    base_key_counts: &HashMap<u64, usize>,
    parent_id: MctsNodeId,
    edge_idx: usize,
    mv: Move,
    child_id: MctsNodeId,
    completed_visits: &mut i32,
) {
    let mut visit_path = path.to_vec();
    visit_path.push(PathStep {
        node_id: parent_id,
        edge_idx,
    });
    let mut visit_pos = parent_pos.clone_for_search();
    visit_pos.do_move(mv);
    let mut visit_positions = scratch.path_positions.clone();
    let mut visit_key_counts = scratch.path_key_counts.clone();
    let repeated = PositionHistory::push_search_path_position(
        base_key_counts,
        &mut visit_key_counts,
        &visit_pos,
    );
    visit_positions.push(visit_pos.clone_for_search());
    if repeated {
        receiver.push(PendingNode::new(
            PendingKey::NewEdge(parent_id, edge_idx),
            visit_path.clone(),
            PendingKind::NewTerminal {
                state_key: visit_pos.key(),
                wl: 0.0,
                d: 1.0,
                m: visit_path.len() as f32,
                terminal_kind: TerminalKind::TwoFold,
            },
            1,
            0,
        ));
        *completed_visits += 1;
        return;
    }
    let visit_scratch = SelectionScratch {
        path_positions: visit_positions,
        path_key_counts: visit_key_counts,
    };
    if let Some(pending) = pending_from_node(
        tree,
        root_id,
        root_history,
        &visit_scratch,
        &visit_path,
        &visit_pos,
        base_key_counts,
        child_id,
    ) {
        receiver.push(pending);
        *completed_visits += 1;
    }
}

fn rebuild_position(root_history: &PositionHistory, scratch: &SelectionScratch) -> Position {
    scratch
        .path_positions
        .last()
        .cloned()
        .unwrap_or_else(|| root_history.current().clone_for_search())
}

fn collision_max_count(
    cur_limit: i32,
    collision_limit: i32,
    max_limit: i32,
    is_root_node: bool,
) -> u32 {
    if is_root_node && cur_limit == collision_limit && max_limit > cur_limit {
        max_limit as u32
    } else {
        0
    }
}

fn pending_key_for_node(node_id: MctsNodeId, path: &[PathStep]) -> PendingKey {
    if let Some(step) = path.last() {
        PendingKey::NewEdge(step.node_id, step.edge_idx)
    } else {
        PendingKey::ExistingLeaf(node_id)
    }
}

fn pending_from_node(
    tree: &MctsTree,
    _root_id: MctsNodeId,
    _root_history: &PositionHistory,
    _scratch: &SelectionScratch,
    path: &[PathStep],
    pos: &Position,
    _base_key_counts: &HashMap<u64, usize>,
    node_id: MctsNodeId,
) -> Option<PendingNode> {
    let node = tree.get(node_id)?;
    if let Some(value) = node.terminal_value {
        let (wl, d, m) = terminal_wdl(value);
        return Some(PendingNode::new(
            PendingKey::ExistingLeaf(node_id),
            path.to_vec(),
            PendingKind::ExistingTerminal {
                leaf_id: node_id,
                wl,
                d,
                m,
            },
            1,
            0,
        ));
    }
    if node.children.is_empty() {
        if !node.expanded {
            let mut buf = vec![ExtMove {
                mv: Move::none(),
                value: 0,
            }; MAX_MOVES];
            let n = generate(pos, GenType::Legal, &mut buf);
            if n == 0 {
                let (wl, d, m) = terminal_wdl(-1.0);
                let parent = path.last()?;
                return Some(PendingNode::new(
                    PendingKey::NewEdge(parent.node_id, parent.edge_idx),
                    path.to_vec(),
                    PendingKind::NewTerminal {
                        state_key: pos.key(),
                        wl,
                        d,
                        m,
                        terminal_kind: TerminalKind::Generic,
                    },
                    1,
                    0,
                ));
            }
            return Some(PendingNode::new(
                PendingKey::ExistingLeaf(node_id),
                path.to_vec(),
                PendingKind::ExpandInPlace { node_id },
                1,
                0,
            ));
        }
        return None;
    }
    None
}

pub(crate) fn pick_edge_visits(tree: &MctsTree, edge: &EdgeStats) -> u32 {
    edge.child
        .and_then(|id| tree.get(id))
        .map(|child| child.visits)
        .unwrap_or(0)
}

fn edge_n_started(
    tree: &MctsTree,
    edge: &EdgeStats,
    is_root: bool,
    config: MctsConfig,
) -> f32 {
    if let Some(child_id) = edge.child {
        if let Some(child) = tree.get(child_id) {
            let started = child.n_started() as f32;
            if is_root {
                return started * config.root_inflight_fraction.clamp(0.0, 1.0);
            }
            return started;
        }
    }
    if is_root {
        edge.visits as f32 + edge.in_flight as f32 * config.root_inflight_fraction.clamp(0.0, 1.0)
    } else {
        edge.n_started() as f32
    }
}

fn edge_q(tree: &MctsTree, edge: &EdgeStats, default_q: f32, draw_score: f32) -> f32 {
    if let Some(child_id) = edge.child {
        if let Some(child) = tree.get(child_id) {
            if child.visits > 0 {
                return child.mean_value_with_draw(draw_score);
            }
        }
    }
    default_q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::config::MctsConfig;
    use std::time::Duration;

    #[test]
    fn budget_exhausted_or_combines_movetime_and_nodes() {
        let deadline = Instant::now() + Duration::from_millis(50);
        let budget = MctsBudget {
            max_playouts: None,
            max_nodes: Some(4),
            max_depth: None,
            max_mate: None,
            deadline: Some(deadline),
            stop: None,
        };
        assert!(!budget_exhausted(&budget, 0, 0, 0, None));
        assert!(budget_exhausted(&budget, 3, 1, 0, None));
        std::thread::sleep(Duration::from_millis(60));
        assert!(budget_exhausted(&budget, 0, 0, 0, None));
    }

    #[test]
    fn max_out_of_order_scales_with_batch() {
        let config = MctsConfig::default();
        assert_eq!(config.max_out_of_order(256), 614);
        assert_eq!(config.max_out_of_order(1), 2);
    }

    #[test]
    fn apply_out_of_order_backups_removes_terminal_pending() {
        let mut tree = MctsTree::default();
        let root_id = tree.add_node(MctsNode {
            state_key: 0,
            visits: 1,
            in_flight: 0,
            wl: 0.0,
            d: 0.0,
            m: 0.0,
            expanded: false,
            terminal_kind: TerminalKind::NonTerminal,
            terminal_value: None,
            children: Vec::new(),
        });
        let mut iteration = SearchIteration {
            minibatch_size: 1,
            ..SearchIteration::default()
        };
        iteration.pending.push(PendingNode {
            key: PendingKey::ExistingLeaf(root_id),
            kind: PendingKind::ExistingTerminal {
                leaf_id: root_id,
                wl: 1.0,
                d: 0.0,
                m: 0.0,
            },
            multivisit: 1,
            collision_upsize: 0,
            path: Vec::new(),
            nn_queried: false,
            is_cache_hit: false,
            ooo_completed: true,
            eval: None,
            task: None,
        });
        let mut minibatch_size = iteration.minibatch_size;
        apply_out_of_order_backups(&mut tree, &mut iteration, None, 0, &mut minibatch_size);
        assert!(iteration.pending.is_empty());
        assert_eq!(iteration.number_out_of_order, 1);
        assert_eq!(minibatch_size, 0);
        assert_eq!(tree.get(root_id).map(|n| n.visits).unwrap_or(0), 2);
    }
}
