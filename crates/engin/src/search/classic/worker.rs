//! px0 `src/search/classic/search.h:201-448` 的 P4 worker。
//!
//! P3 仍由 `SearchSession` 单线程直连 `Backend::evaluate()`。
//! P4 worker 七阶段流水线已可单线程跑通；碰撞/task workers/OOO 完整语义
//! 与 UCI 接线仍属开放项。

use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use xiangqi_core::{GameResult, Move, MoveList, PositionHistory};

use crate::neural::backend::{AddInputResult, Backend, BackendComputation, EvalPosition, EvalResult, EvalTicket};
use crate::EnginError;

use super::node::{NodeTree, Terminal};
use super::params::SearchParams;

/// px0 `Search` 中与 worker 相关的计数子集 (`search.h:49-200`)。
#[derive(Debug)]
pub struct WorkerSearchState {
    pub stop: Arc<AtomicBool>,
    pub remaining_playouts: AtomicU64,
    /// `go nodes` 预算；0 表示无限制。
    pub nodes_budget: AtomicU64,
    pub thread_count: AtomicUsize,
    pub shared_collisions: Mutex<Vec<(usize, u32)>>,
    pub total_playouts: AtomicU64,
    pub total_batches: AtomicU64,
    pub network_evaluations: AtomicU64,
    pub cum_depth: AtomicU64,
    pub max_depth: AtomicU16,
}

impl Default for WorkerSearchState {
    fn default() -> Self {
        Self::new(Arc::new(AtomicBool::new(false)), i64::MAX)
    }
}

impl WorkerSearchState {
    pub fn new(stop: Arc<AtomicBool>, remaining_playouts: i64) -> Self {
        Self {
            stop,
            remaining_playouts: AtomicU64::new(remaining_playouts.max(0) as u64),
            nodes_budget: AtomicU64::new(0),
            thread_count: AtomicUsize::new(1),
            shared_collisions: Mutex::new(Vec::new()),
            total_playouts: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            cum_depth: AtomicU64::new(0),
            max_depth: AtomicU16::new(0),
        }
    }

    pub fn remaining_playouts(&self) -> i64 {
        self.remaining_playouts.load(Ordering::Acquire) as i64
    }

    pub fn set_remaining_playouts(&self, value: i64) {
        self.remaining_playouts.store(value.max(0) as u64, Ordering::Release);
    }

    pub fn set_nodes_budget(&self, nodes: u32) {
        self.nodes_budget.store(nodes as u64, Ordering::Release);
    }

    pub fn nodes_budget_reached(&self, root_visits: u32) -> bool {
        let budget = self.nodes_budget.load(Ordering::Acquire);
        budget > 0 && root_visits >= budget as u32
    }
}

/// px0 `SearchWorker::NodeToProcess` (`search.h:288-347`)。
#[derive(Clone, Debug)]
pub struct NodeToProcess {
    pub node_idx: usize,
    pub eval: EvalResult,
    pub multivisit: u32,
    pub maxvisit: u32,
    pub depth: u16,
    pub nn_queried: bool,
    pub is_cache_hit: bool,
    pub is_collision: bool,
    pub moves_to_visit: MoveList,
    pub ooo_completed: bool,
    pub eval_ticket: Option<EvalTicket>,
}

impl NodeToProcess {
    /// px0 `NodeToProcess::Visit` (`search.h:334-336`)。
    pub fn visit(node_idx: usize, depth: u16) -> Self {
        Self {
            node_idx,
            eval: EvalResult::default(),
            multivisit: 1,
            maxvisit: 0,
            depth,
            nn_queried: false,
            is_cache_hit: false,
            is_collision: false,
            moves_to_visit: Vec::new(),
            ooo_completed: false,
            eval_ticket: None,
        }
    }

    /// px0 `NodeToProcess::Collision` (`search.h:321-332`)。
    pub fn collision(node_idx: usize, depth: u16, multivisit: u32, maxvisit: u32) -> Self {
        Self {
            node_idx,
            eval: EvalResult::default(),
            multivisit,
            maxvisit,
            depth,
            nn_queried: false,
            is_cache_hit: false,
            is_collision: true,
            moves_to_visit: Vec::new(),
            ooo_completed: false,
            eval_ticket: None,
        }
    }

    /// px0 `NodeToProcess::IsExtendable` (`search.h:289`)。
    pub const fn is_extendable(&self, is_terminal: bool) -> bool {
        !self.is_collision && !is_terminal
    }

    /// px0 `NodeToProcess::CanEvalOutOfOrder` (`search.h:303-305`)。
    pub const fn can_eval_out_of_order(&self, is_terminal: bool) -> bool {
        self.is_cache_hit || is_terminal
    }
}

/// px0 `SearchWorker::TaskWorkspace` (`src/search/classic/search.h:348-365`)。
///
/// px0 的 `cur_iters` 是指针迭代器；arena 索引可在读取时重建，故 Rust 只保留
/// 跨层分配和路径状态。每个 task workspace 都必须拥有独立 history，供
/// `ProcessPickedTask` 复原根局面后扩展叶子。
struct TaskWorkspace {
    vtp_buffer: Vec<Vec<u32>>,
    visits_to_perform: Vec<Vec<u32>>,
    vtp_last_filled: Vec<isize>,
    current_path: Vec<isize>,
    moves_to_path: MoveList,
    history: PositionHistory,
}

/// px0 `SearchWorker::PickTask` (`src/search/classic/search.h:367-393`).
/// A gathering task owns a disjoint subtree root; a processing task owns a
/// non-overlapping minibatch range.
#[derive(Debug)]
pub struct PickTask {
    pub kind: PickTaskKind,
    pub start: Option<usize>,
    pub base_depth: u16,
    pub collision_limit: u32,
    pub moves_to_base: MoveList,
    pub results: Vec<NodeToProcess>,
    pub start_idx: usize,
    pub end_idx: usize,
    pub complete: bool,
}

/// px0 `PickTask::PickTaskType` (`src/search/classic/search.h:368-370`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickTaskKind {
    Gathering,
    Processing,
}

impl PickTask {
    /// px0 gathering constructor (`src/search/classic/search.h:384-390`).
    pub fn gathering(start: usize, base_depth: u16, moves_to_base: MoveList, collision_limit: u32) -> Self {
        Self {
            kind: PickTaskKind::Gathering,
            start: Some(start),
            base_depth,
            collision_limit,
            moves_to_base,
            results: Vec::new(),
            start_idx: 0,
            end_idx: 0,
            complete: false,
        }
    }

    /// px0 processing constructor (`src/search/classic/search.h:391-392`).
    pub fn processing(start_idx: usize, end_idx: usize) -> Self {
        Self {
            kind: PickTaskKind::Processing,
            start: None,
            base_depth: 0,
            collision_limit: 0,
            moves_to_base: Vec::new(),
            results: Vec::new(),
            start_idx,
            end_idx,
            complete: false,
        }
    }
}

/// px0 task queue state (`src/search/classic/search.h:435-445`,
/// `search.cc:1464-1483`). Task execution is wired separately.
#[derive(Default)]
pub struct PickTaskQueue {
    tasks: Mutex<Vec<PickTask>>,
    task_count: AtomicUsize,
    tasks_taken: AtomicUsize,
    completed_tasks: AtomicUsize,
    task_added: Condvar,
}

impl PickTaskQueue {
    const MAX_TASKS: usize = 100;

    /// px0 `SearchWorker::ResetTasks` (`src/search/classic/search.cc:1466-1473`).
    pub fn reset(&self) {
        self.task_count.store(0, Ordering::Release);
        self.tasks_taken.store(0, Ordering::Release);
        self.completed_tasks.store(0, Ordering::Release);
        self.tasks.lock().expect("pick task queue lock").clear();
    }

    /// px0 task enqueue (`src/search/classic/search.cc:1843-1856`).
    pub fn push(&self, task: PickTask) -> bool {
        let mut tasks = self.tasks.lock().expect("pick task queue lock");
        if tasks.len() >= Self::MAX_TASKS {
            return false;
        }
        tasks.push(task);
        drop(tasks);
        self.task_count.fetch_add(1, Ordering::AcqRel);
        self.task_added.notify_all();
        true
    }

    /// px0 task claim (`src/search/classic/search.cc:1076-1093`).
    pub fn take(&self) -> Option<(usize, PickTask)> {
        let index = loop {
            let taken = self.tasks_taken.load(Ordering::Acquire);
            if taken >= self.task_count.load(Ordering::Acquire) {
                return None;
            }
            if self
                .tasks_taken
                .compare_exchange_weak(taken, taken + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break taken;
            }
        };
        self.tasks.lock().expect("pick task queue lock").get(index).map(|task| {
            (
                index,
                PickTask {
                    kind: task.kind,
                    start: task.start,
                    base_depth: task.base_depth,
                    collision_limit: task.collision_limit,
                    moves_to_base: task.moves_to_base.clone(),
                    results: Vec::new(),
                    start_idx: task.start_idx,
                    end_idx: task.end_idx,
                    complete: false,
                },
            )
        })
    }

    /// px0 completion accounting (`src/search/classic/search.cc:1136-1137`).
    pub fn complete(&self, index: usize, results: Vec<NodeToProcess>) {
        if let Some(task) = self.tasks.lock().expect("pick task queue lock").get_mut(index) {
            task.results = results;
            task.complete = true;
            self.completed_tasks.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// px0 `SearchWorker::WaitForTasks` (`src/search/classic/search.cc:1475-1483`).
    pub fn wait(&self) {
        while self.completed_tasks.load(Ordering::Acquire) < self.task_count.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    /// px0 result merge (`src/search/classic/search.cc:1501-1507`).
    pub fn drain_results_into(&self, receiver: &mut Vec<NodeToProcess>) {
        self.wait();
        let mut tasks = self.tasks.lock().expect("pick task queue lock");
        for task in tasks.iter_mut() {
            receiver.append(&mut task.results);
        }
    }
}

impl Default for TaskWorkspace {
    /// px0 `TaskWorkspace::TaskWorkspace` (`src/search/classic/search.h:357-364`).
    fn default() -> Self {
        const INITIAL_DEPTH: usize = 30;
        let mut workspace = Self {
            vtp_buffer: Vec::with_capacity(INITIAL_DEPTH),
            visits_to_perform: Vec::with_capacity(INITIAL_DEPTH),
            vtp_last_filled: Vec::with_capacity(INITIAL_DEPTH),
            current_path: Vec::with_capacity(INITIAL_DEPTH),
            moves_to_path: MoveList::with_capacity(INITIAL_DEPTH),
            history: PositionHistory::default(),
        };
        workspace.history.reserve(INITIAL_DEPTH);
        workspace
    }
}

/// px0 `SearchWorker` (`src/search/classic/search.h:203-448`)。
pub struct SearchWorker<'a> {
    tree: &'a mut NodeTree,
    backend: &'a dyn Backend,
    params: &'a SearchParams,
    search_state: &'a WorkerSearchState,
    minibatch: Vec<NodeToProcess>,
    computation: Option<Box<dyn BackendComputation>>,
    history: PositionHistory,
    target_minibatch_size: usize,
    max_out_of_order: usize,
    task_workers: i32,
    number_out_of_order: usize,
    played_history_len: usize,
    task_queue: PickTaskQueue,
    picking_workspace: TaskWorkspace,
}

impl<'a> SearchWorker<'a> {
    /// px0 `SearchWorker::SearchWorker` (`search.h:205-233`)。
    pub fn new(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        Self::from_parts(tree, backend, params, search_state)
    }

    pub fn with_context(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        Self::from_parts(tree, backend, params, search_state)
    }

    fn from_parts(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        let mut target_minibatch_size = if params.minibatch_size > 0 {
            params.minibatch_size as usize
        } else {
            backend.attributes().recommended_batch_size
        };
        if target_minibatch_size == 0 {
            target_minibatch_size = 1;
        }
        let mut task_workers = params.task_workers_per_search_worker;
        if task_workers < 0 {
            if backend.attributes().runs_on_cpu {
                task_workers = 0;
            } else {
                let working_threads =
                    std::cmp::max(search_state.thread_count.load(Ordering::Acquire).saturating_sub(1), 1);
                task_workers = std::thread::available_parallelism()
                    .map_or(1, usize::from)
                    .saturating_div(working_threads)
                    .saturating_sub(1)
                    .min(4) as i32;
            }
        }
        let max_out_of_order = std::cmp::max(
            1,
            (params.max_out_of_order_evals_factor * target_minibatch_size as f32) as usize,
        );
        Self {
            history: tree.history().clone(),
            played_history_len: tree.history().len(),
            tree,
            backend,
            params,
            search_state,
            minibatch: Vec::new(),
            computation: None,
            target_minibatch_size,
            max_out_of_order,
            task_workers,
            number_out_of_order: 0,
            task_queue: PickTaskQueue::default(),
            picking_workspace: TaskWorkspace::default(),
        }
    }

    /// px0 `SearchWorker::ExecuteOneIteration` (`search.cc:1142-1231`)。
    pub fn execute_one_iteration(&mut self) -> Result<(), EnginError> {
        let root = self.tree.current_head();
        if self.search_state.nodes_budget_reached(self.tree.node(root).n()) {
            self.search_state.stop.store(true, Ordering::Release);
            return Ok(());
        }
        self.initialize_iteration()?;
        self.gather_minibatch()?;
        self.collect_collisions()?;
        self.maybe_prefetch_into_cache()?;
        self.run_nn_computation()?;
        self.fetch_minibatch_results()?;
        self.do_backup_update()?;
        self.update_counters()
    }

    /// 单线程测试入口：重复执行 iteration 直到 root N 达标。
    pub fn run_until_root_visits(&mut self, target: u32) -> Result<(), EnginError> {
        while self.tree.node(self.tree.current_head()).n() < target {
            if self.search_state.stop.load(Ordering::Acquire) {
                break;
            }
            self.execute_one_iteration()?;
        }
        Ok(())
    }

    /// px0 `SearchWorker::InitializeIteration` (`search.cc:1233-1266`)。
    pub fn initialize_iteration(&mut self) -> Result<(), EnginError> {
        self.computation = Some(self.backend.create_computation()?);
        self.minibatch.clear();
        self.minibatch.reserve(2 * self.target_minibatch_size);
        self.history = self.tree.history().clone();
        self.played_history_len = self.history.len();
        Ok(())
    }

    /// px0 `SearchWorker::GatherMinibatch` (`search.cc:1268-1363`) 单线程子集。
    pub fn gather_minibatch(&mut self) -> Result<(), EnginError> {
        let root = self.tree.current_head();
        let cur_n = self.tree.node(root).n();
        let remaining_n = self.search_state.remaining_playouts();
        let nodes = cur_n.min(remaining_n.max(0) as u32) as i64;
        let mut collisions_left = self.params.collisions_left(nodes);
        self.number_out_of_order = 0;

        let mut minibatch_size = 0usize;
        while minibatch_size < self.target_minibatch_size && self.number_out_of_order < self.max_out_of_order {
            if minibatch_size > 0
                && self
                    .computation
                    .as_ref()
                    .map_or(0, |computation| computation.used_batch_size())
                    == 0
            {
                return Ok(());
            }

            let new_start = self.minibatch.len();
            let pick_budget = collisions_left
                .min(self.target_minibatch_size as i32 - minibatch_size as i32)
                .min(self.max_out_of_order as i32 - self.number_out_of_order as i32);
            self.pick_nodes_to_extend(pick_budget.max(0) as u32)?;
            let mut picked_visits = 0usize;
            for item in &self.minibatch[new_start..] {
                if !item.is_collision {
                    minibatch_size += 1;
                    picked_visits += 1;
                }
            }
            // px0 `search.cc:1322-1347`: split the initial contiguous work
            // ranges into processing tasks, retaining the final range for the
            // main worker. Until persistent task threads are translated, this
            // worker consumes that same queue synchronously after its range.
            let mut main_start = new_start;
            let mut needs_wait = false;
            if self.task_workers > 0
                && picked_visits
                    >= usize::try_from(self.params.minimum_work_size_for_processing)
                        .expect("px0 MinimumProcessingWork is non-negative")
            {
                let min_per_task = usize::try_from(self.params.minimum_work_per_task_for_processing)
                    .expect("px0 MinimumPerTaskProcessing is non-negative");
                assert!(min_per_task > 0, "px0 MinimumPerTaskProcessing is positive");
                let task_workers = usize::try_from(self.task_workers).expect("positive task worker count");
                let num_tasks = (picked_visits / min_per_task).clamp(2, task_workers + 1);
                let per_worker = picked_visits / num_tasks;
                self.task_queue.reset();
                let mut found = 0usize;
                let mut queued = 0usize;
                for index in new_start..self.minibatch.len() {
                    if self.minibatch[index].is_collision {
                        continue;
                    }
                    found += 1;
                    if found == per_worker {
                        if !self.task_queue.push(PickTask::processing(main_start, index + 1)) {
                            break;
                        }
                        main_start = index + 1;
                        found = 0;
                        queued += 1;
                        if queued == num_tasks - 1 {
                            break;
                        }
                    }
                }
                needs_wait = queued > 0;
            }

            let mut workspace = std::mem::take(&mut self.picking_workspace);
            let process_result = self.process_picked_task(main_start, self.minibatch.len(), &mut workspace);
            self.picking_workspace = workspace;
            process_result?;
            if needs_wait {
                let mut workspace = std::mem::take(&mut self.picking_workspace);
                let task_result = self.run_queued_tasks(&mut workspace);
                self.picking_workspace = workspace;
                task_result?;
                self.task_queue.wait();
            }

            let mut some_ooo = false;
            for item in &self.minibatch[new_start..] {
                if item.ooo_completed {
                    some_ooo = true;
                    break;
                }
            }
            if some_ooo {
                let mut i = self.minibatch.len();
                while i > new_start {
                    i -= 1;
                    if self.minibatch[i].is_collision {
                        let node_idx = self.minibatch[i].node_idx;
                        let multivisit = self.minibatch[i].multivisit;
                        let mut node = node_idx;
                        while let Some(parent) = self.tree.node(node).parent() {
                            self.tree.node_mut(parent).cancel_score_update(multivisit);
                            node = parent;
                            if node == self.tree.current_head() {
                                break;
                            }
                        }
                        self.minibatch.remove(i);
                    } else if self.minibatch[i].ooo_completed {
                        // px0 backs up completed out-of-order entries while
                        // reconciling collisions in GatherMinibatch
                        // (`search.cc:1372-1393`), not in ProcessPickedTask.
                        let item = self.minibatch[i].clone();
                        self.do_backup_update_single_node(&item);
                        self.minibatch.remove(i);
                        minibatch_size = minibatch_size.saturating_sub(1);
                        self.number_out_of_order += 1;
                    }
                }
            }

            // px0 `search.cc:1400-1419`: consume collision work even when a
            // gather produced no independent NN leaf. A root collision may be
            // safely enlarged to its precomputed `maxvisit` bound, updating
            // every ancestor's in-flight count before it is shared.
            for index in new_start..self.minibatch.len() {
                if !self.minibatch[index].is_collision {
                    continue;
                }
                let (node_idx, extra) = {
                    let item = &mut self.minibatch[index];
                    let desired = item.maxvisit.min(collisions_left.max(0) as u32);
                    let extra = desired.saturating_sub(item.multivisit);
                    item.multivisit += extra;
                    (item.node_idx, extra)
                };
                if extra > 0 {
                    let mut node = node_idx;
                    while let Some(parent) = self.tree.node(node).parent() {
                        self.tree.node_mut(parent).increment_n_in_flight(extra);
                        node = parent;
                        if node == self.tree.current_head() {
                            break;
                        }
                    }
                }
                collisions_left -= self.minibatch[index].multivisit as i32;
                if collisions_left <= 0 || self.search_state.stop.load(Ordering::Acquire) {
                    return Ok(());
                }
            }
            if minibatch_size >= self.target_minibatch_size {
                break;
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::PickNodesToExtend` (`search.cc:1485-1508`) 单线程子集。
    fn pick_nodes_to_extend(&mut self, collision_limit: u32) -> Result<(), EnginError> {
        if collision_limit == 0 {
            return Ok(());
        }
        // px0 `SearchWorker::PickNodesToExtend` begins every gather with
        // `ResetTasks` (`src/search/classic/search.cc:1485-1492`).
        self.task_queue.reset();
        let mut receiver = std::mem::take(&mut self.minibatch);
        let result = self.pick_nodes_to_extend_task(
            self.tree.current_head(),
            0,
            collision_limit,
            &MoveList::new(),
            &mut receiver,
            true,
        );
        self.minibatch = receiver;
        let mut workspace = std::mem::take(&mut self.picking_workspace);
        let task_result = self.run_queued_tasks(&mut workspace);
        self.picking_workspace = workspace;
        task_result?;
        self.task_queue.drain_results_into(&mut self.minibatch);
        result
    }

    /// px0 `SearchWorker::RunTasks` (`src/search/classic/search.cc:1069-1140`).
    ///
    /// This is the task dispatch body without px0's persistent worker-thread
    /// wait loop. It deliberately accepts a caller-owned workspace: the later
    /// worker pool will give each task worker one stable `TaskWorkspace`.
    fn run_queued_tasks(&mut self, workspace: &mut TaskWorkspace) -> Result<(), EnginError> {
        while let Some((task_id, task)) = self.task_queue.take() {
            let results = match task.kind {
                PickTaskKind::Gathering => {
                    let mut results = Vec::new();
                    self.pick_nodes_to_extend_task_with_workspace(
                        task.start.expect("gathering task start"),
                        task.base_depth,
                        task.collision_limit,
                        &task.moves_to_base,
                        &mut results,
                        workspace,
                        false,
                    )?;
                    results
                }
                PickTaskKind::Processing => {
                    self.process_picked_task(task.start_idx, task.end_idx, workspace)?;
                    Vec::new()
                }
            };
            self.task_queue.complete(task_id, results);
        }
        Ok(())
    }

    /// px0 `PickNodesToExtendTask` (`src/search/classic/search.cc:1551-1897`)
    /// 的单 worker 路径。
    ///
    /// `task_workers_ == 0` 时 px0 仍使用同一显式 DFS/path-backtrack
    /// 状态机；不要把它替换成递归逐 child 调用，否则 collision 内的策略前缀
    /// 和 visit 分配会发生漂移。
    fn pick_nodes_to_extend_task(
        &mut self,
        root_idx: usize,
        base_depth: u16,
        collision_limit: u32,
        moves_to_base: &[Move],
        receiver: &mut Vec<NodeToProcess>,
        is_root: bool,
    ) -> Result<(), EnginError> {
        let mut workspace = std::mem::take(&mut self.picking_workspace);
        let result = self.pick_nodes_to_extend_task_with_workspace(
            root_idx,
            base_depth,
            collision_limit,
            moves_to_base,
            receiver,
            &mut workspace,
            is_root,
        );
        self.picking_workspace = workspace;
        result
    }

    /// px0 `SearchWorker::PickNodesToExtendTask`
    /// (`src/search/classic/search.cc:1551-1827`).
    ///
    /// px0 gives every gathering task an independent `TaskWorkspace`. Keeping
    /// that workspace explicit here prevents task execution from sharing the
    /// main worker's DFS path state.
    #[allow(
        clippy::too_many_arguments,
        reason = "Keeps the px0 PickNodesToExtendTask input contract explicit."
    )]
    fn pick_nodes_to_extend_task_with_workspace(
        &mut self,
        root_idx: usize,
        base_depth: u16,
        collision_limit: u32,
        moves_to_base: &[Move],
        receiver: &mut Vec<NodeToProcess>,
        workspace: &mut TaskWorkspace,
        is_root: bool,
    ) -> Result<(), EnginError> {
        workspace.current_path.clear();
        workspace.moves_to_path.clear();
        workspace.moves_to_path.extend_from_slice(moves_to_base);
        workspace.current_path.push(-1);

        let mut node_idx = Some(root_idx);
        let mut is_root_node = is_root;
        let mut max_limit = u32::MAX;
        let mut passed_off = 0u32;
        let mut completed_visits = 0u32;

        while !workspace.current_path.is_empty() {
            let current_idx = node_idx.expect("path has a node");
            if *workspace.current_path.last().expect("path entry") == -1 {
                let mut cur_limit = if workspace.current_path.len() == 1 {
                    collision_limit
                } else {
                    let parent_edge = workspace.current_path[workspace.current_path.len() - 2] as usize;
                    *workspace
                        .visits_to_perform
                        .last()
                        .expect("parent visits")
                        .get(parent_edge)
                        .expect("selected parent edge")
                };

                if self.tree.node(current_idx).n() == 0 || self.tree.node(current_idx).is_terminal() {
                    if is_root_node && self.tree.node_mut(current_idx).try_start_score_update() {
                        cur_limit -= 1;
                        receiver.push(NodeToProcess::visit(
                            current_idx,
                            (workspace.current_path.len() + base_depth as usize) as u16,
                        ));
                        completed_visits += 1;
                    }
                    if cur_limit > 0 {
                        let maxvisit = if cur_limit == collision_limit && base_depth == 0 && max_limit > cur_limit {
                            max_limit
                        } else {
                            0
                        };
                        receiver.push(NodeToProcess::collision(
                            current_idx,
                            (workspace.current_path.len() + base_depth as usize) as u16,
                            cur_limit,
                            maxvisit,
                        ));
                        completed_visits += cur_limit;
                    }
                    node_idx = self.tree.node(current_idx).parent();
                    workspace.current_path.pop();
                    continue;
                }

                if is_root_node {
                    self.tree.node_mut(current_idx).increment_n_in_flight(cur_limit);
                }

                // px0 `search.cc:1657-1671`: only a bounded policy prefix can
                // affect this collision batch because edges were policy-sorted.
                let max_needed = self
                    .tree
                    .node(current_idx)
                    .num_edges()
                    .min(self.tree.node(current_idx).n_started() as usize + cur_limit as usize + 2);
                let mut visits = workspace.vtp_buffer.pop().unwrap_or_default();
                visits.clear();
                visits.resize(max_needed, 0);
                workspace.visits_to_perform.push(visits);
                workspace.vtp_last_filled.push(-1);

                // px0 `search.cc:1675-1724`: snapshot policy, child utility,
                // and in-flight visit counters for this tree level.
                let draw_score =
                    self.draw_score((workspace.current_path.len() + base_depth as usize).is_multiple_of(2));
                let cpuct = super::uct::compute_cpuct(self.params, self.tree.node(current_idx).n(), is_root_node);
                let puct_mult = cpuct * (self.tree.node(current_idx).children_visits().max(1) as f32).sqrt();
                let fpu = super::uct::get_fpu(
                    self.params,
                    self.tree.node(current_idx),
                    self.tree.arena(),
                    is_root_node,
                    draw_score,
                );
                let policies = (0..max_needed)
                    .map(|edge_idx| self.tree.node(current_idx).edge(edge_idx).get_p())
                    .collect::<Vec<_>>();
                let mut utilities = Vec::with_capacity(max_needed);
                let mut n_started = Vec::with_capacity(max_needed);
                for edge_idx in 0..max_needed {
                    let edge = self.tree.edge_and_node(current_idx, edge_idx);
                    n_started.push(edge.n_started());
                    utilities.push(
                        edge.child()
                            .filter(|child| child.n() > 0)
                            .map_or(fpu, |child| child.q(draw_score)),
                    );
                }

                while cur_limit > 0 {
                    let mut best_idx = None;
                    let mut best_without_u = f32::NEG_INFINITY;
                    let mut best_score = f32::NEG_INFINITY;
                    let mut second_best = f32::NEG_INFINITY;
                    let mut can_exit = false;
                    for edge_idx in 0..max_needed {
                        if can_exit {
                            break;
                        }
                        let score =
                            utilities[edge_idx] + policies[edge_idx] * puct_mult / (1 + n_started[edge_idx]) as f32;
                        if score > best_score {
                            second_best = best_score;
                            best_score = score;
                            best_without_u = utilities[edge_idx];
                            best_idx = Some(edge_idx);
                        } else if score > second_best {
                            second_best = score;
                        }
                        if n_started[edge_idx] == 0 {
                            can_exit = true;
                        }
                    }
                    let best_idx = best_idx.expect("expanded non-terminal node has an edge");
                    let new_visits = if second_best.is_finite() {
                        let estimate = if best_without_u < second_best {
                            (policies[best_idx] * puct_mult / (second_best - best_without_u)
                                - (n_started[best_idx] + 1) as f32
                                + 1.0)
                                .clamp(1.0, 1.0e9) as u32
                        } else {
                            u32::MAX
                        };
                        max_limit = max_limit.min(estimate);
                        cur_limit.min(estimate)
                    } else {
                        cur_limit
                    };
                    workspace.visits_to_perform.last_mut().expect("visits")[best_idx] += new_visits;
                    cur_limit -= new_visits;

                    let child_idx = self
                        .tree
                        .node(current_idx)
                        .child(best_idx)
                        .unwrap_or_else(|| self.tree.arena_mut().spawn_child(current_idx, best_idx));
                    // px0 `search.cc:1791-1794`: a tree-reused two-fold
                    // terminal may have been reached before the new root.
                    self.ensure_node_twofold_correct_for_depth(
                        child_idx,
                        workspace.current_path.len() as u16 + base_depth,
                    );
                    if self.tree.node_mut(child_idx).try_start_score_update() {
                        n_started[best_idx] += 1;
                        let remaining_visits = new_visits - 1;
                        if self.tree.node(child_idx).n() > 0 && !self.tree.node(child_idx).is_terminal() {
                            self.tree.node_mut(child_idx).increment_n_in_flight(remaining_visits);
                            n_started[best_idx] += remaining_visits;
                        }
                        if self.tree.node(child_idx).n() == 0 || self.tree.node(child_idx).is_terminal() {
                            workspace.visits_to_perform.last_mut().expect("visits")[best_idx] -= 1;
                            let mut item = NodeToProcess::visit(
                                child_idx,
                                (workspace.current_path.len() + 1 + base_depth as usize) as u16,
                            );
                            item.moves_to_visit = workspace.moves_to_path.clone();
                            item.moves_to_visit.push(self.tree.node(current_idx).edge(best_idx).mv);
                            receiver.push(item);
                            completed_visits += 1;
                        }
                    }
                    if best_idx as isize > *workspace.vtp_last_filled.last().expect("last filled")
                        && workspace.visits_to_perform.last().expect("visits")[best_idx] > 0
                    {
                        *workspace.vtp_last_filled.last_mut().expect("last filled") = best_idx as isize;
                    }
                }
                // px0 `search.cc:1828-1864`: pass sufficiently large child
                // subtrees to gathering workers, while retaining enough work
                // for the current DFS task. The queue's MAX_TASKS cap is the
                // same classic-search reservation (100).
                if self.task_workers > 0 {
                    let min_work = u32::try_from(self.params.minimum_work_size_for_picking)
                        .expect("px0 MinimumPickingWork is non-negative");
                    let min_remaining = u32::try_from(self.params.minimum_remaining_work_size_for_picking)
                        .expect("px0 MinimumRemainingPickingWork is non-negative");
                    let last_filled = *workspace.vtp_last_filled.last().expect("last filled");
                    if last_filled >= 0 {
                        for edge_idx in 0..=last_filled as usize {
                            let child_limit = workspace.visits_to_perform.last().expect("visits")[edge_idx];
                            let assigned = passed_off + completed_visits;
                            let Some(remaining) = collision_limit.checked_sub(assigned) else {
                                break;
                            };
                            let Some(required_remaining) = collision_limit.checked_sub(min_remaining) else {
                                break;
                            };
                            if child_limit <= min_work
                                || child_limit >= remaining.saturating_mul(2) / 3
                                || child_limit.saturating_add(assigned) >= required_remaining
                            {
                                continue;
                            }
                            let child_idx = self
                                .tree
                                .node(current_idx)
                                .child(edge_idx)
                                .unwrap_or_else(|| self.tree.arena_mut().spawn_child(current_idx, edge_idx));
                            if self.tree.node(child_idx).n() == 0 || self.tree.node(child_idx).is_terminal() {
                                continue;
                            }
                            let mut moves_to_base = workspace.moves_to_path.clone();
                            moves_to_base.push(self.tree.node(current_idx).edge(edge_idx).mv);
                            let task = PickTask::gathering(
                                child_idx,
                                (workspace.current_path.len() + base_depth as usize) as u16,
                                moves_to_base,
                                child_limit,
                            );
                            if self.task_queue.push(task) {
                                workspace.visits_to_perform.last_mut().expect("visits")[edge_idx] = 0;
                                passed_off += child_limit;
                            }
                        }
                    }
                }
                is_root_node = false;
            }

            let min_idx = *workspace.current_path.last().expect("path entry");
            let last_filled = *workspace.vtp_last_filled.last().expect("last filled");
            let mut found_child = false;
            if last_filled > min_idx {
                for edge_idx in (min_idx + 1) as usize..=last_filled as usize {
                    if workspace.visits_to_perform.last().expect("visits")[edge_idx] == 0 {
                        continue;
                    }
                    let mv = self.tree.node(current_idx).edge(edge_idx).mv;
                    if workspace.moves_to_path.len() != workspace.current_path.len() + base_depth as usize {
                        workspace.moves_to_path.push(mv);
                    } else {
                        *workspace.moves_to_path.last_mut().expect("path move") = mv;
                    }
                    *workspace.current_path.last_mut().expect("path entry") = edge_idx as isize;
                    workspace.current_path.push(-1);
                    node_idx = Some(
                        self.tree
                            .node(current_idx)
                            .child(edge_idx)
                            .unwrap_or_else(|| self.tree.arena_mut().spawn_child(current_idx, edge_idx)),
                    );
                    found_child = true;
                    break;
                }
            }
            if !found_child {
                node_idx = self.tree.node(current_idx).parent();
                workspace.moves_to_path.pop();
                workspace.current_path.pop();
                workspace
                    .vtp_buffer
                    .push(workspace.visits_to_perform.pop().expect("visits"));
                workspace.vtp_last_filled.pop();
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::EnsureNodeTwoFoldCorrectForDepth`
    /// (`src/search/classic/search.cc:1510-1550`)。
    fn ensure_node_twofold_correct_for_depth(&mut self, child_idx: usize, depth: u16) {
        let child = self.tree.node(child_idx);
        if !child.is_twofold_terminal() || depth as f32 >= child.m() {
            return;
        }

        let wl = child.wl();
        let d = child.d();
        let m = child.m();
        let terminal_visits = child.n();
        let mut node_idx = Some(child_idx);
        let mut depth_counter = 0u16;
        while let Some(current_idx) = node_idx {
            let parent = self.tree.node(current_idx).parent();
            self.tree
                .node_mut(current_idx)
                .revert_terminal_visits(wl, d, m + depth_counter as f32, terminal_visits);
            depth_counter += 1;
            if depth_counter > depth {
                break;
            }
            node_idx = parent;
        }
        self.tree.make_not_terminal(child_idx);
    }

    /// px0 `SearchWorker::ProcessPickedTask` (`src/search/classic/search.cc:1423-1462`)。
    fn process_picked_task(
        &mut self,
        start_idx: usize,
        end_idx: usize,
        workspace: &mut TaskWorkspace,
    ) -> Result<(), EnginError> {
        workspace.history = self.tree.history().clone();
        let mut nn_inputs = Vec::new();
        for i in start_idx..end_idx {
            let node_idx = self.minibatch[i].node_idx;
            let depth = self.minibatch[i].depth;
            let moves_to_visit = self.minibatch[i].moves_to_visit.clone();
            let is_terminal = self.tree.node(node_idx).is_terminal();
            if self.minibatch[i].is_extendable(is_terminal) {
                self.extend_node(node_idx, depth, &moves_to_visit, &mut workspace.history)?;
                if !self.tree.node(node_idx).is_terminal() {
                    nn_inputs.push((i, workspace.history.positions().to_vec(), {
                        self.tree
                            .node(node_idx)
                            .edges()
                            .iter()
                            .map(|edge| edge.mv)
                            .collect::<MoveList>()
                    }));
                }
            }
            if self.params.out_of_order_eval
                && self.minibatch[i].can_eval_out_of_order(self.tree.node(node_idx).is_terminal())
            {
                self.fetch_single_node_result(i)?;
                self.minibatch[i].ooo_completed = true;
            }
        }
        let computation = self
            .computation
            .as_mut()
            .ok_or(EnginError::PortIncomplete("P4 ProcessPickedTask without computation"))?;
        for (index, positions, legal_moves) in nn_inputs {
            let (result, ticket) = computation.add_input(EvalPosition { positions, legal_moves })?;
            self.minibatch[index].nn_queried = true;
            self.minibatch[index].is_cache_hit = result == AddInputResult::FetchedImmediately;
            self.minibatch[index].eval_ticket = Some(ticket);
        }
        Ok(())
    }

    /// px0 `SearchWorker::ExtendNode` (`src/search/classic/search.cc:1899-1974`)。
    fn extend_node(
        &mut self,
        node_idx: usize,
        depth: u16,
        moves_to_node: &[Move],
        history: &mut PositionHistory,
    ) -> Result<(), EnginError> {
        let root = self.tree.current_head();
        history.trim(self.played_history_len);
        for mv in moves_to_node {
            history.append(*mv);
        }
        let board = history.last().board();
        let legal_moves = board.generate_legal_moves();
        if legal_moves.is_empty() {
            self.tree.make_terminal(
                node_idx,
                if history.is_black_to_move() {
                    GameResult::WhiteWon
                } else {
                    GameResult::BlackWon
                },
                0.0,
                Terminal::EndOfGame,
            );
            return Ok(());
        }
        if node_idx != root {
            if history.last().repetitions() >= 2 {
                self.tree
                    .make_terminal(node_idx, history.rule_judge(), 0.0, Terminal::EndOfGame);
                return Ok(());
            }
            // px0 `search.cc:1930-1959`: an initial repetition can be a
            // forced two-fold result only after the complete cycle is inside
            // the searched line. The special terminal can later be reverted
            // when tree reuse moves the root into that cycle.
            if history.last().repetitions() == 1
                && depth.saturating_sub(1) >= 4
                && self.params.two_fold_draws
                && u32::from(depth.saturating_sub(1)) >= history.last().cycle_length()
            {
                let cycle_length = history.last().cycle_length();
                let result = history.rule_judge();
                if result == GameResult::Draw {
                    self.tree
                        .make_terminal(node_idx, result, cycle_length as f32, Terminal::TwoFold);
                    return Ok(());
                }

                let mut idx = history.len() - 1;
                let mut idx2 = idx;
                while idx2 > 0 {
                    idx2 -= 1;
                    if history.get(idx2).board() == history.last().board() {
                        break;
                    }
                }
                if idx2 > 0 && history.get(idx - 1).board() == history.get(idx2 - 1).board() {
                    idx -= 1;
                    while idx2 != idx {
                        idx2 += 1;
                        if history.get(idx2).repetitions() > 0 {
                            break;
                        }
                    }
                    if idx2 == idx && history.last().rule60_ply() < 120 {
                        self.tree
                            .make_terminal(node_idx, result, cycle_length as f32, Terminal::TwoFold);
                        return Ok(());
                    }
                }
            }
            if !board.has_mating_material() || history.last().rule60_ply() >= 120 {
                self.tree
                    .make_terminal(node_idx, GameResult::Draw, 0.0, Terminal::EndOfGame);
                return Ok(());
            }
        }
        self.tree.node_mut(node_idx).create_edges(&legal_moves);
        Ok(())
    }

    /// px0 `SearchWorker::CollectCollisions` (`search.cc:1977-1987`)。
    pub fn collect_collisions(&mut self) -> Result<(), EnginError> {
        for item in &self.minibatch {
            if item.is_collision {
                self.search_state
                    .shared_collisions
                    .lock()
                    .expect("collisions lock")
                    .push((item.node_idx, item.multivisit));
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::MaybePrefetchIntoCache` (`search.cc:1989-2007`)。
    pub fn maybe_prefetch_into_cache(&mut self) -> Result<(), EnginError> {
        if self.search_state.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let used = self
            .computation
            .as_ref()
            .map_or(0, |computation| computation.used_batch_size());
        if used == 0 || used >= self.params.max_prefetch_batch as usize {
            return Ok(());
        }
        let budget = self.params.max_prefetch_batch as usize - used;
        let root = self.tree.current_head();
        // px0 resets the workspace history before walking prefetch candidates
        // (`search.cc:1997-2004`). ProcessPickedTask leaves this workspace at
        // its last expanded leaf, so using it directly would encode the wrong
        // position for a root-relative cache probe.
        self.history.trim(self.played_history_len);
        self.prefetch_into_cache(Some(root), budget, false)?;
        Ok(())
    }

    /// px0 `Search::GetDrawScore` (`src/search/classic/search.cc:401-405`)。
    fn draw_score(&self, is_odd_depth: bool) -> f32 {
        if is_odd_depth == self.tree.history().is_black_to_move() {
            self.params.draw_score
        } else {
            -self.params.draw_score
        }
    }

    /// px0 `PrefetchIntoCache` (`search.cc:2010-2099`)。
    fn prefetch_into_cache(
        &mut self,
        node_idx: Option<usize>,
        budget: usize,
        is_odd_depth: bool,
    ) -> Result<usize, EnginError> {
        let draw_score = self.draw_score(is_odd_depth);
        if budget == 0 {
            return Ok(0);
        }

        // px0 also reaches this branch for a missing child edge. It is still a
        // valid future leaf and must be encoded from the current history.
        if node_idx.is_none_or(|idx| self.tree.node(idx).n_started() == 0) {
            if self
                .backend
                .cached_evaluation(&EvalPosition {
                    positions: self.history.positions().to_vec(),
                    legal_moves: Vec::new(),
                })
                .is_some()
            {
                return Ok(1);
            }
            let legal_moves = self.history.last().board().generate_legal_moves();
            if let Some(computation) = self.computation.as_mut() {
                let _ = computation.add_input(EvalPosition {
                    positions: self.history.positions().to_vec(),
                    legal_moves,
                })?;
            }
            return Ok(1);
        }

        let node_idx = node_idx.expect("checked above");
        if self.tree.node(node_idx).n() == 0 || self.tree.node(node_idx).is_terminal() {
            return Ok(0);
        }

        // px0 `search.cc:2036-2051`: score all legal edges using the same
        // EdgeAndNode Q/U proxy as selection. The negated score permits
        // ascending partial sorting below.
        let is_root = node_idx == self.tree.current_head();
        let cpuct = super::uct::compute_cpuct(self.params, self.tree.node(node_idx).n(), is_root);
        let puct_mult = cpuct * (self.tree.node(node_idx).children_visits().max(1) as f32).sqrt();
        let fpu = super::uct::get_fpu(
            self.params,
            self.tree.node(node_idx),
            self.tree.arena(),
            is_root,
            draw_score,
        );
        let mut scores = (0..self.tree.node(node_idx).num_edges())
            .filter_map(|edge_idx| {
                let edge = self.tree.edge_and_node(node_idx, edge_idx);
                (edge.p() != 0.0).then_some((-edge.u(puct_mult) - edge.q(fpu, draw_score), edge_idx))
            })
            .collect::<Vec<_>>();

        let mut first_unsorted_index = 0usize;
        let mut total_budget_spent = 0usize;
        let mut budget_to_spend = budget;
        for index in 0..scores.len() {
            if self.search_state.stop.load(Ordering::Acquire) || budget == total_budget_spent {
                break;
            }

            // px0 `std::partial_sort` sorts only the next 2-3 candidates.
            // `select_nth_unstable_by` + sorting its selected prefix gives the
            // same ordered prefix without sorting the remainder.
            if first_unsorted_index != scores.len() && index + 2 >= first_unsorted_index {
                let remaining_budget = budget - total_budget_spent;
                let new_unsorted_index = std::cmp::min(
                    scores.len(),
                    if remaining_budget < 2 {
                        first_unsorted_index + 2
                    } else {
                        first_unsorted_index + 3
                    },
                );
                let selected = new_unsorted_index - first_unsorted_index;
                let tail = &mut scores[first_unsorted_index..];
                tail.select_nth_unstable_by(selected - 1, |left, right| left.0.total_cmp(&right.0));
                tail[..selected].sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
                first_unsorted_index = new_unsorted_index;
            }

            let edge_idx = scores[index].1;
            if index != scores.len() - 1 {
                let next_score = -scores[index + 1].0;
                let edge = self.tree.edge_and_node(node_idx, edge_idx);
                let q = edge.q(-fpu, draw_score);
                if next_score > q {
                    let estimated = edge.p() * puct_mult / (next_score - q) - edge.n_started() as f32;
                    budget_to_spend = std::cmp::min(budget - total_budget_spent, estimated as usize + 1);
                } else {
                    budget_to_spend = budget - total_budget_spent;
                }
            }

            let (mv, child_idx) = {
                let edge = self.tree.edge_and_node(node_idx, edge_idx);
                (edge.mv(), self.tree.node(node_idx).child(edge_idx))
            };
            self.history.append(mv);
            let result = self.prefetch_into_cache(child_idx, budget_to_spend, !is_odd_depth);
            self.history.pop();
            let budget_spent = result?;
            total_budget_spent += budget_spent;
        }
        Ok(total_budget_spent)
    }

    /// px0 `SearchWorker::RunNNComputation` (`search.cc:2103-2107`)。
    pub fn run_nn_computation(&mut self) -> Result<(), EnginError> {
        if let Some(computation) = self.computation.as_mut() {
            if computation.used_batch_size() > 0 {
                computation.compute_blocking()?;
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::FetchMinibatchResults` (`search.cc:2109-2156`)。
    pub fn fetch_minibatch_results(&mut self) -> Result<(), EnginError> {
        for i in 0..self.minibatch.len() {
            self.fetch_single_node_result(i)?;
        }
        Ok(())
    }

    /// px0 `SearchWorker::FetchSingleNodeResult` (`search.cc:2117-2154`)。
    fn fetch_single_node_result(&mut self, index: usize) -> Result<(), EnginError> {
        if self.minibatch[index].is_collision {
            return Ok(());
        }
        let node_idx = self.minibatch[index].node_idx;
        if !self.minibatch[index].nn_queried {
            self.minibatch[index].eval = EvalResult {
                wl: self.tree.node(node_idx).wl(),
                d: self.tree.node(node_idx).d(),
                m: self.tree.node(node_idx).m(),
                policies: Vec::new(),
            };
            return Ok(());
        }
        let ticket = self.minibatch[index]
            .eval_ticket
            .ok_or(EnginError::PortIncomplete("P4 FetchSingleNodeResult missing ticket"))?;
        let mut eval = self
            .computation
            .as_mut()
            .ok_or(EnginError::PortIncomplete(
                "P4 FetchSingleNodeResult without computation",
            ))?
            .take_result(ticket)?;
        eval.wl = -eval.wl;
        if self.tree.node(node_idx).n() == 0 {
            for (edge_idx, policy) in eval.policies.iter().enumerate() {
                self.tree.node_mut(node_idx).edge_mut(edge_idx).set_p(*policy);
            }
            // px0 sorts the just-initialized policy before any child node can
            // be spawned (`node.cc:291-298`, `search.cc:2145-2153`).
            self.tree.node_mut(node_idx).sort_edges();
        }
        self.minibatch[index].eval = eval;
        Ok(())
    }

    /// px0 `SearchWorker::DoBackupUpdate` (`search.cc:2158-2258`) 单线程子集。
    pub fn do_backup_update(&mut self) -> Result<(), EnginError> {
        let mut work_done = self.number_out_of_order > 0;
        let items: Vec<_> = self
            .minibatch
            .iter()
            .filter(|item| !item.is_collision)
            .cloned()
            .collect();
        for item in items {
            self.do_backup_update_single_node(&item);
            work_done = true;
        }
        if work_done {
            self.cancel_shared_collisions();
            self.search_state.total_batches.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// px0 `Search::CancelSharedCollisions` (`search.cc:1044-1053`).
    fn cancel_shared_collisions(&mut self) {
        let collisions = std::mem::take(&mut *self.search_state.shared_collisions.lock().expect("collisions lock"));
        for (node_idx, multivisit) in collisions {
            let mut current = self.tree.node(node_idx).parent();
            while let Some(node_idx) = current {
                self.tree.node_mut(node_idx).cancel_score_update(multivisit);
                current = self.tree.node(node_idx).parent();
            }
        }
    }

    /// px0 `SearchWorker::DoBackupUpdateSingleNode` (`search.cc:2175-2258`) 子集。
    fn do_backup_update_single_node(&mut self, item: &NodeToProcess) {
        let mut node_idx = item.node_idx;
        let mut v = item.eval.wl;
        let mut d = item.eval.d;
        let mut m = item.eval.m;
        let root = self.tree.current_head();
        loop {
            if self.tree.node(node_idx).is_terminal() {
                v = self.tree.node(node_idx).wl();
                d = self.tree.node(node_idx).d();
                m = self.tree.node(node_idx).m();
            }
            self.tree
                .node_mut(node_idx)
                .finalize_score_update(v, d, m, item.multivisit);
            if node_idx == root {
                break;
            }
            let parent = self.tree.node(node_idx).parent().expect("non-root has parent");
            node_idx = parent;
            v = -v;
            m += 1.0;
        }
        self.search_state
            .total_playouts
            .fetch_add(item.multivisit as u64, Ordering::AcqRel);
        if item.nn_queried && !item.is_cache_hit {
            self.search_state.network_evaluations.fetch_add(1, Ordering::AcqRel);
        }
        self.search_state
            .cum_depth
            .fetch_add(item.depth as u64 * item.multivisit as u64, Ordering::AcqRel);
        loop {
            let current = self.search_state.max_depth.load(Ordering::Acquire);
            if item.depth <= current {
                break;
            }
            if self
                .search_state
                .max_depth
                .compare_exchange(current, item.depth, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    /// px0 `SearchWorker::UpdateCounters` (`search.cc:2331-2364`) 子集。
    pub fn update_counters(&mut self) -> Result<(), EnginError> {
        let _ = self.search_state.stop.load(Ordering::Acquire);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::sync::Once;

    use xiangqi_core::{initialize_magic_bitboards, GameResult, GameState, STARTPOS_FEN};

    use super::*;
    use crate::neural::backend::UniformBackend;

    static INIT: Once = Once::new();

    fn ensure_init() {
        INIT.call_once(initialize_magic_bitboards);
    }

    #[test]
    fn out_of_order_terminal_stays_explicit() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        tree.make_terminal(root, GameResult::Draw, 0.0, Terminal::EndOfGame);
        let backend = UniformBackend::default();
        let params = SearchParams {
            out_of_order_eval: true,
            ..SearchParams::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let search_state = WorkerSearchState::new(stop, i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.initialize_iteration().expect("init");
        // `PickNodesToExtendTask` always reserves the path before handing an
        // item to ProcessPickedTask (`px0 search.cc:1613-1636`).  Keep this
        // focused OOO test on that post-selection contract.
        assert!(worker.tree.node_mut(root).try_start_score_update());
        worker.minibatch.push(NodeToProcess::visit(root, 1));
        let mut workspace = TaskWorkspace::default();
        worker.process_picked_task(0, 1, &mut workspace).expect("ooo terminal");
        assert!(worker.minibatch[0].ooo_completed);
    }

    #[test]
    fn collision_multivisits_are_cancelled_after_backup() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let only_move = tree.history().last().board().generate_legal_moves()[0];
        tree.node_mut(root).create_edges(&vec![only_move]);
        assert!(tree.node_mut(root).try_start_score_update());
        tree.node_mut(root).finalize_score_update(0.0, 0.0, 0.0, 1);

        let backend = UniformBackend::default();
        let params = SearchParams {
            minibatch_size: 4,
            max_collision_visits: 4,
            max_collision_visits_scaling_start: 0,
            max_collision_visits_scaling_end: 1,
            ..SearchParams::default()
        };
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);

        worker.execute_one_iteration().expect("collision iteration");

        assert_eq!(tree.node(root).n_in_flight(), 0);
        assert_eq!(state.shared_collisions.lock().expect("collisions lock").len(), 0);
    }

    #[test]
    fn prefetch_resets_workspace_history_to_root() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.initialize_iteration().expect("init");

        let root_moves = worker.history.last().board().generate_legal_moves();
        worker.history.append(root_moves[0]);
        worker
            .computation
            .as_mut()
            .expect("computation")
            .add_input(EvalPosition {
                positions: worker.history.positions().to_vec(),
                legal_moves: worker.history.last().board().generate_legal_moves(),
            })
            .expect("seed computation");

        worker.maybe_prefetch_into_cache().expect("prefetch");

        assert_eq!(worker.history.len(), worker.played_history_len);
    }

    #[test]
    fn prefetch_descends_from_expanded_root_to_missing_child() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let mv = tree.history().last().board().generate_legal_moves()[0];
        tree.node_mut(root).create_single_child_node(mv);
        tree.node_mut(root).edge_mut(0).set_p(1.0);
        assert!(tree.node_mut(root).try_start_score_update());
        tree.node_mut(root).finalize_score_update(0.0, 0.0, 0.0, 1);

        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.initialize_iteration().expect("init");

        let spent = worker
            .prefetch_into_cache(Some(root), 1, false)
            .expect("recursive prefetch");

        assert_eq!(spent, 1);
        assert_eq!(worker.history.len(), worker.played_history_len);
        assert_eq!(worker.computation.as_ref().expect("computation").used_batch_size(), 1);
    }

    #[test]
    fn draw_score_flips_with_root_side_and_depth() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let backend = UniformBackend::default();
        let params = SearchParams {
            draw_score: 0.25,
            ..SearchParams::default()
        };
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let worker = SearchWorker::new(&mut tree, &backend, &params, &state);

        assert!((worker.draw_score(false) - 0.25).abs() < f32::EPSILON);
        assert!((worker.draw_score(true) + 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn selection_workspace_keeps_full_path_to_two_ply_leaf() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let first = tree.history().last().board().generate_legal_moves()[0];
        tree.node_mut(root).create_single_child_node(first);
        tree.node_mut(root).edge_mut(0).set_p(1.0);
        assert!(tree.node_mut(root).try_start_score_update());
        tree.node_mut(root).finalize_score_update(0.0, 0.0, 0.0, 1);

        let child = tree.arena_mut().spawn_child(root, 0);
        let mut history = tree.history().clone();
        history.append(first);
        let second = history.last().board().generate_legal_moves()[0];
        tree.node_mut(child).create_single_child_node(second);
        tree.node_mut(child).edge_mut(0).set_p(1.0);
        assert!(tree.node_mut(child).try_start_score_update());
        tree.node_mut(child).finalize_score_update(0.0, 0.0, 0.0, 1);

        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.pick_nodes_to_extend(1).expect("pick nodes");

        let visit = worker
            .minibatch
            .iter()
            .find(|item| !item.is_collision)
            .expect("leaf visit");
        assert_eq!(visit.moves_to_visit, vec![first, second]);
        assert_eq!(visit.depth, 3);
    }

    #[test]
    fn queued_gathering_task_merges_into_minibatch() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let mv = tree.history().last().board().generate_legal_moves()[0];
        tree.node_mut(root).create_single_child_node(mv);
        tree.node_mut(root).edge_mut(0).set_p(1.0);
        assert!(tree.node_mut(root).try_start_score_update());
        tree.node_mut(root).finalize_score_update(0.0, 0.0, 0.0, 1);

        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.task_queue.reset();
        assert!(worker.task_queue.push(PickTask::gathering(root, 0, Vec::new(), 1)));
        let mut workspace = TaskWorkspace::default();

        worker.run_queued_tasks(&mut workspace).expect("run gathering task");
        worker.task_queue.drain_results_into(&mut worker.minibatch);

        assert_eq!(worker.minibatch.len(), 1);
        assert!(!worker.minibatch[0].is_collision);
        assert_eq!(worker.minibatch[0].moves_to_visit, vec![mv]);
    }

    #[test]
    fn gathering_uses_px0_processing_task_ranges() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let moves = tree.history().last().board().generate_legal_moves();
        tree.node_mut(root).create_edges(&moves);
        for edge_idx in 0..tree.node(root).num_edges() {
            tree.node_mut(root).edge_mut(edge_idx).set_p(1.0 / moves.len() as f32);
        }
        assert!(tree.node_mut(root).try_start_score_update());
        tree.node_mut(root).finalize_score_update(0.0, 0.0, 0.0, 1);

        let backend = UniformBackend::default();
        let params = SearchParams {
            minibatch_size: 32,
            task_workers_per_search_worker: 1,
            minimum_work_size_for_processing: 2,
            minimum_work_per_task_for_processing: 1,
            out_of_order_eval: false,
            ..SearchParams::default()
        };
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.initialize_iteration().expect("init");
        worker.gather_minibatch().expect("gather processing tasks");

        assert!(worker.minibatch.len() >= 2);
        assert!(worker.computation.as_ref().expect("computation").used_batch_size() >= 2);
    }

    #[test]
    fn reused_twofold_terminal_is_reverted_before_selection() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let mv = tree.history().last().board().generate_legal_moves()[0];
        tree.node_mut(root).create_single_child_node(mv);
        let child = tree.arena_mut().spawn_child(root, 0);
        tree.make_terminal(child, GameResult::Draw, 3.0, Terminal::TwoFold);
        assert!(tree.node_mut(child).try_start_score_update());
        tree.node_mut(child).finalize_score_update(0.0, 1.0, 3.0, 1);
        assert!(tree.node_mut(root).try_start_score_update());
        tree.node_mut(root).finalize_score_update(0.0, 1.0, 4.0, 1);

        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.ensure_node_twofold_correct_for_depth(child, 1);

        assert!(!worker.tree.node(child).is_terminal());
        assert_eq!(worker.tree.node(child).n(), 0);
        assert_eq!(worker.tree.node(root).n(), 0);
    }

    #[test]
    fn extend_node_marks_complete_first_repetition_as_twofold() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let mut history = tree.history().clone();
        let mut moves = Vec::new();
        for uci in ["a0a1", "a9a8", "a1a0", "a8a9"] {
            let mv = history.last().board().parse_move(uci).expect("legal cycle move");
            history.append(mv);
            moves.push(mv);
        }
        assert_eq!(history.last().repetitions(), 1);
        assert_eq!(history.last().cycle_length(), 4);

        let first = moves[0];
        tree.node_mut(root).create_single_child_node(first);
        let child = tree.arena_mut().spawn_child(root, 0);
        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)), i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);

        worker
            .extend_node(child, 5, &moves, &mut history)
            .expect("extend twofold leaf");

        assert!(worker.tree.node(child).is_twofold_terminal());
        assert!((worker.tree.node(child).m() - 4.0).abs() < f32::EPSILON);
    }
}
