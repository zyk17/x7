//! px0 `src/search/classic/search.h:201-448` 的 P4 worker。
//!
//! P4 worker 七阶段流水线、碰撞和 task queue 数据结构已按 px0 源码翻译。
//! px0 task thread 共享一个 `SearchWorker`。Rust 先将 task/workspace/result
//! 所有权拆开；后台 tree phase 仍在逐段重译。

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, AtomicU16, AtomicU64, AtomicUsize, Ordering};
use std::sync::Condvar;
use std::sync::{Arc, Mutex};

use parking_lot::RwLock;

use xiangqi_core::{GameResult, Move, MoveList, PositionHistory};

use crate::neural::backend::{AddInputResult, Backend, BackendComputation, EvalPosition, EvalResult, EvalTicket};
use crate::EnginError;

use super::node::{NodeTree, Terminal};
use super::params::{ContemptMode, SearchParams};
use super::search::{best_child_edge, wdl_rescale, SearchStopController};
use super::stoppers::StoppersHints;

/// px0 `Search` 中与 worker 相关的计数子集 (`search.h:49-200`)。
#[derive(Debug)]
pub struct WorkerSearchState {
    pub stop: Arc<AtomicBool>,
    /// px0 `Search::pending_searchers_` (`search.h:183-184`), the number of
    /// workers allowed to remain in the gather/process tree phase.
    pub pending_searchers: AtomicI32,
    /// px0 `Search::backend_waiting_counter_` (`search.h:181-182`).
    pub backend_waiting_counter: AtomicI32,
    pub thread_count: AtomicUsize,
    pub shared_collisions: Mutex<Vec<(usize, u32)>>,
    /// px0 `Search::current_best_edge_` (`search.h:174`, `search.cc:2212-2249`).
    /// Rust stores the root edge index; edge ordering stays stable for one
    /// `ClassicSearch::StartSearch` lifetime.
    pub current_best_edge: Mutex<Option<usize>>,
    pub total_playouts: AtomicU64,
    pub total_batches: AtomicU64,
    pub network_evaluations: AtomicU64,
    pub cum_depth: AtomicU64,
    pub max_depth: AtomicU16,
}

impl Default for WorkerSearchState {
    fn default() -> Self {
        Self::new(Arc::new(AtomicBool::new(false)))
    }
}

impl WorkerSearchState {
    pub fn new(stop: Arc<AtomicBool>) -> Self {
        Self {
            stop,
            pending_searchers: AtomicI32::new(1),
            backend_waiting_counter: AtomicI32::new(0),
            thread_count: AtomicUsize::new(1),
            shared_collisions: Mutex::new(Vec::new()),
            current_best_edge: Mutex::new(None),
            total_playouts: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            cum_depth: AtomicU64::new(0),
            max_depth: AtomicU16::new(0),
        }
    }

    /// px0 initializes `pending_searchers_` on every `StartSearch`
    /// (`src/search/classic/search.cc:153`). A zero limit disables this
    /// throttle, matching px0 `MaxConcurrentSearchers=0`.
    pub fn set_max_concurrent_searchers(&self, limit: i32) {
        self.pending_searchers.store(limit.max(0), Ordering::Release);
    }
}

/// px0 `SearchWorker::NodeToProcess` (`search.h:288-347`)。
#[derive(Clone, Debug)]
pub struct NodeToProcess {
    pub node_idx: usize,
    pub eval: Arc<EvalResult>,
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
            eval: Arc::new(EvalResult::default()),
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
            eval: Arc::new(EvalResult::default()),
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
    // px0 uses fixed `std::array<_, 256>` scratch storage because
    // `Node::num_edges_` is uint8_t (`search.h:350`, `node.h:320-321`).
    current_policy: [f32; 256],
    current_utility: [f32; 256],
    current_score: [f32; 256],
    current_n_started: [u32; 256],
    vtp_buffer: Vec<Vec<u32>>,
    visits_to_perform: Vec<Vec<u32>>,
    vtp_last_filled: Vec<isize>,
    current_path: Vec<isize>,
    moves_to_path: MoveList,
    history: PositionHistory,
}

/// px0 keeps `nodes_mutex_` across each tree phase
/// (`search.cc:1142-1211,1494-1508`). Rust represents that boundary by
/// borrowing the tree explicitly for every phase instead of storing an active
/// raw pointer inside `SearchWorker`.
enum TreeStorage<'a> {
    Direct(&'a mut NodeTree),
    Shared(Arc<RwLock<NodeTree>>),
    /// Only installed while `SearchWorker::with_tree*` moves a direct or
    /// shared tree out of the worker. No operation can observe this state.
    Detached,
}

struct WorkerTree<'a> {
    storage: TreeStorage<'a>,
}

impl<'a> WorkerTree<'a> {
    fn direct(tree: &'a mut NodeTree) -> Self {
        Self {
            storage: TreeStorage::Direct(tree),
        }
    }

    fn shared(tree: Arc<RwLock<NodeTree>>) -> Self {
        Self {
            storage: TreeStorage::Shared(tree),
        }
    }
}

/// px0 `SearchWorker::PickTask` (`src/search/classic/search.h:367-393`).
/// A gathering task owns a disjoint subtree root; a processing task owns a
/// non-overlapping minibatch range.
#[derive(Debug)]
struct PickTask {
    kind: PickTaskKind,
    start: Option<usize>,
    base_depth: u16,
    collision_limit: u32,
    moves_to_base: MoveList,
    results: Vec<NodeToProcess>,
    start_idx: usize,
    end_idx: usize,
    complete: bool,
}

/// px0 `PickTask::PickTaskType` (`src/search/classic/search.h:368-370`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PickTaskKind {
    Gathering,
    Processing,
}

/// px0 `RunTasks` claims both a stable task index and its task storage
/// (`src/search/classic/search.cc:1076-1093`). Rust keeps that pair together
/// until completion so an owned task cannot be returned into a different slot.
struct ClaimedTask {
    id: usize,
    task: PickTask,
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
/// `search.cc:1069-1140,1464-1483`). Task execution is wired separately.
#[derive(Default)]
struct PickTaskQueue {
    // px0 keeps stable `PickTask` storage because task threads hold pointers
    // into `picking_tasks_` (`search.cc:1088-1090,1469-1473`). Rust instead
    // moves a claimed task out of its slot, then returns that same task on
    // completion. This preserves the task/result lifecycle without cloning a
    // task or aliasing its mutable result vector across threads.
    tasks: Mutex<Vec<Option<PickTask>>>,
    /// Rust-visible proof of px0's gathering-task split invariant. A task
    /// keeps its claimed root until the whole phase ends, so another task
    /// cannot be published for the same subtree or one of its ancestors.
    /// px0 relies on this implicitly in `PickNodesToExtendTask`
    /// (`src/search/classic/search.cc:1828-1864`).
    gathering_roots: Mutex<Vec<usize>>,
    task_count: AtomicIsize,
    /// px0 `task_taking_started_`: a tiny claim lock around the task index.
    /// Rust keeps task storage behind a mutex and retains the same claim
    /// serialization for the synchronous task phase.
    task_taking_started: AtomicBool,
    tasks_taken: AtomicUsize,
    completed_tasks: AtomicUsize,
    /// Scoped Rust task phases seal after their producer has finished
    /// publishing work. This maps px0's long-lived worker sleep boundary
    /// (`search.cc:1069-1124`) without making a task thread persistent yet.
    phase_sealed: AtomicBool,
    // This is px0's `task_count_ == -1` sentinel (`search.cc:1097-1119`,
    // `search.h:435-445`). The active Rust path consumes it synchronously;
    // blocking sleep/exit remains test-only until task ownership is complete.
    #[cfg(test)]
    exiting: AtomicBool,
    task_added: Condvar,
}

impl PickTaskQueue {
    const MAX_TASKS: usize = 100;

    /// px0 `SearchWorker::ResetTasks` (`src/search/classic/search.cc:1466-1473`).
    fn reset(&self) {
        self.task_count.store(0, Ordering::Release);
        self.task_taking_started.store(false, Ordering::Release);
        self.tasks_taken.store(0, Ordering::Release);
        self.completed_tasks.store(0, Ordering::Release);
        self.phase_sealed.store(false, Ordering::Release);
        let mut tasks = self.tasks.lock().expect("pick task queue lock");
        tasks.clear();
        // px0 reserves `MAX_TASKS` every reset because task workers retain
        // pointers into this vector while they execute.
        tasks.reserve(Self::MAX_TASKS);
        drop(tasks);
        self.gathering_roots.lock().expect("gathering root lock").clear();
    }

    /// px0 task enqueue (`src/search/classic/search.cc:1843-1856`).
    fn push(&self, task: PickTask) -> bool {
        if self.task_count.load(Ordering::Acquire) < 0 {
            return false;
        }
        let mut tasks = self.tasks.lock().expect("pick task queue lock");
        if tasks.len() >= Self::MAX_TASKS {
            return false;
        }
        tasks.push(Some(task));
        drop(tasks);
        self.task_count.fetch_add(1, Ordering::AcqRel);
        self.task_added.notify_all();
        true
    }

    /// Publishes a gathering task only when its tree root is disjoint from
    /// every subtree already handed to this phase. This is not a search
    /// heuristic: it exposes the ownership precondition that px0's task split
    /// assumes before task workers mutate the tree concurrently
    /// (`src/search/classic/search.cc:1828-1864`).
    fn push_gathering(&self, tree: &NodeTree, task: PickTask) -> bool {
        debug_assert_eq!(task.kind, PickTaskKind::Gathering);
        let start = task.start.expect("gathering task has a start node");
        if self.task_count.load(Ordering::Acquire) < 0 {
            return false;
        }

        // Keep the lock order identical to `reset`: task slots first, then
        // root claims. Future gathering workers may publish nested tasks.
        let mut tasks = self.tasks.lock().expect("pick task queue lock");
        if tasks.len() >= Self::MAX_TASKS {
            return false;
        }
        let mut roots = self.gathering_roots.lock().expect("gathering root lock");
        if roots
            .iter()
            .copied()
            .any(|existing| Self::subtrees_overlap(tree, existing, start))
        {
            return false;
        }

        tasks.push(Some(task));
        roots.push(start);
        drop(roots);
        drop(tasks);
        self.task_count.fetch_add(1, Ordering::AcqRel);
        self.task_added.notify_all();
        true
    }

    /// Two task roots overlap exactly when either root is an ancestor of the
    /// other in px0's parent chain (`src/search/classic/node.h:234-239`).
    fn subtrees_overlap(tree: &NodeTree, left: usize, right: usize) -> bool {
        Self::is_ancestor_of(tree, left, right) || Self::is_ancestor_of(tree, right, left)
    }

    fn is_ancestor_of(tree: &NodeTree, ancestor: usize, mut node: usize) -> bool {
        loop {
            if node == ancestor {
                return true;
            }
            let Some(parent) = tree.node(node).parent() else {
                return false;
            };
            node = parent;
        }
    }

    /// px0 task claim (`src/search/classic/search.cc:1076-1093`).
    fn take(&self) -> Option<ClaimedTask> {
        let index = loop {
            let taken = self.tasks_taken.load(Ordering::Acquire);
            let task_count = self.task_count.load(Ordering::Acquire);
            if task_count < 0 || taken >= task_count as usize {
                return None;
            }

            if self
                .task_taking_started
                .compare_exchange_weak(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                let taken = self.tasks_taken.load(Ordering::Acquire);
                let task_count = self.task_count.load(Ordering::Acquire);
                if task_count >= 0 && taken < task_count as usize {
                    let index = self.tasks_taken.fetch_add(1, Ordering::AcqRel);
                    self.task_taking_started.store(false, Ordering::Release);
                    break index;
                }
                self.task_taking_started.store(false, Ordering::Release);
            }
            std::hint::spin_loop();
        };
        self.tasks
            .lock()
            .expect("pick task queue lock")
            .get_mut(index)
            .and_then(Option::take)
            .map(|task| ClaimedTask { id: index, task })
    }

    /// px0 completion accounting (`src/search/classic/search.cc:1136-1137`).
    fn complete(&self, mut claimed: ClaimedTask) {
        if let Some(slot) = self.tasks.lock().expect("pick task queue lock").get_mut(claimed.id) {
            claimed.task.complete = true;
            *slot = Some(claimed.task);
            self.completed_tasks.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// px0 `SearchWorker::WaitForTasks` (`src/search/classic/search.cc:1475-1483`).
    fn wait(&self) {
        while self.completed_tasks.load(Ordering::Acquire) < self.task_count.load(Ordering::Acquire).max(0) as usize {
            std::hint::spin_loop();
        }
    }

    /// px0 `RunTasks` waits for tasks while the main gather producer is still
    /// walking the tree (`src/search/classic/search.cc:1069-1124,1485-1508`).
    /// A scoped Rust worker exits only after the producer seals this phase and
    /// every published task has been claimed.
    fn take_until_phase_sealed(&self) -> Option<ClaimedTask> {
        loop {
            if let Some(task) = self.take() {
                return Some(task);
            }
            if self.phase_sealed.load(Ordering::Acquire) {
                return None;
            }
            let tasks = self.tasks.lock().expect("pick task queue lock");
            if self.phase_sealed.load(Ordering::Acquire) {
                return None;
            }
            drop(self.task_added.wait(tasks).expect("pick task queue wait"));
        }
    }

    /// Completes the producer side of one scoped task phase. Existing tasks
    /// remain claimable; idle workers may exit only after they are exhausted.
    fn seal_phase(&self) {
        self.phase_sealed.store(true, Ordering::Release);
        self.task_added.notify_all();
    }

    /// px0 `task_count_.store(-1)` after `GatherMinibatch`
    /// (`src/search/classic/search.cc:1182-1185`). Persistent task threads
    /// sleep after this publication until the next `reset/push` cycle.
    fn idle(&self) {
        self.task_count.store(-1, Ordering::Release);
        self.task_added.notify_all();
    }

    /// px0 `SearchWorker::RunTasks` sleep/exit path
    /// (`src/search/classic/search.cc:1069-1124`).
    ///
    /// A `-1` task count means either idle or exiting. Workers only return
    /// after `close()` sets `exiting`; otherwise they sleep until `reset/push`
    /// publishes the next iteration's work.
    #[cfg(test)]
    fn take_blocking(&self) -> Option<ClaimedTask> {
        let mut spins = 0usize;
        loop {
            if let Some(task) = self.take() {
                return Some(task);
            }
            if self.exiting.load(Ordering::Acquire) {
                return None;
            }

            if self.task_count.load(Ordering::Acquire) != -1 {
                spins += 1;
                if spins >= 512 {
                    std::thread::yield_now();
                    spins = 0;
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }

            spins = 0;
            let tasks = self.tasks.lock().expect("pick task queue lock");
            if self.exiting.load(Ordering::Acquire) {
                return None;
            }
            if self.task_count.load(Ordering::Acquire) != -1 {
                continue;
            }
            drop(self.task_added.wait(tasks).expect("pick task queue wait"));
        }
    }

    /// px0 `SearchWorker::~SearchWorker` (`src/search/classic/search.h:225-233`).
    #[cfg(test)]
    fn close(&self) {
        self.exiting.store(true, Ordering::Release);
        self.task_count.store(-1, Ordering::Release);
        self.task_added.notify_all();
    }

    /// px0 result merge (`src/search/classic/search.cc:1501-1507`).
    fn drain_results_into(&self, receiver: &mut Vec<NodeToProcess>) {
        self.wait();
        let mut tasks = self.tasks.lock().expect("pick task queue lock");
        for task in tasks.iter_mut().flatten() {
            receiver.append(&mut task.results);
        }
    }
}

impl Default for TaskWorkspace {
    /// px0 `TaskWorkspace::TaskWorkspace` (`src/search/classic/search.h:357-364`).
    fn default() -> Self {
        const INITIAL_DEPTH: usize = 30;
        let mut workspace = Self {
            current_policy: [0.0; 256],
            current_utility: [0.0; 256],
            current_score: [0.0; 256],
            current_n_started: [0; 256],
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

/// px0 `SearchWorker` task-phase fields (`src/search/classic/search.h:433-445`).
///
/// This state is deliberately separate from the worker's minibatch, backend
/// computation, and counters. A future task-worker translation may only move
/// task-owned workspaces and task/result records across threads; it may not
/// move the owning `SearchWorker`.
#[derive(Default)]
struct TaskPhaseState {
    queue: PickTaskQueue,
    main_runner: TaskRunner,
}

/// px0 keeps one `TaskWorkspace` per `RunTasks` thread plus a separate
/// `main_workspace_` (`src/search/classic/search.h:348-365,441-445`). A Rust
/// runner owns exactly one workspace; it must never borrow another runner's
/// path/history buffers.
#[derive(Default)]
struct TaskRunner {
    workspace: TaskWorkspace,
}

impl TaskRunner {
    /// px0 `RunTasks` gathering branch (`src/search/classic/search.cc:1116-1124`).
    /// A runner owns its DFS/history scratch; the queued task owns its result
    /// vector. Neither object needs a mutable `SearchWorker` alias.
    fn run_gathering_task(
        &mut self,
        context: &SelectionContext<'_>,
        tree: &mut NodeTree,
        task: &mut PickTask,
    ) -> Result<(), EnginError> {
        debug_assert_eq!(task.kind, PickTaskKind::Gathering);
        SearchWorker::pick_nodes_to_extend_task_with_workspace(
            context,
            tree,
            task.start.expect("gathering task start"),
            task.base_depth,
            task.collision_limit,
            &task.moves_to_base,
            &mut task.results,
            &mut self.workspace,
            false,
        )
    }

    /// px0 `RunTasks` processing branch (`src/search/classic/search.cc:1125-1129`).
    /// The caller provides an already-disjoint minibatch range, matching px0's
    /// task split before a runner mutates its private history scratch.
    fn run_processing_range(
        &mut self,
        context: &ProcessingContext<'_>,
        tree: &mut NodeTree,
        range: &mut [NodeToProcess],
    ) -> Result<(), EnginError> {
        SearchWorker::process_picked_range(context, tree, range, &mut self.workspace)
    }
}

/// px0 lends `nodes_mutex_`'s gather/processing phase to `RunTasks`
/// (`src/search/classic/search.cc:1069-1140,1485-1508`). Rust cannot express
/// that C++ mutable alias with ordinary references, so this pointer is valid
/// only inside `run_tasks_scoped_in_tree`: the owner joins every task thread
/// before it accesses the tree again.
#[derive(Clone, Copy)]
struct ScopedTaskTree(*mut NodeTree);

// Safety: constructed from the caller's exclusive tree phase and consumed
// only by scoped task threads. No pointer may escape the scope or coexist with
// main-worker tree access. px0 reference: `search.cc:1485-1508`.
unsafe impl Send for ScopedTaskTree {}

impl ScopedTaskTree {
    fn from_tree(tree: &mut NodeTree) -> Self {
        Self(tree)
    }

    unsafe fn with_mut<R>(self, operation: impl FnOnce(&mut NodeTree) -> R) -> R {
        // SAFETY: upheld by the scoped tree-phase contract on `ScopedTaskTree`.
        unsafe { operation(&mut *self.0) }
    }
}

/// px0 processing tasks own disjoint `[start_idx, end_idx)` portions of
/// `minibatch_` (`src/search/classic/search.cc:1322-1362`). The split helper
/// and its regression test prove these ranges do not overlap.
#[derive(Clone, Copy)]
struct ScopedMinibatch(*mut NodeToProcess, usize);

// Safety: only `split_processing_tasks` creates concurrent work ranges; the
// main suffix is processed by the owner after task threads join.
unsafe impl Send for ScopedMinibatch {}

impl ScopedMinibatch {
    fn from_items(items: &mut [NodeToProcess]) -> Self {
        Self(items.as_mut_ptr(), items.len())
    }

    unsafe fn range_mut<'a>(self, start: usize, end: usize) -> &'a mut [NodeToProcess] {
        assert!(
            start <= end && end <= self.1,
            "px0 processing range is in minibatch bounds"
        );
        // SAFETY: caller supplies a range created by `split_processing_tasks`.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), end - start) }
    }
}

/// px0 `SearchWorker` 每轮迭代状态（`src/search/classic/search.h:419-427`）。
///
/// `minibatch_`、`computation_` 和 `number_out_of_order_` 的生命周期只跨越
/// 一次 `InitializeIteration -> UpdateCounters`。在 task worker 的 Rust 所有权
/// 尚未完成翻译前，它们只由主搜索线程持有，不能通过 `SearchWorker` 别名暴露给
/// task。
#[derive(Default)]
struct IterationState {
    minibatch: Vec<NodeToProcess>,
    computation: Option<Box<dyn BackendComputation>>,
    number_out_of_order: usize,
}

/// Immutable inputs consumed by px0 `PickNodesToExtendTask`
/// (`src/search/classic/search.cc:1551-1897`).
///
/// Keeping this separate from `SearchWorker` is a prerequisite for a later
/// task-owned gathering translation: selection may read these values, but it
/// must not gain access to the iteration minibatch, backend computation, or
/// another task's workspace.
struct SelectionContext<'a> {
    params: &'a SearchParams,
    search_state: &'a WorkerSearchState,
    root_move_filter: &'a [Move],
    latest_time_manager_hints: StoppersHints,
    task_workers: i32,
    task_queue: &'a PickTaskQueue,
}

/// Inputs used by px0 `ExtendNode` (`src/search/classic/search.cc:1899-1974`).
/// They are copied from the worker so node extension never needs a mutable
/// `SearchWorker` borrow.
#[derive(Clone, Copy)]
struct ExtendContext {
    played_history_len: usize,
    two_fold_draws: bool,
}

/// Inputs used by px0 `ProcessPickedTask` and `FetchSingleNodeResult`
/// (`src/search/classic/search.cc:1423-1462,2117-2154`). The minibatch range
/// itself remains separately and exclusively borrowed by the caller.
struct ProcessingContext<'a> {
    params: &'a SearchParams,
    computation: Option<&'a dyn BackendComputation>,
    extend: ExtendContext,
}

/// px0 `SearchWorker` (`src/search/classic/search.h:203-448`)。
pub struct SearchWorker<'a> {
    tree: WorkerTree<'a>,
    backend: &'a dyn Backend,
    params: &'a SearchParams,
    search_state: &'a WorkerSearchState,
    stop_controller: Option<Arc<SearchStopController>>,
    /// px0 `Search::root_move_filter_` consumed by
    /// `PickNodesToExtendTask` (`search.cc:1592-1595,1737-1740`).
    root_move_filter: &'a [Move],
    iteration: IterationState,
    history: PositionHistory,
    target_minibatch_size: usize,
    max_out_of_order: usize,
    /// px0-resolved configuration, retained for diagnostics and future
    /// task-thread activation.
    task_workers: i32,
    /// Production activation gate. A real ONNX stop/wait regression found
    /// duplicate `ExtendNode` calls in the scoped raw-pointer path, so task
    /// publication remains synchronous until that px0 tree-phase defect is
    /// resolved.
    active_task_workers: i32,
    /// px0 `SearchWorker::latest_time_manager_hints_`
    /// (`src/search/classic/search.h:368-369`). Each search worker owns this
    /// value; it must not be shared with other gather loops.
    latest_time_manager_hints: StoppersHints,
    played_history_len: usize,
    task_phase: TaskPhaseState,
}

impl<'a> SearchWorker<'a> {
    /// px0 `SearchWorker::SearchWorker` (`search.h:205-233`)。
    pub fn new(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        Self::new_with_stop_controller(tree, backend, params, search_state, None)
    }

    /// px0 constructs every worker with its owning `Search`, so
    /// `UpdateCounters` can call `Search::MaybeTriggerStop`
    /// (`src/search/classic/search.cc:2331-2334`). Unit tests without a
    /// complete search owner intentionally use `None`.
    pub(crate) fn new_with_stop_controller(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
        stop_controller: Option<Arc<SearchStopController>>,
    ) -> Self {
        Self::new_with_stop_controller_and_root_move_filter(tree, backend, params, search_state, stop_controller, &[])
    }

    /// px0 `SearchWorker::SearchWorker` receives its owner `Search`, which
    /// owns `root_move_filter_` for this search lifetime (`search.h:205-233`).
    pub(crate) fn new_with_stop_controller_and_root_move_filter(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
        stop_controller: Option<Arc<SearchStopController>>,
        root_move_filter: &'a [Move],
    ) -> Self {
        let history = tree.history().clone();
        let played_history_len = history.len();
        Self::from_parts(
            WorkerTree::direct(tree),
            history,
            played_history_len,
            backend,
            params,
            search_state,
            stop_controller,
            root_move_filter,
        )
    }

    /// px0 `Search::StartThreads` creates each worker against the shared
    /// search tree, but only locks it for the required tree phases
    /// (`src/search/classic/search.cc:1088-1140,1142-1211`).
    pub(crate) fn new_shared_with_stop_controller_and_root_move_filter(
        tree: Arc<RwLock<NodeTree>>,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
        stop_controller: Option<Arc<SearchStopController>>,
        root_move_filter: &'a [Move],
    ) -> Self {
        let history = tree.read().history().clone();
        let played_history_len = history.len();
        Self::from_parts(
            WorkerTree::shared(tree),
            history,
            played_history_len,
            backend,
            params,
            search_state,
            stop_controller,
            root_move_filter,
        )
    }

    pub fn with_context(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        let history = tree.history().clone();
        let played_history_len = history.len();
        Self::from_parts(
            WorkerTree::direct(tree),
            history,
            played_history_len,
            backend,
            params,
            search_state,
            None,
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)] // px0 SearchWorker constructor dependencies are explicit.
    fn from_parts(
        tree: WorkerTree<'a>,
        history: PositionHistory,
        played_history_len: usize,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
        stop_controller: Option<Arc<SearchStopController>>,
        root_move_filter: &'a [Move],
    ) -> Self {
        let mut target_minibatch_size = if params.minibatch_size > 0 {
            params.minibatch_size as usize
        } else {
            backend.attributes().recommended_batch_size
        };
        if target_minibatch_size == 0 {
            target_minibatch_size = 1;
        }
        let task_workers = Self::resolve_task_workers(params, backend.attributes(), search_state);
        let max_out_of_order = std::cmp::max(
            1,
            (params.max_out_of_order_evals_factor * target_minibatch_size as f32) as usize,
        );
        Self {
            history,
            played_history_len,
            tree,
            backend,
            params,
            search_state,
            stop_controller,
            root_move_filter,
            iteration: IterationState::default(),
            target_minibatch_size,
            max_out_of_order,
            task_workers,
            active_task_workers: task_workers,
            latest_time_manager_hints: StoppersHints::default(),
            task_phase: TaskPhaseState::default(),
        }
    }

    /// px0 `SearchWorker::SearchWorker` task-worker resolution
    /// (`src/search/classic/search.h:205-224`). Negative configuration uses
    /// the backend/CPU heuristic; explicit values are retained verbatim.
    fn resolve_task_workers(
        params: &SearchParams,
        attributes: crate::neural::backend::BackendAttributes,
        search_state: &WorkerSearchState,
    ) -> i32 {
        if params.task_workers_per_search_worker >= 0 {
            return params.task_workers_per_search_worker;
        }
        if attributes.runs_on_cpu {
            return 0;
        }
        let working_threads = search_state
            .thread_count
            .load(Ordering::Acquire)
            .saturating_sub(1)
            .max(1);
        let hardware_threads = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
        i32::try_from((hardware_threads / working_threads).wrapping_sub(1).min(4))
            .expect("px0 task worker count fits i32")
    }

    /// Builds the immutable input view used by px0
    /// `PickNodesToExtendTask` (`src/search/classic/search.cc:1551-1897`).
    fn selection_context(&self) -> SelectionContext<'_> {
        debug_assert!(
            self.active_task_workers <= self.task_workers,
            "the temporary activation gate must never exceed the px0-resolved configuration"
        );
        SelectionContext {
            params: self.params,
            search_state: self.search_state,
            root_move_filter: self.root_move_filter,
            latest_time_manager_hints: self.latest_time_manager_hints.clone(),
            task_workers: self.active_task_workers,
            task_queue: &self.task_phase.queue,
        }
    }

    /// px0 gathering-task handoff (`src/search/classic/search.cc:1828-1864`).
    ///
    /// The parent DFS relinquishes this edge only after the task has been
    /// published successfully. A full queue, an unexpanded child, or a
    /// terminal child leaves the parent's visit budget intact.
    fn hand_off_gathering_task(
        queue: &PickTaskQueue,
        tree: &mut NodeTree,
        workspace: &mut TaskWorkspace,
        current_idx: usize,
        edge_idx: usize,
        task_base_depth: u16,
        child_limit: u32,
    ) -> bool {
        let mv = tree.node(current_idx).edge(edge_idx).mv;
        let child_idx = tree
            .node(current_idx)
            .child(edge_idx)
            .unwrap_or_else(|| tree.arena_mut().spawn_child(current_idx, edge_idx));
        if tree.node(child_idx).n() == 0 || tree.node(child_idx).is_terminal() {
            return false;
        }

        let mut moves_to_base = workspace.moves_to_path.clone();
        moves_to_base.push(mv);
        if !queue.push_gathering(
            tree,
            PickTask::gathering(child_idx, task_base_depth, moves_to_base, child_limit),
        ) {
            return false;
        }
        workspace.visits_to_perform.last_mut().expect("visits")[edge_idx] = 0;
        true
    }

    /// px0 `GatherMinibatch` processing task split
    /// (`src/search/classic/search.cc:1322-1362`). Every queued range ends
    /// before the returned main range, so the caller can prove mutable
    /// minibatch ownership is disjoint before any task is run.
    fn split_processing_tasks(
        queue: &PickTaskQueue,
        params: &SearchParams,
        task_workers: i32,
        minibatch: &[NodeToProcess],
        new_start: usize,
        non_collisions: usize,
    ) -> (usize, bool) {
        let mut main_start = new_start;
        if task_workers <= 0
            || non_collisions
                < usize::try_from(params.minimum_work_size_for_processing)
                    .expect("px0 MinimumProcessingWork is non-negative")
        {
            return (main_start, false);
        }

        let min_per_task = usize::try_from(params.minimum_work_per_task_for_processing)
            .expect("px0 MinimumPerTaskProcessing is non-negative");
        assert!(min_per_task > 0, "px0 MinimumPerTaskProcessing is positive");
        let task_workers = usize::try_from(task_workers).expect("positive task worker count");
        let num_tasks = (non_collisions / min_per_task).clamp(2, task_workers + 1);
        let per_worker = non_collisions / num_tasks;
        queue.reset();
        let mut found = 0usize;
        let mut queued = 0usize;
        for (index, item) in minibatch.iter().enumerate().skip(new_start) {
            if item.is_collision {
                continue;
            }
            found += 1;
            if found == per_worker {
                if !queue.push(PickTask::processing(main_start, index + 1)) {
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
        (main_start, queued > 0)
    }

    /// px0 `SearchWorker::ExecuteOneIteration` (`search.cc:1142-1231`)。
    pub fn execute_one_iteration(&mut self) -> Result<(), EnginError> {
        self.initialize_iteration()?;
        if !self.acquire_searcher_permit() {
            return Ok(());
        }
        let gather_result = self.gather_minibatch();
        // Preserve px0's task-count phase boundary before backend work
        // (`search.cc:1182-1185`). There are no independent task threads yet.
        self.task_phase.queue.idle();
        if let Err(error) = gather_result {
            self.release_searcher_permit();
            return Err(error);
        }
        // px0 marks this worker as waiting on the backend before collision
        // collection/prefetch, then removes it immediately after ComputeBlocking
        // (`search.cc:1187-1199`).
        self.search_state
            .backend_waiting_counter
            .fetch_add(1, Ordering::Relaxed);
        // px0 takes nodes_mutex_ for collision publication before entering the
        // read-only prefetch phase (`search.cc:1977-1987`). Keep that write
        // boundary so another SearchWorker cannot begin backup/cancellation
        // between this iteration's gather and collision hand-off.
        self.with_tree(|worker, tree| worker.collect_collisions_in_tree(tree))?;
        let prefetch_result = self.with_tree_read(|worker, tree| worker.maybe_prefetch_into_cache(tree));
        self.release_searcher_permit();
        if let Err(error) = prefetch_result {
            self.search_state
                .backend_waiting_counter
                .fetch_sub(1, Ordering::Relaxed);
            return Err(error);
        }
        let compute_result = self.run_nn_computation();
        self.search_state
            .backend_waiting_counter
            .fetch_sub(1, Ordering::Relaxed);
        compute_result?;
        self.with_tree(|worker, tree| {
            worker.fetch_minibatch_results_in_tree(tree)?;
            worker.do_backup_update_in_tree(tree)
        })?;
        self.update_counters()
    }

    /// Runs one px0 exclusive `nodes_mutex_` phase. `NodeTree` is an explicit
    /// borrow, so neither direct tests nor the shared production tree need an
    /// active raw-pointer bridge.
    fn with_tree<R>(&mut self, operation: impl FnOnce(&mut Self, &mut NodeTree) -> R) -> R {
        let storage = std::mem::replace(&mut self.tree.storage, TreeStorage::Detached);
        match storage {
            TreeStorage::Direct(tree) => {
                let result = operation(self, tree);
                self.tree.storage = TreeStorage::Direct(tree);
                result
            }
            TreeStorage::Shared(shared) => {
                let mut tree = shared.write();
                let result = operation(self, &mut tree);
                drop(tree);
                self.tree.storage = TreeStorage::Shared(shared);
                result
            }
            TreeStorage::Detached => panic!("px0 tree access outside a tree phase"),
        }
    }

    /// px0 `MaybePrefetchIntoCache` reads its tree under `SharedLock`, while
    /// cache submission remains private to the worker (`search.cc:1989-2007`).
    fn with_tree_read<R>(&mut self, operation: impl FnOnce(&mut Self, &NodeTree) -> R) -> R {
        let storage = std::mem::replace(&mut self.tree.storage, TreeStorage::Detached);
        match storage {
            TreeStorage::Direct(tree) => {
                let result = operation(self, tree);
                self.tree.storage = TreeStorage::Direct(tree);
                result
            }
            TreeStorage::Shared(shared) => {
                let tree = shared.read();
                let result = operation(self, &tree);
                drop(tree);
                self.tree.storage = TreeStorage::Shared(shared);
                result
            }
            TreeStorage::Detached => panic!("px0 tree access outside a tree phase"),
        }
    }

    #[cfg(test)]
    fn with_tree_for_test<R>(&mut self, operation: impl FnOnce(&mut NodeTree) -> R) -> R {
        self.with_tree(|_, tree| operation(tree))
    }

    /// px0 `ExecuteOneIteration` permit acquisition
    /// (`src/search/classic/search.cc:1147-1182`). The default hard-spin is
    /// preserved; `SearchSpinBackoff` only adds periodic yielding after the
    /// same failed CAS loop.
    fn acquire_searcher_permit(&self) -> bool {
        if self.params.max_concurrent_searchers == 0 {
            return true;
        }

        let mut spins = 0usize;
        loop {
            // px0 permits one first iteration even when a stop arrives before
            // this worker reaches the throttle (`search.cc:1156-1160`).
            if self.search_state.stop.load(Ordering::Acquire)
                && self.search_state.total_playouts.load(Ordering::Acquire) > 0
            {
                return false;
            }

            let available = self.search_state.pending_searchers.load(Ordering::Acquire);
            if available > 0
                && self
                    .search_state
                    .pending_searchers
                    .compare_exchange_weak(available, available - 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                return true;
            }

            spins += 1;
            if self.params.search_spin_backoff && spins >= 512 {
                std::thread::yield_now();
                spins = 0;
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// px0 releases the slot immediately after prefetch and before waiting on
    /// the backend (`src/search/classic/search.cc:1192-1195`). A zero limit
    /// represents disabled throttling and therefore owns no slot.
    fn release_searcher_permit(&self) {
        if self.params.max_concurrent_searchers != 0 {
            self.search_state.pending_searchers.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// px0 `GatherMinibatch` backend-idling predicate
    /// (`src/search/classic/search.cc:1290-1301`).
    fn should_yield_for_backend(&self) -> bool {
        let thread_count = self.search_state.thread_count.load(Ordering::Acquire);
        thread_count > 1
            && i32::try_from(thread_count).expect("px0 thread count fits i32")
                - self.search_state.backend_waiting_counter.load(Ordering::Relaxed)
                > self.params.thread_idling_threshold
    }

    /// 单线程测试入口：重复执行 iteration 直到 root N 达标。
    pub fn run_until_root_visits(&mut self, target: u32) -> Result<(), EnginError> {
        while self.with_tree_read(|_, tree| tree.node(tree.current_head()).n()) < target {
            if self.search_state.stop.load(Ordering::Acquire) {
                break;
            }
            self.execute_one_iteration()?;
        }
        Ok(())
    }

    /// px0 `SearchWorker::RunBlocking` (`src/search/classic/search.h:235-249`).
    ///
    /// px0 `SearchWorker::RunBlocking` (`search.h:235-249`). Task splitting
    /// follows the configured px0 count; until scoped task threads are enabled
    /// the owner consumes those tasks synchronously at `WaitForTasks`.
    pub fn run_blocking(&mut self) -> Result<(), EnginError> {
        self.run_blocking_without_task_threads()
    }

    fn run_blocking_without_task_threads(&mut self) -> Result<(), EnginError> {
        (|| loop {
            self.execute_one_iteration()?;
            if self.search_state.stop.load(Ordering::Acquire) {
                break Ok(());
            }
        })()
    }

    /// px0 `SearchWorker::InitializeIteration` (`search.cc:1233-1266`)。
    pub fn initialize_iteration(&mut self) -> Result<(), EnginError> {
        self.with_tree(|worker, tree| worker.initialize_iteration_in_tree(tree))
    }

    fn initialize_iteration_in_tree(&mut self, tree: &NodeTree) -> Result<(), EnginError> {
        // px0 resets the previous computation before asking the backend for a
        // replacement, allowing backend-owned buffers to be recycled.
        self.iteration.computation = None;
        self.iteration.computation = Some(self.backend.create_computation()?);
        self.iteration.minibatch.clear();
        self.iteration.minibatch.reserve(2 * self.target_minibatch_size);
        self.history = tree.history().clone();
        self.played_history_len = self.history.len();
        Ok(())
    }

    /// px0 `SearchWorker::GatherMinibatch` (`search.cc:1268-1363`) 单线程子集。
    pub fn gather_minibatch(&mut self) -> Result<(), EnginError> {
        self.with_tree(|worker, tree| worker.gather_minibatch_in_tree(tree))
    }

    fn gather_minibatch_in_tree(&mut self, tree: &mut NodeTree) -> Result<(), EnginError> {
        let root = tree.current_head();
        let cur_n = tree.node(root).n();
        let remaining_n = self.latest_time_manager_hints.estimated_remaining_playouts();
        let nodes = cur_n.min(remaining_n.max(0) as u32) as i64;
        let mut collisions_left = self.params.collisions_left(nodes);
        self.iteration.number_out_of_order = 0;

        let mut minibatch_size = 0usize;
        while minibatch_size < self.target_minibatch_size && self.iteration.number_out_of_order < self.max_out_of_order
        {
            if minibatch_size > 0
                && self
                    .iteration
                    .computation
                    .as_ref()
                    .map_or(0, |computation| computation.used_batch_size())
                    == 0
            {
                return Ok(());
            }

            // px0 lets another gathering worker fill an idle backend instead
            // of accumulating redundant local work (`search.cc:1290-1301`).
            // With one search worker this is deliberately a no-op.
            if minibatch_size > 0
                && self
                    .iteration
                    .computation
                    .as_ref()
                    .map_or(0, |computation| computation.used_batch_size())
                    > usize::try_from(self.params.idling_minimum_work).expect("px0 IdlingMinimumWork is non-negative")
                && self.should_yield_for_backend()
            {
                return Ok(());
            }

            let new_start = self.iteration.minibatch.len();
            let pick_budget = collisions_left
                .min(self.target_minibatch_size as i32 - minibatch_size as i32)
                .min(self.max_out_of_order as i32 - self.iteration.number_out_of_order as i32);
            self.pick_nodes_to_extend_in_tree(tree, pick_budget.max(0) as u32)?;
            let mut picked_visits = 0usize;
            for item in &self.iteration.minibatch[new_start..] {
                if !item.is_collision {
                    minibatch_size += 1;
                    picked_visits += 1;
                }
            }
            // px0 `search.cc:1322-1347`: split the initial contiguous work
            // ranges into processing tasks, retaining the final range for the
            // main worker. Scoped task threads consume the queued ranges while
            // the enclosing px0 tree phase remains active.
            let (main_start, needs_wait) = Self::split_processing_tasks(
                &self.task_phase.queue,
                self.params,
                self.active_task_workers,
                &self.iteration.minibatch,
                new_start,
                picked_visits,
            );

            let mut runner = std::mem::take(&mut self.task_phase.main_runner);
            let process_result = if needs_wait && self.active_task_workers > 0 {
                self.run_processing_phase_scoped_in_tree(tree, main_start, &mut runner)
            } else {
                let result = self.process_picked_task_in_tree(
                    tree,
                    main_start,
                    self.iteration.minibatch.len(),
                    &mut runner.workspace,
                );
                if needs_wait {
                    self.wait_for_queued_tasks_in_tree(tree)?;
                }
                result
            };
            self.task_phase.main_runner = runner;
            process_result?;

            let mut some_ooo = false;
            for item in &self.iteration.minibatch[new_start..] {
                if item.ooo_completed {
                    some_ooo = true;
                    break;
                }
            }
            if some_ooo {
                let mut i = self.iteration.minibatch.len();
                while i > new_start {
                    i -= 1;
                    if self.iteration.minibatch[i].is_collision {
                        let node_idx = self.iteration.minibatch[i].node_idx;
                        let multivisit = self.iteration.minibatch[i].multivisit;
                        let mut node = node_idx;
                        while let Some(parent) = tree.node(node).parent() {
                            tree.node(parent).cancel_score_update(multivisit);
                            node = parent;
                            if node == tree.current_head() {
                                break;
                            }
                        }
                        self.iteration.minibatch.remove(i);
                    } else if self.iteration.minibatch[i].ooo_completed {
                        // px0 backs up completed out-of-order entries while
                        // reconciling collisions in GatherMinibatch
                        // (`search.cc:1372-1393`), not in ProcessPickedTask.
                        let item = self.iteration.minibatch[i].clone();
                        self.do_backup_update_single_node_in_tree(tree, &item);
                        self.iteration.minibatch.remove(i);
                        minibatch_size = minibatch_size.saturating_sub(1);
                        self.iteration.number_out_of_order += 1;
                    }
                }
            }

            // px0 `search.cc:1400-1419`: consume collision work even when a
            // gather produced no independent NN leaf. A root collision may be
            // safely enlarged to its precomputed `maxvisit` bound, updating
            // every ancestor's in-flight count before it is shared.
            for index in new_start..self.iteration.minibatch.len() {
                if !self.iteration.minibatch[index].is_collision {
                    continue;
                }
                let (node_idx, extra) = {
                    let item = &mut self.iteration.minibatch[index];
                    let desired = item.maxvisit.min(collisions_left.max(0) as u32);
                    let extra = desired.saturating_sub(item.multivisit);
                    item.multivisit += extra;
                    (item.node_idx, extra)
                };
                if extra > 0 {
                    let mut node = node_idx;
                    while let Some(parent) = tree.node(node).parent() {
                        tree.node(parent).increment_n_in_flight(extra);
                        node = parent;
                        if node == tree.current_head() {
                            break;
                        }
                    }
                }
                collisions_left -= self.iteration.minibatch[index].multivisit as i32;
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
    #[cfg(test)]
    fn pick_nodes_to_extend(&mut self, collision_limit: u32) -> Result<(), EnginError> {
        self.with_tree(|worker, tree| worker.pick_nodes_to_extend_in_tree(tree, collision_limit))
    }

    fn pick_nodes_to_extend_in_tree(&mut self, tree: &mut NodeTree, collision_limit: u32) -> Result<(), EnginError> {
        if collision_limit == 0 {
            return Ok(());
        }
        // px0 `SearchWorker::PickNodesToExtend` begins every gather with
        // `ResetTasks` (`src/search/classic/search.cc:1485-1492`).
        self.task_phase.queue.reset();
        let mut receiver = std::mem::take(&mut self.iteration.minibatch);
        let mut runner = std::mem::take(&mut self.task_phase.main_runner);
        let result = if self.active_task_workers == 0 {
            let context = self.selection_context();
            Self::pick_nodes_to_extend_task_with_workspace(
                &context,
                tree,
                tree.current_head(),
                0,
                collision_limit,
                &MoveList::new(),
                &mut receiver,
                &mut runner.workspace,
                true,
            )
        } else {
            self.run_gathering_phase_scoped_in_tree(tree, collision_limit, &mut receiver, &mut runner)
        };
        self.task_phase.main_runner = runner;
        self.iteration.minibatch = receiver;
        if self.active_task_workers == 0 {
            self.wait_for_queued_tasks_in_tree(tree)?;
        } else {
            self.task_phase.queue.wait();
        }
        self.task_phase.queue.drain_results_into(&mut self.iteration.minibatch);
        result
    }

    /// px0 starts `RunTasks` before the main `PickNodesToExtendTask` walks the
    /// tree, then waits before merging task results (`src/search/classic/
    /// search.cc:1069-1140,1485-1508`). The scoped translation preserves that
    /// ordering without sharing `SearchWorker` or task workspaces.
    fn run_gathering_phase_scoped_in_tree(
        &mut self,
        tree: &mut NodeTree,
        collision_limit: u32,
        receiver: &mut Vec<NodeToProcess>,
        main_runner: &mut TaskRunner,
    ) -> Result<(), EnginError> {
        let task_workers = usize::try_from(self.active_task_workers).expect("positive px0 task worker count");
        let context = SelectionContext {
            params: self.params,
            search_state: self.search_state,
            root_move_filter: self.root_move_filter,
            latest_time_manager_hints: self.latest_time_manager_hints.clone(),
            task_workers: self.active_task_workers,
            task_queue: &self.task_phase.queue,
        };
        let queue = &self.task_phase.queue;
        let tree = ScopedTaskTree::from_tree(tree);
        let first_error = Mutex::new(None);
        let context = &context;
        let first_error = &first_error;

        let main_result = std::thread::scope(|scope| {
            for _ in 0..task_workers {
                scope.spawn(move || {
                    let mut runner = TaskRunner::default();
                    while let Some(mut claimed) = queue.take_until_phase_sealed() {
                        debug_assert_eq!(claimed.task.kind, PickTaskKind::Gathering);
                        // SAFETY: the main DFS and these task runners execute
                        // only inside this px0 tree phase; join completes
                        // before the owner reuses `NodeTree`.
                        let result = unsafe {
                            tree.with_mut(|tree| runner.run_gathering_task(context, tree, &mut claimed.task))
                        };
                        if let Err(error) = result {
                            let mut first = first_error.lock().expect("task error lock");
                            if first.is_none() {
                                *first = Some(error);
                            }
                        }
                        queue.complete(claimed);
                    }
                });
            }

            // SAFETY: exactly the same scoped px0 tree phase as the runners.
            // The queue is sealed immediately after this producer returns.
            let result = unsafe {
                tree.with_mut(|tree| {
                    Self::pick_nodes_to_extend_task_with_workspace(
                        context,
                        tree,
                        tree.current_head(),
                        0,
                        collision_limit,
                        &MoveList::new(),
                        receiver,
                        &mut main_runner.workspace,
                        true,
                    )
                })
            };
            queue.seal_phase();
            result
        });

        main_result?;
        if let Some(error) = first_error.lock().expect("task error lock").take() {
            return Err(error);
        }
        Ok(())
    }

    /// px0 `SearchWorker::RunTasks` (`src/search/classic/search.cc:1069-1140`).
    ///
    /// The queue is consumed by its owning worker until px0's independent
    /// task-worker SearchWorker ownership is translated. This is not a
    /// persistent task-thread implementation.
    fn run_queued_tasks_in_tree(&mut self, tree: &mut NodeTree, runner: &mut TaskRunner) -> Result<(), EnginError> {
        while let Some(mut claimed) = self.task_phase.queue.take_until_phase_sealed() {
            match claimed.task.kind {
                PickTaskKind::Gathering => {
                    let context = self.selection_context();
                    runner.run_gathering_task(&context, tree, &mut claimed.task)?;
                }
                PickTaskKind::Processing => {
                    let context = ProcessingContext {
                        params: self.params,
                        computation: self.iteration.computation.as_deref(),
                        extend: ExtendContext {
                            played_history_len: self.played_history_len,
                            two_fold_draws: self.params.two_fold_draws,
                        },
                    };
                    runner.run_processing_range(
                        &context,
                        tree,
                        &mut self.iteration.minibatch[claimed.task.start_idx..claimed.task.end_idx],
                    )?;
                }
            }
            self.task_phase.queue.complete(claimed);
        }
        Ok(())
    }

    /// px0 `SearchWorker::RunTasks` (`src/search/classic/search.cc:1069-1140`).
    ///
    /// This is the scoped Rust translation of px0's task-thread phase. Task
    /// runners own their scratch space, the queue owns claimed task/result
    /// records, and the main worker joins all threads before it resumes tree
    /// access. The raw pointers are confined to this phase; see
    /// `ScopedTaskTree` and `ScopedMinibatch`.
    fn run_tasks_scoped_in_tree(&mut self, tree: &mut NodeTree) -> Result<(), EnginError> {
        debug_assert!(self.active_task_workers > 0);
        let task_workers = usize::try_from(self.active_task_workers).expect("positive px0 task worker count");
        let selection = SelectionContext {
            params: self.params,
            search_state: self.search_state,
            root_move_filter: self.root_move_filter,
            latest_time_manager_hints: self.latest_time_manager_hints.clone(),
            task_workers: self.active_task_workers,
            task_queue: &self.task_phase.queue,
        };
        let processing = ProcessingContext {
            params: self.params,
            computation: self.iteration.computation.as_deref(),
            extend: ExtendContext {
                played_history_len: self.played_history_len,
                two_fold_draws: self.params.two_fold_draws,
            },
        };
        let queue = &self.task_phase.queue;
        let tree = ScopedTaskTree::from_tree(tree);
        let minibatch = ScopedMinibatch::from_items(&mut self.iteration.minibatch);
        let first_error = Mutex::new(None);
        let selection = &selection;
        let processing = &processing;
        let first_error = &first_error;

        std::thread::scope(|scope| {
            for _ in 0..task_workers {
                scope.spawn(move || {
                    let mut runner = TaskRunner::default();
                    while let Some(mut claimed) = queue.take_until_phase_sealed() {
                        let result = match claimed.task.kind {
                            PickTaskKind::Gathering => {
                                // SAFETY: `ScopedTaskTree` is valid for this
                                // entire scope; the main worker is waiting.
                                unsafe {
                                    tree.with_mut(|tree| runner.run_gathering_task(selection, tree, &mut claimed.task))
                                }
                            }
                            PickTaskKind::Processing => {
                                // SAFETY: processing ranges were split before
                                // publication; see `ScopedMinibatch`.
                                unsafe {
                                    let range = minibatch.range_mut(claimed.task.start_idx, claimed.task.end_idx);
                                    tree.with_mut(|tree| runner.run_processing_range(processing, tree, range))
                                }
                            }
                        };
                        if let Err(error) = result {
                            let mut first = first_error.lock().expect("task error lock");
                            if first.is_none() {
                                *first = Some(error);
                            }
                        }
                        queue.complete(claimed);
                    }
                });
            }
        });

        if let Some(error) = first_error.lock().expect("task error lock").take() {
            return Err(error);
        }
        Ok(())
    }

    /// px0 starts processing task threads before the main worker processes the
    /// final minibatch suffix (`src/search/classic/search.cc:1322-1362`).
    /// `split_processing_tasks` proves every queue range ends before
    /// `main_start`, so task runners and the main runner receive disjoint
    /// `NodeToProcess` slices for this scoped tree phase.
    fn run_processing_phase_scoped_in_tree(
        &mut self,
        tree: &mut NodeTree,
        main_start: usize,
        main_runner: &mut TaskRunner,
    ) -> Result<(), EnginError> {
        let task_workers = usize::try_from(self.active_task_workers).expect("positive px0 task worker count");
        let processing = ProcessingContext {
            params: self.params,
            computation: self.iteration.computation.as_deref(),
            extend: ExtendContext {
                played_history_len: self.played_history_len,
                two_fold_draws: self.params.two_fold_draws,
            },
        };
        let queue = &self.task_phase.queue;
        let tree = ScopedTaskTree::from_tree(tree);
        let minibatch = ScopedMinibatch::from_items(&mut self.iteration.minibatch);
        let minibatch_len = self.iteration.minibatch.len();
        let first_error = Mutex::new(None);
        let processing = &processing;
        let first_error = &first_error;

        let main_result = std::thread::scope(|scope| {
            for _ in 0..task_workers {
                scope.spawn(move || {
                    let mut runner = TaskRunner::default();
                    while let Some(claimed) = queue.take_until_phase_sealed() {
                        debug_assert_eq!(claimed.task.kind, PickTaskKind::Processing);
                        // SAFETY: the split helper proved task ranges disjoint
                        // from one another and from the main suffix.
                        let result = unsafe {
                            let range = minibatch.range_mut(claimed.task.start_idx, claimed.task.end_idx);
                            tree.with_mut(|tree| runner.run_processing_range(processing, tree, range))
                        };
                        if let Err(error) = result {
                            let mut first = first_error.lock().expect("task error lock");
                            if first.is_none() {
                                *first = Some(error);
                            }
                        }
                        queue.complete(claimed);
                    }
                });
            }

            // SAFETY: `[main_start, minibatch_len)` is the suffix retained by
            // px0's main worker, outside all published processing ranges.
            let result = unsafe {
                let range = minibatch.range_mut(main_start, minibatch_len);
                tree.with_mut(|tree| main_runner.run_processing_range(processing, tree, range))
            };
            queue.seal_phase();
            result
        });

        main_result?;
        if let Some(error) = first_error.lock().expect("task error lock").take() {
            return Err(error);
        }
        self.task_phase.queue.wait();
        Ok(())
    }

    /// px0 waits for independent task workers here (`search.cc:1494-1508`).
    /// The current ownership translation retains px0's split decision but
    /// drains the queue on its owner until the scoped tree-phase workers are
    /// enabled.
    fn wait_for_queued_tasks_in_tree(&mut self, tree: &mut NodeTree) -> Result<(), EnginError> {
        self.task_phase.queue.seal_phase();
        if self.active_task_workers == 0 {
            self.run_tasks_synchronously_in_tree(tree)?;
        } else {
            self.run_tasks_scoped_in_tree(tree)?;
        }
        self.task_phase.queue.wait();
        Ok(())
    }

    /// CPU fallback for px0 `TaskWorkers=0`: run queued work on the owning
    /// workspace without creating task threads.
    fn run_tasks_synchronously_in_tree(&mut self, tree: &mut NodeTree) -> Result<(), EnginError> {
        let mut runner = std::mem::take(&mut self.task_phase.main_runner);
        let result = self.run_queued_tasks_in_tree(tree, &mut runner);
        self.task_phase.main_runner = runner;
        result
    }

    #[cfg(test)]
    fn run_queued_tasks(&mut self, workspace: &mut TaskWorkspace) -> Result<(), EnginError> {
        let mut runner = TaskRunner {
            workspace: std::mem::take(workspace),
        };
        self.task_phase.queue.seal_phase();
        let result = self.with_tree(|worker, tree| worker.run_queued_tasks_in_tree(tree, &mut runner));
        *workspace = runner.workspace;
        result
    }

    /// px0 `PickNodesToExtendTask` (`src/search/classic/search.cc:1551-1897`)
    /// 的单 worker 路径。
    ///
    /// `task_workers_ == 0` 时 px0 仍使用同一显式 DFS/path-backtrack
    /// 状态机；不要把它替换成递归逐 child 调用，否则 collision 内的策略前缀
    /// 和 visit 分配会发生漂移。
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
        context: &SelectionContext<'_>,
        tree: &mut NodeTree,
        root_idx: usize,
        base_depth: u16,
        collision_limit: u32,
        moves_to_base: &[Move],
        receiver: &mut Vec<NodeToProcess>,
        workspace: &mut TaskWorkspace,
        is_root: bool,
    ) -> Result<(), EnginError> {
        // px0 only reserves when the receiver is still small: the main
        // minibatch and reusable gathering-task result vectors then retain
        // their capacity across iterations (`search.cc:1570-1573`).
        if receiver.capacity() < 30 {
            receiver.reserve(30 - receiver.capacity());
        }
        workspace.current_path.clear();
        workspace.moves_to_path.clear();
        workspace.moves_to_path.extend_from_slice(moves_to_base);
        workspace.current_path.push(-1);

        let mut node_idx = Some(root_idx);
        let mut is_root_node = is_root;
        let mut max_limit = u32::MAX;
        let mut passed_off = 0u32;
        let mut completed_visits = 0u32;
        // px0 snapshots the root best edge before this selection task
        // (`src/search/classic/search.cc:1584-1588`). It is only used for
        // root smart pruning below; nested gathering tasks have no root edge.
        let (current_best_edge, best_node_n) = if is_root {
            let best_edge = *context.search_state.current_best_edge.lock().expect("best edge lock");
            let best_n = best_edge
                .and_then(|edge_idx| tree.node(root_idx).child(edge_idx))
                .map_or(0, |child_idx| tree.node(child_idx).n());
            (best_edge, best_n)
        } else {
            (None, 0)
        };

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

                if tree.node(current_idx).n() == 0 || tree.node(current_idx).is_terminal() {
                    if is_root_node && tree.node(current_idx).try_start_score_update() {
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
                    node_idx = tree.node(current_idx).parent();
                    workspace.current_path.pop();
                    continue;
                }

                if is_root_node {
                    tree.node(current_idx).increment_n_in_flight(cur_limit);
                }

                // px0 `search.cc:1657-1671`: normally a bounded policy prefix
                // is sufficient. With `searchmoves`, px0 must copy every root
                // edge because an allowed move can be outside that prefix.
                let max_needed = if is_root_node && !context.root_move_filter.is_empty() {
                    tree.node(current_idx).num_edges()
                } else {
                    tree.node(current_idx)
                        .num_edges()
                        .min(tree.node(current_idx).n_started() as usize + cur_limit as usize + 2)
                };
                let mut visits = workspace.vtp_buffer.pop().unwrap_or_default();
                visits.clear();
                visits.resize(max_needed, 0);
                workspace.visits_to_perform.push(visits);
                workspace.vtp_last_filled.push(-1);

                // px0 `search.cc:1675-1724`: snapshot policy, child utility,
                // and in-flight visit counters for this tree level.
                let draw_score = Self::draw_score_for_tree(
                    tree,
                    context.params,
                    (workspace.current_path.len() + base_depth as usize).is_multiple_of(2),
                );
                let cpuct = super::uct::compute_cpuct(context.params, tree.node(current_idx).n(), is_root_node);
                let puct_mult = cpuct * (tree.node(current_idx).children_visits().max(1) as f32).sqrt();
                // The formal x7 ONNX contract has no moves-left head. px0
                // constructs the disabled `MEvaluator()` in this case, whose
                // visited and default M utilities are both zero
                // (`search.cc:60-114,1596,1680-1692`).
                let fpu = super::uct::get_fpu(
                    context.params,
                    tree.node(current_idx),
                    tree.arena(),
                    is_root_node,
                    draw_score,
                );
                for edge_idx in 0..max_needed {
                    workspace.current_policy[edge_idx] = tree.node(current_idx).edge(edge_idx).get_p();
                    workspace.current_utility[edge_idx] = f32::NEG_INFINITY;
                }
                for edge_idx in 0..max_needed {
                    let edge = tree.edge_and_node(current_idx, edge_idx);
                    workspace.current_n_started[edge_idx] = edge.n_started();
                    workspace.current_utility[edge_idx] = edge
                        .child()
                        .filter(|child| child.n() > 0)
                        .map_or(fpu, |child| child.q(draw_score));
                    workspace.current_score[edge_idx] = workspace.current_utility[edge_idx]
                        + workspace.current_policy[edge_idx] * puct_mult
                            / (1 + workspace.current_n_started[edge_idx]) as f32;
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
                        // px0 `search.cc:1726-1742`: when the estimated
                        // remaining visits cannot overtake the cached root
                        // best edge, skip the losing root candidate. The
                        // cached edge itself is always retained so a batch
                        // can still make progress.
                        if is_root_node && Some(edge_idx) != current_best_edge {
                            let edge_n = tree
                                .node(current_idx)
                                .child(edge_idx)
                                .map_or(0, |child_idx| tree.node(child_idx).n());
                            if context.latest_time_manager_hints.estimated_remaining_playouts()
                                < i64::from(best_node_n) - i64::from(edge_n)
                            {
                                continue;
                            }
                        }
                        // px0 `search.cc:1737-1740`: root filtering is part of
                        // selection, not only bestmove/PV presentation.
                        if is_root_node
                            && !context.root_move_filter.is_empty()
                            && !context
                                .root_move_filter
                                .contains(&tree.node(current_idx).edge(edge_idx).mv)
                        {
                            continue;
                        }
                        let score = workspace.current_score[edge_idx];
                        if score > best_score {
                            second_best = best_score;
                            best_score = score;
                            best_without_u = workspace.current_utility[edge_idx];
                            best_idx = Some(edge_idx);
                        } else if score > second_best {
                            second_best = score;
                        }
                        if workspace.current_n_started[edge_idx] == 0 {
                            can_exit = true;
                        }
                    }
                    let best_idx = best_idx.expect("expanded non-terminal node has an edge");
                    let new_visits = if second_best.is_finite() {
                        let estimate = if best_without_u < second_best {
                            (workspace.current_policy[best_idx] * puct_mult / (second_best - best_without_u)
                                - (workspace.current_n_started[best_idx] + 1) as f32
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

                    let child_idx = tree
                        .node(current_idx)
                        .child(best_idx)
                        .unwrap_or_else(|| tree.arena_mut().spawn_child(current_idx, best_idx));
                    // px0 `search.cc:1791-1794`: a tree-reused two-fold
                    // terminal may have been reached before the new root.
                    Self::ensure_node_twofold_correct_for_depth_in_tree(
                        tree,
                        child_idx,
                        workspace.current_path.len() as u16 + base_depth,
                    );
                    if tree.node(child_idx).try_start_score_update() {
                        workspace.current_n_started[best_idx] += 1;
                        let remaining_visits = new_visits - 1;
                        if tree.node(child_idx).n() > 0 && !tree.node(child_idx).is_terminal() {
                            tree.node(child_idx).increment_n_in_flight(remaining_visits);
                            workspace.current_n_started[best_idx] += remaining_visits;
                        }
                        workspace.current_score[best_idx] = workspace.current_utility[best_idx]
                            + workspace.current_policy[best_idx] * puct_mult
                                / (1 + workspace.current_n_started[best_idx]) as f32;
                        if tree.node(child_idx).n() == 0 || tree.node(child_idx).is_terminal() {
                            workspace.visits_to_perform.last_mut().expect("visits")[best_idx] -= 1;
                            let mut item = NodeToProcess::visit(
                                child_idx,
                                (workspace.current_path.len() + 1 + base_depth as usize) as u16,
                            );
                            item.moves_to_visit = workspace.moves_to_path.clone();
                            item.moves_to_visit.push(tree.node(current_idx).edge(best_idx).mv);
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
                if context.task_workers > 0 {
                    let min_work = u32::try_from(context.params.minimum_work_size_for_picking)
                        .expect("px0 MinimumPickingWork is non-negative");
                    let min_remaining = u32::try_from(context.params.minimum_remaining_work_size_for_picking)
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
                            if Self::hand_off_gathering_task(
                                context.task_queue,
                                tree,
                                workspace,
                                current_idx,
                                edge_idx,
                                (workspace.current_path.len() + base_depth as usize) as u16,
                                child_limit,
                            ) {
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
                    let mv = tree.node(current_idx).edge(edge_idx).mv;
                    if workspace.moves_to_path.len() != workspace.current_path.len() + base_depth as usize {
                        workspace.moves_to_path.push(mv);
                    } else {
                        *workspace.moves_to_path.last_mut().expect("path move") = mv;
                    }
                    *workspace.current_path.last_mut().expect("path entry") = edge_idx as isize;
                    workspace.current_path.push(-1);
                    node_idx = Some(
                        tree.node(current_idx)
                            .child(edge_idx)
                            .unwrap_or_else(|| tree.arena_mut().spawn_child(current_idx, edge_idx)),
                    );
                    found_child = true;
                    break;
                }
            }
            if !found_child {
                node_idx = tree.node(current_idx).parent();
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
    fn ensure_node_twofold_correct_for_depth_in_tree(tree: &mut NodeTree, child_idx: usize, depth: u16) {
        let child = tree.node(child_idx);
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
            let parent = tree.node(current_idx).parent();
            tree.node_mut(current_idx)
                .revert_terminal_visits(wl, d, m + depth_counter as f32, terminal_visits);
            depth_counter += 1;
            if depth_counter > depth {
                break;
            }
            node_idx = parent;
        }
        tree.make_not_terminal(child_idx);
    }

    #[cfg(test)]
    fn ensure_node_twofold_correct_for_depth(&mut self, child_idx: usize, depth: u16) {
        self.with_tree(|_, tree| Self::ensure_node_twofold_correct_for_depth_in_tree(tree, child_idx, depth));
    }

    /// px0 `SearchWorker::ProcessPickedTask` (`src/search/classic/search.cc:1423-1462`)。
    #[cfg(test)]
    fn process_picked_task(
        &mut self,
        start_idx: usize,
        end_idx: usize,
        workspace: &mut TaskWorkspace,
    ) -> Result<(), EnginError> {
        self.with_tree(|worker, tree| worker.process_picked_task_in_tree(tree, start_idx, end_idx, workspace))
    }

    fn process_picked_task_in_tree(
        &mut self,
        tree: &mut NodeTree,
        start_idx: usize,
        end_idx: usize,
        workspace: &mut TaskWorkspace,
    ) -> Result<(), EnginError> {
        let context = ProcessingContext {
            params: self.params,
            computation: self.iteration.computation.as_deref(),
            extend: ExtendContext {
                played_history_len: self.played_history_len,
                two_fold_draws: self.params.two_fold_draws,
            },
        };
        let range = &mut self.iteration.minibatch[start_idx..end_idx];
        Self::process_picked_range(&context, tree, range, workspace)
    }

    fn process_picked_range(
        context: &ProcessingContext<'_>,
        tree: &mut NodeTree,
        range: &mut [NodeToProcess],
        workspace: &mut TaskWorkspace,
    ) -> Result<(), EnginError> {
        let extend_context = ExtendContext {
            played_history_len: context.extend.played_history_len,
            two_fold_draws: context.extend.two_fold_draws,
        };
        workspace.history = tree.history().clone();
        for item in range {
            // px0 immediately skips collisions here. They only carry an
            // in-flight reservation and must not enter terminal/cache OOO
            // evaluation (`search.cc:1429-1432`).
            if item.is_collision {
                continue;
            }
            let node_idx = item.node_idx;
            let depth = item.depth;
            let moves_to_visit = std::mem::take(&mut item.moves_to_visit);
            let is_terminal = tree.node(node_idx).is_terminal();
            if item.is_extendable(is_terminal) {
                Self::extend_node_in_tree(
                    extend_context,
                    tree,
                    node_idx,
                    depth,
                    &moves_to_visit,
                    &mut workspace.history,
                )?;
                if !tree.node(node_idx).is_terminal() {
                    let position = EvalPosition {
                        positions: workspace.history.positions().to_vec(),
                        legal_moves: tree
                            .node(node_idx)
                            .edges()
                            .iter()
                            .map(|edge| edge.mv)
                            .collect::<MoveList>(),
                    };
                    let (result, ticket) = context
                        .computation
                        .ok_or(EnginError::PortIncomplete("P4 ProcessPickedTask without computation"))?
                        .add_input(position)?;
                    item.nn_queried = true;
                    item.is_cache_hit = result == AddInputResult::FetchedImmediately;
                    item.eval_ticket = Some(ticket);
                }
            }
            if context.params.out_of_order_eval && item.can_eval_out_of_order(tree.node(node_idx).is_terminal()) {
                Self::fetch_single_node_result_in_tree(context, tree, item)?;
                item.ooo_completed = true;
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::ExtendNode` (`src/search/classic/search.cc:1899-1974`)。
    #[cfg(test)]
    fn extend_node(
        &mut self,
        node_idx: usize,
        depth: u16,
        moves_to_node: &[Move],
        history: &mut PositionHistory,
    ) -> Result<(), EnginError> {
        self.with_tree(|worker, tree| {
            Self::extend_node_in_tree(
                ExtendContext {
                    played_history_len: worker.played_history_len,
                    two_fold_draws: worker.params.two_fold_draws,
                },
                tree,
                node_idx,
                depth,
                moves_to_node,
                history,
            )
        })
    }

    fn extend_node_in_tree(
        context: ExtendContext,
        tree: &mut NodeTree,
        node_idx: usize,
        depth: u16,
        moves_to_node: &[Move],
        history: &mut PositionHistory,
    ) -> Result<(), EnginError> {
        let root = tree.current_head();
        history.trim(context.played_history_len);
        for mv in moves_to_node {
            history.append(*mv);
        }
        let board = history.last().board();
        let legal_moves = board.generate_legal_moves();
        if legal_moves.is_empty() {
            tree.make_terminal(
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
                tree.make_terminal(node_idx, history.rule_judge(), 0.0, Terminal::EndOfGame);
                return Ok(());
            }
            // px0 `search.cc:1930-1959`: an initial repetition can be a
            // forced two-fold result only after the complete cycle is inside
            // the searched line. The special terminal can later be reverted
            // when tree reuse moves the root into that cycle.
            if history.last().repetitions() == 1
                && depth.saturating_sub(1) >= 4
                && context.two_fold_draws
                && u32::from(depth.saturating_sub(1)) >= history.last().cycle_length()
            {
                let cycle_length = history.last().cycle_length();
                let result = history.rule_judge();
                if result == GameResult::Draw {
                    tree.make_terminal(node_idx, result, cycle_length as f32, Terminal::TwoFold);
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
                        tree.make_terminal(node_idx, result, cycle_length as f32, Terminal::TwoFold);
                        return Ok(());
                    }
                }
            }
            if !board.has_mating_material() || history.last().rule60_ply() >= 120 {
                tree.make_terminal(node_idx, GameResult::Draw, 0.0, Terminal::EndOfGame);
                return Ok(());
            }
        }
        tree.node_mut(node_idx).create_edges(&legal_moves);
        Ok(())
    }

    /// px0 `SearchWorker::CollectCollisions` (`search.cc:1977-1987`)。
    fn collect_collisions_in_tree(&mut self, _tree: &mut NodeTree) -> Result<(), EnginError> {
        for item in &self.iteration.minibatch {
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

    /// px0 `SearchWorker::MaybePrefetchIntoCache` (`search.cc:1989-2007`).
    /// This phase only reads the tree; history and computation belong to this
    /// worker and remain mutable without upgrading the tree lock.
    fn maybe_prefetch_into_cache(&mut self, tree: &NodeTree) -> Result<(), EnginError> {
        if self.search_state.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let used = self
            .iteration
            .computation
            .as_ref()
            .map_or(0, |computation| computation.used_batch_size());
        if used == 0 || used >= self.params.max_prefetch_batch as usize {
            return Ok(());
        }
        let budget = self.params.max_prefetch_batch as usize - used;
        let root = tree.current_head();
        // px0 resets the workspace history before walking prefetch candidates
        // (`search.cc:1997-2004`). ProcessPickedTask leaves this workspace at
        // its last expanded leaf, so using it directly would encode the wrong
        // position for a root-relative cache probe.
        self.history.trim(self.played_history_len);
        self.prefetch_into_cache(tree, Some(root), budget, false)?;
        Ok(())
    }

    /// px0 `Search::GetDrawScore` (`src/search/classic/search.cc:401-405`)。
    fn draw_score_for_tree(tree: &NodeTree, params: &SearchParams, is_odd_depth: bool) -> f32 {
        if is_odd_depth == tree.history().is_black_to_move() {
            params.draw_score
        } else {
            -params.draw_score
        }
    }

    #[cfg(test)]
    fn draw_score(&mut self, is_odd_depth: bool) -> f32 {
        self.with_tree_read(|worker, tree| Self::draw_score_for_tree(tree, worker.params, is_odd_depth))
    }

    /// px0 `PrefetchIntoCache` (`search.cc:2010-2099`)。
    fn prefetch_into_cache(
        &mut self,
        tree: &NodeTree,
        node_idx: Option<usize>,
        budget: usize,
        is_odd_depth: bool,
    ) -> Result<usize, EnginError> {
        let draw_score = Self::draw_score_for_tree(tree, self.params, is_odd_depth);
        if budget == 0 {
            return Ok(0);
        }

        // px0 also reaches this branch for a missing child edge. It is still a
        // valid future leaf and must be encoded from the current history.
        if node_idx.is_none_or(|idx| tree.node(idx).n_started() == 0) {
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
            if let Some(computation) = self.iteration.computation.as_mut() {
                let _ = computation.add_input(EvalPosition {
                    positions: self.history.positions().to_vec(),
                    legal_moves,
                })?;
            }
            return Ok(1);
        }

        let node_idx = node_idx.expect("checked above");
        if tree.node(node_idx).n() == 0 || tree.node(node_idx).is_terminal() {
            return Ok(0);
        }

        // px0 `search.cc:2036-2051`: score all legal edges using the same
        // EdgeAndNode Q/U proxy as selection. The negated score permits
        // ascending partial sorting below.
        let is_root = node_idx == tree.current_head();
        let cpuct = super::uct::compute_cpuct(self.params, tree.node(node_idx).n(), is_root);
        let puct_mult = cpuct * (tree.node(node_idx).children_visits().max(1) as f32).sqrt();
        let fpu = super::uct::get_fpu(self.params, tree.node(node_idx), tree.arena(), is_root, draw_score);
        let mut scores = (0..tree.node(node_idx).num_edges())
            .filter_map(|edge_idx| {
                let edge = tree.edge_and_node(node_idx, edge_idx);
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
                let edge = tree.edge_and_node(node_idx, edge_idx);
                let q = edge.q(-fpu, draw_score);
                if next_score > q {
                    let estimated = edge.p() * puct_mult / (next_score - q) - edge.n_started() as f32;
                    budget_to_spend = std::cmp::min(budget - total_budget_spent, estimated as usize + 1);
                } else {
                    budget_to_spend = budget - total_budget_spent;
                }
            }

            let (mv, child_idx) = {
                let edge = tree.edge_and_node(node_idx, edge_idx);
                (edge.mv(), tree.node(node_idx).child(edge_idx))
            };
            self.history.append(mv);
            let result = self.prefetch_into_cache(tree, child_idx, budget_to_spend, !is_odd_depth);
            self.history.pop();
            let budget_spent = result?;
            total_budget_spent += budget_spent;
        }
        Ok(total_budget_spent)
    }

    /// px0 `SearchWorker::RunNNComputation` (`search.cc:2103-2107`)。
    pub fn run_nn_computation(&mut self) -> Result<(), EnginError> {
        if let Some(computation) = self.iteration.computation.as_mut() {
            if computation.used_batch_size() > 0 {
                computation.compute_blocking()?;
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::FetchMinibatchResults` (`search.cc:2109-2156`)。
    pub fn fetch_minibatch_results(&mut self) -> Result<(), EnginError> {
        self.with_tree(|worker, tree| worker.fetch_minibatch_results_in_tree(tree))
    }

    fn fetch_minibatch_results_in_tree(&mut self, tree: &mut NodeTree) -> Result<(), EnginError> {
        let context = ProcessingContext {
            params: self.params,
            computation: self.iteration.computation.as_deref(),
            extend: ExtendContext {
                played_history_len: self.played_history_len,
                two_fold_draws: self.params.two_fold_draws,
            },
        };
        for item in &mut self.iteration.minibatch {
            Self::fetch_single_node_result_in_tree(&context, tree, item)?;
        }
        Ok(())
    }

    /// px0 `SearchWorker::FetchSingleNodeResult` (`search.cc:2117-2154`)。
    fn fetch_single_node_result_in_tree(
        context: &ProcessingContext<'_>,
        tree: &mut NodeTree,
        item: &mut NodeToProcess,
    ) -> Result<(), EnginError> {
        if item.is_collision {
            return Ok(());
        }
        let node_idx = item.node_idx;
        if !item.nn_queried {
            item.eval = Arc::new(EvalResult {
                wl: tree.node(node_idx).wl(),
                d: tree.node(node_idx).d(),
                m: tree.node(node_idx).m(),
                policies: Vec::new(),
            });
            return Ok(());
        }
        let ticket = item
            .eval_ticket
            .ok_or(EnginError::PortIncomplete("P4 FetchSingleNodeResult missing ticket"))?;
        let eval = context
            .computation
            .ok_or(EnginError::PortIncomplete(
                "P4 FetchSingleNodeResult without computation",
            ))?
            .take_result(ticket)?;
        let mut wl = -eval.wl;
        let mut d = eval.d;
        // px0 rescales NN WDL before it reaches backup, never afterwards in
        // UCI formatting (`src/search/classic/search.cc:2128-2143`). The
        // neutral defaults are identity in ratio/diff terms; contempt mode is
        // translated separately before non-zero diff becomes configurable.
        if context.params.wdl_rescale_ratio != 1.0
            || (context.params.wdl_rescale_diff != 0.0 && context.params.contempt_mode != ContemptMode::None)
        {
            let root_stm = (context.params.contempt_mode == ContemptMode::Black) == tree.history().is_black_to_move();
            let sign = if root_stm ^ (item.depth % 2 == 1) { 1.0 } else { -1.0 };
            wdl_rescale(
                &mut wl,
                &mut d,
                context.params.wdl_rescale_ratio,
                if context.params.contempt_mode == ContemptMode::None {
                    0.0
                } else {
                    context.params.wdl_rescale_diff
                },
                sign,
                false,
                context.params.wdl_max_s,
            );
        }
        if tree.node(node_idx).n() == 0 {
            for (edge_idx, policy) in eval.policies.iter().enumerate() {
                tree.node_mut(node_idx).edge_mut(edge_idx).set_p(*policy);
            }
            // px0 sorts the just-initialized policy before any child node can
            // be spawned (`node.cc:291-298`, `search.cc:2145-2153`).
            tree.node_mut(node_idx).sort_edges();
        }
        // Policy is consumed by the leaf edges above. Keep only the adjusted
        // scalar fields in the tree item so a cache hit never clones its policy
        // vector.
        item.eval = Arc::new(EvalResult {
            wl,
            d,
            m: eval.m,
            policies: Vec::new(),
        });
        Ok(())
    }

    /// px0 `SearchWorker::DoBackupUpdate` (`search.cc:2158-2258`) 单线程子集。
    pub fn do_backup_update(&mut self) -> Result<(), EnginError> {
        self.with_tree(|worker, tree| worker.do_backup_update_in_tree(tree))
    }

    fn do_backup_update_in_tree(&mut self, tree: &mut NodeTree) -> Result<(), EnginError> {
        let mut work_done = self.iteration.number_out_of_order > 0;
        let items: Vec<_> = self
            .iteration
            .minibatch
            .iter()
            .filter(|item| !item.is_collision)
            .cloned()
            .collect();
        for item in items {
            self.do_backup_update_single_node_in_tree(tree, &item);
            work_done = true;
        }
        if work_done {
            self.cancel_shared_collisions_in_tree(tree);
            self.search_state.total_batches.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// px0 `Search::CancelSharedCollisions` (`search.cc:1044-1053`).
    fn cancel_shared_collisions_in_tree(&mut self, tree: &mut NodeTree) {
        let collisions = std::mem::take(&mut *self.search_state.shared_collisions.lock().expect("collisions lock"));
        for (node_idx, multivisit) in collisions {
            let mut current = tree.node(node_idx).parent();
            while let Some(node_idx) = current {
                tree.node(node_idx).cancel_score_update(multivisit);
                current = tree.node(node_idx).parent();
            }
        }
    }

    /// px0 `SearchWorker::MaybeSetBounds` (`search.cc:2229-2289`).
    #[allow(
        clippy::too_many_arguments,
        reason = "Keeps the px0 MaybeSetBounds output-parameter contract explicit."
    )]
    fn maybe_set_bounds_in_tree(
        &mut self,
        tree: &mut NodeTree,
        parent_idx: usize,
        m: f32,
        n_to_fix: &mut u32,
        v_delta: &mut f32,
        d_delta: &mut f32,
        m_delta: &mut f32,
    ) -> bool {
        let mut losing_m: f32 = 0.0;
        let mut prefer_tablebase = false;
        let mut lower = GameResult::BlackWon;
        let mut upper = GameResult::BlackWon;

        for edge_idx in 0..tree.node(parent_idx).num_edges() {
            let child = tree.node(parent_idx).child(edge_idx).map(|idx| tree.node(idx));
            let edge_lower = child.map_or(GameResult::BlackWon, |node| node.lower_bound());
            let edge_upper = child.map_or(GameResult::WhiteWon, |node| node.upper_bound());
            lower = lower.max(edge_lower);
            upper = upper.max(edge_upper);

            let is_tablebase = child.is_some_and(|node| node.is_tablebase_terminal());
            if edge_lower == GameResult::WhiteWon && !is_tablebase {
                prefer_tablebase = false;
                break;
            }
            if edge_upper == GameResult::BlackWon {
                losing_m = losing_m.max(child.map_or(0.0, |node| node.m()));
            }
            prefer_tablebase |= is_tablebase;
        }

        if lower == GameResult::BlackWon && upper == GameResult::WhiteWon {
            return false;
        }
        if lower == upper {
            let parent = tree.node(parent_idx);
            *n_to_fix = parent.n();
            debug_assert!(*n_to_fix > 0);
            let current_v = parent.wl();
            let current_d = parent.d();
            let current_m = parent.m();
            let result = upper.negate();
            let plies_left = if upper == GameResult::BlackWon {
                losing_m.max(m)
            } else {
                m
            } + 1.0;
            tree.make_terminal(
                parent_idx,
                result,
                plies_left,
                if prefer_tablebase {
                    Terminal::Tablebase
                } else {
                    Terminal::EndOfGame
                },
            );
            let parent = tree.node(parent_idx);
            *v_delta = -(parent.wl() - current_v);
            *d_delta = parent.d() - current_d;
            *m_delta = parent.m() - current_m;
        } else {
            tree.node_mut(parent_idx).set_bounds(upper.negate(), lower.negate());
        }
        true
    }

    /// px0 `SearchWorker::DoBackupUpdateSingleNode` (`search.cc:2175-2289`).
    #[cfg(test)]
    fn do_backup_update_single_node(&mut self, item: &NodeToProcess) {
        self.with_tree(|worker, tree| worker.do_backup_update_single_node_in_tree(tree, item));
    }

    fn do_backup_update_single_node_in_tree(&mut self, tree: &mut NodeTree, item: &NodeToProcess) {
        let mut node_idx = item.node_idx;
        let mut v = item.eval.wl;
        let mut d = item.eval.d;
        let mut m = item.eval.m;
        let root = tree.current_head();
        let mut update_parent_bounds =
            self.params.sticky_endgames && tree.node(node_idx).is_terminal() && tree.node(node_idx).n() == 0;
        let mut n_to_fix = 0;
        let mut v_delta = 0.0;
        let mut d_delta = 0.0;
        let mut m_delta = 0.0;
        loop {
            if tree.node(node_idx).is_terminal() {
                v = tree.node(node_idx).wl();
                d = tree.node(node_idx).d();
                m = tree.node(node_idx).m();
            }
            tree.node_mut(node_idx).finalize_score_update(v, d, m, item.multivisit);
            if n_to_fix > 0 && !tree.node(node_idx).is_terminal() {
                tree.node_mut(node_idx)
                    .adjust_for_terminal(v_delta, d_delta, m_delta, n_to_fix);
            }
            // px0 solidifies a sufficiently visited node after its score is
            // finalized. The root best-edge cache stores an edge iterator, so
            // refresh it if root solidification changed that representation
            // (`search.cc:2211-2217`, `node.cc:245-288`).
            if tree.node(node_idx).n() >= self.params.solid_tree_threshold
                && tree.make_solid(node_idx)
                && node_idx == root
            {
                *self.search_state.current_best_edge.lock().expect("best edge lock") =
                    best_child_edge(tree, root, self.params, 0, self.root_move_filter);
            }
            if node_idx == root {
                break;
            }
            let parent = tree.node(node_idx).parent().expect("non-root has parent");
            let old_update_parent_bounds = update_parent_bounds;
            if tree.node(parent).is_terminal() {
                n_to_fix = 0;
            }
            update_parent_bounds = update_parent_bounds
                && parent != root
                && !tree.node(parent).is_terminal()
                && self.maybe_set_bounds_in_tree(
                    tree,
                    parent,
                    m,
                    &mut n_to_fix,
                    &mut v_delta,
                    &mut d_delta,
                    &mut m_delta,
                );
            // px0 refreshes the root cache only when a terminal bound may
            // have changed the candidate or a non-cached child catches up in
            // visits (`search.cc:2241-2249`). This avoids a full root sort on
            // every backup while preserving the selection pruning invariant.
            if parent == root {
                let cached_edge = *self.search_state.current_best_edge.lock().expect("best edge lock");
                let cached_node = cached_edge.and_then(|edge_idx| tree.node(root).child(edge_idx));
                let cached_n = cached_node.map_or(0, |child_idx| tree.node(child_idx).n());
                if (old_update_parent_bounds && tree.node(node_idx).is_terminal())
                    || (cached_node != Some(node_idx) && cached_n <= tree.node(node_idx).n())
                {
                    *self.search_state.current_best_edge.lock().expect("best edge lock") =
                        best_child_edge(tree, root, self.params, 0, self.root_move_filter);
                }
            }
            node_idx = parent;
            v = -v;
            v_delta = -v_delta;
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

    /// px0 `SearchWorker::UpdateCounters` (`search.cc:2331-2364`).
    pub fn update_counters(&mut self) -> Result<(), EnginError> {
        if let Some(controller) = &self.stop_controller {
            controller.maybe_trigger_stop(self.search_state, &mut self.latest_time_manager_hints);
        }
        // px0 deliberately backs off when an iteration only found collisions
        // (`src/search/classic/search.cc:2337-2351`). Such an iteration does
        // not advance the tree; immediately spinning again only competes with
        // the worker that owns the useful in-flight work.
        let work_done =
            self.iteration.number_out_of_order > 0 || self.iteration.minibatch.iter().any(|item| !item.is_collision);
        if !work_done {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // px0 applies NodesPerSecondLimit after backup/counter publication,
        // never in gather or NN compute. The first completed batch is the
        // timing origin for the same reason as its UCI NPS field.
        if self.params.nps_limit > 0.0 {
            while !self.search_state.stop.load(Ordering::Acquire) {
                // px0 obtains this through Search::GetTimeSinceFirstBatch,
                // falling back to move-start before the watchdog has
                // initialized `nps_start_time_`
                // (`src/search/classic/search.cc:1213-1231`).
                let elapsed_ms = self
                    .stop_controller
                    .as_ref()
                    .map_or(0.0, |controller| controller.nps_elapsed_or_move_start_ms() as f32);
                if elapsed_ms <= 0.0 {
                    break;
                }
                let nps = self.search_state.total_playouts.load(Ordering::Acquire) as f32 * 1_000.0 / elapsed_ms;
                if nps <= self.params.nps_limit {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::sync::Once;
    use std::time::{Duration, Instant};

    use xiangqi_core::{initialize_magic_bitboards, GameResult, GameState, STARTPOS_FEN};

    use super::*;
    use crate::neural::backend::{
        Backend, BackendAttributes, BackendComputation, EvalPosition, EvalResult, UniformBackend,
    };

    static INIT: Once = Once::new();

    fn ensure_init() {
        INIT.call_once(initialize_magic_bitboards);
    }

    /// px0 replenishes `pending_searchers_` before NN work, so a worker that
    /// leaves the gather phase cannot starve later workers
    /// (`search.cc:1147-1195`).
    #[test]
    fn searcher_permit_returns_slot_after_gather_phase() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let search_state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        search_state.set_max_concurrent_searchers(1);
        let worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);

        assert!(worker.acquire_searcher_permit());
        assert_eq!(search_state.pending_searchers.load(Ordering::Acquire), 0);
        worker.release_searcher_permit();
        assert_eq!(search_state.pending_searchers.load(Ordering::Acquire), 1);
    }

    /// px0 only applies the idling shortcut when more than one search worker
    /// could keep the backend busy (`search.cc:1290-1301`).
    #[test]
    fn backend_idling_requires_another_search_worker() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let backend = UniformBackend::default();
        let params = SearchParams {
            thread_idling_threshold: 1,
            ..SearchParams::default()
        };
        let search_state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);

        assert!(!worker.should_yield_for_backend());
        search_state.thread_count.store(4, Ordering::Release);
        search_state.backend_waiting_counter.store(1, Ordering::Release);
        assert!(worker.should_yield_for_backend());
    }

    /// A non-CPU attribute wrapper makes the px0 automatic task-worker path
    /// observable while retaining UniformBackend's deterministic computation.
    #[derive(Clone, Debug, Default)]
    struct GpuUniformBackend(UniformBackend);

    impl Backend for GpuUniformBackend {
        fn evaluate(&self, history: &PositionHistory, legal_moves: &[Move]) -> Arc<EvalResult> {
            self.0.evaluate(history, legal_moves)
        }

        fn attributes(&self) -> BackendAttributes {
            BackendAttributes {
                runs_on_cpu: false,
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

    /// px0 activates an explicit GPU task-worker request for both gathering
    /// and processing phases (`search.h:205-244`,
    /// `search.cc:1322-1362,1494-1508`).
    #[test]
    fn gpu_backend_keeps_requested_task_worker_count() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);

        let backend = GpuUniformBackend::default();
        let params = SearchParams {
            minibatch_size: 4,
            task_workers_per_search_worker: 1,
            minimum_work_size_for_processing: 2,
            minimum_work_per_task_for_processing: 1,
            max_collision_visits: 4,
            max_collision_visits_scaling_start: 0,
            max_collision_visits_scaling_end: 1,
            out_of_order_eval: false,
            ..SearchParams::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let search_state = Arc::new(WorkerSearchState::new(Arc::clone(&stop)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, search_state.as_ref());

        assert_eq!(worker.task_workers, 1);
        assert_eq!(worker.active_task_workers, 1);

        std::thread::scope(|scope| {
            let state = Arc::clone(&search_state);
            let stop = Arc::clone(&stop);
            let stopper = scope.spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(5);
                while state.total_playouts.load(Ordering::Acquire) < 16 && Instant::now() < deadline {
                    std::thread::yield_now();
                }
                stop.store(true, Ordering::Release);
            });
            worker.run_blocking().expect("GPU task split search");
            stopper.join().expect("stopper thread");
        });

        assert!(search_state.total_playouts.load(Ordering::Acquire) >= 16);
        assert_eq!(worker.task_workers, 1);
        assert_eq!(worker.active_task_workers, 1);
        assert_eq!(tree.node(tree.current_head()).n_in_flight(), 0);
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
        let search_state = WorkerSearchState::new(stop);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.initialize_iteration().expect("init");
        // `PickNodesToExtendTask` always reserves the path before handing an
        // item to ProcessPickedTask (`px0 search.cc:1613-1636`).  Keep this
        // focused OOO test on that post-selection contract.
        assert!(worker.with_tree_for_test(|tree| tree.node_mut(root).try_start_score_update()));
        worker.iteration.minibatch.push(NodeToProcess::visit(root, 1));
        let mut workspace = TaskWorkspace::default();
        worker.process_picked_task(0, 1, &mut workspace).expect("ooo terminal");
        assert!(worker.iteration.minibatch[0].ooo_completed);
    }

    /// px0 `ProcessPickedTask` skips a collision before checking its terminal
    /// state (`src/search/classic/search.cc:1429-1436`). A collision is only
    /// cancelled or shared by Gather/backup; it can never become an OOO result.
    #[test]
    fn out_of_order_skips_terminal_collision() {
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
        let search_state = WorkerSearchState::new(stop);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.initialize_iteration().expect("init");
        worker.iteration.minibatch.push(NodeToProcess::collision(root, 1, 1, 1));

        let mut workspace = TaskWorkspace::default();
        worker
            .process_picked_task(0, 1, &mut workspace)
            .expect("skip collision");

        assert!(!worker.iteration.minibatch[0].ooo_completed);
        assert!(!worker.iteration.minibatch[0].nn_queried);
    }

    /// px0 immediately fetches a cache-hit leaf during `ProcessPickedTask`,
    /// before `GatherMinibatch` reconciles and backs it up out of order
    /// (`src/search/classic/search.cc:1423-1462,1370-1393`).
    #[test]
    fn out_of_order_cache_hit_avoids_nn_batch() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let backend = UniformBackend::with_wdl(0.2, 0.3, 4.0);
        let legal_moves = tree.history().last().board().generate_legal_moves();
        let position = EvalPosition {
            positions: tree.history().positions().to_vec(),
            legal_moves: legal_moves.clone(),
        };
        backend.store_cache(&position, backend.evaluate(tree.history(), &legal_moves));
        let params = SearchParams {
            out_of_order_eval: true,
            ..SearchParams::default()
        };
        let search_state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.initialize_iteration().expect("init");
        assert!(worker.with_tree_for_test(|tree| tree.node_mut(root).try_start_score_update()));
        worker.iteration.minibatch.push(NodeToProcess::visit(root, 1));

        let mut workspace = TaskWorkspace::default();
        worker
            .process_picked_task(0, 1, &mut workspace)
            .expect("cache-hit OOO evaluation");

        assert!(worker.iteration.minibatch[0].ooo_completed);
        assert!(worker.iteration.minibatch[0].is_cache_hit);
        assert_eq!(
            worker
                .iteration
                .computation
                .as_ref()
                .expect("computation")
                .used_batch_size(),
            0
        );
        assert!(worker.iteration.minibatch[0].eval.policies.is_empty());
        worker.with_tree_for_test(|tree| {
            assert_eq!(tree.node(root).num_edges(), legal_moves.len());
            assert!(tree.node(root).edges().iter().all(|edge| edge.get_p() > 0.0));
        });
    }

    /// px0 removes an OOO cache result from the minibatch only after its
    /// immediate backup in `GatherMinibatch` (`search.cc:1370-1393`).
    #[test]
    fn gather_backs_up_out_of_order_cache_hit() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let backend = UniformBackend::default();
        let legal_moves = tree.history().last().board().generate_legal_moves();
        let position = EvalPosition {
            positions: tree.history().positions().to_vec(),
            legal_moves: legal_moves.clone(),
        };
        backend.store_cache(&position, backend.evaluate(tree.history(), &legal_moves));
        let params = SearchParams {
            minibatch_size: 1,
            out_of_order_eval: true,
            ..SearchParams::default()
        };
        let search_state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.initialize_iteration().expect("init");

        worker.gather_minibatch().expect("gather cache hit");

        assert_eq!(worker.iteration.number_out_of_order, 1);
        assert_eq!(worker.with_tree_for_test(|tree| tree.node(root).n()), 1);
        // px0 keeps gathering after the OOO backup, so the next ordinary leaf
        // is already reserved by the time this phase returns.
        assert!(worker.with_tree_for_test(|tree| tree.node(root).n_in_flight() > 0));
        assert!(worker.iteration.minibatch.iter().all(|item| !item.ooo_completed));
    }

    #[test]
    fn backup_refreshes_px0_root_best_edge_cache() {
        ensure_init();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let mut tree = NodeTree::default();
        tree.reset_to_position(&state.startpos, &state.moves);
        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let stop = Arc::new(AtomicBool::new(false));
        let search_state = WorkerSearchState::new(stop);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.run_until_root_visits(2).expect("two root visits");

        let root = worker.with_tree_for_test(|tree| tree.current_head());
        let edge = *search_state.current_best_edge.lock().expect("best edge lock");
        assert!(edge.is_some_and(|edge_idx| edge_idx < worker.with_tree_for_test(|tree| tree.node(root).num_edges())));
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
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);

        worker.execute_one_iteration().expect("collision iteration");

        assert_eq!(tree.node(root).n_in_flight(), 0);
        assert_eq!(state.shared_collisions.lock().expect("collisions lock").len(), 0);
    }

    /// px0 `SearchWorker::UpdateCounters` treats collision-only iterations as
    /// idle (`src/search/classic/search.cc:2337-2351`). They must not count
    /// as useful search work merely because the minibatch is non-empty.
    #[test]
    fn collision_only_iteration_is_not_work_done() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let search_state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let root = tree.current_head();
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.iteration.minibatch.push(NodeToProcess::collision(root, 1, 1, 1));

        let started = Instant::now();
        worker.update_counters().expect("collision-only update");
        assert!(started.elapsed() >= Duration::from_millis(8));
    }

    #[test]
    fn sticky_terminal_backup_sets_non_root_parent_bounds_like_px0() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let mv = tree.history().last().board().generate_legal_moves()[0];
        tree.node_mut(root).create_single_child_node(mv);
        let parent = tree.arena_mut().spawn_child(root, 0);
        tree.node_mut(parent).create_single_child_node(mv);
        let leaf = tree.arena_mut().spawn_child(parent, 0);
        tree.make_terminal(leaf, GameResult::WhiteWon, 0.0, Terminal::EndOfGame);

        // A child only exists after its parent was extended, so px0's
        // `MaybeSetBounds` always sees a previously visited parent.
        assert!(tree.node_mut(parent).try_start_score_update());
        tree.node_mut(parent).finalize_score_update(0.0, 0.0, 0.0, 1);

        for node_idx in [root, parent, leaf] {
            assert!(tree.node_mut(node_idx).try_start_score_update());
        }
        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.do_backup_update_single_node(&NodeToProcess::visit(leaf, 2));

        // px0 `MaybeSetBounds` does not terminalize root, but it turns this
        // non-root forced child into a sticky loss for its side to move.
        assert!(!worker.with_tree_for_test(|tree| tree.node(root).is_terminal()));
        assert!(worker.with_tree_for_test(|tree| tree.node(parent).is_terminal()));
        assert!((worker.with_tree_for_test(|tree| tree.node(parent).wl()) + 1.0).abs() < f32::EPSILON);
        assert_eq!(
            worker.with_tree_for_test(|tree| tree.node(parent).lower_bound()),
            GameResult::BlackWon
        );
        assert_eq!(
            worker.with_tree_for_test(|tree| tree.node(parent).upper_bound()),
            GameResult::BlackWon
        );
    }

    #[test]
    fn prefetch_resets_workspace_history_to_root() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let backend = UniformBackend::default();
        let params = SearchParams::default();
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.initialize_iteration().expect("init");

        let root_moves = worker.history.last().board().generate_legal_moves();
        worker.history.append(root_moves[0]);
        worker
            .iteration
            .computation
            .as_mut()
            .expect("computation")
            .add_input(EvalPosition {
                positions: worker.history.positions().to_vec(),
                legal_moves: worker.history.last().board().generate_legal_moves(),
            })
            .expect("seed computation");

        worker
            .with_tree_read(|worker, tree| worker.maybe_prefetch_into_cache(tree))
            .expect("prefetch");

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
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.initialize_iteration().expect("init");

        let spent = worker
            .with_tree_read(|worker, tree| worker.prefetch_into_cache(tree, Some(root), 1, false))
            .expect("recursive prefetch");

        assert_eq!(spent, 1);
        assert_eq!(worker.history.len(), worker.played_history_len);
        assert_eq!(
            worker
                .iteration
                .computation
                .as_ref()
                .expect("computation")
                .used_batch_size(),
            1
        );
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
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);

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
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.pick_nodes_to_extend(1).expect("pick nodes");

        let visit = worker
            .iteration
            .minibatch
            .iter()
            .find(|item| !item.is_collision)
            .expect("leaf visit");
        assert_eq!(visit.moves_to_visit, vec![first, second]);
        assert_eq!(visit.depth, 3);
    }

    /// px0 `PickNodesToExtendTask` only zeros the parent work after the
    /// gathering task is published (`search.cc:1828-1864`).
    #[test]
    fn gathering_handoff_relinquishes_parent_edge_only_after_enqueue() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let mv = tree.history().last().board().generate_legal_moves()[0];
        tree.node_mut(root).create_single_child_node(mv);
        let child = tree.arena_mut().spawn_child(root, 0);
        assert!(tree.node_mut(child).try_start_score_update());
        tree.node_mut(child).finalize_score_update(0.0, 0.0, 0.0, 1);

        let queue = PickTaskQueue::default();
        queue.reset();
        let mut workspace = TaskWorkspace::default();
        workspace.visits_to_perform.push(vec![3]);

        assert!(SearchWorker::hand_off_gathering_task(
            &queue,
            &mut tree,
            &mut workspace,
            root,
            0,
            1,
            3,
        ));
        assert_eq!(workspace.visits_to_perform.last().expect("visits")[0], 0);

        let claimed = queue.take().expect("published gathering task");
        assert_eq!(claimed.task.kind, PickTaskKind::Gathering);
        assert_eq!(claimed.task.start, Some(child));
        assert_eq!(claimed.task.base_depth, 1);
        assert_eq!(claimed.task.moves_to_base, vec![mv]);
        assert_eq!(claimed.task.collision_limit, 3);
        queue.complete(claimed);
    }

    /// px0 only hands independent descendants to concurrent gathering tasks
    /// (`search.cc:1828-1864`). Sibling roots may run together; a duplicate
    /// root or an ancestor/descendant pair must remain with the parent DFS.
    #[test]
    fn gathering_task_roots_are_disjoint_for_the_whole_phase() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        let moves = tree.history().last().board().generate_legal_moves();
        tree.node_mut(root).create_edges(&moves[..2].to_vec());
        let first_child = tree.arena_mut().spawn_child(root, 0);
        let second_child = tree.arena_mut().spawn_child(root, 1);
        tree.node_mut(first_child).create_single_child_node(moves[2]);
        let grandchild = tree.arena_mut().spawn_child(first_child, 0);

        let queue = PickTaskQueue::default();
        queue.reset();
        assert!(queue.push_gathering(&tree, PickTask::gathering(first_child, 1, vec![moves[0]], 8),));
        assert!(queue.push_gathering(&tree, PickTask::gathering(second_child, 1, vec![moves[1]], 8),));
        assert!(!queue.push_gathering(&tree, PickTask::gathering(first_child, 1, vec![moves[0]], 8),));
        assert!(!queue.push_gathering(&tree, PickTask::gathering(grandchild, 2, vec![moves[0], moves[2]], 4),));
        assert_eq!(queue.task_count.load(Ordering::Acquire), 2);
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
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.task_phase.queue.reset();
        assert!(worker
            .task_phase
            .queue
            .push(PickTask::gathering(root, 0, Vec::new(), 1)));
        let mut workspace = TaskWorkspace::default();

        worker.run_queued_tasks(&mut workspace).expect("run gathering task");
        worker
            .task_phase
            .queue
            .drain_results_into(&mut worker.iteration.minibatch);

        assert_eq!(worker.iteration.minibatch.len(), 1);
        assert!(!worker.iteration.minibatch[0].is_collision);
        assert_eq!(worker.iteration.minibatch[0].moves_to_visit, vec![mv]);
    }

    /// px0 executes a processing task against the task worker's own
    /// workspace, never the main worker's scratch (`search.cc:1125-1129`).
    #[test]
    fn task_runner_processing_owns_its_workspace() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        tree.make_terminal(root, GameResult::Draw, 0.0, Terminal::EndOfGame);

        let params = SearchParams::default();
        let context = ProcessingContext {
            params: &params,
            computation: None,
            extend: ExtendContext {
                played_history_len: tree.history().len(),
                two_fold_draws: params.two_fold_draws,
            },
        };
        let mut runner = TaskRunner::default();
        let mut range = vec![NodeToProcess::visit(root, 1)];

        runner
            .run_processing_range(&context, &mut tree, &mut range)
            .expect("terminal processing task");

        assert_eq!(runner.workspace.history.len(), tree.history().len());
        assert!(!range[0].nn_queried);
    }

    #[test]
    fn pick_task_queue_wakes_and_closes_like_px0_run_tasks() {
        let queue = Arc::new(PickTaskQueue::default());
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let worker_queue = Arc::clone(&queue);
        let worker = std::thread::spawn(move || {
            ready_tx.send(()).expect("ready");
            worker_queue.take_blocking()
        });
        ready_rx.recv().expect("worker ready");
        assert!(queue.push(PickTask::processing(3, 5)));
        let claimed = worker.join().expect("task worker").expect("queued task");
        assert_eq!(claimed.id, 0);
        assert_eq!(claimed.task.kind, PickTaskKind::Processing);
        assert_eq!((claimed.task.start_idx, claimed.task.end_idx), (3, 5));

        let closing_queue = Arc::new(PickTaskQueue::default());
        let waiter_queue = Arc::clone(&closing_queue);
        let waiter = std::thread::spawn(move || waiter_queue.take_blocking());
        closing_queue.close();
        assert!(waiter.join().expect("closing worker").is_none());
    }

    /// px0 task threads stay available while the main gather walk publishes
    /// work, then leave the scoped phase once publication is sealed
    /// (`search.cc:1069-1124,1485-1508`).
    #[test]
    fn pick_task_queue_waits_for_phase_publish_then_seal() {
        let queue = Arc::new(PickTaskQueue::default());
        queue.reset();
        let worker_queue = Arc::clone(&queue);
        let worker = std::thread::spawn(move || {
            let mut completed = 0;
            while let Some(task) = worker_queue.take_until_phase_sealed() {
                worker_queue.complete(task);
                completed += 1;
            }
            completed
        });

        queue.push(PickTask::processing(0, 1));
        queue.seal_phase();

        assert_eq!(worker.join().expect("scoped task worker"), 1);
        queue.wait();
    }

    /// px0 serializes the `tasks_taken_` increment with
    /// `task_taking_started_`, so parallel runners cannot claim one task twice
    /// (`src/search/classic/search.cc:1076-1093`).
    #[test]
    fn pick_task_queue_claims_each_published_task_once() {
        let queue = Arc::new(PickTaskQueue::default());
        queue.reset();
        for index in 0..32 {
            assert!(queue.push(PickTask::processing(index, index + 1)));
        }

        let claimed = Arc::new(Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let queue = Arc::clone(&queue);
                let claimed = Arc::clone(&claimed);
                scope.spawn(move || {
                    while let Some(task) = queue.take() {
                        claimed.lock().expect("claimed task lock").push(task.id);
                    }
                });
            }
        });

        let mut claimed = claimed.lock().expect("claimed task lock").clone();
        claimed.sort_unstable();
        assert_eq!(claimed, (0..32).collect::<Vec<_>>());
    }

    /// A claimed gathering task keeps ownership of its result vector until it
    /// is explicitly completed, matching px0's `PickTask::results` handoff
    /// (`src/search/classic/search.cc:1124-1137,1501-1507`).
    #[test]
    fn pick_task_queue_returns_owned_gather_results_once() {
        let queue = PickTaskQueue::default();
        queue.reset();
        assert!(queue.push(PickTask::gathering(7, 2, Vec::new(), 1)));

        let mut claimed = queue.take().expect("claimed gathering task");
        claimed.task.results.push(NodeToProcess::visit(11, 3));
        queue.complete(claimed);

        let mut results = Vec::new();
        queue.drain_results_into(&mut results);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_idx, 11);
        assert_eq!(results[0].depth, 3);
    }

    /// px0 splits processing work into contiguous, non-overlapping ranges and
    /// leaves the final suffix for the main worker (`search.cc:1322-1362`).
    #[test]
    fn processing_task_ranges_are_disjoint_from_main_suffix() {
        let queue = PickTaskQueue::default();
        let params = SearchParams {
            minimum_work_size_for_processing: 2,
            minimum_work_per_task_for_processing: 2,
            ..SearchParams::default()
        };
        let minibatch = vec![
            NodeToProcess::visit(0, 1),
            NodeToProcess::collision(1, 1, 1, 1),
            NodeToProcess::visit(2, 1),
            NodeToProcess::visit(3, 1),
            NodeToProcess::collision(4, 1, 1, 1),
            NodeToProcess::visit(5, 1),
            NodeToProcess::visit(6, 1),
            NodeToProcess::visit(7, 1),
        ];

        let (main_start, needs_wait) = SearchWorker::split_processing_tasks(&queue, &params, 2, &minibatch, 0, 6);
        assert!(needs_wait);

        let mut ranges = Vec::new();
        while let Some(claimed) = queue.take() {
            ranges.push((claimed.task.start_idx, claimed.task.end_idx));
            queue.complete(claimed);
        }
        assert_eq!(ranges, vec![(0, 3), (3, 6)]);
        assert!(ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        assert!(ranges.last().is_some_and(|range| range.1 <= main_start));
        assert_eq!(main_start, 6);
    }

    /// px0 sets `task_count_ = -1` after one gather without destroying its
    /// persistent task workers; the next `ResetTasks` reuses the same worker
    /// (`src/search/classic/search.cc:1182-1185,1464-1492`).
    #[test]
    fn pick_task_queue_idles_then_reuses_worker_next_iteration() {
        let queue = Arc::new(PickTaskQueue::default());
        let worker_queue = Arc::clone(&queue);
        let (first_tx, first_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let first = worker_queue.take_blocking()?.id;
            first_tx.send(()).expect("first task claimed");
            let second = worker_queue.take_blocking()?.id;
            Some((first, second))
        });

        assert!(queue.push(PickTask::processing(0, 1)));
        first_rx.recv().expect("first task");
        queue.idle();
        queue.reset();
        assert!(queue.push(PickTask::processing(1, 2)));

        assert_eq!(worker.join().expect("persistent task worker"), Some((0, 0)));
    }

    #[test]
    fn gathering_processes_full_batch_without_task_workers() {
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
            out_of_order_eval: false,
            ..SearchParams::default()
        };
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.initialize_iteration().expect("init");
        worker.gather_minibatch().expect("gather full batch");

        assert!(worker.iteration.minibatch.len() >= 2);
        assert!(
            worker
                .iteration
                .computation
                .as_ref()
                .expect("computation")
                .used_batch_size()
                >= 2
        );
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
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);
        worker.ensure_node_twofold_correct_for_depth(child, 1);

        assert!(!worker.with_tree_for_test(|tree| tree.node(child).is_terminal()));
        assert_eq!(worker.with_tree_for_test(|tree| tree.node(child).n()), 0);
        assert_eq!(worker.with_tree_for_test(|tree| tree.node(root).n()), 0);
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
        let state = WorkerSearchState::new(Arc::new(AtomicBool::new(false)));
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &state);

        worker
            .extend_node(child, 5, &moves, &mut history)
            .expect("extend twofold leaf");

        assert!(worker.with_tree_for_test(|tree| tree.node(child).is_twofold_terminal()));
        assert!((worker.with_tree_for_test(|tree| tree.node(child).m()) - 4.0).abs() < f32::EPSILON);
    }
}
