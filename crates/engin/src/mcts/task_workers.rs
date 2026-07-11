//! lc0 task workers：gather 内 picking / processing 拆分（search.cc:1091-1161,1507-1573）。

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use super::backend::SharedBackendComputation;
use super::node::MctsNodeId;
use super::tree::MctsTree;
use super::worker::{
    is_collision_kind, pick_nodes_to_extend_task, process_picked_range_shared, PathStep, PendingNode,
    PickWorkspace, SearchIteration, SelectionScratch,
};
use super::{MctsBudget, MctsConfig, PolicyValueEval, SearchStats};
use crate::history::PositionHistory;
use crate::policy_onnx::BackendAttributes;

/// 并行 search worker 上的 task worker 上下文。
pub(crate) struct TaskWorkerGatherCtx<E: PolicyValueEval + Send + Sync + 'static> {
    pub pool: SearchWorkerTaskPool<E>,
    pub backend_shared: Arc<SharedBackendComputation<E>>,
}

impl<E: PolicyValueEval + Send + Sync + 'static> TaskWorkerGatherCtx<E> {
    pub fn new(
        task_workers: usize,
        backend_shared: Arc<SharedBackendComputation<E>>,
    ) -> Self {
        Self {
            pool: SearchWorkerTaskPool::new(task_workers),
            backend_shared,
        }
    }
}

/// lc0 `TaskWorkspace`（search.h:354-371）。
pub(crate) struct TaskWorkspace {
    pub pick: PickWorkspace,
    pub scratch: SelectionScratch,
}

impl Default for TaskWorkspace {
    fn default() -> Self {
        Self {
            pick: PickWorkspace::default(),
            scratch: SelectionScratch::default(),
        }
    }
}

/// lc0 `PickTask::kGathering` payload（search.h:377-396）。
#[derive(Clone)]
pub(crate) struct GatheringPickTask {
    pub start: MctsNodeId,
    pub base_depth: usize,
    pub collision_limit: i32,
    pub moves_to_base: Vec<PathStep>,
    pub results: Vec<PendingNode>,
}

/// lc0 `PickTask::kProcessing` 索引区间。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProcessingTaskRange {
    pub start: usize,
    pub end: usize,
}

/// lc0 `PickTask`（search.h:373-398）。
enum PickTask {
    Gathering(GatheringPickTask),
    Processing {
        start_idx: usize,
        end_idx: usize,
    },
}

#[derive(Clone)]
pub(crate) struct PickingDispatch {
    pub tree: Arc<Mutex<MctsTree>>,
    pub root_id: MctsNodeId,
    pub root_history: PositionHistory,
    pub config: MctsConfig,
    pub budget: MctsBudget,
    pub session_playouts: u32,
    pub stats: Option<Arc<SearchStats>>,
}

/// gather processing 阶段共享上下文。
pub(crate) struct ProcessingDispatch<E: PolicyValueEval + Send + Sync + 'static> {
    pub tree: Arc<Mutex<MctsTree>>,
    pub iteration: Arc<Mutex<SearchIteration>>,
    pub root_history: PositionHistory,
    pub config: MctsConfig,
    pub root_id: MctsNodeId,
    pub backend: Arc<SharedBackendComputation<E>>,
}

pub(crate) struct TaskPoolInner<E: PolicyValueEval + Send + Sync + 'static> {
    picking_tasks: Mutex<Vec<PickTask>>,
    task_count: AtomicI32,
    tasks_taken: AtomicI32,
    completed_tasks: AtomicI32,
    task_taking_started: AtomicI32,
    task_added: (Mutex<()>, Condvar),
    exiting: AtomicBool,
    picking_dispatch: Mutex<Option<PickingDispatch>>,
    processing_dispatch: Mutex<Option<ProcessingDispatch<E>>>,
    workspaces: Arc<Vec<Mutex<TaskWorkspace>>>,
}

/// lc0 SearchWorker 级 task 线程池（search.h:439-451；search.cc:1091-1161,1488-1505）。
pub(crate) struct SearchWorkerTaskPool<E: PolicyValueEval + Send + Sync + 'static> {
    pub(crate) inner: Arc<TaskPoolInner<E>>,
    threads: Vec<JoinHandle<()>>,
}

const MAX_TASKS: usize = 100;

impl<E: PolicyValueEval + Send + Sync + 'static> SearchWorkerTaskPool<E> {
    /// lc0 SearchWorker ctor task_threads_（search.h:227-229）。
    pub fn new(task_workers: usize) -> Self {
        let workspaces: Arc<Vec<Mutex<TaskWorkspace>>> = Arc::new(
            (0..task_workers)
                .map(|_| Mutex::new(TaskWorkspace::default()))
                .collect(),
        );
        let inner = Arc::new(TaskPoolInner {
            picking_tasks: Mutex::new(Vec::with_capacity(MAX_TASKS)),
            task_count: AtomicI32::new(-1),
            tasks_taken: AtomicI32::new(0),
            completed_tasks: AtomicI32::new(0),
            task_taking_started: AtomicI32::new(0),
            task_added: (Mutex::new(()), Condvar::new()),
            exiting: AtomicBool::new(false),
            picking_dispatch: Mutex::new(None),
            processing_dispatch: Mutex::new(None),
            workspaces: Arc::clone(&workspaces),
        });
        let mut threads = Vec::with_capacity(task_workers);
        for tid in 0..task_workers {
            let inner = Arc::clone(&inner);
            threads.push(thread::spawn(move || run_tasks_loop::<E>(inner, tid)));
        }
        Self { inner, threads }
    }

    /// lc0 `ResetTasks`（search.cc:1488-1495）。
    pub fn reset_tasks(&self) {
        self.inner.task_count.store(0, Ordering::Release);
        self.inner.tasks_taken.store(0, Ordering::Release);
        self.inner.completed_tasks.store(0, Ordering::Release);
        let mut tasks = self
            .inner
            .picking_tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        tasks.clear();
        tasks.reserve(MAX_TASKS);
    }

    pub fn set_picking_dispatch(&self, dispatch: PickingDispatch) {
        *self
            .inner
            .picking_dispatch
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(dispatch);
    }

    pub fn clear_picking_dispatch(&self) {
        *self
            .inner
            .picking_dispatch
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
    }

    pub fn set_processing_dispatch(&self, dispatch: ProcessingDispatch<E>) {
        *self
            .inner
            .processing_dispatch
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(dispatch);
    }

    pub fn clear_processing_dispatch(&self) {
        *self
            .inner
            .processing_dispatch
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// lc0 processing split 入队（search.cc:1370-1371）。
    pub fn enqueue_processing(&self, start_idx: usize, end_idx: usize) {
        self.inner
            .picking_tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(PickTask::Processing {
                start_idx,
                end_idx,
            });
        self.inner
            .task_count
            .fetch_add(1, Ordering::AcqRel);
    }

    /// lc0 PickNodesToExtend 前唤醒（search.cc:1509-1513）。
    pub fn wake_workers(&self) {
        let _guard = self
            .inner
            .task_added
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.inner.task_added.1.notify_all();
    }

    /// lc0 主线程 `ProcessPickedTask` / `PickNodesToExtend` 后 `WaitForTasks`（search.cc:1382-1383,1523,1497-1505）。
    pub fn wait_for_tasks(&self) {
        loop {
            let completed = self.inner.completed_tasks.load(Ordering::Acquire);
            let todo = self.inner.task_count.load(Ordering::Acquire);
            if todo == completed {
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// lc0 `PickNodesToExtend` 汇总 worker results（search.cc:1524-1528）。
    pub fn merge_gathering_results(&self, receiver: &mut Vec<PendingNode>) {
        let tasks = self
            .inner
            .picking_tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for task in tasks.iter() {
            if let PickTask::Gathering(gather) = task {
                receiver.extend(gather.results.iter().cloned());
            }
        }
    }
}

/// lc0 picking task 入队接口（search.cc:1867-1877）。
pub(crate) trait PickTaskEnqueue {
    fn try_enqueue_gathering(
        &self,
        start: MctsNodeId,
        base_depth: usize,
        moves_to_base: Vec<PathStep>,
        collision_limit: i32,
    ) -> bool;
}

impl<E: PolicyValueEval + Send + Sync + 'static> PickTaskEnqueue for TaskPoolInner<E> {
    fn try_enqueue_gathering(
        &self,
        start: MctsNodeId,
        base_depth: usize,
        moves_to_base: Vec<PathStep>,
        collision_limit: i32,
    ) -> bool {
        TaskPoolInner::try_enqueue_gathering(self, start, base_depth, moves_to_base, collision_limit)
    }
}

impl<E: PolicyValueEval + Send + Sync + 'static> TaskPoolInner<E> {
    pub(crate) fn try_enqueue_gathering(
        &self,
        start: MctsNodeId,
        base_depth: usize,
        moves_to_base: Vec<PathStep>,
        collision_limit: i32,
    ) -> bool {
        let mut tasks = self
            .picking_tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if tasks.len() >= MAX_TASKS {
            return false;
        }
        tasks.push(PickTask::Gathering(GatheringPickTask {
            start,
            base_depth,
            collision_limit,
            moves_to_base,
            results: Vec::new(),
        }));
        self.task_count.fetch_add(1, Ordering::AcqRel);
        let _guard = self
            .task_added
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.task_added.1.notify_all();
        true
    }
}

impl<E: PolicyValueEval + Send + Sync + 'static> Drop for SearchWorkerTaskPool<E> {
    fn drop(&mut self) {
        self.inner.task_count.store(-1, Ordering::Release);
        {
            let _guard = self
                .inner
                .task_added
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            self.inner.exiting.store(true, Ordering::Release);
            self.inner.task_added.1.notify_all();
        }
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

/// lc0 `SearchWorker::RunTasks`（search.cc:1091-1161）。
fn run_tasks_loop<E: PolicyValueEval + Send + Sync + 'static>(
    inner: Arc<TaskPoolInner<E>>,
    tid: usize,
) {
    loop {
        let mut spins = 0i32;
        let task_id = loop {
            let nta = inner.tasks_taken.load(Ordering::Acquire);
            let tc = inner.task_count.load(Ordering::Acquire);
            if nta < tc {
                if inner
                    .task_taking_started
                    .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    let nta = inner.tasks_taken.load(Ordering::Acquire);
                    let tc = inner.task_count.load(Ordering::Acquire);
                    if nta < tc {
                        let id = inner.tasks_taken.fetch_add(1, Ordering::AcqRel) as usize;
                        inner.task_taking_started.store(0, Ordering::Release);
                        break Some(id);
                    }
                    inner.task_taking_started.store(0, Ordering::Release);
                }
                std::hint::spin_loop();
                spins = 0;
                continue;
            } else if tc != -1 {
                spins += 1;
                if spins >= 512 {
                    thread::yield_now();
                    spins = 0;
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }
            spins = 0;
            let _guard = inner.task_added.0.lock().unwrap_or_else(|e| e.into_inner());
            let nta = inner.tasks_taken.load(Ordering::Acquire);
            let tc = inner.task_count.load(Ordering::Acquire);
            if tc != -1 {
                continue;
            }
            if nta >= tc && inner.exiting.load(Ordering::Acquire) {
                return;
            }
            inner.task_added.1.wait(_guard).ok();
            let nta = inner.tasks_taken.load(Ordering::Acquire);
            let tc = inner.task_count.load(Ordering::Acquire);
            if nta >= tc && inner.exiting.load(Ordering::Acquire) {
                return;
            }
        };

        let Some(task_id) = task_id else {
            continue;
        };

        let task_kind = {
            inner
                .picking_tasks
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(task_id)
                .map(|task| match task {
                    PickTask::Gathering(g) => TaskKind::Gathering {
                        start: g.start,
                        base_depth: g.base_depth,
                        collision_limit: g.collision_limit,
                        moves_to_base: g.moves_to_base.clone(),
                    },
                    PickTask::Processing {
                        start_idx,
                        end_idx,
                    } => TaskKind::Processing {
                        start_idx: *start_idx,
                        end_idx: *end_idx,
                    },
                })
        };

        let Some(task_kind) = task_kind else {
            inner
                .completed_tasks
                .fetch_add(1, Ordering::AcqRel);
            continue;
        };

        match task_kind {
            TaskKind::Gathering {
                start,
                base_depth,
                collision_limit,
                moves_to_base,
            } => {
                let picking = inner
                    .picking_dispatch
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                if let Some(dispatch) = picking {
                    let mut workspace = inner.workspaces[tid]
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let TaskWorkspace {
                        scratch: ref mut task_scratch,
                        pick: ref mut task_pick,
                    } = &mut *workspace;
                    let mut results = Vec::new();
                    let mut tree = dispatch.tree.lock().unwrap_or_else(|e| e.into_inner());
                    pick_nodes_to_extend_task(
                        &mut tree,
                        dispatch.root_id,
                        &dispatch.root_history,
                        start,
                        base_depth,
                        &moves_to_base,
                        dispatch.config,
                        dispatch.stats.as_deref(),
                        &dispatch.budget,
                        dispatch.session_playouts,
                        collision_limit,
                        &mut results,
                        task_scratch,
                        task_pick,
                        Some(inner.as_ref() as &dyn PickTaskEnqueue),
                        collision_limit,
                        &mut 0i32,
                    );
                    drop(tree);
                    let mut tasks = inner
                        .picking_tasks
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if let Some(PickTask::Gathering(gather)) = tasks.get_mut(task_id) {
                        gather.results = results;
                    }
                }
            }
            TaskKind::Processing {
                start_idx,
                end_idx,
            } => {
                if let Some(dispatch) = inner
                    .processing_dispatch
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                {
                    process_picked_range_shared(dispatch, start_idx, end_idx);
                }
            }
        }

        inner
            .completed_tasks
            .fetch_add(1, Ordering::AcqRel);
    }
}

enum TaskKind {
    Gathering {
        start: MctsNodeId,
        base_depth: usize,
        collision_limit: i32,
        moves_to_base: Vec<PathStep>,
    },
    Processing {
        start_idx: usize,
        end_idx: usize,
    },
}

/// lc0 `TaskWorkersPerSearchWorker`（params.cc:622；search.h:216-226）。
pub(crate) fn resolve_task_workers(
    config: MctsConfig,
    search_threads: usize,
    attrs: Option<&BackendAttributes>,
) -> usize {
    let configured = config.task_workers;
    if configured == 0 {
        return 0;
    }
    if configured > 0 {
        return configured as usize;
    }
    // configured == -1: auto
    let runs_on_cpu = attrs.map(|a| a.runs_on_cpu).unwrap_or(true);
    if runs_on_cpu {
        return 0;
    }
    let working_threads = search_threads.saturating_sub(1).max(1);
    let hw = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    (hw / working_threads).saturating_sub(1).min(4)
}

/// lc0 `GatherMinibatch` processing split（search.cc:1353-1378）。
pub(crate) fn plan_processing_task_ranges(
    new_start: usize,
    pending_len: usize,
    pending: &[PendingNode],
    config: MctsConfig,
    task_workers: usize,
) -> (Vec<ProcessingTaskRange>, usize) {
    if task_workers == 0 {
        return (Vec::new(), new_start);
    }

    let non_collisions = pending[new_start..pending_len]
        .iter()
        .filter(|p| !is_collision_kind(&p.kind))
        .count();
    if non_collisions < config.minimum_work_size_for_processing as usize {
        return (Vec::new(), new_start);
    }

    let num_tasks = (non_collisions / config.minimum_work_per_task_for_processing as usize)
        .clamp(2, task_workers + 1);
    let per_worker = non_collisions / num_tasks;

    let mut worker_ranges = Vec::new();
    let mut ppt_start = new_start;
    let mut found = 0usize;

    for i in new_start..pending_len {
        if is_collision_kind(&pending[i].kind) {
            continue;
        }
        found += 1;
        if found == per_worker {
            worker_ranges.push(ProcessingTaskRange {
                start: ppt_start,
                end: i + 1,
            });
            ppt_start = i + 1;
            found = 0;
            if worker_ranges.len() == num_tasks - 1 {
                break;
            }
        }
    }

    (worker_ranges, ppt_start)
}

#[cfg(test)]
mod tests {
    use super::{plan_processing_task_ranges, resolve_task_workers, SearchWorkerTaskPool};
    use crate::mcts::node::MctsNodeId;
    use crate::mcts::worker::{PendingKey, PendingKind, PendingNode};
    use crate::mcts::MctsConfig;
    use crate::policy_onnx::BackendAttributes;

    fn pending_collision() -> PendingNode {
        PendingNode {
            key: PendingKey::ExistingLeaf(MctsNodeId(0)),
            kind: PendingKind::Collision { max_count: 1 },
            path: Vec::new(),
            multivisit: 1,
            collision_upsize: 0,
            nn_queried: false,
            is_cache_hit: false,
            ooo_completed: false,
            eval: None,
            task: None,
        }
    }

    fn pending_expand() -> PendingNode {
        PendingNode {
            key: PendingKey::ExistingLeaf(MctsNodeId(0)),
            kind: PendingKind::ExpandInPlace {
                node_id: MctsNodeId(0),
            },
            path: Vec::new(),
            multivisit: 1,
            collision_upsize: 0,
            nn_queried: false,
            is_cache_hit: false,
            ooo_completed: false,
            eval: None,
            task: None,
        }
    }

    #[test]
    fn resolve_task_workers_zero_on_cpu_auto() {
        let config = MctsConfig {
            task_workers: -1,
            ..MctsConfig::default()
        };
        let attrs = BackendAttributes {
            runs_on_cpu: true,
            suggested_num_search_threads: 1,
            recommended_batch_size: 256,
            maximum_batch_size: 1024,
        };
        assert_eq!(resolve_task_workers(config, 2, Some(&attrs)), 0);
    }

    #[test]
    fn resolve_task_workers_positive_respects_config() {
        let config = MctsConfig {
            task_workers: 3,
            ..MctsConfig::default()
        };
        assert_eq!(resolve_task_workers(config, 2, None), 3);
    }

    #[test]
    fn plan_processing_splits_into_worker_and_main_ranges() {
        let config = MctsConfig::default();
        let mut pending = vec![pending_collision()];
        pending.extend(std::iter::repeat_n(pending_expand(), 24));
        let new_start = 1;
        let (ranges, main_start) =
            plan_processing_task_ranges(new_start, pending.len(), &pending, config, 2);
        assert!(!ranges.is_empty());
        assert!(main_start > new_start);
        assert!(main_start <= pending.len());
    }

    #[test]
    fn task_pool_reset_and_wait_with_no_tasks() {
        let pool = SearchWorkerTaskPool::<crate::mcts::OnnxPolicyValueEval>::new(1);
        pool.reset_tasks();
        pool.wait_for_tasks();
    }
}
