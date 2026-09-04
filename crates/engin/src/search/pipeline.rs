//! Shared / Stats + Gather 流程组装 + `Search` API。
//!
//! 树走组装（claim / collision / 选边下降）在本文件；选边公式在 `select`，
//! worker 线程循环在 `workerpool`，发边 / 回传在 `eval` / `backprop`。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{SendTimeoutError, TrySendError, bounded};
use parking_lot::{Condvar, Mutex};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;
use crate::neural::MOVE_HISTORY;
use crate::neural::backend::Backend;

use super::eval::cancel_evaluation;
use super::expand::path_terminal_value;
use super::observer::{NoopObserver, SearchObserver};
use super::param::{SearchConfig, SearchParams};
use super::select::select_edge;
use super::workerpool::{BackpropEvent, Event, GatherEvent, NnRequest, WorkerPool};
use super::{EdgeReservation, ExpansionState, Node, NodeArena, NodeId, SearchTree};

pub(crate) const RECEIVE_POLL: Duration = Duration::from_millis(10);

const SHORT_CLOCK: Duration = Duration::from_millis(500);
const SHORT_CLOCK_IN_FLIGHT_CAP: usize = 8;

// --- Stats / Shared ----------------------------------------------------------

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    pub completed_playouts: u64,
    pub average_depth: u64,
    pub max_depth: u64,
    /// 实际走 NN 的叶子数（不含 cache hit）；UCI `eps` 用。
    pub network_evaluations: u64,
}

pub(crate) struct Shared<O: SearchObserver = NoopObserver> {
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) arena: Arc<NodeArena>,
    pub(crate) params: SearchParams,
    pub(crate) root_move_filter: Mutex<Vec<Move>>,
    pub(crate) stopping: AtomicBool,
    /// 未 `finish` 的 owned event：drain / 节点预算。
    pub(crate) outstanding: AtomicUsize,
    /// 已交给 Eval 的叶子（编码或 NN）。满窗口就不再 claim 新叶子。
    pub(crate) nn_inflight: AtomicUsize,
    pub(crate) completed: AtomicU64,
    pub(crate) completed_depth: AtomicU64,
    pub(crate) max_depth: AtomicU64,
    pub(crate) network_evaluations: AtomicU64,
    pub(crate) observer: O,
    pub(crate) error: Mutex<Option<EnginError>>,
    pub(crate) idle_lock: Mutex<()>,
    pub(crate) idle: Condvar,
    pub(crate) gather_tx: crossbeam_channel::Sender<GatherEvent<O::Stamp>>,
    pub(crate) eval_tx: crossbeam_channel::Sender<GatherEvent<O::Stamp>>,
    pub(crate) backprop_tx: crossbeam_channel::Sender<BackpropEvent<O::Stamp>>,
    /// 撞上 `Evaluating` 叶子的 playout。先留着 reservation / μ；该叶子自己的
    /// backprop `complete` 之后再按 `node_id` 摘出来 cancel。
    pub(crate) collision_waiters: Mutex<Vec<Event>>,
}

impl<O: SearchObserver> Shared<O> {
    pub(crate) fn start_playout(&self) {
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        if O::ENABLED {
            self.observer.on_submitted();
        }
    }

    pub(crate) fn finish(&self, n: usize, completed: bool) {
        if n == 0 {
            return;
        }
        if completed {
            self.completed.fetch_add(n as u64, Ordering::AcqRel);
        }
        let previous = self.outstanding.fetch_sub(n, Ordering::AcqRel);
        debug_assert!(previous >= n, "stream outstanding task underflow");
        // 唤醒等待本轮真实 leaf 完成的 owner。
        let _guard = self.idle_lock.lock();
        self.idle.notify_all();
        let _ = previous;
    }

    pub(crate) fn release_eval_claims(&self, count: usize) {
        if count == 0 {
            return;
        }
        let previous = self.nn_inflight.fetch_sub(count, Ordering::AcqRel);
        debug_assert!(previous >= count, "stream eval claim underflow");
        let _guard = self.idle_lock.lock();
        self.idle.notify_all();
    }

    pub(crate) fn wait_while(
        &self,
        deadline: Option<Instant>,
        stop_on_stopping: bool,
        mut busy: impl FnMut(&Self) -> bool,
    ) {
        let mut guard = self.idle_lock.lock();
        while busy(self) && !(stop_on_stopping && self.stopping.load(Ordering::Acquire)) && self.error.lock().is_none()
        {
            let Some(deadline) = deadline else {
                self.idle.wait(&mut guard);
                continue;
            };
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let wait = deadline.saturating_duration_since(now).min(RECEIVE_POLL);
            if wait.is_zero() {
                break;
            }
            self.idle.wait_for(&mut guard, wait);
        }
    }

    /// 撞上正在评估的叶子：挂起整条 reservation，让 μ 继续分流。
    pub(crate) fn park_collision(&self, event: GatherEvent<O::Stamp>) {
        if O::ENABLED {
            self.observer.on_collision(event.variation.moves().len());
        }
        let event = event.into_event();
        {
            let mut waiters = self.collision_waiters.lock();
            if !self.stopping.load(Ordering::Acquire)
                && self
                    .arena
                    .get(event.node_id)
                    .is_some_and(|node| node.expansion_state() == ExpansionState::Evaluating)
            {
                waiters.push(event);
                return;
            }
        }
        event.cancel();
        self.finish(1, false);
    }

    pub(crate) fn cancel_collisions(&self, id: NodeId) {
        let mut waiters = self.collision_waiters.lock();
        let mut i = 0;
        let mut parked = Vec::new();
        while i < waiters.len() {
            if waiters[i].node_id == id {
                parked.push(waiters.swap_remove(i));
            } else {
                i += 1;
            }
        }
        drop(waiters);
        let n = parked.len();
        for event in parked {
            event.cancel();
        }
        self.finish(n, false);
    }

    pub(crate) fn cancel_all_collisions(&self) {
        let parked = std::mem::take(&mut *self.collision_waiters.lock());
        let n = parked.len();
        for event in parked {
            event.cancel();
        }
        self.finish(n, false);
    }

    pub(crate) fn fail(&self, error: EnginError) {
        let mut current = self.error.lock();
        if current.is_none() {
            *current = Some(error);
        }
        self.stopping.store(true, Ordering::Release);
        self.idle.notify_all();
    }

    /// 先增加 `nn_inflight` 再入队：图复用后 cache hit 极快，若 `try_send` 成功后才 +1，
    /// Eval 可能已经 `release_eval_claims`，第二手 `go` 就会 underflow 闪退。
    pub(crate) fn send_eval(&self, mut event: GatherEvent<O::Stamp>) {
        event.mark_queued();
        let inflight = self.nn_inflight.fetch_add(1, Ordering::AcqRel) + 1;
        if O::ENABLED {
            self.observer.on_peak_inflight(inflight);
        }
        loop {
            if self.stopping.load(Ordering::Acquire) {
                cancel_evaluation(self, event.into_event());
                return;
            }
            match self.eval_tx.try_send(event) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    event = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    cancel_evaluation(self, returned.into_event());
                    return;
                }
            }
        }
    }

    pub(crate) fn send_backprop(&self, mut event: BackpropEvent<O::Stamp>) {
        event.mark_queued();
        loop {
            if self.stopping.load(Ordering::Acquire) {
                self.release_eval_claims(usize::from(event.held_eval_claim));
                let id = event.event.node_id;
                event.cancel();
                self.finish(1, false);
                self.cancel_collisions(id);
                return;
            }
            match self.backprop_tx.try_send(event) {
                Ok(()) => return,
                Err(TrySendError::Full(returned)) => {
                    event = returned;
                    thread::yield_now();
                }
                Err(TrySendError::Disconnected(returned)) => {
                    self.release_eval_claims(usize::from(returned.held_eval_claim));
                    let id = returned.event.node_id;
                    returned.cancel();
                    self.finish(1, false);
                    self.cancel_collisions(id);
                    return;
                }
            }
        }
    }

    pub(crate) fn stats(&self) -> Stats {
        let completed_playouts = self.completed.load(Ordering::Acquire);
        Stats {
            completed_playouts,
            average_depth: self.completed_depth.load(Ordering::Acquire) / completed_playouts.max(1),
            max_depth: self.max_depth.load(Ordering::Acquire),
            network_evaluations: self.network_evaluations.load(Ordering::Acquire),
        }
    }
}

// --- Gather 组装 -------------------------------------------------------------

fn branch_at_expanded_node<O: SearchObserver>(
    shared: &Shared<O>,
    node: &Node,
    depth: usize,
) -> Option<(NodeId, EdgeReservation)> {
    let (edge_index, virtual_mean) = select_edge(
        &node.edges(),
        node.completed_visits(),
        node.q(),
        depth,
        &shared.params,
        &shared.root_move_filter.lock(),
        shared.arena.as_ref(),
    )?;
    let edge = &node.edges()[edge_index];
    let child = shared.arena.child_or_create(edge);
    let reservation = node
        .reserve_edge_with_virtual_mean(edge_index, virtual_mean)
        .expect("selected stream edge");
    Some((child, reservation))
}

pub(crate) fn process_gather_event<O: SearchObserver>(
    shared: &Shared<O>,
    mut event: GatherEvent<O::Stamp>,
    eval_claim_limit: usize,
) {
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            event.cancel();
            shared.finish(1, false);
            return;
        }
        let node = shared
            .arena
            .get(event.event.node_id)
            .expect("event node lives until job drain");
        match node.expansion_state() {
            ExpansionState::Unexpanded => {
                if shared.nn_inflight.load(Ordering::Acquire) >= eval_claim_limit {
                    shared.wait_while(None, true, |shared| {
                        shared.nn_inflight.load(Ordering::Acquire) >= eval_claim_limit
                    });
                    continue;
                }
                if node.try_begin_evaluation() {
                    shared.send_eval(event);
                    return;
                }
            }
            ExpansionState::Evaluating => {
                shared.park_collision(event);
                return;
            }
            ExpansionState::Terminal => {
                let (wl, draw, plies_left) = node.terminal_value().unwrap_or((0.0, 1.0, 0.0));
                shared.send_backprop(BackpropEvent::from_gather(event.into_event(), wl, draw, plies_left));
                return;
            }
            ExpansionState::Expanded => {
                let depth = event.variation.moves().len();
                let Some((child, reservation)) = branch_at_expanded_node(shared, node, depth) else {
                    // 当前搜索范围内的 child 都已 terminal；root 因而已精确解完。
                    if event.node_path().len() == 1 {
                        shared.stopping.store(true, Ordering::Release);
                        shared.idle.notify_all();
                    }
                    thread::yield_now();
                    continue;
                };
                event = event.descend(child, reservation);
            }
        }
    }
}

// --- Search API --------------------------------------------------------------

/// 这一手 `go` 的停止条件与根着过滤。
///
/// `searchmoves` 挂在这里，不进 `SearchConfig`（拓扑）或 `SearchParams`（算法）。
#[derive(Clone, Debug, Default)]
pub struct SearchLimits {
    pub max_playouts: Option<u64>,
    pub deadline: Option<Instant>,
    /// UCI `go searchmoves`；空表示不限制。
    pub root_move_filter: Vec<Move>,
}

impl SearchLimits {
    fn is_exhausted(&self, completed: u64, target: u64, now: Instant) -> bool {
        completed >= target || self.deadline.is_some_and(|deadline| now >= deadline)
    }
}

/// 剩余思考时间不足 `SHORT_CLOCK` 时，不要按推荐 MiniBatch 填满窗口。
/// 否则第一刀 GPU 推理就会单独超过 `go movetime`。
fn deadline_in_flight_limit(deadline: Option<Instant>, window: usize, batch: usize, now: Instant) -> usize {
    let Some(deadline) = deadline else {
        return window;
    };
    if deadline.saturating_duration_since(now) >= SHORT_CLOCK {
        return window;
    }
    batch.clamp(1, SHORT_CLOCK_IN_FLIGHT_CAP).min(window)
}

/// 运行中搜索可克隆的 stop 句柄。
#[derive(Clone)]
pub struct SearchControl<O: SearchObserver = NoopObserver> {
    shared: Arc<Shared<O>>,
}

impl<O: SearchObserver> SearchControl<O> {
    pub fn request_stop(&self) {
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.idle.notify_all();
    }

    pub fn stats(&self) -> Stats {
        self.shared.stats()
    }
}

/// 连续流式搜索：Gather / Eval / NN / Backprop。
/// Eval：terminal | cache → Backprop；否则稀疏编码 → NN queue；NN expand + 合批推理后
/// 整批交回 → Eval 切行/softmax/edges → Backprop。
pub struct Search<O: SearchObserver = NoopObserver> {
    shared: Arc<Shared<O>>,
    root_history: Arc<PositionHistory>,
    root_id: NodeId,
    /// 启动本次 job 前 root 已有的 completed N。它计入 UCI `go nodes`，但不计入本次 NPS。
    initial_visits: u64,
    worker_pool: Arc<WorkerPool<O>>,
    workers_idle: bool,
}

impl Search<NoopObserver> {
    pub fn new(backend: Arc<dyn Backend>, root_history: Arc<PositionHistory>, config: SearchConfig) -> Self {
        let tree = SearchTree::new(root_history);
        Self::new_with_graph(backend, &tree, config)
    }

    /// 从保留树创建独立搜索；该搜索自己创建、销毁 worker pool。
    pub fn new_with_graph(backend: Arc<dyn Backend>, graph: &SearchTree, config: SearchConfig) -> Self {
        Self::new_with_graph_with_observer(backend, graph, config, NoopObserver)
    }
}

impl<O: SearchObserver> Search<O> {
    pub fn new_with_observer(
        backend: Arc<dyn Backend>,
        root_history: Arc<PositionHistory>,
        config: SearchConfig,
        observer: O,
    ) -> Self {
        let tree = SearchTree::new(root_history);
        Self::new_with_graph_with_observer(backend, &tree, config, observer)
    }

    pub fn new_with_graph_with_observer(
        backend: Arc<dyn Backend>,
        graph: &SearchTree,
        config: SearchConfig,
        observer: O,
    ) -> Self {
        let worker_pool = Arc::new(WorkerPool::new(backend.as_ref(), &config));
        Self::new_with_graph_in_pool(backend, graph, config, observer, worker_pool)
    }

    pub(crate) fn new_with_graph_in_pool(
        backend: Arc<dyn Backend>,
        graph: &SearchTree,
        config: SearchConfig,
        observer: O,
        worker_pool: Arc<WorkerPool<O>>,
    ) -> Self {
        config.validate();
        let resolved = config.resolve(backend.as_ref());
        worker_pool.assert_compatible(&resolved);
        let (gather_tx, gather_rx) = bounded(resolved.queue_capacity);
        let (eval_tx, eval_rx) = bounded(resolved.queue_capacity);
        let (backprop_tx, backprop_rx) = bounded(resolved.queue_capacity);
        // UCI/graph 持有完整 history 用于跨回合定位；每个 event 只需要重复规则自
        // 最近零化着以来的后缀，以及 NN 的最近 8 层。这里一次裁剪后由整次 job 共享。
        let root_history = Arc::new(graph.root_history().search_window(MOVE_HISTORY));
        let root_id = graph.root_id();
        let initial_visits = graph
            .arena()
            .get(root_id)
            .map_or(0, |root| root.completed_visits() as u64);
        let shared = Arc::new(Shared {
            backend,
            arena: Arc::clone(graph.arena()),
            params: resolved.params,
            root_move_filter: Mutex::new(Vec::new()),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(0),
            nn_inflight: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            observer,
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            gather_tx,
            eval_tx,
            backprop_tx,
            collision_waiters: Mutex::new(Vec::new()),
        });
        let (nn_tx, nn_rx) = bounded::<NnRequest<O::Stamp>>(resolved.queue_capacity);
        worker_pool.start_job(&shared, &gather_rx, &eval_rx, &nn_tx, &nn_rx, &backprop_rx);
        drop(nn_tx);
        Self {
            shared,
            root_history,
            root_id,
            initial_visits,
            worker_pool,
            workers_idle: false,
        }
    }

    pub fn arena(&self) -> &Arc<NodeArena> {
        &self.shared.arena
    }

    pub fn root_id(&self) -> NodeId {
        self.root_id
    }

    pub fn initial_visits(&self) -> u64 {
        self.initial_visits
    }

    pub fn observer(&self) -> &O {
        &self.shared.observer
    }

    /// 只以 root history 的规则终局作为搜前门禁。
    ///
    /// `ExpansionState::Terminal` 还表示子树已证明的胜负；该 node 跨回合成为
    /// root 后仍可能有合法着可输出，不能把它误作棋局已经结束。
    pub(crate) fn root_is_terminal(&self) -> bool {
        path_terminal_value(self.root_history.as_ref(), 0).is_some()
    }

    pub fn stats(&self) -> Stats {
        self.shared.stats()
    }

    pub fn control(&self) -> SearchControl<O> {
        SearchControl {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Requests a normal stream-search stop without tearing down worker
    /// threads. Gather/Eval/Backprop cancel every unfinished event and its
    /// edge reservation before becoming idle. This is the boundary a later
    /// UCI controller uses this for `stop`; owner cleanup drains this job and
    /// returns its workers to the pool.
    ///
    /// Reference: LC3 overview, "Watchdog" and worker stop coordination.
    pub fn request_stop(&self) {
        self.control().request_stop();
    }

    pub fn is_stopping(&self) -> bool {
        self.shared.stopping.load(Ordering::Acquire)
    }

    fn submit_playout(&self) -> Result<(), EnginError> {
        if self.shared.stopping.load(Ordering::Acquire) {
            return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
        }
        self.shared.start_playout();
        let mut event = GatherEvent::at_root(self.root_id, Arc::clone(&self.root_history));
        event.mark_queued();
        loop {
            if self.shared.stopping.load(Ordering::Acquire) {
                event.cancel();
                self.shared.finish(1, false);
                return Err(EnginError::PortIncomplete("stream worker pipeline is stopped"));
            }
            match self.shared.gather_tx.send_timeout(event, RECEIVE_POLL) {
                Ok(()) => return Ok(()),
                Err(SendTimeoutError::Timeout(returned)) => event = returned,
                Err(SendTimeoutError::Disconnected(returned)) => {
                    returned.cancel();
                    self.shared.finish(1, false);
                    return Err(EnginError::PortIncomplete("stream gather queue disconnected"));
                }
            }
        }
    }

    pub fn run_playouts(&self, count: u64) -> Result<Stats, EnginError> {
        self.run_with_limits(SearchLimits {
            max_playouts: Some(count),
            ..Default::default()
        })
    }

    /// Runs logical batches until a cumulative root-visit budget, deadline, or explicit stop.
    pub fn run_with_limits(&self, limits: SearchLimits) -> Result<Stats, EnginError> {
        self.run_with_limits_reporting(limits, None, |_| {})
    }

    /// 与 `run_with_limits` 相同，但在不 drain 在途流水线的前提下定期归还一次 owner
    /// 控制权。UCI owner 用它判断是否需要输出 `info`；搜索层不解释输出语义。
    pub(crate) fn run_with_limits_reporting(
        &self,
        limits: SearchLimits,
        report_interval: Option<Duration>,
        mut report: impl FnMut(Stats),
    ) -> Result<Stats, EnginError> {
        *self.shared.root_move_filter.lock() = limits.root_move_filter.clone();
        // root 终局 / 共享 Terminal：不进流水线，避免 Gather 再特判。
        if self.root_is_terminal() {
            return Ok(self.stats());
        }
        let target = limits.max_playouts.unwrap_or(u64::MAX);
        let mut next_report = report_interval.and_then(|interval| Instant::now().checked_add(interval));
        loop {
            let now = Instant::now();
            if self.is_stopping()
                || limits.is_exhausted(
                    self.initial_visits.saturating_add(self.stats().completed_playouts),
                    target,
                    now,
                )
            {
                break;
            }
            if next_report.is_some_and(|deadline| now >= deadline) {
                report(self.stats());
                next_report = report_interval.and_then(|interval| now.checked_add(interval));
            }
            let root_state = self.shared.arena.get(self.root_id).map(|root| root.expansion_state());
            if root_state == Some(ExpansionState::Terminal) {
                break;
            }
            if root_state == Some(ExpansionState::Evaluating) {
                self.wait_until_outstanding_below(1, limits.deadline)?;
                continue;
            }
            if root_state != Some(ExpansionState::Expanded) {
                // root 展开前只需要一个真实 leaf。
                if let Err(error) = self.submit_playout() {
                    if self.is_stopping() {
                        break;
                    }
                    return Err(error);
                }
                self.wait_until_outstanding_below(1, limits.deadline)?;
                continue;
            }
            let outstanding = self.shared.outstanding.load(Ordering::Acquire);
            let completed = self.stats().completed_playouts;
            if self
                .initial_visits
                .saturating_add(completed)
                .saturating_add(outstanding as u64)
                >= target
            {
                if outstanding == 0 {
                    break;
                }
                self.wait_until_outstanding_below(outstanding, limits.deadline)?;
                continue;
            }
            let in_flight_limit = deadline_in_flight_limit(
                limits.deadline,
                self.worker_pool.eval_claim_limit(),
                self.worker_pool.eval_batch_size(),
                now,
            );
            if self.shared.nn_inflight.load(Ordering::Acquire) >= in_flight_limit {
                self.shared.wait_while(limits.deadline, true, |shared| {
                    shared.nn_inflight.load(Ordering::Acquire) >= in_flight_limit
                });
                if let Some(error) = self.shared.error.lock().clone() {
                    return Err(error);
                }
                continue;
            }
            if let Err(error) = self.submit_playout() {
                if self.is_stopping() {
                    break;
                }
                return Err(error);
            }
        }
        // 时钟到期后必须 request_stop：否则 Eval 会等当前 GPU 整批跑完，200ms
        // 的 go 就会变成一次推荐 batch 的推理时间。节点预算仍等在途完成。
        if limits.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            self.request_stop();
        }
        // 请求停止是正常搜索结果。`wait_for_idle()` 已保证每个入队 event 都已完成或取消
        // reservation，因此调用方可安全快照部分 graph。
        self.wait_for_idle()?;
        Ok(self.stats())
    }

    pub fn wait_for_idle(&self) -> Result<(), EnginError> {
        self.wait_until_outstanding_below(1, None)
    }

    fn wait_until_outstanding_below(&self, limit: usize, deadline: Option<Instant>) -> Result<(), EnginError> {
        self.shared.wait_while(deadline, false, |shared| {
            shared.outstanding.load(Ordering::Acquire) >= limit
        });
        if let Some(error) = self.shared.error.lock().clone() {
            return Err(error);
        }
        Ok(())
    }

    /// 结束当前 job；常驻 worker 完成 drain 后回到等待，不在这里退出线程。
    pub fn stop_and_finish(&mut self) {
        if self.workers_idle {
            return;
        }
        self.request_stop();
        let _ = self.wait_for_idle();
        self.worker_pool.finish_job();
        self.workers_idle = true;
    }
}

impl<O: SearchObserver> Drop for Search<O> {
    fn drop(&mut self) {
        self.stop_and_finish();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

    use super::Search;
    use crate::neural::backend::UniformBackend;
    use crate::search::decision::{best_move, root_stats};
    use crate::search::param::SearchConfig;

    #[test]
    fn search_completes_batched_playouts() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        let mut pipeline = Search::new(Arc::new(UniformBackend::default()), history, SearchConfig::default());
        let stats = pipeline.run_playouts(64).expect("search");
        assert_eq!(stats.completed_playouts, 64);
        assert!(stats.network_evaluations > 0);
        let root = root_stats(pipeline.arena(), pipeline.root_id()).expect("root");
        assert!(root.completed_visits >= 64);
        assert!(root.edges.iter().all(|e| e.started_visits == e.completed_visits));
        assert!(best_move(pipeline.arena(), pipeline.root_id(), root_is_black).is_some());
        pipeline.stop_and_finish();
    }

    #[test]
    fn advance_prunes_sibling_and_reuses_child() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = super::SearchTree::new(history);
        let mut first = Search::new_with_graph(Arc::new(UniformBackend::default()), &tree, SearchConfig::default());
        first.run_playouts(32).expect("first");
        let played = best_move(first.arena(), first.root_id(), false).expect("best");
        first.stop_and_finish();
        let old_root = tree.root_id();
        tree.advance(played).expect("advance");
        assert_ne!(tree.root_id(), old_root);
        assert!(tree.arena().get(old_root).is_some());
        assert!(tree.arena().get(tree.root_id()).is_some());
    }

    #[test]
    fn reused_proven_win_is_not_a_root_game_terminal() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = super::SearchTree::new(history);
        let played = tree
            .root_history()
            .last()
            .board()
            .parse_move("b2b3")
            .expect("legal move");
        let reply = Move::new(
            xiangqi_core::Square::parse("a9").expect("square"),
            xiangqi_core::Square::parse("a8").expect("square"),
        );

        let root = tree.arena().get(tree.root_id()).expect("root");
        assert!(root.try_begin_evaluation());
        root.publish_edges([(played, 1.0)]);
        let child = tree.arena().child_or_create(&root.edges()[0]);
        let child_node = tree.arena().get(child).expect("child");
        assert!(child_node.try_begin_evaluation());
        child_node.publish_edges([(reply, 1.0)]);
        let terminal = tree.arena().child_or_create(&child_node.edges()[0]);
        let terminal_node = tree.arena().get(terminal).expect("terminal child");
        assert!(terminal_node.try_begin_evaluation());
        terminal_node.mark_terminal(1.0, 0.0, 0.0);
        tree.arena()
            .propagate_proven_terminals(&[tree.root_id(), child, terminal], tree.root_id());
        assert_eq!(child_node.expansion_state(), super::ExpansionState::Terminal);

        tree.advance(played).expect("advance to proven child");
        let mut reused = Search::new_with_graph(Arc::new(UniformBackend::default()), &tree, SearchConfig::default());
        assert!(!reused.root_is_terminal());
        assert_eq!(best_move(reused.arena(), reused.root_id(), true), Some(reply.flip()));
        reused.stop_and_finish();
    }
}
