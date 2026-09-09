//! Shared / Stats + Select/Expand 流程组装 + `Search` API。
//!
//! 树走组装（claim / collision / 选边下降）在本文件；选边公式在 `select`，
//! 事件编排在本文件，worker 线程循环在 `workerpool`；NN 编码/回包与回传算术分别在
//! `eval` / `backprop`。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, unbounded};
use parking_lot::{Condvar, Mutex};
use xiangqi_core::{Move, PositionHistory};

use crate::EnginError;
use crate::neural::MOVE_HISTORY;
use crate::neural::backend::Backend;

use super::backprop::{complete_batch, complete_one};
use super::expand::{ExpandKind, classify_expand, game_terminal_value};
use super::observer::{ExecutionKind, ExecutionTimer, NoopObserver, SearchObserver};
use super::param::{SearchConfig, SearchParams};
use super::select::select_edge;
use super::workerpool::{BackpropEvent, EvalEvent, Event, ExpandEvent, NnRequest, SelectEvent, WorkerJob, WorkerPool};
use super::{EdgeReservation, ExpansionState, Node, NodeArena, NodeId, SearchTree};
use crate::neural::backend::EvalCacheKey;

pub(crate) const RECEIVE_POLL: Duration = Duration::from_millis(10);

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
    stopping: AtomicBool,
    /// 未 `finish` 的 owned event：drain 与派发上限。
    pub(crate) outstanding: AtomicUsize,
    /// Backprop 或取消归还给 NN scheduler 的 credit。
    pub(crate) nn_credit_tx: Sender<usize>,
    pub(crate) completed: AtomicU64,
    pub(crate) completed_depth: AtomicU64,
    pub(crate) max_depth: AtomicU64,
    pub(crate) network_evaluations: AtomicU64,
    pub(crate) observer: O,
    pub(crate) error: Mutex<Option<EnginError>>,
    pub(crate) idle_lock: Mutex<()>,
    pub(crate) idle: Condvar,
    pub(crate) select_tx: Sender<SelectEvent<O::Stamp>>,
    pub(crate) expand_tx: Sender<ExpandEvent<O::Stamp>>,
    pub(crate) eval_tx: Sender<EvalEvent<O::Stamp>>,
    /// 撞上 `Claimed` 叶子的 playout。先留着 reservation / μ；该叶子自己的
    /// backprop `complete` 之后再按 `node_id` 摘出来 cancel。
    pub(crate) collision_waiters: Mutex<Vec<Event>>,
}

impl<O: SearchObserver> Shared<O> {
    pub(crate) fn request_stop(&self) {
        self.stopping.store(true, Ordering::Release);
        self.idle.notify_all();
    }

    pub(crate) fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }

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

    /// 取消 owned event，并归还其 reservation。
    pub(crate) fn cancel_event(&self, event: Event) {
        event.cancel();
        self.finish(1, false);
    }

    /// 取消尚未取得 NN permit 的叶子，并归还其 reservation。
    pub(crate) fn cancel_claim(&self, event: Event) {
        self.abort_claim(event.node_id);
        self.cancel_event(event);
    }

    /// 取消已获 NN credit 的 Eval event。
    pub(crate) fn cancel_evaluation(&self, event: Event) {
        self.abort_claim(event.node_id);
        self.return_nn_credits(1);
        self.cancel_event(event);
    }

    /// 清除 Claimed 状态及其所有 collision reservation。
    fn abort_claim(&self, id: NodeId) {
        self.arena.get(id).expect("claimed event node lives until drain").abort_claim();
        self.cancel_collisions(id);
    }

    pub(crate) fn return_nn_credits(&self, count: usize) {
        if count == 0 {
            return;
        }
        let _ = self.nn_credit_tx.send(count);
    }

    fn wait_until_outstanding_below(&self, limit: usize, deadline: Option<Instant>) {
        let mut guard = self.idle_lock.lock();
        while self.outstanding.load(Ordering::Acquire) >= limit && self.error.lock().is_none() {
            let Some(deadline) = deadline else {
                self.idle.wait(&mut guard);
                continue;
            };
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let wait = deadline.saturating_duration_since(now);
            if wait.is_zero() {
                break;
            }
            self.idle.wait_for(&mut guard, wait);
        }
    }

    /// 撞上正在评估的叶子：挂起整条 reservation，让 μ 继续分流。
    pub(crate) fn park_collision(&self, event: SelectEvent<O::Stamp>) {
        if O::ENABLED {
            self.observer.on_collision(event.variation.moves().len());
        }
        let event = event.into_event();
        {
            let mut waiters = self.collision_waiters.lock();
            // owner 可在本次 Select 看到 Claimed 后、waiter 入队前完成或取消；此处复查
            // 避免把永远不会再由 owner 唤醒的 reservation 留在 waiter 表。
            if !self.is_stopping()
                && self.arena.get(event.node_id).expect("collision event node lives until drain").expansion_state()
                    == ExpansionState::Claimed
            {
                waiters.push(event);
                return;
            }
        }
        self.cancel_event(event);
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

    pub(crate) fn fail(&self, error: EnginError) {
        let mut current = self.error.lock();
        if current.is_none() {
            *current = Some(error);
        }
        self.request_stop();
    }

    pub(crate) fn send_eval(&self, mut event: EvalEvent<O::Stamp>) {
        event.mark_queued();
        if let Err(error) = self.eval_tx.send(event) {
            self.cancel_claim(error.0.event);
        }
    }

    pub(crate) fn send_expand(&self, mut event: ExpandEvent<O::Stamp>) {
        event.mark_queued();
        if let Err(error) = self.expand_tx.send(event) {
            self.cancel_claim(error.0.into_event());
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

// --- Select / Expand ---------------------------------------------------------

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
    )?;
    let edge = &node.edges()[edge_index];
    let child = shared.arena.child_or_create(edge);
    let reservation = node.reserve_edge(edge_index, virtual_mean).expect("selected stream edge");
    Some((child, reservation))
}

pub(crate) fn process_select_event<O: SearchObserver>(shared: &Shared<O>, mut event: SelectEvent<O::Stamp>) {
    let _timer = ExecutionTimer::new(&shared.observer, ExecutionKind::Select);
    loop {
        if shared.is_stopping() {
            shared.cancel_event(event.into_event());
            return;
        }
        let node = shared.arena.get(event.event.node_id).expect("event node lives until job drain");
        match node.expansion_state() {
            ExpansionState::Unexpanded => {
                if node.try_claim() {
                    shared.send_expand(event);
                    return;
                }
            }
            ExpansionState::Claimed => {
                shared.park_collision(event);
                return;
            }
            ExpansionState::Terminal => {
                let (wl, draw, plies_left) = node.terminal_value().expect("terminal node has exact value");
                drop(_timer);
                process_backprop_one(
                    shared,
                    BackpropEvent::without_nn_credit(event.into_event(), wl, draw, plies_left),
                );
                return;
            }
            ExpansionState::Expanded => {
                let depth = event.variation.moves().len();
                let Some((child, reservation)) = branch_at_expanded_node(shared, node, depth) else {
                    // 非 root 没有空 edges；root 只有被 searchmoves 排空时才会走到这里。
                    if event.node_path().len() == 1 {
                        shared.request_stop();
                    }
                    thread::yield_now();
                    continue;
                };
                event = event.descend(child, reservation);
            }
        }
    }
}

/// 只处理已 claim 叶子的规则、合法着与后续任务分流。
pub(crate) fn process_expand_event<O: SearchObserver>(shared: &Shared<O>, event: ExpandEvent<O::Stamp>) {
    let timer = ExecutionTimer::new(&shared.observer, ExecutionKind::Expand);
    if shared.is_stopping() {
        shared.cancel_claim(event.into_event());
        return;
    }
    let node = shared.arena.get(event.event.node_id).expect("expand node lives until job drain");
    let history = event.variation.history();
    match classify_expand(&history) {
        ExpandKind::Terminal { wl, draw, plies_left } => {
            node.mark_terminal(wl, draw, plies_left);
            let root = event.node_path()[0];
            shared.arena.propagate_proven_terminals(event.node_path(), root);
            drop(timer);
            process_backprop_one(shared, BackpropEvent::without_nn_credit(event.into_event(), wl, draw, plies_left));
        }
        ExpandKind::Evaluate { legal_moves } => {
            shared.send_eval(EvalEvent {
                event: event.into_event(),
                cache_key: EvalCacheKey::new(history.last(), legal_moves.len()),
                legal_moves,
                history,
                queued_at: O::Stamp::default(),
            });
        }
    }
}

/// 同一 NN batch 的回传直接进入这里，合并 node 写入。
pub(crate) fn process_backprop_batch<O: SearchObserver>(shared: &Shared<O>, mut events: Vec<BackpropEvent>) {
    match events.len() {
        0 => return,
        1 => {
            process_backprop_one(shared, events.pop().expect("one backprop event"));
            return;
        }
        _ => {}
    }
    let _timer = ExecutionTimer::new(&shared.observer, ExecutionKind::Backprop);
    if O::ENABLED {
        shared.observer.on_backprop_batch(events.len());
    }
    let claims: Vec<(bool, NodeId)> = events.iter().map(|event| (event.holds_nn_credit, event.event.node_id)).collect();
    let result = complete_batch(events, &shared.arena);
    for (_, id) in &claims {
        shared.cancel_collisions(*id);
    }
    shared.return_nn_credits(claims.iter().filter(|(held, _)| *held).count());
    shared.completed_depth.fetch_add(result.completed_depth, Ordering::AcqRel);
    shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
    drop(_timer);
    shared.finish(result.completed_playouts as usize, true);
}

pub(crate) fn process_backprop_one<O: SearchObserver>(shared: &Shared<O>, event: BackpropEvent) {
    let _timer = ExecutionTimer::new(&shared.observer, ExecutionKind::Backprop);
    if O::ENABLED {
        shared.observer.on_backprop_batch(1);
    }
    let held_nn_credit = event.holds_nn_credit;
    let node_id = event.event.node_id;
    let result = complete_one(event, &shared.arena);
    shared.cancel_collisions(node_id);
    shared.return_nn_credits(usize::from(held_nn_credit));
    shared.completed_depth.fetch_add(result.completed_depth, Ordering::AcqRel);
    shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
    drop(_timer);
    shared.finish(1, true);
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

/// 运行中搜索可克隆的 stop 句柄。
#[derive(Clone)]
pub(crate) struct StopHandle<O: SearchObserver = NoopObserver> {
    shared: Arc<Shared<O>>,
}

impl<O: SearchObserver> StopHandle<O> {
    pub(crate) fn request_stop(&self) {
        self.shared.request_stop();
    }
}

/// 连续流式搜索：Select / Expand / Eval / NN / Reply / Backprop。
/// Expand 裁决规则终局和合法着；Eval 先查 cache，仅 miss 编码并送 NN；NN 合批结果交回
/// Reply 发布 edge，再由 Backprop 完成 reservation。
pub struct Search<O: SearchObserver = NoopObserver> {
    shared: Arc<Shared<O>>,
    root_id: NodeId,
    search_history: Arc<PositionHistory>,
    /// 启动本次 job 前 root 已有的 completed N。它计入 UCI `go nodes`，但不计入本次 NPS。
    initial_visits: u64,
    worker_pool: Arc<WorkerPool<O>>,
    finished: bool,
}

impl<O: SearchObserver> Search<O> {
    /// 启动一次独立 Search job；该 job 自己创建并持有 worker pool。
    pub fn start(backend: Arc<dyn Backend>, graph: &SearchTree, config: SearchConfig, observer: O) -> Self {
        let worker_pool = Arc::new(WorkerPool::new(backend.as_ref(), &config));
        Self::start_with_pool(backend, graph, config, observer, worker_pool)
    }

    /// 启动一次 job，复用 Engine 持有的固定 worker pool。
    pub(crate) fn start_with_pool(
        backend: Arc<dyn Backend>,
        graph: &SearchTree,
        config: SearchConfig,
        observer: O,
        worker_pool: Arc<WorkerPool<O>>,
    ) -> Self {
        config.validate();
        let resolved = config.resolve(backend.as_ref());
        let (select_tx, select_rx) = unbounded();
        let (expand_tx, expand_rx) = unbounded();
        let (eval_tx, eval_rx) = unbounded();
        let (nn_reply_tx, nn_reply_rx) = unbounded();
        let (nn_credit_tx, nn_credit_rx) = unbounded();
        // UCI/graph 持有完整 history 用于跨回合定位；每个 event 只需要重复规则自
        // 最近零化着以来的后缀，以及 NN 的最近 8 层。这里一次裁剪后由整次 job 共享。
        let root_id = graph.root_id();
        let search_history = Arc::new(graph.root_history().search_window(MOVE_HISTORY));
        let initial_visits = graph.arena().get(root_id).map_or(0, |root| root.completed_visits() as u64);
        let shared = Arc::new(Shared {
            backend,
            arena: Arc::clone(graph.arena()),
            params: resolved.params,
            root_move_filter: Mutex::new(Vec::new()),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(0),
            nn_credit_tx,
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            observer,
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            select_tx,
            expand_tx,
            eval_tx,
            collision_waiters: Mutex::new(Vec::new()),
        });
        let (nn_tx, nn_rx) = crossbeam_channel::unbounded::<NnRequest<O::Stamp>>();
        worker_pool.start(WorkerJob {
            shared: Arc::clone(&shared),
            select_rx,
            expand_rx,
            eval_rx,
            nn_reply_rx,
            nn_tx,
            nn_rx,
            nn_credit_rx,
            nn_reply_tx,
        });
        Self { shared, root_id, search_history, initial_visits, worker_pool, finished: false }
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
        game_terminal_value(self.search_history.as_ref()).is_some()
    }

    pub fn stats(&self) -> Stats {
        self.shared.stats()
    }

    pub(crate) fn stop_handle(&self) -> StopHandle<O> {
        StopHandle { shared: Arc::clone(&self.shared) }
    }

    fn submit_select(&self) -> Result<(), EnginError> {
        self.shared.start_playout();
        let mut event = SelectEvent::at_root(self.root_id, Arc::clone(&self.search_history));
        event.mark_queued();
        if let Err(error) = self.shared.select_tx.send(event) {
            self.shared.cancel_event(error.0.into_event());
            return Err(EnginError::Internal("stream select queue disconnected"));
        }
        Ok(())
    }

    /// Runs logical batches until a cumulative root-visit budget, deadline, or explicit stop.
    pub fn run(&self, limits: SearchLimits) -> Result<Stats, EnginError> {
        self.run_with_report(limits, None, |_| {})
    }

    /// 与 `run` 相同，但在不 drain 在途流水线的前提下定期归还一次 owner
    /// 控制权。UCI owner 用它判断是否需要输出 `info`；搜索层不解释输出语义。
    pub(crate) fn run_with_report(
        &self,
        limits: SearchLimits,
        report_interval: Option<Duration>,
        mut report: impl FnMut(Stats),
    ) -> Result<Stats, EnginError> {
        *self.shared.root_move_filter.lock() = limits.root_move_filter.clone();
        // root 终局 / 共享 Terminal：不进流水线，避免 Select 再特判。
        if self.root_is_terminal() {
            return Ok(self.stats());
        }
        let target = limits.max_playouts.unwrap_or(u64::MAX);
        let mut next_report = report_interval.and_then(|interval| Instant::now().checked_add(interval));
        loop {
            let now = Instant::now();
            if self.shared.is_stopping()
                || limits.is_exhausted(self.initial_visits.saturating_add(self.stats().completed_playouts), target, now)
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
            if root_state == Some(ExpansionState::Claimed) {
                self.wait_until_outstanding_below(1, limits.deadline)?;
                continue;
            }
            if root_state != Some(ExpansionState::Expanded) {
                // root 展开前只需要一个真实 leaf。
                self.submit_select()?;
                self.wait_until_outstanding_below(1, limits.deadline)?;
                continue;
            }
            let outstanding = self.shared.outstanding.load(Ordering::Acquire);
            let dispatch_limit = self.worker_pool.nn_credit_limit().saturating_add(self.worker_pool.worker_count());
            if outstanding >= dispatch_limit {
                self.wait_until_outstanding_below(dispatch_limit, limits.deadline)?;
                continue;
            }
            self.submit_select()?;
        }
        // 时钟到期后必须 request_stop：否则 Eval 会等当前 GPU 整批跑完，200ms
        // 的 go 就会变成一次推荐 batch 的推理时间。节点预算仍等在途完成。
        if limits.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            self.shared.request_stop();
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
        self.shared.wait_until_outstanding_below(limit, deadline);
        if let Some(error) = self.shared.error.lock().clone() {
            return Err(error);
        }
        Ok(())
    }

    /// 所有 `run` 阶段结束后，停止并归还本 job 占用的常驻 worker。
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.shared.request_stop();
        let _ = self.wait_for_idle();
        self.worker_pool.finish();
        self.finished = true;
    }
}

impl<O: SearchObserver> Drop for Search<O> {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    use crossbeam_channel::{Receiver, bounded};
    use parking_lot::{Condvar, Mutex};
    use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

    use super::{Search, SearchLimits, Shared, process_select_event};
    use crate::neural::backend::{Backend, UniformBackend};
    use crate::search::decision::{best_move, root_stats};
    use crate::search::observer::NoopObserver;
    use crate::search::param::{SearchConfig, SearchParams};
    use crate::search::workerpool::ExpandEvent;
    use crate::search::{NoQueueStamp, NodeArena, SearchTree};

    fn test_shared(arena: Arc<NodeArena>, outstanding: usize) -> (Arc<Shared>, Receiver<ExpandEvent<NoQueueStamp>>) {
        let (select_tx, _) = bounded(1);
        let (expand_tx, expand_rx) = bounded(1);
        let (eval_tx, _) = bounded(1);
        let shared = Arc::new(Shared {
            backend: Arc::new(UniformBackend::default()) as Arc<dyn Backend>,
            arena,
            params: SearchParams::default(),
            root_move_filter: Mutex::new(Vec::new()),
            stopping: AtomicBool::new(false),
            outstanding: AtomicUsize::new(outstanding),
            nn_credit_tx: crossbeam_channel::unbounded().0,
            completed: AtomicU64::new(0),
            completed_depth: AtomicU64::new(0),
            max_depth: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            observer: NoopObserver,
            error: Mutex::new(None),
            idle_lock: Mutex::new(()),
            idle: Condvar::new(),
            select_tx,
            expand_tx,
            eval_tx,
            collision_waiters: Mutex::new(Vec::new()),
        });
        (shared, expand_rx)
    }

    #[test]
    fn terminal_select_backprops_its_exact_value() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let arena = Arc::new(NodeArena::default());
        let root_id = arena.allocate();
        let root = arena.get(root_id).expect("root");
        assert!(root.try_claim());
        let mv =
            Move::new(xiangqi_core::Square::parse("b2").expect("from"), xiangqi_core::Square::parse("b3").expect("to"));
        root.publish_edges([(mv, 1.0)]);
        let terminal_id = arena.child_or_create(&root.edges()[0]);
        let terminal = arena.get(terminal_id).expect("terminal");
        assert!(terminal.try_claim());
        terminal.mark_terminal(1.0, 0.0, 1.0);

        let (shared, _) = test_shared(Arc::clone(&arena), 1);
        let event = super::SelectEvent::<NoQueueStamp>::at_root(root_id, history)
            .descend(terminal_id, root.reserve_edge(0, 0.0).expect("reservation"));

        process_select_event(&shared, event);
        assert_eq!(root.edges()[0].visits(), 1);
        assert_eq!(root.edges()[0].completed_visits(), 1);
        assert_eq!(root.edges()[0].q(), 1.0);
        assert_eq!(root.completed_visits(), 1);
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 0);
        assert_eq!(shared.completed.load(Ordering::Acquire), 1);
    }

    #[test]
    fn select_claims_then_queues_an_expand_event() {
        let arena = Arc::new(NodeArena::default());
        let root = arena.allocate();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let (shared, expand_rx) = test_shared(Arc::clone(&arena), 1);

        process_select_event(&shared, super::SelectEvent::at_root(root, history));

        assert_eq!(arena.get(root).expect("root").expansion_state(), super::ExpansionState::Claimed);
        assert_eq!(expand_rx.try_recv().expect("expand event").event.node_id, root);
        assert_eq!(shared.outstanding.load(Ordering::Acquire), 1);
    }

    #[test]
    fn search_drains_batched_playouts() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let root_is_black = history.is_black_to_move();
        let tree = SearchTree::new(history);
        let mut pipeline =
            Search::start(Arc::new(UniformBackend::default()), &tree, SearchConfig::default(), NoopObserver);
        let stats = pipeline.run(SearchLimits { max_playouts: Some(64), ..Default::default() }).expect("search");
        assert!(stats.completed_playouts >= 64);
        assert!(stats.network_evaluations > 0);
        let root = root_stats(pipeline.arena(), pipeline.root_id()).expect("root");
        assert!(root.completed_visits >= 64);
        assert!(root.edges.iter().all(|e| e.started_visits == e.completed_visits));
        assert!(best_move(pipeline.arena(), pipeline.root_id(), root_is_black).is_some());
        pipeline.finish();
    }

    #[test]
    fn advance_prunes_sibling_and_reuses_child() {
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        let history = Arc::new(PositionHistory::from_positions(state.positions()));
        let mut tree = SearchTree::new(history);
        let mut first =
            Search::start(Arc::new(UniformBackend::default()), &tree, SearchConfig::default(), NoopObserver);
        first.run(SearchLimits { max_playouts: Some(32), ..Default::default() }).expect("first");
        let played = best_move(first.arena(), first.root_id(), false).expect("best");
        first.finish();
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
        let mut tree = SearchTree::new(history);
        let played = tree.root_history().last().board().parse_move("b2b3").expect("legal move");
        let reply = Move::new(
            xiangqi_core::Square::parse("a9").expect("square"),
            xiangqi_core::Square::parse("a8").expect("square"),
        );

        let root = tree.arena().get(tree.root_id()).expect("root");
        assert!(root.try_claim());
        root.publish_edges([(played, 1.0)]);
        let child = tree.arena().child_or_create(&root.edges()[0]);
        let child_node = tree.arena().get(child).expect("child");
        assert!(child_node.try_claim());
        child_node.publish_edges([(reply, 1.0)]);
        let terminal = tree.arena().child_or_create(&child_node.edges()[0]);
        let terminal_node = tree.arena().get(terminal).expect("terminal child");
        assert!(terminal_node.try_claim());
        terminal_node.mark_terminal(1.0, 0.0, 0.0);
        tree.arena().propagate_proven_terminals(&[tree.root_id(), child, terminal], tree.root_id());
        assert_eq!(child_node.expansion_state(), super::ExpansionState::Terminal);

        tree.advance(played).expect("advance to proven child");
        let mut reused =
            Search::start(Arc::new(UniformBackend::default()), &tree, SearchConfig::default(), NoopObserver);
        assert!(!reused.root_is_terminal());
        assert_eq!(best_move(reused.arena(), reused.root_id(), true), Some(reply.flip()));
        reused.finish();
    }
}
