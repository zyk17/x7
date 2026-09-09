//! 事件定义与固定任务池。
//!
//! worker 不再绑定 Select/Expand/Eval 角色：每次只取一个就绪任务，优先让 NN 回包
//! 释放 reservation；NN inference 仍是独立设备 worker。Proof 的任务位
//! 留在这个调度边界，暂不赋予搜索语义。

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, unbounded};
use parking_lot::Mutex;
use xiangqi_core::{LegalMoveList, Move, PositionHistory};

use crate::EnginError;
use crate::neural::backend::{Backend, EvalCacheKey};
use crate::neural::{EncodedBatch, InputPlanes};
use crate::search::EdgeReservation;
use crate::search::backprop::ValueDelta;

use super::NodeId;
use super::eval::{handle_nn_reply_batch, infer_nn_batch, process_eval_event};
use super::observer::{
    ExecutionKind, ExecutionTimer, NoQueueStamp, NoopObserver, QueueKind, QueueStamp, SearchObserver,
    observe_queue_wait,
};
use super::param::{ResolvedSearchConfig, SearchConfig};
use super::pipeline::{RECEIVE_POLL, Shared, process_expand_event, process_select_event};

/// 第一份 NN request 到达后，给并发 Eval 极短的汇聚时间；避免退化成连续 batch=1。
const NN_BATCH_GATHER: Duration = Duration::from_micros(50);

#[derive(Clone, Debug)]
pub struct Variation {
    base_history: Arc<PositionHistory>,
    moves: smallvec::SmallVec<[Move; 32]>,
}

impl Variation {
    pub fn root(root_history: Arc<PositionHistory>) -> Self {
        Self { base_history: root_history, moves: smallvec::SmallVec::new() }
    }
    pub fn moves(&self) -> &[Move] {
        &self.moves
    }
    pub(crate) fn history(&self) -> PositionHistory {
        let mut history = self.base_history.as_ref().clone();
        for &mv in &self.moves {
            history.append(mv);
        }
        history
    }
    pub fn push(&mut self, mv: Move) {
        self.moves.push(mv);
    }
}

#[derive(Debug)]
pub struct Event {
    pub(crate) node_id: NodeId,
    pub(crate) node_path: Vec<NodeId>,
    pub(crate) reservations: Vec<EdgeReservation>,
}

impl Event {
    pub fn cancel(self) {
        for reservation in self.reservations.into_iter().rev() {
            reservation.cancel();
        }
    }
}

#[derive(Debug)]
pub struct SelectEvent<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: Event,
    pub variation: Variation,
    pub(crate) queued_at: S,
}

/// Select 已 claim 的未展开叶子；保留完整 variation，供规则裁决和后续 forced 展开使用。
pub(crate) type ExpandEvent<S = NoQueueStamp> = SelectEvent<S>;

impl<S: QueueStamp> SelectEvent<S> {
    pub fn at_root(root_id: NodeId, root_history: Arc<PositionHistory>) -> Self {
        Self {
            event: Event { node_id: root_id, node_path: vec![root_id], reservations: Vec::new() },
            variation: Variation::root(root_history),
            queued_at: S::default(),
        }
    }
    pub fn descend(mut self, child_id: NodeId, reservation: EdgeReservation) -> Self {
        self.variation.push(reservation.mv());
        self.event.node_id = child_id;
        self.event.node_path.push(child_id);
        self.event.reservations.push(reservation);
        self
    }
    pub fn node_path(&self) -> &[NodeId] {
        self.event.node_path.as_slice()
    }
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
    pub(crate) fn into_event(self) -> Event {
        self.event
    }
}

/// Expand 已完成规则分类的普通叶子。Eval 只处理 cache、编码、NN 和发布结果。
#[derive(Debug)]
pub(crate) struct EvalEvent<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: Event,
    pub(crate) legal_moves: LegalMoveList,
    pub(crate) cache_key: EvalCacheKey,
    pub(crate) history: PositionHistory,
    pub(crate) queued_at: S,
}
impl<S: QueueStamp> EvalEvent<S> {
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

/// Eval worker 编码后的 cache-miss 请求；NN scheduler 按 FIFO admission。
pub(crate) struct NnRequest<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: EvalEvent<S>,
    pub(crate) planes: InputPlanes,
    pub(crate) queued_at: S,
}
impl<S: QueueStamp> NnRequest<S> {
    pub(crate) fn new(event: EvalEvent<S>, planes: InputPlanes) -> Self {
        Self { event, planes, queued_at: S::default() }
    }
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

/// NN worker 回交给任一 worker 的一整个物理 batch；所有 event 都持有一个 NN credit。
pub(crate) struct NnReplyBatch<S: QueueStamp = NoQueueStamp> {
    pub(crate) events: Vec<EvalEvent<S>>,
    pub(crate) result: Result<Arc<EncodedBatch>, EnginError>,
    pub(crate) queued_at: S,
}
impl<S: QueueStamp> NnReplyBatch<S> {
    pub(crate) fn new(events: Vec<EvalEvent<S>>, result: Result<Arc<EncodedBatch>, EnginError>) -> Self {
        Self { events, result, queued_at: S::default() }
    }
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

#[derive(Debug)]
pub struct BackpropEvent {
    pub(crate) event: Event,
    pub(crate) value: ValueDelta,
    /// 该 event 是否持有 NN scheduler credit。
    ///
    /// cache miss 从 NN scheduler admission 起一直持有到 Backprop 或取消；不在 NN Reply
    /// 时提前释放，确保新的 NN 工作只在当前评估已写入树的 Evidence 后再占 slot。
    pub(crate) holds_nn_credit: bool,
}

impl BackpropEvent {
    pub(crate) fn with_nn_credit(event: Event, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self { event, value: ValueDelta::one(wl, draw, plies_left), holds_nn_credit: true }
    }
    pub(crate) fn without_nn_credit(event: Event, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self { event, value: ValueDelta::one(wl, draw, plies_left), holds_nn_credit: false }
    }
    pub fn cancel(self) {
        self.event.cancel();
    }
}

enum WorkerCommand<O: SearchObserver> {
    Run {
        shared: Arc<Shared<O>>,
        select_rx: Receiver<SelectEvent<O::Stamp>>,
        expand_rx: Receiver<ExpandEvent<O::Stamp>>,
        eval_rx: Receiver<EvalEvent<O::Stamp>>,
        nn_reply_rx: Receiver<NnReplyBatch<O::Stamp>>,
        nn_tx: Sender<NnRequest<O::Stamp>>,
    },
    Shutdown,
}
enum NnCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<NnRequest<O::Stamp>>, Receiver<usize>, Sender<NnReplyBatch<O::Stamp>>),
    Shutdown,
}

/// 固定容量的任务池；NN 另占一个设备 worker。
pub(crate) struct WorkerPool<O: SearchObserver = NoopObserver> {
    worker_commands: Vec<Sender<WorkerCommand<O>>>,
    nn_commands: Sender<NnCommand<O>>,
    job_done: Receiver<()>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    eval_batch_size: usize,
    nn_credit_limit: usize,
}

impl<O: SearchObserver> WorkerPool<O> {
    pub(crate) fn new(backend: &dyn Backend, config: &SearchConfig) -> Self {
        config.validate();
        Self::from_resolved(&config.resolve(backend))
    }
    pub(crate) fn matches_config(&self, backend: &dyn Backend, config: &SearchConfig) -> bool {
        let config = config.resolve(backend);
        self.eval_batch_size == config.eval_batch_size
            && self.nn_credit_limit == config.nn_permit_limit
            && self.worker_commands.len() == config.threads
    }
    fn from_resolved(config: &ResolvedSearchConfig) -> Self {
        let (job_done_tx, job_done) = unbounded();
        let (nn_commands, nn_rx) = unbounded();
        let mut worker_commands = Vec::with_capacity(config.threads);
        let mut threads = Vec::with_capacity(config.threads + 1);
        for _ in 0..config.threads {
            let (tx, rx) = unbounded();
            let done = job_done_tx.clone();
            threads.push(thread::spawn(move || persistent_worker::<O>(rx, done)));
            worker_commands.push(tx);
        }
        let batch_size = config.eval_batch_size;
        let credit_limit = config.nn_permit_limit;
        threads.push(thread::spawn({
            let done = job_done_tx.clone();
            move || persistent_nn_worker::<O>(nn_rx, done, batch_size, credit_limit)
        }));
        Self {
            worker_commands,
            nn_commands,
            job_done,
            threads: Mutex::new(threads),
            eval_batch_size: config.eval_batch_size,
            nn_credit_limit: config.nn_permit_limit,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_job(
        &self,
        shared: &Arc<Shared<O>>,
        select_rx: &Receiver<SelectEvent<O::Stamp>>,
        expand_rx: &Receiver<ExpandEvent<O::Stamp>>,
        eval_rx: &Receiver<EvalEvent<O::Stamp>>,
        nn_reply_rx: &Receiver<NnReplyBatch<O::Stamp>>,
        nn_tx: &Sender<NnRequest<O::Stamp>>,
        nn_rx: &Receiver<NnRequest<O::Stamp>>,
        nn_credit_rx: &Receiver<usize>,
        nn_reply_tx: &Sender<NnReplyBatch<O::Stamp>>,
    ) {
        self.nn_commands
            .send(NnCommand::Run(Arc::clone(shared), nn_rx.clone(), nn_credit_rx.clone(), nn_reply_tx.clone()))
            .expect("persistent nn worker is alive");
        for sender in &self.worker_commands {
            sender
                .send(WorkerCommand::Run {
                    shared: Arc::clone(shared),
                    select_rx: select_rx.clone(),
                    expand_rx: expand_rx.clone(),
                    eval_rx: eval_rx.clone(),
                    nn_reply_rx: nn_reply_rx.clone(),
                    nn_tx: nn_tx.clone(),
                })
                .expect("persistent worker is alive");
        }
    }
    pub(crate) fn finish_job(&self) {
        for _ in 0..self.worker_commands.len() + 1 {
            self.job_done.recv().expect("persistent worker completion");
        }
    }
    pub(crate) fn nn_credit_limit(&self) -> usize {
        self.nn_credit_limit
    }
    pub(crate) fn worker_count(&self) -> usize {
        self.worker_commands.len()
    }
}
impl<O: SearchObserver> Drop for WorkerPool<O> {
    fn drop(&mut self) {
        for sender in &self.worker_commands {
            let _ = sender.send(WorkerCommand::Shutdown);
        }
        let _ = self.nn_commands.send(NnCommand::Shutdown);
        for worker in self.threads.get_mut().drain(..) {
            let _ = worker.join();
        }
    }
}

/// stop 后不再启动上游工作；已经得到结果的 NN reply 仍完成并写入缓存。
fn cancel_pending_worker_queues<O: SearchObserver>(
    shared: &Shared<O>,
    select_rx: &Receiver<SelectEvent<O::Stamp>>,
    expand_rx: &Receiver<ExpandEvent<O::Stamp>>,
    eval_rx: &Receiver<EvalEvent<O::Stamp>>,
) {
    while let Ok(event) = select_rx.try_recv() {
        shared.cancel_event(event.into_event());
    }
    while let Ok(event) = expand_rx.try_recv() {
        shared.cancel_claim(event.into_event());
    }
    while let Ok(job) = eval_rx.try_recv() {
        shared.cancel_claim(job.event);
    }
}

fn process_nn_reply<O: SearchObserver>(shared: &Shared<O>, mut reply: NnReplyBatch<O::Stamp>) {
    if O::ENABLED {
        observe_queue_wait(&mut reply.queued_at, &shared.observer, QueueKind::NnReply);
    }
    if let Err(error) = handle_nn_reply_batch(shared, reply) {
        shared.fail(error);
    }
}

fn worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    select_rx: Receiver<SelectEvent<O::Stamp>>,
    expand_rx: Receiver<ExpandEvent<O::Stamp>>,
    eval_rx: Receiver<EvalEvent<O::Stamp>>,
    reply_rx: Receiver<NnReplyBatch<O::Stamp>>,
    nn_tx: Sender<NnRequest<O::Stamp>>,
) {
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            cancel_pending_worker_queues(&shared, &select_rx, &expand_rx, &eval_rx);
            if let Ok(reply) = reply_rx.try_recv() {
                process_nn_reply(&shared, reply);
                continue;
            }
            if shared.outstanding.load(Ordering::Acquire) == 0 {
                break;
            }
            if let Ok(reply) = reply_rx.recv_timeout(RECEIVE_POLL) {
                process_nn_reply(&shared, reply);
            }
            continue;
        }
        if let Ok(reply) = reply_rx.try_recv() {
            process_nn_reply(&shared, reply);
            continue;
        }
        if let Ok(mut job) = eval_rx.try_recv() {
            if O::ENABLED {
                observe_queue_wait(&mut job.queued_at, &shared.observer, QueueKind::Eval);
            }
            if let Err(error) = process_eval_event(&shared, &nn_tx, job) {
                shared.fail(error);
            }
            continue;
        }
        if let Ok(mut event) = expand_rx.try_recv() {
            if O::ENABLED {
                observe_queue_wait(&mut event.queued_at, &shared.observer, QueueKind::Expand);
            }
            process_expand_event(&shared, event);
            continue;
        }
        if let Ok(mut event) = select_rx.try_recv() {
            if O::ENABLED {
                observe_queue_wait(&mut event.queued_at, &shared.observer, QueueKind::Select);
            }
            process_select_event(&shared, event);
            continue;
        }
        crossbeam_channel::select! {
            recv(reply_rx) -> result => {
                if let Ok(reply) = result { process_nn_reply(&shared, reply); }
            },
            recv(eval_rx) -> result => {
                if let Ok(mut job) = result {
                    if O::ENABLED { observe_queue_wait(&mut job.queued_at, &shared.observer, QueueKind::Eval); }
                    let outcome = process_eval_event(&shared, &nn_tx, job);
                    if let Err(error) = outcome { shared.fail(error); }
                }
            },
            recv(expand_rx) -> result => {
                if let Ok(mut event) = result {
                    if O::ENABLED { observe_queue_wait(&mut event.queued_at, &shared.observer, QueueKind::Expand); }
                    process_expand_event(&shared, event);
                }
            },
            recv(select_rx) -> result => {
                if let Ok(mut event) = result {
                    if O::ENABLED { observe_queue_wait(&mut event.queued_at, &shared.observer, QueueKind::Select); }
                    process_select_event(&shared, event);
                }
            },
            default(RECEIVE_POLL) => {},
        }
    }
}

fn persistent_worker<O: SearchObserver>(commands: Receiver<WorkerCommand<O>>, done: Sender<()>) {
    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::Run { shared, select_rx, expand_rx, eval_rx, nn_reply_rx, nn_tx } => {
                worker(shared, select_rx, expand_rx, eval_rx, nn_reply_rx, nn_tx);
                let _ = done.send(());
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

fn nn_worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    request_rx: Receiver<NnRequest<O::Stamp>>,
    credit_rx: Receiver<usize>,
    reply_tx: Sender<NnReplyBatch<O::Stamp>>,
    batch_size: usize,
    credit_limit: usize,
) {
    let mut requests = Vec::with_capacity(batch_size);
    let mut samples = Vec::with_capacity(batch_size);
    let mut in_flight = 0;
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            while let Ok(request) = request_rx.try_recv() {
                shared.cancel_claim(request.event.event);
            }
            break;
        }
        while let Ok(count) = credit_rx.try_recv() {
            debug_assert!(count <= in_flight, "NN credit underflow");
            in_flight -= count;
        }

        if in_flight == credit_limit {
            match credit_rx.recv_timeout(RECEIVE_POLL) {
                Ok(count) => {
                    debug_assert!(count <= in_flight, "NN credit underflow");
                    in_flight -= count;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            continue;
        }

        let request = match request_rx.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        requests.push(request);
        let deadline = Instant::now() + NN_BATCH_GATHER;
        while requests.len() < batch_size && Instant::now() < deadline {
            while let Ok(count) = credit_rx.try_recv() {
                debug_assert!(count <= in_flight, "NN credit underflow");
                in_flight -= count;
            }
            if in_flight + requests.len() >= credit_limit {
                std::hint::spin_loop();
                continue;
            }
            match request_rx.try_recv() {
                Ok(request) => requests.push(request),
                Err(TryRecvError::Empty) => std::hint::spin_loop(),
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if O::ENABLED {
            for (index, request) in requests.iter_mut().enumerate() {
                observe_queue_wait(&mut request.queued_at, &shared.observer, QueueKind::Nn);
                shared.observer.on_peak_inflight(in_flight + index + 1);
            }
        }
        in_flight += requests.len();
        if shared.stopping.load(Ordering::Acquire) {
            for request in requests.drain(..) {
                shared.cancel_evaluation(request.event.event);
            }
            continue;
        }
        let _timer = ExecutionTimer::new(&shared.observer, ExecutionKind::Nn);
        samples.clear();
        samples.extend(requests.iter().map(|request| request.planes));
        match infer_nn_batch(&shared, &samples) {
            Ok(output) => {
                send_nn_reply_batch(&reply_tx, requests.drain(..).map(|request| request.event).collect(), Ok(output))
            }
            Err(error) => {
                send_nn_reply_batch(&reply_tx, requests.drain(..).map(|request| request.event).collect(), Err(error))
            }
        }
    }
}

fn send_nn_reply_batch<S: QueueStamp>(
    reply_tx: &Sender<NnReplyBatch<S>>,
    events: Vec<EvalEvent<S>>,
    result: Result<Arc<EncodedBatch>, EnginError>,
) {
    let mut reply = NnReplyBatch::new(events, result);
    reply.mark_queued();
    let _ = reply_tx.send(reply);
}

fn persistent_nn_worker<O: SearchObserver>(
    commands: Receiver<NnCommand<O>>,
    done: Sender<()>,
    batch_size: usize,
    credit_limit: usize,
) {
    while let Ok(command) = commands.recv() {
        match command {
            NnCommand::Run(shared, request_rx, credit_rx, reply_tx) => {
                nn_worker(shared, request_rx, credit_rx, reply_tx, batch_size, credit_limit);
                let _ = done.send(());
            }
            NnCommand::Shutdown => break,
        }
    }
}
