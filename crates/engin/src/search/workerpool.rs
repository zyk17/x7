//! 事件定义与固定任务池。
//!
//! worker 不再绑定 Select/Expand/Eval/Backprop 角色：每次只取一个就绪任务，优先让
//! Backprop 和 NN 回包释放 reservation；NN inference 仍是独立设备 worker。Proof 的任务位
//! 留在这个调度边界，暂不赋予搜索语义。

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, unbounded};
use parking_lot::Mutex;
use xiangqi_core::{LegalMoveList, Move, PositionHistory};

use crate::EnginError;
use crate::neural::backend::{Backend, EvalCacheKey};
use crate::neural::{EncodedBatch, InputPlanes};
use crate::search::EdgeReservation;

use super::eval::{cancel_evaluation, handle_nn_reply, infer_nn_batch, process_eval_event, send_nn_reply};
use super::observer::{NoQueueStamp, NoopObserver, QueueKind, QueueStamp, SearchObserver, observe_queue_wait};
use super::param::{ResolvedSearchConfig, SearchConfig};
use super::pipeline::{RECEIVE_POLL, Shared, process_backprop_event, process_expand_event, process_select_event};
use super::{NodeId, ValueDelta};

#[derive(Clone, Debug)]
pub struct Variation {
    base_history: Arc<PositionHistory>,
    moves: smallvec::SmallVec<[Move; 32]>,
}

impl Variation {
    pub fn root(root_history: Arc<PositionHistory>) -> Self {
        Self {
            base_history: root_history,
            moves: smallvec::SmallVec::new(),
        }
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
            event: Event {
                node_id: root_id,
                node_path: vec![root_id],
                reservations: Vec::new(),
            },
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
    pub fn cancel(self) {
        self.event.cancel();
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

/// 已取得 NN window slot 的设备请求。
pub(crate) struct NnRequest<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: EvalEvent<S>,
    pub(crate) planes: InputPlanes,
    pub(crate) queued_at: S,
}
impl<S: QueueStamp> NnRequest<S> {
    pub(crate) fn new(event: EvalEvent<S>, planes: InputPlanes) -> Self {
        Self {
            event,
            planes,
            queued_at: S::default(),
        }
    }
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

/// NN worker 回交给任一 worker 的结果；其 event 持有一个 NN permit。
pub(crate) struct NnReply<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: EvalEvent<S>,
    pub(crate) result: Result<(Arc<EncodedBatch>, usize), EnginError>,
    pub(crate) queued_at: S,
}
impl<S: QueueStamp> NnReply<S> {
    pub(crate) fn new(event: EvalEvent<S>, result: Result<(Arc<EncodedBatch>, usize), EnginError>) -> Self {
        Self {
            event,
            result,
            queued_at: S::default(),
        }
    }
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

#[derive(Debug)]
pub struct BackpropEvent<S: QueueStamp = NoQueueStamp> {
    pub(crate) event: Event,
    pub(crate) value: ValueDelta,
    /// 此 event 是否持有 NN permit；Backprop 完成后释放对应 window slot。
    pub(crate) holds_nn_permit: bool,
    pub(crate) queued_at: S,
}

impl<S: QueueStamp> BackpropEvent<S> {
    pub(crate) fn with_nn_permit(event: Event, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self {
            event,
            value: ValueDelta::with_plies_left(wl, draw, plies_left),
            holds_nn_permit: true,
            queued_at: S::default(),
        }
    }
    pub(crate) fn without_nn_permit(event: Event, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self {
            event,
            value: ValueDelta::with_plies_left(wl, draw, plies_left),
            holds_nn_permit: false,
            queued_at: S::default(),
        }
    }
    pub fn cancel(self) {
        self.event.cancel();
    }
    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }
}

enum WorkerCommand<O: SearchObserver> {
    Run {
        shared: Arc<Shared<O>>,
        select_rx: Receiver<SelectEvent<O::Stamp>>,
        expand_rx: Receiver<ExpandEvent<O::Stamp>>,
        eval_rx: Receiver<EvalEvent<O::Stamp>>,
        nn_reply_rx: Receiver<NnReply<O::Stamp>>,
        nn_tx: Sender<NnRequest<O::Stamp>>,
        backprop_rx: Receiver<BackpropEvent<O::Stamp>>,
    },
    Shutdown,
}
enum NnCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<NnRequest<O::Stamp>>, Sender<NnReply<O::Stamp>>),
    Shutdown,
}

/// 固定容量的任务池；NN 另占一个设备 worker。
pub(crate) struct WorkerPool<O: SearchObserver = NoopObserver> {
    worker_commands: Vec<Sender<WorkerCommand<O>>>,
    nn_commands: Sender<NnCommand<O>>,
    job_done: Receiver<()>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    eval_batch_size: usize,
    nn_permit_limit: usize,
}

impl<O: SearchObserver> WorkerPool<O> {
    pub(crate) fn new(backend: &dyn Backend, config: &SearchConfig) -> Self {
        config.validate();
        Self::from_resolved(&config.resolve(backend))
    }
    pub(crate) fn matches_config(&self, backend: &dyn Backend, config: &SearchConfig) -> bool {
        let config = config.resolve(backend);
        self.eval_batch_size == config.eval_batch_size
            && self.nn_permit_limit == config.nn_permit_limit
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
        threads.push(thread::spawn({
            let done = job_done_tx.clone();
            move || persistent_nn_worker::<O>(nn_rx, done, batch_size)
        }));
        Self {
            worker_commands,
            nn_commands,
            job_done,
            threads: Mutex::new(threads),
            eval_batch_size: config.eval_batch_size,
            nn_permit_limit: config.nn_permit_limit,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_job(
        &self,
        shared: &Arc<Shared<O>>,
        select_rx: &Receiver<SelectEvent<O::Stamp>>,
        expand_rx: &Receiver<ExpandEvent<O::Stamp>>,
        eval_rx: &Receiver<EvalEvent<O::Stamp>>,
        nn_reply_rx: &Receiver<NnReply<O::Stamp>>,
        nn_tx: &Sender<NnRequest<O::Stamp>>,
        nn_rx: &Receiver<NnRequest<O::Stamp>>,
        nn_reply_tx: &Sender<NnReply<O::Stamp>>,
        backprop_rx: &Receiver<BackpropEvent<O::Stamp>>,
    ) {
        self.nn_commands
            .send(NnCommand::Run(Arc::clone(shared), nn_rx.clone(), nn_reply_tx.clone()))
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
                    backprop_rx: backprop_rx.clone(),
                })
                .expect("persistent worker is alive");
        }
    }
    pub(crate) fn finish_job(&self) {
        for _ in 0..self.worker_commands.len() + 1 {
            self.job_done.recv().expect("persistent worker completion");
        }
    }
    pub(crate) fn nn_permit_limit(&self) -> usize {
        self.nn_permit_limit
    }
    pub(crate) fn eval_batch_size(&self) -> usize {
        self.eval_batch_size
    }

    pub(crate) fn worker_count(&self) -> usize {
        self.worker_commands.len()
    }
    pub(crate) fn assert_compatible(&self, config: &ResolvedSearchConfig) {
        debug_assert_eq!(
            self.worker_commands.len(),
            config.threads,
            "worker pool capacity changed"
        );
        debug_assert_eq!(
            self.eval_batch_size, config.eval_batch_size,
            "worker pool batch size changed"
        );
        debug_assert_eq!(
            self.nn_permit_limit, config.nn_permit_limit,
            "worker pool nn window changed"
        );
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

fn cancel_worker_queues<O: SearchObserver>(
    shared: &Shared<O>,
    select_rx: &Receiver<SelectEvent<O::Stamp>>,
    expand_rx: &Receiver<ExpandEvent<O::Stamp>>,
    eval_rx: &Receiver<EvalEvent<O::Stamp>>,
    reply_rx: &Receiver<NnReply<O::Stamp>>,
    backprop_rx: &Receiver<BackpropEvent<O::Stamp>>,
) {
    shared.cancel_deferred_eval_events();
    while let Ok(event) = select_rx.try_recv() {
        event.cancel();
        shared.finish(1, false);
    }
    while let Ok(event) = expand_rx.try_recv() {
        shared.cancel_expansion(event.into_event());
    }
    while let Ok(job) = eval_rx.try_recv() {
        shared.cancel_expansion(job.event);
    }
    while let Ok(reply) = reply_rx.try_recv() {
        cancel_evaluation(shared, reply.event.event);
    }
    while let Ok(event) = backprop_rx.try_recv() {
        let held = usize::from(event.holds_nn_permit);
        event.cancel();
        shared.release_nn_permits(held);
        shared.finish(1, false);
    }
}

fn worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    select_rx: Receiver<SelectEvent<O::Stamp>>,
    expand_rx: Receiver<ExpandEvent<O::Stamp>>,
    eval_rx: Receiver<EvalEvent<O::Stamp>>,
    reply_rx: Receiver<NnReply<O::Stamp>>,
    nn_tx: Sender<NnRequest<O::Stamp>>,
    backprop_rx: Receiver<BackpropEvent<O::Stamp>>,
) {
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            cancel_worker_queues(&shared, &select_rx, &expand_rx, &eval_rx, &reply_rx, &backprop_rx);
            if shared.outstanding.load(Ordering::Acquire) == 0 {
                break;
            }
            thread::yield_now();
            continue;
        }
        if let Some(event) = shared.take_deferred_eval_event() {
            if let Err(error) = process_eval_event(&shared, &nn_tx, event, true) {
                shared.fail(error);
            }
            continue;
        }
        if let Ok(event) = backprop_rx.try_recv() {
            process_backprop_event(&shared, event, &backprop_rx);
            continue;
        }
        if let Ok(mut reply) = reply_rx.try_recv() {
            if O::ENABLED {
                observe_queue_wait(&mut reply.queued_at, &shared.observer, QueueKind::NnReply);
            }
            if let Err(error) = handle_nn_reply(&shared, reply) {
                shared.fail(error);
            }
            continue;
        }
        if let Ok(mut job) = eval_rx.try_recv() {
            if O::ENABLED {
                observe_queue_wait(&mut job.queued_at, &shared.observer, QueueKind::Eval);
            }
            if let Err(error) = process_eval_event(&shared, &nn_tx, job, false) {
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
            recv(backprop_rx) -> result => {
                if let Ok(event) = result { process_backprop_event(&shared, event, &backprop_rx); }
            },
            recv(reply_rx) -> result => {
                if let Ok(mut reply) = result {
                    if O::ENABLED { observe_queue_wait(&mut reply.queued_at, &shared.observer, QueueKind::NnReply); }
                    let outcome = handle_nn_reply(&shared, reply);
                    if let Err(error) = outcome { shared.fail(error); }
                }
            },
            recv(eval_rx) -> result => {
                if let Ok(mut job) = result {
                    if O::ENABLED { observe_queue_wait(&mut job.queued_at, &shared.observer, QueueKind::Eval); }
                    let outcome = process_eval_event(&shared, &nn_tx, job, false);
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
            WorkerCommand::Run {
                shared,
                select_rx,
                expand_rx,
                eval_rx,
                nn_reply_rx,
                nn_tx,
                backprop_rx,
            } => {
                worker(shared, select_rx, expand_rx, eval_rx, nn_reply_rx, nn_tx, backprop_rx);
                let _ = done.send(());
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

fn nn_worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    receiver: Receiver<NnRequest<O::Stamp>>,
    reply_tx: Sender<NnReply<O::Stamp>>,
    batch_size: usize,
) {
    loop {
        let mut first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => {
                while let Ok(request) = receiver.try_recv() {
                    send_nn_reply(
                        &reply_tx,
                        NnReply::new(request.event, Err(EnginError::PortIncomplete("stream nn stopping"))),
                    );
                }
                break;
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        if O::ENABLED {
            observe_queue_wait(&mut first.queued_at, &shared.observer, QueueKind::Nn);
        }
        let mut requests = vec![first];
        while requests.len() < batch_size {
            match receiver.try_recv() {
                Ok(mut request) => {
                    if O::ENABLED {
                        observe_queue_wait(&mut request.queued_at, &shared.observer, QueueKind::Nn);
                    }
                    requests.push(request);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        infer_nn_batch(&shared, requests, &reply_tx);
    }
}
fn persistent_nn_worker<O: SearchObserver>(commands: Receiver<NnCommand<O>>, done: Sender<()>, batch_size: usize) {
    while let Ok(command) = commands.recv() {
        match command {
            NnCommand::Run(shared, receiver, reply_tx) => {
                nn_worker(shared, receiver, reply_tx, batch_size);
                let _ = done.send(());
            }
            NnCommand::Shutdown => break,
        }
    }
}
