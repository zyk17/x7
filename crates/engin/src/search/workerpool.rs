//! 事件定义 + WorkerPool + Gather/Eval/NN/Backprop 线程循环壳。
//!
//! 循环只调度；算法在 `pipeline`（Gather 树走）/ `eval` / `backprop`。

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, unbounded};
use parking_lot::Mutex;
use xiangqi_core::{Move, Position, PositionHistory};

use crate::neural::backend::Backend;
use crate::search::EdgeReservation;

use super::backprop::complete_batch;
use super::eval::{
    NnRequest, WaitingNn, drain_waiting, handle_eval_event, infer_nn_batch, poll_nn_completions, wait_one_nn_completion,
};
use super::observer::{NoQueueStamp, NoopObserver, QueueKind, QueueStamp, SearchObserver};
use super::param::{ResolvedSearchConfig, SearchConfig};
use super::pipeline::{RECEIVE_POLL, Shared, process_gather_event};
use super::{NodeId, ValueDelta};

// --- Event -------------------------------------------------------------------

/// root history 加上从 root 到 repository node 的走法。
#[derive(Clone, Debug)]
pub struct Variation {
    root_history: Arc<PositionHistory>,
    moves: Vec<Move>,
    history: Option<PositionHistory>,
    position: Position,
}

impl Variation {
    pub fn root(root_history: Arc<PositionHistory>) -> Self {
        Self {
            position: root_history.last().clone(),
            root_history,
            moves: Vec::new(),
            history: None,
        }
    }

    pub fn root_history(&self) -> &Arc<PositionHistory> {
        &self.root_history
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    pub(crate) fn history(&mut self) -> &PositionHistory {
        if self.history.is_none() {
            let mut history = self.root_history.as_ref().clone();
            for &mv in &self.moves {
                history.append(mv);
            }
            self.history = Some(history);
        }
        self.history.as_ref().expect("variation history is initialized")
    }

    pub fn push(&mut self, mv: Move) {
        self.position = Position::after(&self.position, mv);
        self.moves.push(mv);
        if let Some(history) = self.history.as_mut() {
            history.append(mv);
        }
    }
}

/// 一次完整 playout：从 root 到 leaf 的路径、reservation 与 variation 上下文。
#[derive(Debug)]
pub struct PlayoutEvent<S: QueueStamp = NoQueueStamp> {
    pub generation: u64,
    pub node_id: NodeId,
    pub(crate) node_path: Vec<NodeId>,
    pub variation: Variation,
    pub reservations: Vec<EdgeReservation>,
    queued_at: S,
}

impl<S: QueueStamp> PlayoutEvent<S> {
    pub fn at_root(generation: u64, root_id: NodeId, root_history: Arc<PositionHistory>) -> Self {
        Self {
            generation,
            node_id: root_id,
            node_path: vec![root_id],
            variation: Variation::root(root_history),
            reservations: Vec::new(),
            queued_at: S::default(),
        }
    }

    pub fn descend(mut self, child_id: NodeId, reservation: EdgeReservation) -> Self {
        self.variation.push(reservation.mv());
        self.node_id = child_id;
        self.node_path.push(child_id);
        self.reservations.push(reservation);
        self
    }

    pub fn cancel(self) {
        for reservation in self.reservations.into_iter().rev() {
            reservation.cancel();
        }
    }

    pub fn node_path(&self) -> &[NodeId] {
        &self.node_path
    }

    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }

    pub(crate) fn observe_wait<O: SearchObserver<Stamp = S>>(&mut self, obs: &O, kind: QueueKind) {
        if let Some(wait) = self.queued_at.take_wait() {
            obs.on_queue_wait(kind, wait);
        }
    }
}

/// 由 Gather/Eval 路由给 Backprop 的结果（算法在 `backprop::complete_batch`）。
#[derive(Debug)]
pub struct BackpropEvent<S: QueueStamp = NoQueueStamp> {
    pub(crate) playout: PlayoutEvent<S>,
    pub(crate) value: ValueDelta,
    queued_at: S,
}

impl<S: QueueStamp> BackpropEvent<S> {
    pub(crate) fn evaluation(playout: PlayoutEvent<S>, wl: f32, draw: f32, plies_left: f32) -> Self {
        Self {
            playout,
            value: ValueDelta::with_plies_left(wl, draw, plies_left),
            queued_at: S::default(),
        }
    }

    pub fn cancel(self) {
        self.playout.cancel();
    }

    pub(crate) fn mark_queued(&mut self) {
        self.queued_at.mark();
    }

    pub(crate) fn observe_wait<O: SearchObserver<Stamp = S>>(&mut self, obs: &O, kind: QueueKind) {
        if let Some(wait) = self.queued_at.take_wait() {
            obs.on_queue_wait(kind, wait);
        }
    }
}

// --- Pool --------------------------------------------------------------------

enum GatherCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<PlayoutEvent<O::Stamp>>),
    Shutdown,
}

enum EvalCommand<O: SearchObserver> {
    Run(
        Arc<Shared<O>>,
        Receiver<PlayoutEvent<O::Stamp>>,
        Sender<NnRequest<O::Stamp>>,
    ),
    Shutdown,
}

enum NnCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<NnRequest<O::Stamp>>),
    Shutdown,
}

enum BackpropCommand<O: SearchObserver> {
    Run(Arc<Shared<O>>, Receiver<BackpropEvent<O::Stamp>>),
    Shutdown,
}

/// Engine 持有的固定 worker 拓扑。
///
/// 每个 job 独占树视图、队列与 generation；线程池只跨 job 保留线程。
pub(crate) struct WorkerPool<O: SearchObserver = NoopObserver> {
    gather_commands: Vec<Sender<GatherCommand<O>>>,
    eval_commands: Vec<Sender<EvalCommand<O>>>,
    nn_commands: Sender<NnCommand<O>>,
    backprop_commands: Vec<Sender<BackpropCommand<O>>>,
    job_done: Receiver<()>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    eval_batch_size: usize,
    eval_claim_limit: usize,
}

impl<O: SearchObserver> WorkerPool<O> {
    pub(crate) fn new(backend: &dyn Backend, config: &SearchConfig) -> Self {
        config.validate();
        Self::from_resolved(&config.resolve(backend))
    }

    /// 对齐 LC3 Overview 的固定 worker job：batch 或任一 worker 数改变时，必须
    /// 使用相应拓扑的新 pool。
    pub(crate) fn matches_config(&self, backend: &dyn Backend, config: &SearchConfig) -> bool {
        let config = config.resolve(backend);
        self.eval_batch_size == config.eval_batch_size
            && self.eval_claim_limit == config.eval_claim_limit
            && self.gather_commands.len() == config.gather_workers
            && self.eval_commands.len() == config.eval_workers
    }

    fn from_resolved(config: &ResolvedSearchConfig) -> Self {
        let (job_done_tx, job_done) = unbounded();
        let eval_batch_size = config.eval_batch_size;
        let eval_claim_limit = config.eval_claim_limit;
        let mut gather_commands = Vec::with_capacity(config.gather_workers);
        let mut eval_commands = Vec::with_capacity(config.eval_workers);
        let (nn_commands, nn_rx) = unbounded();
        let (backprop_commands, backprop_rx) = unbounded();
        let mut threads = Vec::with_capacity(config.gather_workers + config.eval_workers + 2);
        for _ in 0..config.gather_workers {
            let (tx, rx) = unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || {
                persistent_gather_worker::<O>(rx, job_done, eval_claim_limit)
            }));
            gather_commands.push(tx);
        }
        for _ in 0..config.eval_workers {
            let (tx, rx) = unbounded();
            let job_done = job_done_tx.clone();
            threads.push(thread::spawn(move || persistent_eval_worker::<O>(rx, job_done)));
            eval_commands.push(tx);
        }
        threads.push(thread::spawn({
            let job_done = job_done_tx.clone();
            move || persistent_nn_worker::<O>(nn_rx, job_done, eval_batch_size)
        }));
        threads.push(thread::spawn({
            let job_done = job_done_tx.clone();
            move || persistent_backprop_worker::<O>(backprop_rx, job_done)
        }));
        Self {
            gather_commands,
            eval_commands,
            nn_commands,
            backprop_commands: vec![backprop_commands],
            job_done,
            threads: Mutex::new(threads),
            eval_batch_size: config.eval_batch_size,
            eval_claim_limit: config.eval_claim_limit,
        }
    }

    pub(crate) fn start_job(
        &self,
        shared: &Arc<Shared<O>>,
        gather_rx: &Receiver<PlayoutEvent<O::Stamp>>,
        eval_rx: &Receiver<PlayoutEvent<O::Stamp>>,
        nn_tx: &Sender<NnRequest<O::Stamp>>,
        nn_rx: &Receiver<NnRequest<O::Stamp>>,
        backprop_rx: &Receiver<BackpropEvent<O::Stamp>>,
    ) {
        for sender in &self.gather_commands {
            sender
                .send(GatherCommand::Run(Arc::clone(shared), gather_rx.clone()))
                .expect("persistent search worker is alive");
        }
        self.nn_commands
            .send(NnCommand::Run(Arc::clone(shared), nn_rx.clone()))
            .expect("persistent nn worker is alive");
        for sender in &self.eval_commands {
            sender
                .send(EvalCommand::Run(Arc::clone(shared), eval_rx.clone(), nn_tx.clone()))
                .expect("persistent eval worker is alive");
        }
        for sender in &self.backprop_commands {
            sender
                .send(BackpropCommand::Run(Arc::clone(shared), backprop_rx.clone()))
                .expect("persistent backprop worker is alive");
        }
    }

    pub(crate) fn finish_job(&self) {
        for _ in 0..self.gather_commands.len() + self.eval_commands.len() + self.backprop_commands.len() + 1 {
            self.job_done.recv().expect("persistent worker completion");
        }
    }

    pub(crate) fn eval_claim_limit(&self) -> usize {
        self.eval_claim_limit
    }

    pub(crate) fn eval_batch_size(&self) -> usize {
        self.eval_batch_size
    }

    pub(crate) fn assert_compatible(&self, config: &ResolvedSearchConfig) {
        assert_eq!(
            self.gather_commands.len(),
            config.gather_workers,
            "worker pool gather topology changed"
        );
        assert_eq!(
            self.eval_commands.len(),
            config.eval_workers,
            "worker pool eval topology changed"
        );
        assert_eq!(
            self.eval_batch_size, config.eval_batch_size,
            "worker pool batch size changed"
        );
        assert_eq!(
            self.eval_claim_limit, config.eval_claim_limit,
            "worker pool nn window changed"
        );
    }
}

impl<O: SearchObserver> Drop for WorkerPool<O> {
    fn drop(&mut self) {
        for sender in &self.gather_commands {
            let _ = sender.send(GatherCommand::Shutdown);
        }
        let _ = self.nn_commands.send(NnCommand::Shutdown);
        for sender in &self.eval_commands {
            let _ = sender.send(EvalCommand::Shutdown);
        }
        for sender in &self.backprop_commands {
            let _ = sender.send(BackpropCommand::Shutdown);
        }
        for worker in self.threads.get_mut().drain(..) {
            let _ = worker.join();
        }
    }
}

// --- Gather / Eval / NN / Backprop loops

fn gather_worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    receiver: Receiver<PlayoutEvent<O::Stamp>>,
    eval_claim_limit: usize,
) {
    loop {
        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(mut event) => {
                event.observe_wait(&shared.observer, QueueKind::Gather);
                if shared.stopping.load(Ordering::Acquire) {
                    event.cancel();
                    shared.finish(1, false);
                } else {
                    process_gather_event(&shared, event, eval_claim_limit);
                }
            }
            Err(RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn persistent_gather_worker<O: SearchObserver>(
    commands: Receiver<GatherCommand<O>>,
    job_done: Sender<()>,
    eval_claim_limit: usize,
) {
    while let Ok(command) = commands.recv() {
        match command {
            GatherCommand::Run(shared, receiver) => {
                gather_worker(shared, receiver, eval_claim_limit);
                let _ = job_done.send(());
            }
            GatherCommand::Shutdown => break,
        }
    }
}

fn eval_worker<O: SearchObserver>(
    shared: Arc<Shared<O>>,
    receiver: Receiver<PlayoutEvent<O::Stamp>>,
    nn_tx: Sender<NnRequest<O::Stamp>>,
) {
    let mut waiting: Vec<WaitingNn<O::Stamp>> = Vec::new();
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            drain_waiting(&shared, &mut waiting);
            while let Ok(event) = receiver.try_recv() {
                shared.release_eval_claim();
                shared.cancel_claimed_evaluation(event);
            }
            shared.cancel_all_collisions();
            break;
        }

        poll_nn_completions(&shared, &mut waiting);

        match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(mut event) => {
                event.observe_wait(&shared.observer, QueueKind::Eval);
                if let Err(error) = handle_eval_event(&shared, &nn_tx, &mut waiting, event) {
                    shared.fail(error);
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if waiting.is_empty() || shared.stopping.load(Ordering::Acquire) {
                    continue;
                }
                wait_one_nn_completion(&shared, &mut waiting);
            }
            Err(RecvTimeoutError::Disconnected) => {
                drain_waiting(&shared, &mut waiting);
                while let Ok(event) = receiver.try_recv() {
                    shared.release_eval_claim();
                    shared.cancel_claimed_evaluation(event);
                }
                shared.cancel_all_collisions();
                break;
            }
        }
    }
}

fn persistent_eval_worker<O: SearchObserver>(commands: Receiver<EvalCommand<O>>, job_done: Sender<()>) {
    while let Ok(command) = commands.recv() {
        match command {
            EvalCommand::Run(shared, receiver, nn_tx) => {
                eval_worker(shared, receiver, nn_tx);
                let _ = job_done.send(());
            }
            EvalCommand::Shutdown => break,
        }
    }
}

fn nn_worker<O: SearchObserver>(shared: Arc<Shared<O>>, receiver: Receiver<NnRequest<O::Stamp>>, batch_size: usize) {
    loop {
        let mut first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(request) => request,
            Err(RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        first.observe_wait(&shared.observer, QueueKind::Nn);
        let mut requests = vec![first];
        while requests.len() < batch_size {
            match receiver.try_recv() {
                Ok(mut request) => {
                    request.observe_wait(&shared.observer, QueueKind::Nn);
                    requests.push(request);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        infer_nn_batch(&shared, requests);
    }
}

fn persistent_nn_worker<O: SearchObserver>(commands: Receiver<NnCommand<O>>, job_done: Sender<()>, batch_size: usize) {
    while let Ok(command) = commands.recv() {
        match command {
            NnCommand::Run(shared, receiver) => {
                nn_worker(shared, receiver, batch_size);
                let _ = job_done.send(());
            }
            NnCommand::Shutdown => break,
        }
    }
}

fn backprop_worker<O: SearchObserver>(shared: Arc<Shared<O>>, receiver: Receiver<BackpropEvent<O::Stamp>>) {
    loop {
        let first = match receiver.recv_timeout(RECEIVE_POLL) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) if shared.stopping.load(Ordering::Acquire) => break,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let mut events = Vec::with_capacity(1 + receiver.len());
        events.push(first);
        events.extend(receiver.try_iter());
        if events.is_empty() {
            continue;
        }
        if shared.stopping.load(Ordering::Acquire) {
            let n = events.len();
            for event in events {
                event.cancel();
            }
            shared.finish(n, false);
            continue;
        }
        for event in &mut events {
            event.observe_wait(&shared.observer, QueueKind::Backprop);
        }
        let leaf_ids: Vec<NodeId> = events.iter().map(|event| event.playout.node_id).collect();
        let result = complete_batch(events, &shared.arena);
        for id in leaf_ids {
            shared.cancel_collisions(id);
        }
        shared
            .completed_depth
            .fetch_add(result.completed_depth, Ordering::AcqRel);
        shared.max_depth.fetch_max(result.max_depth, Ordering::AcqRel);
        shared.finish(result.completed_playouts as usize, true);
    }
}

fn persistent_backprop_worker<O: SearchObserver>(commands: Receiver<BackpropCommand<O>>, job_done: Sender<()>) {
    while let Ok(command) = commands.recv() {
        match command {
            BackpropCommand::Run(shared, receiver) => {
                backprop_worker(shared, receiver);
                let _ = job_done.send(());
            }
            BackpropCommand::Shutdown => break,
        }
    }
}
