//! Eval 算法：叶子终局 | cache | 编码 | 发边 | NN 合批推理。
//!
//! MCTS 叶子侧实验改这里。worker 循环壳在 `workerpool`。

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError, bounded};
use xiangqi_core::LegalMoveList;

use crate::EnginError;
use crate::neural::backend::{EvalCacheKey, EvalResult};
use crate::neural::{
    EncodedBatch, FillEmptyHistory, InputPlanes, encode_position_input_planes, eval_result_from_encoded_row,
};

use super::expand::{ExpandKind, classify_expand};
use super::observer::{QueueKind, QueueStamp, SearchObserver};
use super::pipeline::{RECEIVE_POLL, Shared};
use super::workerpool::{BackpropEvent, PlayoutEvent};

type NnReply = Result<(Arc<EncodedBatch>, usize), EnginError>;

/// 交给 NN 线程的一个已编码局面（稀疏 InputPlanes；ORT 前再 expand）。
pub(crate) struct NnRequest<S: QueueStamp = super::observer::NoQueueStamp> {
    planes: InputPlanes,
    reply: Sender<NnReply>,
    queued_at: S,
}

impl<S: QueueStamp> NnRequest<S> {
    fn mark_queued(&mut self) {
        self.queued_at.mark();
    }

    pub(crate) fn observe_wait<O: SearchObserver<Stamp = S>>(&mut self, obs: &O, kind: QueueKind) {
        if let Some(wait) = self.queued_at.take_wait() {
            obs.on_queue_wait(kind, wait);
        }
    }
}

/// Eval 正在等待此 node 的 NN。
pub(crate) struct WaitingNn<S: QueueStamp = super::observer::NoQueueStamp> {
    event: PlayoutEvent<S>,
    legal_moves: LegalMoveList,
    cache_key: EvalCacheKey,
    reply: Receiver<NnReply>,
}

pub(crate) fn poll_nn_completions<O: SearchObserver>(shared: &Shared<O>, waiting: &mut Vec<WaitingNn<O::Stamp>>) {
    let mut i = 0;
    while i < waiting.len() {
        match waiting[i].reply.try_recv() {
            Ok(Ok((batch, row))) => {
                let item = waiting.swap_remove(i);
                if let Err(error) = complete_nn_item(shared, item, batch, row) {
                    shared.fail(error);
                    return;
                }
            }
            Ok(Err(error)) => {
                let item = waiting.swap_remove(i);
                cancel_evaluation(shared, item.event);
                if !shared.stopping.load(Ordering::Acquire) {
                    shared.fail(error);
                }
                return;
            }
            Err(TryRecvError::Empty) => i += 1,
            Err(TryRecvError::Disconnected) => {
                let item = waiting.swap_remove(i);
                cancel_evaluation(shared, item.event);
                if !shared.stopping.load(Ordering::Acquire) {
                    shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
                }
                return;
            }
        }
    }
}

pub(crate) fn wait_one_nn_completion<O: SearchObserver>(shared: &Shared<O>, waiting: &mut Vec<WaitingNn<O::Stamp>>) {
    if waiting.is_empty() || shared.stopping.load(Ordering::Acquire) {
        return;
    }
    match waiting[0].reply.recv_timeout(RECEIVE_POLL) {
        Ok(Ok((batch, row))) => {
            let item = waiting.remove(0);
            if let Err(error) = complete_nn_item(shared, item, batch, row) {
                shared.fail(error);
            }
        }
        Ok(Err(error)) => {
            let item = waiting.remove(0);
            cancel_evaluation(shared, item.event);
            if !shared.stopping.load(Ordering::Acquire) {
                shared.fail(error);
            }
        }
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
            let item = waiting.remove(0);
            cancel_evaluation(shared, item.event);
            if !shared.stopping.load(Ordering::Acquire) {
                shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
            }
        }
    }
}

pub(crate) fn drain_waiting<O: SearchObserver>(shared: &Shared<O>, waiting: &mut Vec<WaitingNn<O::Stamp>>) {
    for item in waiting.drain(..) {
        cancel_evaluation(shared, item.event);
    }
}

/// 处理一个 Gather claim 来的叶子：终局 / cache / 排队 NN。
pub(crate) fn handle_eval_event<O: SearchObserver>(
    shared: &Shared<O>,
    nn_tx: &Sender<NnRequest<O::Stamp>>,
    waiting: &mut Vec<WaitingNn<O::Stamp>>,
    mut event: PlayoutEvent<O::Stamp>,
) -> Result<(), EnginError> {
    if shared.stopping.load(Ordering::Acquire) {
        cancel_evaluation(shared, event);
        return Ok(());
    }
    let node = shared
        .arena
        .get(event.node_id)
        .expect("eval node lives until job drain");
    let depth = event.variation.moves().len();
    let history = event.variation.history();
    match classify_expand(history, depth) {
        ExpandKind::Terminal { wl, draw, plies_left } => {
            node.mark_terminal(wl, draw, plies_left);
            let root = event.node_path()[0];
            shared.arena.propagate_proven_terminals(event.node_path(), root);
            shared.send_backprop(BackpropEvent::from_eval(event, wl, draw, plies_left));
            Ok(())
        }
        ExpandKind::Evaluate { legal_moves } => {
            let cache_key = EvalCacheKey::new(history.last(), legal_moves.len());
            if let Some(eval) = shared.backend.cached_evaluation(cache_key) {
                if O::ENABLED {
                    shared.observer.on_cache_hit();
                }
                return publish_eval(shared, event, legal_moves, eval);
            }
            let planes = encode_position_input_planes(history, FillEmptyHistory::FenOnly);
            let (reply_tx, reply_rx) = bounded(1);
            if let Err(error) = send_nn_request(
                shared,
                nn_tx,
                NnRequest {
                    planes,
                    reply: reply_tx,
                    queued_at: Default::default(),
                },
            ) {
                cancel_evaluation(shared, event);
                return if shared.stopping.load(Ordering::Acquire) {
                    Ok(())
                } else {
                    Err(error)
                };
            }
            waiting.push(WaitingNn {
                event,
                legal_moves,
                cache_key,
                reply: reply_rx,
            });
            Ok(())
        }
    }
}

fn send_nn_request<O: SearchObserver>(
    shared: &Shared<O>,
    nn_tx: &Sender<NnRequest<O::Stamp>>,
    mut request: NnRequest<O::Stamp>,
) -> Result<(), EnginError> {
    request.mark_queued();
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            return Err(EnginError::PortIncomplete("stream nn stopping"));
        }
        match nn_tx.try_send(request) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned)) => {
                request = returned;
                thread::yield_now();
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(EnginError::PortIncomplete("stream nn queue disconnected"));
            }
        }
    }
}

fn complete_nn_item<O: SearchObserver>(
    shared: &Shared<O>,
    item: WaitingNn<O::Stamp>,
    batch: Arc<EncodedBatch>,
    row: usize,
) -> Result<(), EnginError> {
    let eval = match eval_result_from_encoded_row(&batch, row, &item.legal_moves) {
        Ok(eval) => eval,
        Err(error) => {
            cancel_evaluation(shared, item.event);
            return Err(error);
        }
    };
    shared.backend.store_evaluation(item.cache_key, Arc::clone(&eval));
    publish_eval(shared, item.event, item.legal_moves, eval)
}

fn publish_eval<O: SearchObserver>(
    shared: &Shared<O>,
    event: PlayoutEvent<O::Stamp>,
    legal_moves: LegalMoveList,
    eval: Arc<EvalResult>,
) -> Result<(), EnginError> {
    let value_is_valid = eval.wl.is_finite()
        && eval.d.is_finite()
        && (0.0..=1.0).contains(&eval.d)
        && eval.wl.abs() <= 1.0 - eval.d + f32::EPSILON
        && eval.plies_left.is_finite()
        && eval.plies_left >= 0.0;
    let policy_sum: f32 = eval.policies.iter().sum();
    let policy_is_valid = eval.policies.len() == legal_moves.len()
        && eval.policies.iter().all(|policy| policy.is_finite() && *policy >= 0.0)
        && policy_sum.is_finite()
        && (policy_sum - 1.0).abs() <= 1e-3;
    if !value_is_valid || !policy_is_valid {
        cancel_evaluation(shared, event);
        return Err(EnginError::Onnx("stream backend evaluation is invalid".into()));
    }
    shared
        .arena
        .get(event.node_id)
        .expect("eval node lives until job drain")
        .publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()));
    shared.send_backprop(BackpropEvent::from_eval(event, -eval.wl, eval.d, eval.plies_left));
    Ok(())
}

/// 释放已 claim 但不会发布结果的 evaluation event。
pub(crate) fn cancel_evaluation<O: SearchObserver>(shared: &Shared<O>, event: PlayoutEvent<O::Stamp>) {
    shared.release_eval_claims(1);
    let id = event.node_id;
    event.cancel();
    if let Some(node) = shared.arena.get(id) {
        node.abort_evaluation();
    }
    shared.cancel_collisions(id);
    shared.finish(1, false);
}

/// 合批推理一批已编码请求（不含取队列循环）。
pub(crate) fn infer_nn_batch<O: SearchObserver>(shared: &Shared<O>, requests: Vec<NnRequest<O::Stamp>>) {
    if requests.is_empty() {
        return;
    }
    if shared.stopping.load(Ordering::Acquire) {
        reject_nn_requests(requests, EnginError::PortIncomplete("stream nn stopping"));
        return;
    }
    let batch = requests.len();
    let mut samples = Vec::with_capacity(batch);
    for request in &requests {
        samples.push(request.planes);
    }
    let mut logits = Vec::new();
    let mut wdl = Vec::new();
    let mut moves_left = Vec::new();
    match shared
        .backend
        .infer_input_planes_into(&samples, &mut logits, &mut wdl, &mut moves_left)
    {
        Ok(()) => {
            let output = EncodedBatch::take_from(&mut logits, &mut wdl, &mut moves_left);
            if let Err(error) = output.ensure_batch_len(batch) {
                reject_nn_requests(requests, error);
                return;
            }
            shared.network_evaluations.fetch_add(batch as u64, Ordering::AcqRel);
            if O::ENABLED {
                shared.observer.on_batch(batch);
            }
            let output = Arc::new(output);
            for (row, request) in requests.into_iter().enumerate() {
                let _ = request.reply.send(Ok((Arc::clone(&output), row)));
            }
        }
        Err(error) => reject_nn_requests(requests, error),
    }
}

fn reject_nn_requests<S: QueueStamp>(requests: Vec<NnRequest<S>>, error: EnginError) {
    for request in requests {
        let _ = request.reply.send(Err(error.clone()));
    }
}
