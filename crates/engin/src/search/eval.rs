//! Eval 算法：叶子终局 | cache | 编码 | 发边 | NN 合批推理。
//!
//! MCTS 叶子侧实验改这里。worker 循环壳在 `workerpool`。

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
#[cfg(feature = "benchmark")]
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError, bounded};
use xiangqi_core::Move;

use crate::EnginError;
use crate::neural::backend::{EvalCacheKey, EvalResult};
use crate::neural::{
    EncodedBatch, FillEmptyHistory, InputPlanes, encode_position_input_planes, eval_result_from_encoded_row,
};

use super::Node;
use super::expand::{ExpandKind, classify_expand};
use super::pipeline::{RECEIVE_POLL, Shared};
use super::workerpool::{BackpropEvent, PlayoutEvent};

type NnReply = Result<(Arc<EncodedBatch>, usize), EnginError>;

/// 交给 NN 线程的一个已编码局面（稀疏 InputPlanes；ORT 前再 expand）。
pub(crate) struct NnRequest {
    planes: InputPlanes,
    reply: Sender<NnReply>,
    #[cfg(feature = "benchmark")]
    queued_at: Option<Instant>,
}

impl NnRequest {
    #[cfg(feature = "benchmark")]
    fn mark_queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    #[cfg(feature = "benchmark")]
    pub(crate) fn take_queue_wait(&mut self) -> Option<Duration> {
        self.queued_at.take().map(|queued_at| queued_at.elapsed())
    }
}

/// Eval 正在等待此 node 的 NN。
pub(crate) struct WaitingNn {
    event: PlayoutEvent,
    node: Arc<Node>,
    legal_moves: Vec<Move>,
    cache_key: EvalCacheKey,
    reply: Receiver<NnReply>,
}

pub(crate) fn poll_nn_completions(shared: &Shared, waiting: &mut Vec<WaitingNn>) {
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
                cancel_waiting_item(shared, item);
                if !shared.stopping.load(Ordering::Acquire) {
                    shared.fail(error);
                }
                return;
            }
            Err(TryRecvError::Empty) => i += 1,
            Err(TryRecvError::Disconnected) => {
                let item = waiting.swap_remove(i);
                cancel_waiting_item(shared, item);
                if !shared.stopping.load(Ordering::Acquire) {
                    shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
                }
                return;
            }
        }
    }
}

pub(crate) fn wait_one_nn_completion(shared: &Shared, waiting: &mut Vec<WaitingNn>) {
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
            cancel_waiting_item(shared, item);
            if !shared.stopping.load(Ordering::Acquire) {
                shared.fail(error);
            }
        }
        Err(RecvTimeoutError::Timeout) => {}
        Err(RecvTimeoutError::Disconnected) => {
            let item = waiting.remove(0);
            cancel_waiting_item(shared, item);
            if !shared.stopping.load(Ordering::Acquire) {
                shared.fail(EnginError::PortIncomplete("stream nn reply disconnected"));
            }
        }
    }
}

pub(crate) fn drain_waiting(shared: &Shared, waiting: &mut Vec<WaitingNn>) {
    for item in waiting.drain(..) {
        cancel_waiting_item(shared, item);
    }
}

/// 处理一个 Gather claim 来的叶子：终局 / cache / 排队 NN。
pub(crate) fn handle_eval_event(
    shared: &Shared,
    nn_tx: &Sender<NnRequest>,
    waiting: &mut Vec<WaitingNn>,
    mut event: PlayoutEvent,
) -> Result<(), EnginError> {
    if shared.stopping.load(Ordering::Acquire) {
        shared.release_eval_claim();
        shared.cancel_claimed_evaluation(event);
        return Ok(());
    }
    let node = shared.repository.get_or_insert(event.node_key);
    let depth = event.variation.moves().len();
    let history = event.variation.history();
    match classify_expand(history, depth) {
        ExpandKind::Terminal { wl, draw, plies_left } => {
            node.mark_terminal(wl, draw, plies_left);
            let root = event.node_path()[0];
            shared.repository.propagate_proven_terminals(event.node_path(), root);
            shared.release_eval_claim();
            shared.send_backprop(BackpropEvent::evaluation(event, wl, draw, plies_left));
            Ok(())
        }
        ExpandKind::Evaluate { legal_moves } => {
            let cache_key = EvalCacheKey::new(history.last(), legal_moves.len());
            if let Some(eval) = shared.backend.cached_evaluation(cache_key) {
                shared.cache_hits.fetch_add(1, Ordering::AcqRel);
                shared.release_eval_claim();
                return publish_eval(shared, event, node, legal_moves, eval);
            }
            let planes = encode_position_input_planes(history, FillEmptyHistory::FenOnly);
            let (reply_tx, reply_rx) = bounded(1);
            if let Err(error) = send_nn_request(
                shared,
                nn_tx,
                NnRequest {
                    planes,
                    reply: reply_tx,
                    #[cfg(feature = "benchmark")]
                    queued_at: None,
                },
            ) {
                cancel_evaluation(shared, event, node);
                shared.release_eval_claim();
                return if shared.stopping.load(Ordering::Acquire) {
                    Ok(())
                } else {
                    Err(error)
                };
            }
            waiting.push(WaitingNn {
                event,
                node,
                legal_moves,
                cache_key,
                reply: reply_rx,
            });
            Ok(())
        }
    }
}

fn send_nn_request(shared: &Shared, nn_tx: &Sender<NnRequest>, mut request: NnRequest) -> Result<(), EnginError> {
    #[cfg(feature = "benchmark")]
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

fn complete_nn_item(shared: &Shared, item: WaitingNn, batch: Arc<EncodedBatch>, row: usize) -> Result<(), EnginError> {
    shared.release_eval_claim();
    let eval = match eval_result_from_encoded_row(&batch, row, &item.legal_moves) {
        Ok(eval) => eval,
        Err(error) => {
            cancel_evaluation(shared, item.event, item.node);
            return Err(error);
        }
    };
    shared.backend.store_evaluation(item.cache_key, Arc::clone(&eval));
    publish_eval(shared, item.event, item.node, item.legal_moves, eval)
}

fn publish_eval(
    shared: &Shared,
    event: PlayoutEvent,
    node: Arc<Node>,
    legal_moves: Vec<xiangqi_core::Move>,
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
        cancel_evaluation(shared, event, node);
        return Err(EnginError::Onnx("stream backend evaluation is invalid".into()));
    }
    node.publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()).collect());
    shared.send_backprop(BackpropEvent::evaluation(event, -eval.wl, eval.d, eval.plies_left));
    Ok(())
}

fn cancel_waiting_item(shared: &Shared, item: WaitingNn) {
    shared.release_eval_claim();
    cancel_evaluation(shared, item.event, item.node);
}

/// 释放已 claim 但不会发布结果的 evaluation event。
pub(crate) fn cancel_evaluation(shared: &Shared, event: PlayoutEvent, node: Arc<Node>) {
    let key = event.node_key;
    event.cancel();
    node.abort_evaluation();
    shared.cancel_collisions(key);
    shared.finish(1, false);
}

/// 合批推理一批已编码请求（不含取队列循环）。
pub(crate) fn infer_nn_batch(shared: &Shared, requests: Vec<NnRequest>) {
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
            shared.network_batches.fetch_add(1, Ordering::AcqRel);
            shared.network_evaluations.fetch_add(batch as u64, Ordering::AcqRel);
            shared.network_batch_size_max.fetch_max(batch as u64, Ordering::AcqRel);
            let output = Arc::new(output);
            for (row, request) in requests.into_iter().enumerate() {
                let _ = request.reply.send(Ok((Arc::clone(&output), row)));
            }
        }
        Err(error) => reject_nn_requests(requests, error),
    }
}

fn reject_nn_requests(requests: Vec<NnRequest>, error: EnginError) {
    for request in requests {
        let _ = request.reply.send(Err(error.clone()));
    }
}
