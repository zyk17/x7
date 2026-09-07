//! Eval 算法：cache | 已编码 NN 请求 | NN 回包发布。
//!
//! 规则终局和合法着生成属于独立 Expand task；这里不持有 worker-local 等待列表，
//! 因此任一 CPU worker 都可继续处理下一项任务。

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;

use crossbeam_channel::{Sender, TrySendError};

use crate::EnginError;
use crate::neural::backend::EvalResult;
use crate::neural::{EncodedBatch, FillEmptyHistory, encode_position_input_planes, eval_result_from_encoded_row};

use super::observer::{QueueStamp, SearchObserver};
use super::pipeline::Shared;
use super::workerpool::{BackpropEvent, EvalJob, Event, NnReply, NnRequest};

/// 处理一个 Gather 已分类的普通叶子。
///
/// `claim_held` 只会来自 deferred job：它在上次 cache miss 时已经取得了 NN slot。
pub(crate) fn handle_eval_job<O: SearchObserver>(
    shared: &Shared<O>,
    nn_tx: &Sender<NnRequest<O::Stamp>>,
    job: EvalJob<O::Stamp>,
    claim_held: bool,
) -> Result<(), EnginError> {
    if shared.stopping.load(Ordering::Acquire) {
        if claim_held {
            cancel_evaluation(shared, job.event);
        } else {
            shared.cancel_expansion(job.event);
        }
        return Ok(());
    }
    if let Some(eval) = shared.backend.cached_evaluation(job.cache_key) {
        if O::ENABLED {
            shared.observer.on_cache_hit();
        }
        // 延后期间，另一条非合并路径可能填入相同 cache key。slot 已取得，仍由
        // 一次 held-claim backprop 释放，不能在这里提前减计数。
        return publish_eval(shared, job.event, job.legal_moves, eval, claim_held);
    }
    if !claim_held && !shared.try_acquire_eval_claim() {
        shared.defer_eval(job);
        return Ok(());
    }
    let planes = encode_position_input_planes(&job.history, FillEmptyHistory::FenOnly);
    send_nn_request(shared, nn_tx, NnRequest::new(job, planes))
}

fn send_nn_request<O: SearchObserver>(
    shared: &Shared<O>,
    nn_tx: &Sender<NnRequest<O::Stamp>>,
    mut request: NnRequest<O::Stamp>,
) -> Result<(), EnginError> {
    request.mark_queued();
    loop {
        if shared.stopping.load(Ordering::Acquire) {
            cancel_evaluation(shared, request.job.event);
            return Ok(());
        }
        match nn_tx.try_send(request) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned)) => {
                request = returned;
                thread::yield_now();
            }
            Err(TrySendError::Disconnected(returned)) => {
                cancel_evaluation(shared, returned.job.event);
                return Err(EnginError::PortIncomplete("stream nn queue disconnected"));
            }
        }
    }
}

pub(crate) fn handle_nn_reply<O: SearchObserver>(
    shared: &Shared<O>,
    reply: NnReply<O::Stamp>,
) -> Result<(), EnginError> {
    match reply.result {
        Ok((batch, row)) => complete_nn_item(shared, reply.job, batch, row),
        Err(error) => {
            cancel_evaluation(shared, reply.job.event);
            Err(error)
        }
    }
}

fn complete_nn_item<O: SearchObserver>(
    shared: &Shared<O>,
    job: EvalJob<O::Stamp>,
    batch: Arc<EncodedBatch>,
    row: usize,
) -> Result<(), EnginError> {
    let eval = match eval_result_from_encoded_row(&batch, row, &job.legal_moves) {
        Ok(eval) => eval,
        Err(error) => {
            cancel_evaluation(shared, job.event);
            return Err(error);
        }
    };
    shared.backend.store_evaluation(job.cache_key, Arc::clone(&eval));
    publish_eval(shared, job.event, job.legal_moves, eval, true)
}

fn publish_eval<O: SearchObserver>(
    shared: &Shared<O>,
    event: Event,
    legal_moves: xiangqi_core::LegalMoveList,
    eval: Arc<EvalResult>,
    held_claim: bool,
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
        if held_claim {
            cancel_evaluation(shared, event);
        } else {
            shared.cancel_expansion(event);
        }
        return Err(EnginError::Onnx("stream backend evaluation is invalid".into()));
    }
    shared
        .arena
        .get(event.node_id)
        .expect("eval node lives until job drain")
        .publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()));
    let backprop = if held_claim {
        BackpropEvent::from_eval(event, -eval.wl, eval.d, eval.plies_left)
    } else {
        BackpropEvent::without_eval_claim(event, -eval.wl, eval.d, eval.plies_left)
    };
    shared.send_backprop(backprop);
    Ok(())
}

/// 取消一个已占 NN slot 的 evaluation。
pub(crate) fn cancel_evaluation<O: SearchObserver>(shared: &Shared<O>, event: Event) {
    shared.release_eval_claims(1);
    shared.cancel_expansion(event);
}

/// 合批推理一批已编码请求；结果回交通用 CPU reply 队列，不阻塞提交它的 worker。
pub(crate) fn infer_nn_batch<O: SearchObserver>(
    shared: &Shared<O>,
    requests: Vec<NnRequest<O::Stamp>>,
    reply_tx: &Sender<NnReply<O::Stamp>>,
) {
    if requests.is_empty() {
        return;
    }
    if shared.stopping.load(Ordering::Acquire) {
        reject_nn_requests(requests, EnginError::PortIncomplete("stream nn stopping"), reply_tx);
        return;
    }
    let batch = requests.len();
    let samples: Vec<_> = requests.iter().map(|request| request.planes).collect();
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
                reject_nn_requests(requests, error, reply_tx);
                return;
            }
            shared.network_evaluations.fetch_add(batch as u64, Ordering::AcqRel);
            if O::ENABLED {
                shared.observer.on_batch(batch);
            }
            let output = Arc::new(output);
            for (row, request) in requests.into_iter().enumerate() {
                let mut reply = NnReply::new(request.job, Ok((Arc::clone(&output), row)));
                reply.mark_queued();
                let _ = reply_tx.send(reply);
            }
        }
        Err(error) => reject_nn_requests(requests, error, reply_tx),
    }
}

fn reject_nn_requests<S: QueueStamp>(requests: Vec<NnRequest<S>>, error: EnginError, reply_tx: &Sender<NnReply<S>>) {
    for request in requests {
        let mut reply = NnReply::new(request.job, Err(error.clone()));
        reply.mark_queued();
        let _ = reply_tx.send(reply);
    }
}
