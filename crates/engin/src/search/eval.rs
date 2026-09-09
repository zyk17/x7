//! Eval 算法：cache | 已编码 NN 请求 | NN 回包发布。
//!
//! 规则终局和合法着生成属于独立 Expand task；这里不持有 worker-local 等待列表，
//! 因此任一 worker 都可继续处理下一项任务。

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crossbeam_channel::Sender;

use crate::EnginError;
use crate::neural::backend::EvalResult;
use crate::neural::{
    EncodedBatch, FillEmptyHistory, InputPlanes, encode_position_input_planes, eval_result_from_encoded_row,
};

use super::observer::{ExecutionKind, ExecutionTimer, SearchObserver};
use super::pipeline::{Shared, process_backprop_batch};
use super::workerpool::{BackpropEvent, EvalEvent, Event, NnReplyBatch, NnRequest};

/// 处理一个 Expand 已分类的普通叶子。
///
pub(crate) fn process_eval_event<O: SearchObserver>(
    shared: &Shared<O>,
    nn_tx: &Sender<NnRequest<O::Stamp>>,
    event: EvalEvent<O::Stamp>,
) -> Result<(), EnginError> {
    let _timer = ExecutionTimer::new(&shared.observer, ExecutionKind::Eval);
    if shared.stopping.load(Ordering::Acquire) {
        shared.cancel_claim(event.event);
        return Ok(());
    }
    if let Some(eval) = shared.backend.cached_evaluation(event.cache_key) {
        if O::ENABLED {
            shared.observer.on_cache_hit();
        }
        shared.send_backprop(publish_eval(shared, event.event, event.legal_moves, eval, false));
        return Ok(());
    }
    let planes = encode_position_input_planes(&event.history, FillEmptyHistory::No);
    let mut request = NnRequest::new(event, planes);
    request.mark_queued();
    match nn_tx.send(request) {
        Ok(()) => Ok(()),
        Err(error) => {
            shared.cancel_claim(error.0.event.event);
            Err(EnginError::Internal("stream nn queue disconnected"))
        }
    }
}

pub(crate) fn infer_nn_batch<O: SearchObserver>(
    shared: &Shared<O>,
    samples: &[InputPlanes],
) -> Result<Arc<EncodedBatch>, EnginError> {
    let mut logits = Vec::new();
    let mut wdl = Vec::new();
    let mut moves_left = Vec::new();
    shared.backend.infer_input_planes_into(samples, &mut logits, &mut wdl, &mut moves_left)?;
    let output = EncodedBatch::take_from(&mut logits, &mut wdl, &mut moves_left);
    output.ensure_batch_len(samples.len())?;
    shared.network_evaluations.fetch_add(samples.len() as u64, Ordering::AcqRel);
    if O::ENABLED {
        shared.observer.on_batch(samples.len());
    }
    Ok(Arc::new(output))
}

pub(crate) fn handle_nn_reply_batch<O: SearchObserver>(
    shared: &Shared<O>,
    reply: NnReplyBatch<O::Stamp>,
) -> Result<(), EnginError> {
    let timer = ExecutionTimer::new(&shared.observer, ExecutionKind::NnReply);
    let batch = match reply.result {
        Ok(batch) => batch,
        Err(error) => {
            for event in reply.events {
                shared.cancel_evaluation(event.event);
            }
            return Err(error);
        }
    };
    let count = reply.events.len();
    let mut events = reply.events.into_iter();
    let mut backprops = Vec::with_capacity(count);
    for row in 0..count {
        let event = events.next().expect("NN batch output matches requests");
        let eval = match eval_result_from_encoded_row(&batch, row, &event.legal_moves) {
            Ok(eval) => eval,
            Err(error) => {
                shared.cancel_evaluation(event.event);
                for event in events {
                    shared.cancel_evaluation(event.event);
                }
                return Err(error);
            }
        };
        shared.backend.store_evaluation(event.cache_key, Arc::clone(&eval));
        if shared.stopping.load(Ordering::Acquire) {
            shared.cancel_evaluation(event.event);
        } else {
            backprops.push(publish_eval(shared, event.event, event.legal_moves, eval, true));
        }
    }
    debug_assert!(events.next().is_none(), "NN batch output matches requests");
    drop(timer);
    process_backprop_batch(shared, backprops);
    Ok(())
}

fn publish_eval<O: SearchObserver>(
    shared: &Shared<O>,
    event: Event,
    legal_moves: xiangqi_core::LegalMoveList,
    eval: Arc<EvalResult>,
    holds_nn_credit: bool,
) -> BackpropEvent<O::Stamp> {
    shared
        .arena
        .get(event.node_id)
        .expect("eval node lives until job drain")
        .publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()));
    if holds_nn_credit {
        BackpropEvent::with_nn_credit(event, -eval.wl, eval.d, eval.plies_left)
    } else {
        BackpropEvent::without_nn_credit(event, -eval.wl, eval.d, eval.plies_left)
    }
}
