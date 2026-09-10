//! NN inference 与回包发布。

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::EnginError;
use crate::neural::backend::EvalResult;
use crate::neural::{EncodedBatch, InputPlanes, eval_result_from_encoded_row};

use super::backprop::Backprop;
use super::observer::{ExecutionKind, ExecutionTimer, SearchObserver};
use super::pipeline::{Shared, process_backprop_batch};
use super::workerpool::{NnReplyBatch, Selection};

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
            for item in reply.items {
                shared.cancel_evaluation(item.selection);
            }
            return Err(error);
        }
    };
    let count = reply.items.len();
    let mut items = reply.items.into_iter();
    let mut backprops = Vec::with_capacity(count);
    for row in 0..count {
        let item = items.next().expect("NN batch output matches requests");
        let eval = match eval_result_from_encoded_row(&batch, row, &item.legal_moves) {
            Ok(eval) => eval,
            Err(error) => {
                shared.cancel_evaluation(item.selection);
                for item in items {
                    shared.cancel_evaluation(item.selection);
                }
                return Err(error);
            }
        };
        shared.cache.insert_evaluation(item.cache_key, Arc::clone(&eval));
        if shared.is_stopping() {
            shared.cancel_evaluation(item.selection);
        } else {
            backprops.push(publish_eval(shared, item.selection, item.legal_moves, eval, true));
        }
    }
    debug_assert!(items.next().is_none(), "NN batch output matches requests");
    drop(timer);
    process_backprop_batch(shared, backprops);
    Ok(())
}

pub(crate) fn publish_eval<O: SearchObserver>(
    shared: &Shared<O>,
    selection: Selection,
    legal_moves: xiangqi_core::LegalMoveList,
    eval: Arc<EvalResult>,
    holds_nn_credit: bool,
) -> Backprop {
    shared
        .arena
        .get(selection.node_id)
        .expect("NN node lives until job drain")
        .publish_edges(legal_moves.iter().copied().zip(eval.policies.iter().copied()));
    if holds_nn_credit {
        Backprop::with_nn_credit(selection, -eval.wl, eval.d, eval.plies_left)
    } else {
        Backprop::without_nn_credit(selection, -eval.wl, eval.d, eval.plies_left)
    }
}
