//! px0 `src/neural/wrapper.cc:49-172` 的 ONNX Runtime backend。
//!
//! 本模块只负责 `NetworkAsBackendComputation` 的 batch 输入、网络执行和
//! 合法着 softmax。cache 是 px0 `neural/memcache.cc` 的独立包装层，不能在此
//! 偷偷实现。

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use crate::search::classic::{
    AddInputResult, Backend, BackendAttributes, BackendComputation, EvalPosition, EvalResult, EvalTicket,
};
use crate::EnginError;

use super::{
    encode_position_for_nn, move_to_nn_index, FillEmptyHistory, BOARD_COLS, BOARD_ROWS, INPUT_PLANES, POLICY_SIZE,
};

/// px0 `NetworkAsBackend` (`wrapper.cc:49-98`) 的最小 Rust 对应物。
#[derive(Clone)]
pub struct OnnxBackend {
    session: Arc<Mutex<Session>>,
    attributes: BackendAttributes,
}

impl OnnxBackend {
    /// px0 `NetworkAsBackendFactory::Create` (`wrapper.cc:177-195`) 的本地权重入口。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, EnginError> {
        let session = Session::builder()
            .map_err(onnx_error)?
            .commit_from_file(path)
            .map_err(onnx_error)?;
        validate_model_io(&session)?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            attributes: BackendAttributes {
                has_mlh: false,
                has_wdl: true,
                runs_on_cpu: true,
                suggested_num_search_threads: 1,
                recommended_batch_size: 1,
                maximum_batch_size: 1024,
            },
        })
    }
}

impl Backend for OnnxBackend {
    fn evaluate(&self, history: &xiangqi_core::PositionHistory, legal_moves: &[xiangqi_core::Move]) -> EvalResult {
        let mut computation = self.create_computation().expect("create ONNX computation");
        let (_, ticket) = computation
            .add_input(EvalPosition {
                positions: history.positions().to_vec(),
                legal_moves: legal_moves.to_vec(),
            })
            .expect("enqueue ONNX evaluation");
        computation.compute_blocking().expect("run ONNX evaluation");
        computation.take_result(ticket).expect("fetch ONNX evaluation")
    }

    fn attributes(&self) -> BackendAttributes {
        self.attributes
    }

    fn create_computation(&self) -> Result<Box<dyn BackendComputation>, EnginError> {
        Ok(Box::new(OnnxBackendComputation::new(self.session.clone())))
    }
}

/// px0 `NetworkAsBackendComputation` (`wrapper.cc:100-172`)。
struct OnnxBackendComputation {
    session: Arc<Mutex<Session>>,
    entries: Vec<(EvalTicket, EvalPosition)>,
    results: HashMap<usize, EvalResult>,
    next_ticket: usize,
}

impl OnnxBackendComputation {
    fn new(session: Arc<Mutex<Session>>) -> Self {
        Self {
            session,
            entries: Vec::new(),
            results: HashMap::new(),
            next_ticket: 0,
        }
    }
}

impl BackendComputation for OnnxBackendComputation {
    fn used_batch_size(&self) -> usize {
        self.entries.len()
    }

    fn add_input(&mut self, position: EvalPosition) -> Result<(AddInputResult, EvalTicket), EnginError> {
        let ticket = EvalTicket(self.next_ticket);
        self.next_ticket += 1;
        self.entries.push((ticket, position));
        Ok((AddInputResult::EnqueuedForEval, ticket))
    }

    fn compute_blocking(&mut self) -> Result<(), EnginError> {
        if self.entries.is_empty() {
            return Ok(());
        }
        let batch = self.entries.len();
        let mut input = Vec::with_capacity(batch * INPUT_PLANES * BOARD_ROWS * BOARD_COLS);
        for (_, entry) in &self.entries {
            let history = xiangqi_core::PositionHistory::from_positions(entry.positions.clone());
            input.extend(encode_position_for_nn(&history, FillEmptyHistory::FenOnly));
        }
        let board = Array4::from_shape_vec((batch, INPUT_PLANES, BOARD_ROWS, BOARD_COLS), input)
            .map_err(|error| EnginError::Onnx(format!("invalid batch input: {error}")))?;
        let mut session = self.session.lock().expect("ONNX session lock");
        let tensor = TensorRef::from_array_view(&board).map_err(onnx_error)?;
        let outputs = session.run(ort::inputs!["board" => tensor]).map_err(onnx_error)?;
        let logits = tensor_output(&outputs, "logits", batch, POLICY_SIZE)?;
        let wdl = tensor_output(&outputs, "value", batch, 3)?;

        for (index, (ticket, entry)) in self.entries.drain(..).enumerate() {
            let values = &logits[index * POLICY_SIZE..(index + 1) * POLICY_SIZE];
            let policies = softmax_legal_policy(values, &entry.legal_moves)?;
            let value = &wdl[index * 3..(index + 1) * 3];
            self.results.insert(
                ticket.0,
                EvalResult {
                    wl: value[0] - value[2],
                    d: value[1],
                    m: 0.0,
                    policies,
                },
            );
        }
        Ok(())
    }

    fn take_result(&mut self, ticket: EvalTicket) -> Result<EvalResult, EnginError> {
        self.results
            .remove(&ticket.0)
            .ok_or(EnginError::PortIncomplete("OnnxBackendComputation missing result"))
    }
}

/// px0 `NetworkAsBackendComputation::SoftmaxPolicy` (`wrapper.cc:135-164`)。
fn softmax_legal_policy(logits: &[f32], legal_moves: &[xiangqi_core::Move]) -> Result<Vec<f32>, EnginError> {
    let mut selected = Vec::with_capacity(legal_moves.len());
    let mut maximum = f32::NEG_INFINITY;
    for &mv in legal_moves {
        let index = move_to_nn_index(mv)
            .ok_or_else(|| EnginError::Onnx(format!("legal move absent from px0 policy table: {mv}")))?;
        let logit = logits[index];
        maximum = maximum.max(logit);
        selected.push(logit);
    }
    let total: f32 = selected
        .iter_mut()
        .map(|value| {
            *value = (*value - maximum).exp();
            *value
        })
        .sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(EnginError::Onnx("legal policy softmax is invalid".into()));
    }
    for value in &mut selected {
        *value /= total;
    }
    Ok(selected)
}

fn validate_model_io(session: &Session) -> Result<(), EnginError> {
    let inputs = session.inputs();
    if inputs.len() != 1 || inputs[0].name() != "board" {
        return Err(EnginError::Onnx("expected one ONNX input named board".into()));
    }
    let output_names: Vec<_> = session.outputs().iter().map(|output| output.name()).collect();
    if !output_names.contains(&"logits") || !output_names.contains(&"value") {
        return Err(EnginError::Onnx("expected ONNX outputs logits and value".into()));
    }
    Ok(())
}

fn tensor_output(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    batch: usize,
    width: usize,
) -> Result<Vec<f32>, EnginError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| EnginError::Onnx(format!("ONNX output missing {name}")))?;
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(onnx_error)?;
    if **shape != [batch as i64, width as i64] {
        return Err(EnginError::Onnx(format!(
            "{name} shape must be [{batch},{width}], got {shape:?}"
        )));
    }
    Ok(data.to_vec())
}

fn onnx_error(error: ort::Error) -> EnginError {
    EnginError::Onnx(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn softmax_is_limited_to_legal_policy_entries() {
        let a0 = xiangqi_core::Square::parse("a0").unwrap();
        let a1 = xiangqi_core::Square::parse("a1").unwrap();
        let a2 = xiangqi_core::Square::parse("a2").unwrap();
        let mut logits = vec![f32::NEG_INFINITY; POLICY_SIZE];
        logits[move_to_nn_index(xiangqi_core::Move::new(a0, a1)).unwrap()] = 0.0;
        logits[move_to_nn_index(xiangqi_core::Move::new(a0, a2)).unwrap()] = 1.0;
        let policy = softmax_legal_policy(
            &logits,
            &[xiangqi_core::Move::new(a0, a1), xiangqi_core::Move::new(a0, a2)],
        )
        .unwrap();
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(policy[1] > policy[0]);
    }

    #[test]
    fn local_x7_onnx_smoke_if_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/x7.onnx");
        if !path.is_file() {
            eprintln!("skip: {} is absent", path.display());
            return;
        }
        let backend = OnnxBackend::from_file(&path).expect("load local x7.onnx");
        let history = xiangqi_core::PositionHistory::from_positions(vec![xiangqi_core::Position::from_fen(
            xiangqi_core::STARTPOS_FEN,
        )
        .unwrap()]);
        let legal = history.last().board().generate_legal_moves();
        let eval = backend.evaluate(&history, &legal);
        assert_eq!(eval.policies.len(), legal.len());
        assert!((eval.policies.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(eval.wl.is_finite() && eval.d.is_finite());
    }
}
