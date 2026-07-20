//! px0 `src/neural/wrapper.cc:49-172` 的 ONNX Runtime backend。
//!
//! 本模块只负责 `NetworkAsBackendComputation` 的 batch 输入、网络执行和
//! 合法着 softmax。cache 是 px0 `neural/memcache.cc` 的独立包装层，不能在此
//! 偷偷实现。

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::Array4;
use ort::ep::{directml::PerformancePreference, DirectML};
use ort::session::Session;
use ort::value::TensorRef;

use crate::neural::backend::{
    AddInputResult, Backend, BackendAttributes, BackendComputation, EvalPosition, EvalResult, EvalTicket,
};
use crate::EnginError;

use super::{
    encode_position_for_nn, move_to_nn_index, FillEmptyHistory, BOARD_COLS, BOARD_ROWS, INPUT_PLANES, POLICY_SIZE,
};

/// px0 `NetworkAsBackend` (`wrapper.cc:49-98`) 的最小 Rust 对应物。
#[derive(Clone)]
pub struct OnnxBackend {
    sessions: Arc<Mutex<OnnxSessions>>,
    attributes: BackendAttributes,
    provider: OnnxProvider,
}

/// px0 ONNX backend selects a configured provider and reports its real CPU
/// capability through `NetworkAsBackend` (`src/neural/backends/onnx/
/// network_onnx.cc:140-176`, `src/neural/wrapper.cc:49-68`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxProvider {
    DirectMl,
    Cpu,
}

impl OnnxProvider {
    pub const fn name(self) -> &'static str {
        match self {
            Self::DirectMl => "DmlExecutionProvider",
            Self::Cpu => "CPUExecutionProvider",
        }
    }
}

const DML_BATCH_STEP: usize = 16;
const DML_BATCH_STEPS: usize = 4;
const DML_MAX_BATCH: usize = DML_BATCH_STEP * DML_BATCH_STEPS;

/// Fixed-shape DirectML sessions, matching px0's ONNX session set.
///
/// px0 `src/neural/backends/onnx/network_onnx.cc:629-677,810-838,878-901`
/// creates one session per `batch_size * step`; DirectML needs a static input
/// shape to compile an efficient graph.
struct OnnxSessions {
    provider: OnnxProvider,
    sessions: Vec<(usize, Session)>,
}

impl OnnxSessions {
    fn single(provider: OnnxProvider, session: Session) -> Self {
        Self {
            provider,
            sessions: vec![(usize::MAX, session)],
        }
    }

    fn direct_ml(sessions: Vec<(usize, Session)>) -> Self {
        Self {
            provider: OnnxProvider::DirectMl,
            sessions,
        }
    }

    fn validation_session(&self) -> &Session {
        &self.sessions[0].1
    }

    fn chunk_size(&self) -> usize {
        if self.provider == OnnxProvider::DirectMl {
            DML_MAX_BATCH
        } else {
            usize::MAX
        }
    }

    fn run_batch_size(&self, entries: usize) -> usize {
        if self.provider == OnnxProvider::DirectMl {
            entries.div_ceil(DML_BATCH_STEP).min(DML_BATCH_STEPS) * DML_BATCH_STEP
        } else {
            entries
        }
    }

    fn session_for(&mut self, batch_size: usize) -> Result<&mut Session, EnginError> {
        let expected = if self.provider == OnnxProvider::DirectMl {
            batch_size
        } else {
            usize::MAX
        };
        self.sessions
            .iter_mut()
            .find_map(|(size, session)| (*size == expected).then_some(session))
            .ok_or_else(|| EnginError::Onnx(format!("missing ONNX session for batch size {batch_size}")))
    }
}

impl OnnxBackend {
    /// px0 `NetworkAsBackendFactory::Create` (`wrapper.cc:177-195`) 的本地权重入口。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, EnginError> {
        let (sessions, provider) = create_sessions(path.as_ref())?;
        validate_model_io(sessions.validation_session())?;
        Ok(Self {
            sessions: Arc::new(Mutex::new(sessions)),
            attributes: BackendAttributes {
                has_mlh: false,
                has_wdl: true,
                runs_on_cpu: provider == OnnxProvider::Cpu,
                suggested_num_search_threads: 1,
                // px0 DirectML defaults to batch=16 and steps=4, yielding
                // a 64-position minibatch (`network_onnx.cc:810-838`).
                recommended_batch_size: if provider == OnnxProvider::DirectMl {
                    DML_MAX_BATCH
                } else {
                    256
                },
                maximum_batch_size: 1024,
            },
            provider,
        })
    }

    pub const fn provider(&self) -> OnnxProvider {
        self.provider
    }
}

/// Creates DirectML sessions and explicitly falls back to CPU when DirectML
/// cannot be initialized.
///
/// px0's ONNX backend receives a provider at construction and derives
/// `IsCpu`/batch attributes from it (`src/neural/backends/onnx/
/// network_onnx.cc:140-176`). This Windows build intentionally supports one
/// GPU execution provider only: DirectML. That keeps the bundled ORT runtime
/// independent of locally installed CUDA and cuDNN versions.
fn create_sessions(path: &Path) -> Result<(OnnxSessions, OnnxProvider), EnginError> {
    match create_direct_ml_sessions(path) {
        Ok(sessions) => Ok((OnnxSessions::direct_ml(sessions), OnnxProvider::DirectMl)),
        Err(direct_ml_error) => {
            let session = Session::builder()
                .map_err(onnx_error)?
                .commit_from_file(path)
                .map_err(onnx_error)?;
            eprintln!("ONNX DirectML unavailable; using CPUExecutionProvider: {direct_ml_error}");
            Ok((OnnxSessions::single(OnnxProvider::Cpu, session), OnnxProvider::Cpu))
        }
    }
}

/// px0 `network_onnx.cc:629-677,810-838,878-901`: create fixed DirectML
/// sessions for each batch step instead of making DirectML compile a dynamic
/// batch graph on every search workload.
fn create_direct_ml_sessions(path: &Path) -> Result<Vec<(usize, Session)>, EnginError> {
    let mut sessions = Vec::with_capacity(DML_BATCH_STEPS);
    for step in 1..=DML_BATCH_STEPS {
        let batch_size = step * DML_BATCH_STEP;
        let session = Session::builder()
            .map_err(onnx_error)?
            .with_dimension_override("batch", batch_size as i64)
            .map_err(onnx_error)?
            .with_execution_providers([DirectML::default()
                .with_performance_preference(PerformancePreference::HighPerformance)
                .build()
                .error_on_failure()])
            .map_err(onnx_error)?
            .commit_from_file(path)
            .map_err(onnx_error)?;
        sessions.push((batch_size, session));
    }
    Ok(sessions)
}

impl Backend for OnnxBackend {
    fn evaluate(&self, history: &xiangqi_core::PositionHistory, legal_moves: &[xiangqi_core::Move]) -> Arc<EvalResult> {
        let computation = self.create_computation().expect("create ONNX computation");
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
        Ok(Box::new(OnnxBackendComputation::new(self.sessions.clone())))
    }

    fn infer_encoded(&self, planes: &[f32], batch: usize) -> Result<(Vec<f32>, Vec<f32>), EnginError> {
        let mut sessions = self.sessions.lock().expect("ONNX session lock");
        infer_encoded_planes(&mut sessions, planes, batch)
    }
}

/// px0 `NetworkAsBackendComputation` (`wrapper.cc:100-172`)。
struct OnnxBackendComputation {
    sessions: Arc<Mutex<OnnxSessions>>,
    state: Mutex<OnnxComputationState>,
}

struct OnnxComputationState {
    entries: Vec<(EvalTicket, EvalPosition)>,
    results: HashMap<usize, Arc<EvalResult>>,
    next_ticket: usize,
}

impl OnnxBackendComputation {
    fn new(sessions: Arc<Mutex<OnnxSessions>>) -> Self {
        Self {
            sessions,
            state: Mutex::new(OnnxComputationState {
                entries: Vec::new(),
                results: HashMap::new(),
                next_ticket: 0,
            }),
        }
    }
}

impl BackendComputation for OnnxBackendComputation {
    fn used_batch_size(&self) -> usize {
        self.state.lock().expect("ONNX computation lock").entries.len()
    }

    fn add_input(&self, position: EvalPosition) -> Result<(AddInputResult, EvalTicket), EnginError> {
        let mut state = self.state.lock().expect("ONNX computation lock");
        let ticket = EvalTicket(state.next_ticket);
        state.next_ticket += 1;
        state.entries.push((ticket, position));
        Ok((AddInputResult::EnqueuedForEval, ticket))
    }

    fn compute_blocking(&self) -> Result<(), EnginError> {
        let entries = std::mem::take(&mut self.state.lock().expect("ONNX computation lock").entries);
        if entries.is_empty() {
            return Ok(());
        }
        let plane_len = INPUT_PLANES * BOARD_ROWS * BOARD_COLS;
        let mut packed = vec![0.0; entries.len() * plane_len];
        for (index, (_, entry)) in entries.iter().enumerate() {
            let history = xiangqi_core::PositionHistory::from_positions(entry.positions.clone());
            let planes = encode_position_for_nn(&history, FillEmptyHistory::FenOnly);
            packed[index * plane_len..(index + 1) * plane_len].copy_from_slice(&planes);
        }
        let mut sessions = self.sessions.lock().expect("ONNX session lock");
        let (logits, wdl) = infer_encoded_planes(&mut sessions, &packed, entries.len())?;
        let mut results = HashMap::with_capacity(entries.len());
        for (index, (ticket, entry)) in entries.iter().enumerate() {
            let values = &logits[index * POLICY_SIZE..(index + 1) * POLICY_SIZE];
            let policies = softmax_legal_policy(values, &entry.legal_moves)?;
            let value = &wdl[index * 3..(index + 1) * 3];
            results.insert(
                ticket.0,
                Arc::new(EvalResult {
                    wl: value[0] - value[2],
                    d: value[1],
                    m: 0.0,
                    policies,
                }),
            );
        }
        self.state
            .lock()
            .expect("ONNX computation lock")
            .results
            .extend(results);
        Ok(())
    }

    fn take_result(&self, ticket: EvalTicket) -> Result<Arc<EvalResult>, EnginError> {
        self.state
            .lock()
            .expect("ONNX computation lock")
            .results
            .remove(&ticket.0)
            .ok_or(EnginError::PortIncomplete("OnnxBackendComputation missing result"))
    }
}

/// px0 `NetworkAsBackendComputation::SoftmaxPolicy` (`wrapper.cc:135-164`)。
pub fn softmax_legal_policy(logits: &[f32], legal_moves: &[xiangqi_core::Move]) -> Result<Vec<f32>, EnginError> {
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


/// ONNX-only: padded DirectML/CPU session run on pre-encoded planes.
fn infer_encoded_planes(
    sessions: &mut OnnxSessions,
    planes: &[f32],
    batch: usize,
) -> Result<(Vec<f32>, Vec<f32>), EnginError> {
    let plane_len = INPUT_PLANES * BOARD_ROWS * BOARD_COLS;
    if planes.len() != batch * plane_len {
        return Err(EnginError::Onnx(format!(
            "encoded planes length {} != batch {batch} * {plane_len}",
            planes.len()
        )));
    }
    if batch == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut all_logits = Vec::with_capacity(batch * POLICY_SIZE);
    let mut all_wdl = Vec::with_capacity(batch * 3);
    let chunk_size = sessions.chunk_size();
    for chunk_start in (0..batch).step_by(chunk_size.min(batch).max(1)) {
        let end = (chunk_start + chunk_size).min(batch);
        let actual_batch = end - chunk_start;
        let run_batch = sessions.run_batch_size(actual_batch);
        // px0 zero-pads each DirectML minibatch (`network_onnx.cc:398-447`).
        let mut input = vec![0.0; run_batch * plane_len];
        input[..actual_batch * plane_len]
            .copy_from_slice(&planes[chunk_start * plane_len..end * plane_len]);
        let board = Array4::from_shape_vec((run_batch, INPUT_PLANES, BOARD_ROWS, BOARD_COLS), input)
            .map_err(|error| EnginError::Onnx(format!("invalid batch input: {error}")))?;
        let tensor = TensorRef::from_array_view(&board).map_err(onnx_error)?;
        let outputs = sessions
            .session_for(run_batch)?
            .run(ort::inputs!["board" => tensor])
            .map_err(onnx_error)?;
        let logits = tensor_output(&outputs, "logits", run_batch, POLICY_SIZE)?;
        let wdl = tensor_output(&outputs, "value", run_batch, 3)?;
        all_logits.extend_from_slice(&logits[..actual_batch * POLICY_SIZE]);
        all_wdl.extend_from_slice(&wdl[..actual_batch * 3]);
    }
    Ok((all_logits, all_wdl))
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

fn onnx_error<T>(error: ort::Error<T>) -> EnginError {
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
