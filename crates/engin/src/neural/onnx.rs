//! ONNX Runtime backend。
//!
//! 负责 batch 输入与网络执行。`BackendComputation` 路径仍在此做合法着 softmax；
//! stream NN worker 只走 `infer_input_planes_into` / `infer_encoded*`，softmax/组装在 Eval。
//! NN cache 是独立包装层。热路径复用 host `input_scratch`：稀疏路径在 scratch 内 expand+pad；
//! dense 路径在无 DirectML pad 时可对 planes 零拷贝建 TensorRef。

#[cfg(all(feature = "cuda", feature = "directml"))]
compile_error!("ONNX runtime build must select either `cuda` or `directml`, not both");

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::EnginError;
use crate::neural::backend::{
    AddInputResult, Backend, BackendAttributes, BackendComputation, EncodedInference, EvalPosition, EvalResult,
    EvalTicket,
};
#[cfg(feature = "cuda")]
use ndarray::Array4;
#[cfg(feature = "cuda")]
use ort::ep::{CUDA, cuda::ConvAlgorithmSearch};
#[cfg(feature = "directml")]
use ort::ep::{DirectML, directml::PerformancePreference};
use ort::session::Session;
use ort::value::TensorRef;
use xiangqi_core::PositionHistory;

use super::{
    BOARD_COLS, BOARD_ROWS, ENCODED_PLANE_FLOATS, FillEmptyHistory, INPUT_PLANES, InputPlanes, POLICY_SIZE,
    encode_position_input_planes, expand_input_planes_into_zeroed, softmax_legal_policy,
};

/// ONNX Runtime backend。
#[derive(Clone)]
pub struct OnnxBackend {
    sessions: Arc<Mutex<OnnxSessions>>,
    attributes: BackendAttributes,
    provider: OnnxProvider,
}

/// ONNX backend 选择配置的 provider，并报告实际 CPU
/// 能力（`src/neural/backends/onnx/network_onnx.cc:140-176`、
/// `src/neural/wrapper.cc:49-68`）。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxProvider {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "directml")]
    DirectMl,
    Cpu,
}

impl OnnxProvider {
    pub const fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => "CUDAExecutionProvider",
            #[cfg(feature = "directml")]
            Self::DirectMl => "DmlExecutionProvider",
            Self::Cpu => "CPUExecutionProvider",
        }
    }
}

#[cfg(feature = "cuda")]
const CUDA_RECOMMENDED_BATCH: usize = 256;
#[cfg(feature = "directml")]
const DML_BATCH_STEP: usize = 16;
#[cfg(feature = "directml")]
const DML_BATCH_STEPS: usize = 4;
#[cfg(feature = "directml")]
const DML_MAX_BATCH: usize = DML_BATCH_STEP * DML_BATCH_STEPS;

/// CUDA 使用一个动态 batch session；DirectML 使用固定形状 session 集合。
struct OnnxSessions {
    #[cfg(feature = "directml")]
    provider: OnnxProvider,
    sessions: Vec<(usize, Session)>,
    /// DirectML pad / 偶发 owned 输入的常驻 host scratch；容量按需增长，跨次 `run` 复用。
    input_scratch: Vec<f32>,
}

impl OnnxSessions {
    fn single(session: Session) -> Self {
        Self {
            #[cfg(feature = "directml")]
            provider: OnnxProvider::Cpu,
            sessions: vec![(usize::MAX, session)],
            input_scratch: Vec::new(),
        }
    }

    #[cfg(feature = "directml")]
    fn direct_ml(sessions: Vec<(usize, Session)>) -> Self {
        Self {
            provider: OnnxProvider::DirectMl,
            sessions,
            input_scratch: Vec::new(),
        }
    }

    fn validation_session(&self) -> &Session {
        &self.sessions[0].1
    }

    fn chunk_size(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            usize::MAX
        }
        #[cfg(feature = "directml")]
        {
            if self.provider == OnnxProvider::DirectMl {
                DML_MAX_BATCH
            } else {
                usize::MAX
            }
        }
    }

    fn run_batch_size(&self, entries: usize) -> usize {
        #[cfg(feature = "cuda")]
        {
            entries
        }
        #[cfg(feature = "directml")]
        {
            if self.provider == OnnxProvider::DirectMl {
                entries.div_ceil(DML_BATCH_STEP).min(DML_BATCH_STEPS) * DML_BATCH_STEP
            } else {
                entries
            }
        }
    }

    fn session_index(&self, batch_size: usize) -> Result<usize, EnginError> {
        #[cfg(feature = "cuda")]
        let expected = usize::MAX;
        #[cfg(feature = "directml")]
        let expected = if self.provider == OnnxProvider::DirectMl {
            batch_size
        } else {
            usize::MAX
        };
        self.sessions
            .iter()
            .position(|(size, _)| *size == expected)
            .ok_or_else(|| EnginError::Onnx(format!("missing ONNX session for batch size {batch_size}")))
    }
}

impl OnnxBackend {
    /// 从本地权重文件创建 ONNX backend。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, EnginError> {
        let (sessions, provider) = create_sessions(path.as_ref())?;
        validate_model_io(sessions.validation_session())?;
        Ok(Self {
            sessions: Arc::new(Mutex::new(sessions)),
            attributes: BackendAttributes {
                has_mlh: true,
                has_wdl: true,
                runs_on_cpu: provider == OnnxProvider::Cpu,
                suggested_num_search_threads: 1,
                recommended_batch_size: recommended_batch_size(provider),
                maximum_batch_size: 1024,
            },
            provider,
        })
    }

    pub const fn provider(&self) -> OnnxProvider {
        self.provider
    }
}

#[cfg(feature = "cuda")]
fn create_sessions(path: &Path) -> Result<(OnnxSessions, OnnxProvider), EnginError> {
    match create_cuda_session(path).and_then(|mut session| {
        validate_cuda_session(&mut session)?;
        Ok(session)
    }) {
        Ok(session) => Ok((OnnxSessions::single(session), OnnxProvider::Cuda)),
        Err(cuda_error) => {
            let session = Session::builder()
                .map_err(onnx_error)?
                .commit_from_file(path)
                .map_err(onnx_error)?;
            eprintln!("ONNX CUDA unavailable; using CPUExecutionProvider: {cuda_error}");
            Ok((OnnxSessions::single(session), OnnxProvider::Cpu))
        }
    }
}

#[cfg(feature = "directml")]
fn create_sessions(path: &Path) -> Result<(OnnxSessions, OnnxProvider), EnginError> {
    match create_direct_ml_sessions(path) {
        Ok(sessions) => Ok((OnnxSessions::direct_ml(sessions), OnnxProvider::DirectMl)),
        Err(direct_ml_error) => {
            let session = Session::builder()
                .map_err(onnx_error)?
                .commit_from_file(path)
                .map_err(onnx_error)?;
            eprintln!("ONNX DirectML unavailable; using CPUExecutionProvider: {direct_ml_error}");
            Ok((OnnxSessions::single(session), OnnxProvider::Cpu))
        }
    }
}

#[cfg(feature = "cuda")]
fn recommended_batch_size(provider: OnnxProvider) -> usize {
    if provider == OnnxProvider::Cuda {
        CUDA_RECOMMENDED_BATCH
    } else {
        256
    }
}

#[cfg(feature = "directml")]
fn recommended_batch_size(provider: OnnxProvider) -> usize {
    if provider == OnnxProvider::DirectMl {
        DML_MAX_BATCH
    } else {
        256
    }
}

/// ORT CUDA EP 使用动态输入形状；heuristic 避免首次加载为卷积算法做完整穷举。
/// 参考：ORT CUDA EP `cudnn_conv_algo_search` 文档。
#[cfg(feature = "cuda")]
fn create_cuda_session(path: &Path) -> Result<Session, EnginError> {
    Session::builder()
        .map_err(onnx_error)?
        .with_execution_providers([CUDA::default()
            .with_device_id(0)
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
            // ORT CUDA EP：卷积网络在 Tensor Core GPU 上优先 NHWC；本机 A/B 有收益。
            // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#prefer_nhwc
            .with_prefer_nhwc(true)
            .build()
            .error_on_failure()])
        .map_err(onnx_error)?
        .commit_from_file(path)
        .map_err(onnx_error)
}

/// CUDA EP 注册成功不代表首个 kernel 能在当前 GPU 架构执行。启动时以一个最小输入
/// 提前验证，避免搜索线程在第一批推理时崩溃；参考 ORT CUDA EP 的运行时兼容要求。
#[cfg(feature = "cuda")]
fn validate_cuda_session(session: &mut Session) -> Result<(), EnginError> {
    let board = Array4::<f32>::zeros((1, INPUT_PLANES, BOARD_ROWS, BOARD_COLS));
    let tensor = TensorRef::from_array_view(&board).map_err(onnx_error)?;
    session.run(ort::inputs!["board" => tensor]).map_err(onnx_error)?;
    Ok(())
}

/// 固定形状 DirectML / CUDA session。
#[cfg(feature = "directml")]
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
    fn evaluate(&self, history: &PositionHistory, legal_moves: &[xiangqi_core::Move]) -> Arc<EvalResult> {
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

    fn infer_encoded(&self, planes: &[f32], batch: usize) -> Result<EncodedInference, EnginError> {
        let mut logits = Vec::new();
        let mut wdl = Vec::new();
        let mut moves_left = Vec::new();
        self.infer_encoded_into(planes, batch, &mut logits, &mut wdl, &mut moves_left)?;
        Ok((logits, wdl, moves_left))
    }

    fn infer_encoded_into(
        &self,
        planes: &[f32],
        batch: usize,
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError> {
        let mut sessions = self.sessions.lock().expect("ONNX session lock");
        infer_encoded_planes(&mut sessions, planes, batch, logits, wdl, moves_left)
    }

    fn infer_input_planes_into(
        &self,
        samples: &[InputPlanes],
        logits: &mut Vec<f32>,
        wdl: &mut Vec<f32>,
        moves_left: &mut Vec<f32>,
    ) -> Result<(), EnginError> {
        let mut sessions = self.sessions.lock().expect("ONNX session lock");
        infer_input_planes(&mut sessions, samples, logits, wdl, moves_left)
    }
}

/// 一次 ONNX batch computation。
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
        let mut samples = Vec::with_capacity(entries.len());
        for (_, entry) in &entries {
            let history = PositionHistory::from_positions(entry.positions.clone());
            samples.push(encode_position_input_planes(&history, FillEmptyHistory::FenOnly));
        }
        let mut sessions = self.sessions.lock().expect("ONNX session lock");
        let mut logits = Vec::new();
        let mut wdl = Vec::new();
        let mut moves_left = Vec::new();
        infer_input_planes(&mut sessions, &samples, &mut logits, &mut wdl, &mut moves_left)?;
        let mut results = HashMap::with_capacity(entries.len());
        for (index, (ticket, entry)) in entries.iter().enumerate() {
            let values = &logits[index * POLICY_SIZE..(index + 1) * POLICY_SIZE];
            let policies = softmax_legal_policy(values, &entry.legal_moves).map_err(|error| {
                let position = entry.positions.last().expect("EvalPosition has a position");
                EnginError::Onnx(format!("{error}; position: {}", position.to_fen()))
            })?;
            let value = &wdl[index * 3..(index + 1) * 3];
            results.insert(
                ticket.0,
                Arc::new(EvalResult {
                    wl: value[0] - value[2],
                    d: value[1],
                    plies_left: moves_left[index],
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

/// 稀疏合批：expand 直接写入常驻 `input_scratch`（含 DirectML pad 清零），再 `Session::run`。
///
/// 参考：px0 ONNX `PrepareInputs` 在 ORT 边界 densify；此处额外把 pad 与 expand 合并到同一 scratch，
/// 去掉 stream 路径上的中间 dense `packed`。
fn infer_input_planes(
    sessions: &mut OnnxSessions,
    samples: &[InputPlanes],
    all_logits: &mut Vec<f32>,
    all_wdl: &mut Vec<f32>,
    all_moves_left: &mut Vec<f32>,
) -> Result<(), EnginError> {
    let batch = samples.len();
    all_logits.clear();
    all_wdl.clear();
    all_moves_left.clear();
    if batch == 0 {
        return Ok(());
    }
    all_logits.reserve(batch * POLICY_SIZE);
    all_wdl.reserve(batch * 3);
    all_moves_left.reserve(batch);
    let plane_len = ENCODED_PLANE_FLOATS;
    let chunk_size = sessions.chunk_size();
    for chunk_start in (0..batch).step_by(chunk_size.min(batch).max(1)) {
        let end = (chunk_start + chunk_size).min(batch);
        let actual_batch = end - chunk_start;
        let run_batch = sessions.run_batch_size(actual_batch);
        let session_index = sessions.session_index(run_batch)?;
        let shape = [run_batch, INPUT_PLANES, BOARD_ROWS, BOARD_COLS];
        let need = run_batch * plane_len;

        let mut scratch = std::mem::take(&mut sessions.input_scratch);
        if scratch.len() < need {
            scratch.resize(need, 0.0);
        }
        scratch[..need].fill(0.0);
        for (row, sample) in samples[chunk_start..end].iter().enumerate() {
            let offset = row * plane_len;
            expand_input_planes_into_zeroed(sample, &mut scratch[offset..offset + plane_len]);
        }

        {
            let input = &scratch[..need];
            let tensor = TensorRef::from_array_view((shape, input)).map_err(onnx_error)?;
            let outputs = sessions.sessions[session_index]
                .1
                .run(ort::inputs!["board" => tensor])
                .map_err(onnx_error)?;
            append_tensor_rows(&outputs, "logits", run_batch, POLICY_SIZE, actual_batch, all_logits)?;
            append_tensor_rows(&outputs, "value", run_batch, 3, actual_batch, all_wdl)?;
            append_tensor_rows(&outputs, "moves_left", run_batch, 1, actual_batch, all_moves_left)?;
        }
        sessions.input_scratch = scratch;
    }
    Ok(())
}

/// 仅供 ONNX 使用：在预编码 dense planes 上运行当前发行包的 ONNX session。
///
/// - `run_batch == actual_batch`：对 `planes` 切片零拷贝建 `TensorRef`
/// - DirectML pad：写入常驻 `input_scratch`，只拷贝有效样本并清零 pad
/// - 输出只把 `actual_batch` 行追加到调用方缓冲（先 `clear`），避免整段 pad 的 `to_vec`
fn infer_encoded_planes(
    sessions: &mut OnnxSessions,
    planes: &[f32],
    batch: usize,
    all_logits: &mut Vec<f32>,
    all_wdl: &mut Vec<f32>,
    all_moves_left: &mut Vec<f32>,
) -> Result<(), EnginError> {
    let plane_len = INPUT_PLANES * BOARD_ROWS * BOARD_COLS;
    if planes.len() != batch * plane_len {
        return Err(EnginError::Onnx(format!(
            "encoded planes length {} != batch {batch} * {plane_len}",
            planes.len()
        )));
    }
    all_logits.clear();
    all_wdl.clear();
    all_moves_left.clear();
    if batch == 0 {
        return Ok(());
    }
    all_logits.reserve(batch * POLICY_SIZE);
    all_wdl.reserve(batch * 3);
    all_moves_left.reserve(batch);
    let chunk_size = sessions.chunk_size();
    for chunk_start in (0..batch).step_by(chunk_size.min(batch).max(1)) {
        let end = (chunk_start + chunk_size).min(batch);
        let actual_batch = end - chunk_start;
        let run_batch = sessions.run_batch_size(actual_batch);
        let session_index = sessions.session_index(run_batch)?;
        let shape = [run_batch, INPUT_PLANES, BOARD_ROWS, BOARD_COLS];
        let src = &planes[chunk_start * plane_len..end * plane_len];

        // 先填 scratch（若需要），再 `take` 出来，避免与 `&mut Session` 同时借用 `OnnxSessions`。
        let mut owned_input = if run_batch == actual_batch {
            None
        } else {
            let need = run_batch * plane_len;
            let mut scratch = std::mem::take(&mut sessions.input_scratch);
            if scratch.len() < need {
                scratch.resize(need, 0.0);
            }
            scratch[..src.len()].copy_from_slice(src);
            scratch[src.len()..need].fill(0.0);
            Some(scratch)
        };

        let input: &[f32] = match owned_input.as_deref() {
            Some(scratch) => &scratch[..run_batch * plane_len],
            None => src,
        };
        {
            let tensor = TensorRef::from_array_view((shape, input)).map_err(onnx_error)?;
            let outputs = sessions.sessions[session_index]
                .1
                .run(ort::inputs!["board" => tensor])
                .map_err(onnx_error)?;
            append_tensor_rows(&outputs, "logits", run_batch, POLICY_SIZE, actual_batch, all_logits)?;
            append_tensor_rows(&outputs, "value", run_batch, 3, actual_batch, all_wdl)?;
            append_tensor_rows(&outputs, "moves_left", run_batch, 1, actual_batch, all_moves_left)?;
        }
        if let Some(scratch) = owned_input.take() {
            sessions.input_scratch = scratch;
        }
    }
    Ok(())
}

fn validate_model_io(session: &Session) -> Result<(), EnginError> {
    let inputs = session.inputs();
    if inputs.len() != 1 || inputs[0].name() != "board" {
        return Err(EnginError::Onnx("expected one ONNX input named board".into()));
    }
    let output_names: Vec<_> = session.outputs().iter().map(|output| output.name()).collect();
    if output_names != ["logits", "value", "moves_left"] {
        return Err(EnginError::Onnx(
            "expected ONNX outputs logits, value, and moves_left".into(),
        ));
    }
    Ok(())
}

/// 将输出 tensor 的前 `actual_batch` 行追加到 `dest`，不分配整段 pad 缓冲。
fn append_tensor_rows(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    run_batch: usize,
    width: usize,
    actual_batch: usize,
    dest: &mut Vec<f32>,
) -> Result<(), EnginError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| EnginError::Onnx(format!("ONNX output missing {name}")))?;
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(onnx_error)?;
    if **shape != [run_batch as i64, width as i64] {
        return Err(EnginError::Onnx(format!(
            "{name} shape must be [{run_batch},{width}], got {shape:?}"
        )));
    }
    if actual_batch > run_batch {
        return Err(EnginError::Onnx(format!(
            "{name}: actual_batch {actual_batch} > run_batch {run_batch}"
        )));
    }
    dest.extend_from_slice(&data[..actual_batch * width]);
    Ok(())
}

fn onnx_error<T>(error: ort::Error<T>) -> EnginError {
    EnginError::Onnx(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn local_x7_onnx_smoke_if_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/x7.onnx");
        if !path.is_file() {
            eprintln!("skip: {} is absent", path.display());
            return;
        }
        let backend = OnnxBackend::from_file(&path).expect("load local x7.onnx");
        let history = PositionHistory::from_positions(vec![
            xiangqi_core::Position::from_fen(xiangqi_core::STARTPOS_FEN).unwrap(),
        ]);
        let legal = history.last().board().generate_legal_moves();
        let eval = backend.evaluate(&history, &legal);
        assert_eq!(eval.policies.len(), legal.len());
        assert!((eval.policies.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(eval.wl.is_finite() && eval.d.is_finite() && eval.plies_left.is_finite());
    }
}
