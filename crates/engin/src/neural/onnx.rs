//! ONNX Runtime backend。
//!
//! DirectML：host expand + `Session::run`。TensorRT：GPU expand + IoBinding。

#[cfg(all(feature = "tensorrt", feature = "directml"))]
compile_error!(
    "enable exactly one EP: DirectML is default; for TensorRT use \
     `--no-default-features --features tensorrt`"
);

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::EnginError;
use crate::neural::backend::{
    AddInputResult, Backend, BackendAttributes, BackendComputation, EncodedInference, EvalPosition, EvalResult,
    EvalTicket,
};
#[cfg(feature = "tensorrt")]
use crate::neural::cuda_trt::{CudaStream, DeviceBuffer, download_device_async, expand_planes_async};
#[cfg(feature = "tensorrt")]
use ndarray::Array4;
#[cfg(feature = "tensorrt")]
use ort::ep::TensorRT;
#[cfg(feature = "directml")]
use ort::ep::{DirectML, directml::PerformancePreference};
#[cfg(feature = "tensorrt")]
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::TensorRef;
#[cfg(feature = "tensorrt")]
use ort::value::{Shape, Tensor, TensorRefMut, TensorValueType};
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
    #[cfg(feature = "tensorrt")]
    TensorRt,
    #[cfg(feature = "directml")]
    DirectMl,
    Cpu,
}

impl OnnxProvider {
    pub const fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "tensorrt")]
            Self::TensorRt => "TensorrtExecutionProvider",
            #[cfg(feature = "directml")]
            Self::DirectMl => "DmlExecutionProvider",
            Self::Cpu => "CPUExecutionProvider",
        }
    }
}

#[cfg(feature = "tensorrt")]
const TRT_PROFILE_MIN_BATCH: usize = 1;
/// recommended / opt / max 对齐 stream 稳态合批。
#[cfg(feature = "tensorrt")]
const TRT_MAX_BATCH: usize = 256;
#[cfg(feature = "directml")]
const DML_BATCH_STEP: usize = 16;
#[cfg(feature = "directml")]
const DML_BATCH_STEPS: usize = 4;
#[cfg(feature = "directml")]
const DML_MAX_BATCH: usize = DML_BATCH_STEP * DML_BATCH_STEPS;

/// TensorRT GPU 状态。
///
/// 字段声明顺序 = drop 顺序：先释放输出 Tensor，再 Allocator（其 Arc 可能拖住 Session），
/// `stream` 必须最后销毁（ORT TRT 仍可能引用 user compute stream）。
#[cfg(feature = "tensorrt")]
struct TrtGpu {
    logits: Option<Tensor<f32>>,
    value: Option<Tensor<f32>>,
    moves_left: Option<Tensor<f32>>,
    allocator: Allocator,
    sparse_host: Vec<u8>,
    sparse_dev: DeviceBuffer,
    dense_dev: DeviceBuffer,
    logits_host: Vec<f32>,
    value_host: Vec<f32>,
    moves_left_host: Vec<f32>,
    /// 与 `logits/value/moves_left` 对应的 batch；0 表示需重建。
    out_batch: usize,
    max_batch: usize,
    stream: CudaStream,
}

#[cfg(feature = "tensorrt")]
impl TrtGpu {
    fn new(stream: CudaStream, allocator: Allocator, max_batch: usize) -> Result<Self, EnginError> {
        let n = max_batch * INPUT_PLANES;
        Ok(Self {
            logits: None,
            value: None,
            moves_left: None,
            allocator,
            sparse_host: Vec::new(),
            sparse_dev: DeviceBuffer::new(n * (size_of::<u128>() + size_of::<f32>()))?,
            dense_dev: DeviceBuffer::new(max_batch * ENCODED_PLANE_FLOATS * size_of::<f32>())?,
            logits_host: Vec::new(),
            value_host: Vec::new(),
            moves_left_host: Vec::new(),
            out_batch: 0,
            max_batch,
            stream,
        })
    }

    fn take_outputs(&mut self, batch: usize) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), EnginError> {
        if self.out_batch != batch || self.logits.is_none() || self.value.is_none() || self.moves_left.is_none() {
            self.logits = Some(Tensor::<f32>::new(&self.allocator, [batch, POLICY_SIZE]).map_err(onnx_error)?);
            self.value = Some(Tensor::<f32>::new(&self.allocator, [batch, 3]).map_err(onnx_error)?);
            self.moves_left = Some(Tensor::<f32>::new(&self.allocator, [batch, 1]).map_err(onnx_error)?);
            self.out_batch = batch;
        }
        Ok((
            self.logits.take().expect("logits"),
            self.value.take().expect("value"),
            self.moves_left.take().expect("moves_left"),
        ))
    }

    fn put_outputs(&mut self, logits: Tensor<f32>, value: Tensor<f32>, moves_left: Tensor<f32>) {
        self.logits = Some(logits);
        self.value = Some(value);
        self.moves_left = Some(moves_left);
    }

    fn clear_outputs(&mut self) {
        self.logits = None;
        self.value = None;
        self.moves_left = None;
        self.out_batch = 0;
    }
}

/// TensorRT：单动态 batch session；DirectML：固定形状 session 集。
struct OnnxSessions {
    #[cfg(feature = "directml")]
    provider: OnnxProvider,
    sessions: Vec<(usize, Session)>,
    input_scratch: Vec<f32>,
    #[cfg(feature = "tensorrt")]
    trt_gpu: Option<TrtGpu>,
}

impl OnnxSessions {
    fn single(session: Session) -> Self {
        Self {
            #[cfg(feature = "directml")]
            provider: OnnxProvider::Cpu,
            sessions: vec![(usize::MAX, session)],
            input_scratch: Vec::new(),
            #[cfg(feature = "tensorrt")]
            trt_gpu: None,
        }
    }

    #[cfg(feature = "tensorrt")]
    fn tensor_rt(session: Session, gpu: TrtGpu) -> Self {
        Self {
            sessions: vec![(usize::MAX, session)],
            input_scratch: Vec::new(),
            trt_gpu: Some(gpu),
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
        #[cfg(feature = "tensorrt")]
        {
            if self.trt_gpu.is_some() {
                TRT_MAX_BATCH
            } else {
                usize::MAX
            }
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
        #[cfg(feature = "tensorrt")]
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
        #[cfg(feature = "tensorrt")]
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
                maximum_batch_size: maximum_batch_size(provider),
            },
            provider,
        })
    }

    pub const fn provider(&self) -> OnnxProvider {
        self.provider
    }
}

#[cfg(feature = "tensorrt")]
fn create_sessions(path: &Path) -> Result<(OnnxSessions, OnnxProvider), EnginError> {
    match create_tensorrt_sessions(path) {
        Ok(sessions) => Ok((sessions, OnnxProvider::TensorRt)),
        Err(trt_error) => {
            let session = Session::builder()
                .map_err(onnx_error)?
                .commit_from_file(path)
                .map_err(onnx_error)?;
            eprintln!("ONNX TensorRT unavailable; using CPUExecutionProvider: {trt_error}");
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

#[cfg(feature = "tensorrt")]
fn recommended_batch_size(provider: OnnxProvider) -> usize {
    if provider == OnnxProvider::TensorRt {
        TRT_MAX_BATCH
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

/// UCI `MiniBatchSize` 硬顶与 option spin 对齐；单次 ORT run 仍按 provider 内部分块。
fn maximum_batch_size(_provider: OnnxProvider) -> usize {
    1024
}

/// ORT TensorRT EP：动态 batch、engine cache、user compute stream（px0 TRT 选项语义）。
#[cfg(feature = "tensorrt")]
fn create_tensorrt_sessions(path: &Path) -> Result<OnnxSessions, EnginError> {
    let stream = CudaStream::new()?;
    let cache_dir = trt_cache_dir()?;
    let board = format!("board:{TRT_PROFILE_MIN_BATCH}x{INPUT_PLANES}x{BOARD_ROWS}x{BOARD_COLS}");
    let board_opt = format!("board:{TRT_MAX_BATCH}x{INPUT_PLANES}x{BOARD_ROWS}x{BOARD_COLS}");
    let board_max = format!("board:{TRT_MAX_BATCH}x{INPUT_PLANES}x{BOARD_ROWS}x{BOARD_COLS}");
    let mut session = Session::builder()
        .map_err(onnx_error)?
        .with_execution_providers([unsafe {
            TensorRT::default()
                .with_device_id(0)
                .with_fp16(true)
                .with_builder_optimization_level(5)
                .with_engine_cache(true)
                .with_engine_cache_path(cache_dir.to_string_lossy())
                .with_timing_cache(true)
                .with_timing_cache_path(cache_dir.to_string_lossy())
                .with_force_sequential_engine_build(true)
                .with_context_memory_sharing(true)
                .with_layer_norm_fp32_fallback(true)
                .with_profile_min_shapes(&board)
                .with_profile_opt_shapes(&board_opt)
                .with_profile_max_shapes(&board_max)
                .with_compute_stream(stream.as_ptr().cast())
                .build()
                .error_on_failure()
        }])
        .map_err(onnx_error)?
        .commit_from_file(path)
        .map_err(onnx_error)?;
    validate_tensorrt_session(&mut session)?;
    let allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default).map_err(onnx_error)?,
    )
    .map_err(onnx_error)?;
    Ok(OnnxSessions::tensor_rt(
        session,
        TrtGpu::new(stream, allocator, TRT_MAX_BATCH)?,
    ))
}

#[cfg(feature = "tensorrt")]
fn trt_cache_dir() -> Result<std::path::PathBuf, EnginError> {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.join("trt_cache")))
        .unwrap_or_else(|| Path::new("trt_cache").to_path_buf());
    std::fs::create_dir_all(&dir).map_err(|error| EnginError::Onnx(format!("create trt_cache: {error}")))?;
    Ok(dir)
}

/// TensorRT EP 注册成功不代表首个 engine 能构建/执行；启动时以最小输入验证。
#[cfg(feature = "tensorrt")]
fn validate_tensorrt_session(session: &mut Session) -> Result<(), EnginError> {
    let board = Array4::<f32>::zeros((1, INPUT_PLANES, BOARD_ROWS, BOARD_COLS));
    let tensor = TensorRef::from_array_view(&board).map_err(onnx_error)?;
    session.run(ort::inputs!["board" => tensor]).map_err(onnx_error)?;
    Ok(())
}

/// 固定形状 DirectML session。
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

/// 稀疏合批推理。
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

    #[cfg(feature = "tensorrt")]
    if let Some(gpu) = sessions.trt_gpu.as_mut() {
        let session = &mut sessions.sessions[0].1;
        let chunk = gpu.max_batch.max(1);
        for chunk_start in (0..batch).step_by(chunk) {
            let end = (chunk_start + chunk).min(batch);
            infer_input_planes_trt(
                session,
                gpu,
                &samples[chunk_start..end],
                all_logits,
                all_wdl,
                all_moves_left,
            )?;
        }
        return Ok(());
    }

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

#[cfg(feature = "tensorrt")]
fn infer_input_planes_trt(
    session: &mut Session,
    gpu: &mut TrtGpu,
    samples: &[InputPlanes],
    all_logits: &mut Vec<f32>,
    all_wdl: &mut Vec<f32>,
    all_moves_left: &mut Vec<f32>,
) -> Result<(), EnginError> {
    let batch = samples.len();
    if batch > gpu.max_batch {
        return Err(EnginError::Onnx(format!("TRT batch {batch} > {}", gpu.max_batch)));
    }
    pack_sparse_planes(samples, &mut gpu.sparse_host);
    let n_planes = (batch * INPUT_PLANES) as u32;
    gpu.sparse_dev.upload_async(&gpu.sparse_host, &gpu.stream)?;
    expand_planes_async(&gpu.dense_dev, &gpu.sparse_dev, n_planes, &gpu.stream)?;
    gpu.stream.sync()?;

    let board = unsafe {
        // ORT 不拥有 dense_dev；见 TrtGpu 字段序（stream 最后销毁）。
        TensorRefMut::<f32>::from_raw(
            MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)
                .map_err(onnx_error)?,
            gpu.dense_dev.as_ptr(),
            Shape::new([batch as i64, INPUT_PLANES as i64, BOARD_ROWS as i64, BOARD_COLS as i64]),
        )
    }
    .map_err(onnx_error)?;

    let (logits_t, value_t, mlh_t) = gpu.take_outputs(batch)?;
    let restored = (|| -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), EnginError> {
        let mut binding = session.create_binding().map_err(onnx_error)?;
        binding.bind_input("board", &board).map_err(onnx_error)?;
        binding.bind_output("logits", logits_t).map_err(onnx_error)?;
        binding.bind_output("value", value_t).map_err(onnx_error)?;
        binding.bind_output("moves_left", mlh_t).map_err(onnx_error)?;
        let mut outputs = session.run_binding(&binding).map_err(onnx_error)?;
        binding.synchronize_outputs().map_err(onnx_error)?;

        gpu.logits_host.resize(batch * POLICY_SIZE, 0.0);
        gpu.value_host.resize(batch * 3, 0.0);
        gpu.moves_left_host.resize(batch, 0.0);
        {
            let logits = outputs
                .get("logits")
                .ok_or_else(|| EnginError::Onnx("missing logits".into()))?
                .downcast_ref::<TensorValueType<f32>>()
                .map_err(onnx_error)?;
            let value = outputs
                .get("value")
                .ok_or_else(|| EnginError::Onnx("missing value".into()))?
                .downcast_ref::<TensorValueType<f32>>()
                .map_err(onnx_error)?;
            let mlh = outputs
                .get("moves_left")
                .ok_or_else(|| EnginError::Onnx("missing moves_left".into()))?
                .downcast_ref::<TensorValueType<f32>>()
                .map_err(onnx_error)?;
            download_device_async(f32_bytes_mut(&mut gpu.logits_host), logits.data_ptr(), &gpu.stream)?;
            download_device_async(f32_bytes_mut(&mut gpu.value_host), value.data_ptr(), &gpu.stream)?;
            download_device_async(f32_bytes_mut(&mut gpu.moves_left_host), mlh.data_ptr(), &gpu.stream)?;
            gpu.stream.sync()?;
        }

        all_logits.extend_from_slice(&gpu.logits_host);
        all_wdl.extend_from_slice(&gpu.value_host);
        all_moves_left.extend_from_slice(&gpu.moves_left_host);

        let logits_t = outputs
            .remove("logits")
            .ok_or_else(|| EnginError::Onnx("missing logits".into()))?
            .downcast::<TensorValueType<f32>>()
            .map_err(onnx_error)?;
        let value_t = outputs
            .remove("value")
            .ok_or_else(|| EnginError::Onnx("missing value".into()))?
            .downcast::<TensorValueType<f32>>()
            .map_err(onnx_error)?;
        let mlh_t = outputs
            .remove("moves_left")
            .ok_or_else(|| EnginError::Onnx("missing moves_left".into()))?
            .downcast::<TensorValueType<f32>>()
            .map_err(onnx_error)?;
        Ok((logits_t, value_t, mlh_t))
    })();
    match restored {
        Ok((logits_t, value_t, mlh_t)) => {
            gpu.put_outputs(logits_t, value_t, mlh_t);
            Ok(())
        }
        Err(error) => {
            // take 之后失败：tensor 已随 binding/outputs drop；下次重建。
            gpu.clear_outputs();
            Err(error)
        }
    }
}

#[cfg(feature = "tensorrt")]
fn f32_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    let len = values.len() * size_of::<f32>();
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len) }
}

#[cfg(feature = "tensorrt")]
fn pack_sparse_planes(samples: &[InputPlanes], dest: &mut Vec<u8>) {
    let n = samples.len() * INPUT_PLANES;
    dest.clear();
    dest.reserve(n * (size_of::<u128>() + size_of::<f32>()));
    for plane in samples.iter().flat_map(|sample| sample.iter()) {
        dest.extend_from_slice(&plane.mask.to_le_bytes());
    }
    for plane in samples.iter().flat_map(|sample| sample.iter()) {
        dest.extend_from_slice(&plane.value.to_le_bytes());
    }
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
