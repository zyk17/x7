//! ONNX Runtime 加载 `x7.onnx`（`board` → `logits` + 可选 `value(WDL)`）。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::Array4;
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use ort::Error;

/// 单次推理结果（与 `export_onnx.py` 输出名一致）。
#[derive(Debug, Clone)]
pub struct PolicyOutputs {
    pub logits: Vec<f32>,
    /// WDL 概率 `[win, draw, loss]`。
    pub wdl: Option<[f32; 3]>,
}

/// 批量推理结果。
#[derive(Debug, Clone)]
pub struct BatchPolicyOutputs {
    pub logits: Vec<Vec<f32>>,
    pub wdl: Option<Vec<[f32; 3]>>,
}

/// ORT 会话实际选用的 execution provider（lc0 `OnnxProvider` / `network_->IsCpu()` 口径）。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ActiveExecutionProvider {
    Cuda,
    DirectMl,
    Cpu,
}

impl ActiveExecutionProvider {
    fn display_chain(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::DirectMl => "DirectML",
            Self::Cpu => "CPU",
        }
    }
}

/// 封装 ORT [`Session`]，batch 固定为 1。
pub struct PolicyOnnx {
    session: Session,
    model_path: PathBuf,
    active_provider: ActiveExecutionProvider,
}

/// lc0 backend 属性（线程 / batch 推算）。
#[derive(Clone, Copy, Debug)]
pub struct BackendAttributes {
    pub runs_on_cpu: bool,
    pub suggested_num_search_threads: usize,
    pub recommended_batch_size: usize,
    pub maximum_batch_size: usize,
}

/// lc0 `Network::GetMiniBatchSize()` 默认值（network.h:123）。
const LC0_NETWORK_DEFAULT_MINIBATCH: usize = 256;

/// lc0 ONNX `max_batch_size_` / `NetworkAsBackend`（network_onnx.cc:212, wrapper.cc:64）。
const LC0_ONNX_MAX_BATCH_SIZE: usize = 1024;

impl BackendAttributes {
    /// lc0 `NetworkAsBackend` ctor（wrapper.cc:57-64）+ ONNX `GetMiniBatchSize()`（network_onnx.cc:164-167）。
    pub(crate) fn from_active_provider(provider: ActiveExecutionProvider) -> Self {
        let runs_on_cpu = matches!(provider, ActiveExecutionProvider::Cpu);
        let suggested_num_search_threads = if runs_on_cpu {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            1
        };
        let (recommended_batch_size, maximum_batch_size) = onnx_minibatch_sizes(provider);
        Self {
            runs_on_cpu,
            suggested_num_search_threads,
            recommended_batch_size,
            maximum_batch_size,
        }
    }
}

/// lc0 ONNX backend `batch` / `steps` 选项（network_onnx.cc:844-874）。
fn onnx_minibatch_sizes(provider: ActiveExecutionProvider) -> (usize, usize) {
    let (default_batch, default_steps) = match provider {
        // DML / MIGraphX / CoreML：default_batch=16, steps=4 → GetMiniBatchSize()=64
        ActiveExecutionProvider::DirectMl => (16, 4),
        ActiveExecutionProvider::Cuda | ActiveExecutionProvider::Cpu => (-1, 1),
    };

    let mut batch = std::env::var("ENGIN_ONNX_BATCH")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(default_batch);
    let steps = std::env::var("ENGIN_ONNX_STEPS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(default_steps)
        .max(1);

    if batch <= 0 {
        return (LC0_NETWORK_DEFAULT_MINIBATCH, LC0_ONNX_MAX_BATCH_SIZE);
    }

    if batch as usize * steps as usize > LC0_ONNX_MAX_BATCH_SIZE {
        batch = (LC0_ONNX_MAX_BATCH_SIZE / steps as usize) as i32;
    }

    (
        batch as usize * steps as usize,
        LC0_ONNX_MAX_BATCH_SIZE,
    )
}

/// lc0 `StartThreads(0)`：`suggested_num_search_threads + !runs_on_cpu`。
pub fn resolved_search_threads(requested: usize, attrs: &BackendAttributes) -> usize {
    if requested == 0 {
        attrs.suggested_num_search_threads + usize::from(!attrs.runs_on_cpu)
    } else {
        requested.clamp(1, 1024)
    }
}

pub struct PolicySessionPool {
    sessions: Mutex<Vec<Arc<Mutex<PolicyOnnx>>>>,
    attributes: BackendAttributes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProviderMode {
    Auto,
    CudaThenCpu,
    CudaThenDmlThenCpu,
    DmlThenCpu,
    CpuOnly,
}

impl PolicyOnnx {
    /// 从 `.onnx` 路径构建会话。
    ///
    /// 按候选 EP 顺序逐个尝试创建会话，以**实际 commit 成功**的 provider 为准（对齐 lc0
    /// `network_->IsCpu()`，而非期望 chain 字符串）。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let provider_mode = ProviderMode::from_env();
        let (session, active_provider) = open_session(path, provider_mode)?;
        Self::check_io(&session)?;
        Ok(Self {
            session,
            model_path: path.to_path_buf(),
            active_provider,
        })
    }

    pub fn clone_session(&self) -> Result<Self, Error> {
        let session = build_session(&self.model_path, self.active_provider)?;
        Self::check_io(&session)?;
        Ok(Self {
            session,
            model_path: self.model_path.clone(),
            active_provider: self.active_provider,
        })
    }

    fn check_io(session: &Session) -> Result<(), Error> {
        let ins = session.inputs();
        if ins.len() != 1 {
            return Err(Error::new(format!("期望 1 个 ONNX 输入，得到 {}", ins.len())));
        }
        if ins[0].name() != "board" {
            return Err(Error::new(format!("期望输入名为 board，得到 {}", ins[0].name())));
        }
        Ok(())
    }

    /// 对局面张量推理（形状 `float32[1,124,10,9]`）。
    pub fn eval_board(&mut self, board: &Array4<f32>) -> Result<PolicyOutputs, Error> {
        Self::run_session_single(&mut self.session, board)
    }

    pub fn eval_boards(&mut self, boards: &Array4<f32>) -> Result<BatchPolicyOutputs, Error> {
        Self::run_session_batch(&mut self.session, boards)
    }

    fn run_session_single(session: &mut Session, board: &Array4<f32>) -> Result<PolicyOutputs, Error> {
        let tensor_ref = TensorRef::from_array_view(board)?;
        let outputs = session.run(ort::inputs!["board" => tensor_ref])?;

        let logits_val = outputs
            .get("logits")
            .ok_or_else(|| Error::new("ONNX 输出缺少 logits"))?;
        let (log_shape, log_slice) = logits_val.try_extract_tensor::<f32>()?;
        if log_shape.len() != 2 || log_shape[0] != 1 || log_shape[1] <= 0 {
            return Err(Error::new(format!("logits 形状期望 [1,V]，得到 {:?}", log_shape)));
        }
        let vocab = log_shape[1] as usize;
        if log_slice.len() != vocab {
            return Err(Error::new("logits 张量长度与形状不一致"));
        }
        let logits = log_slice.to_vec();

        let wdl = if let Some(v) = outputs.get("value") {
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            if shape.len() != 2 || shape[0] != 1 || shape[1] != 3i64 {
                return Err(Error::new(format!("value 形状期望 [1,3]，得到 {:?}", shape)));
            }
            let sum = data[0] + data[1] + data[2];
            if !sum.is_finite() || sum <= 0.0 {
                return Err(Error::new("value WDL 概率非法"));
            }
            Some([data[0], data[1], data[2]])
        } else {
            None
        };

        Ok(PolicyOutputs { logits, wdl })
    }

    fn run_session_batch(session: &mut Session, board: &Array4<f32>) -> Result<BatchPolicyOutputs, Error> {
        let tensor_ref = TensorRef::from_array_view(board)?;
        let outputs = session.run(ort::inputs!["board" => tensor_ref])?;

        let logits_val = outputs
            .get("logits")
            .ok_or_else(|| Error::new("ONNX 输出缺少 logits"))?;
        let (log_shape, log_slice) = logits_val.try_extract_tensor::<f32>()?;
        if log_shape.len() != 2 || log_shape[0] <= 0 || log_shape[1] <= 0 {
            return Err(Error::new(format!("logits 形状期望 [B,V]，得到 {:?}", log_shape)));
        }
        let batch = log_shape[0] as usize;
        let vocab = log_shape[1] as usize;
        if log_slice.len() != batch * vocab {
            return Err(Error::new("logits 张量长度与形状不一致"));
        }
        let logits = log_slice.chunks(vocab).map(|chunk| chunk.to_vec()).collect::<Vec<_>>();

        let wdl = if let Some(v) = outputs.get("value") {
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            if shape.len() != 2 || shape[0] != batch as i64 || shape[1] != 3i64 {
                return Err(Error::new(format!("value 形状期望 [{batch},3]，得到 {:?}", shape)));
            }
            let mut wdls = Vec::with_capacity(batch);
            for chunk in data.chunks(3) {
                let sum = chunk[0] + chunk[1] + chunk[2];
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(Error::new("value WDL 概率非法"));
                }
                wdls.push([chunk[0], chunk[1], chunk[2]]);
            }
            Some(wdls)
        } else {
            None
        };

        Ok(BatchPolicyOutputs { logits, wdl })
    }

    /// FEN → 平面 → 推理。
    pub fn eval_fen(&mut self, fen: &str) -> Result<PolicyOutputs, Error> {
        let board = crate::fen_tensor::fen_to_planes(fen).map_err(Error::new)?;
        self.eval_board(&board)
    }

    /// [`xiangqi_core::Position`] → 平面 → 推理（搜索热路径，无 FEN 分配）。
    pub fn eval_position(&mut self, pos: &xiangqi_core::Position) -> Result<PolicyOutputs, Error> {
        let board = crate::fen_tensor::position_to_planes(pos).map_err(Error::new)?;
        self.eval_board(&board)
    }

    /// 实际加载成功的 execution provider（非期望 chain）。
    pub fn provider_chain(&self) -> &'static str {
        self.active_provider.display_chain()
    }
}

impl PolicySessionPool {
    pub fn new(net: PolicyOnnx) -> Self {
        let attributes = BackendAttributes::from_active_provider(net.active_provider);
        Self {
            sessions: Mutex::new(vec![Arc::new(Mutex::new(net))]),
            attributes,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self::new(PolicyOnnx::from_file(path)?))
    }

    pub fn primary(&self) -> Arc<Mutex<PolicyOnnx>> {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .first()
            .cloned()
            .expect("policy session pool must contain at least one session")
    }

    pub fn provider_chain(&self) -> &'static str {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .first()
            .map(|session| {
                session
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .provider_chain()
            })
            .unwrap_or("CPU")
    }

    pub fn backend_attributes(&self) -> BackendAttributes {
        self.attributes
    }

    pub fn resize_sessions(&self, count: usize) -> Result<Vec<Arc<Mutex<PolicyOnnx>>>, Error> {
        let target = count.max(1);
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        while sessions.len() < target {
            let cloned = sessions
                .first()
                .expect("policy session pool must contain at least one session")
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone_session()?;
            sessions.push(Arc::new(Mutex::new(cloned)));
        }
        sessions.truncate(target);
        Ok(sessions.iter().take(target).cloned().collect())
    }
}

fn open_session(
    path: &Path,
    mode: ProviderMode,
) -> Result<(Session, ActiveExecutionProvider), Error> {
    let mut last_err = None;
    for provider in mode.candidate_providers() {
        match build_session(path, *provider) {
            Ok(session) => return Ok((session, *provider)),
            Err(err) => last_err = Some(err),
        }
    }
    Err(last_err.unwrap_or_else(|| Error::new("无法创建 ONNX 会话")))
}

fn build_session(path: &Path, provider: ActiveExecutionProvider) -> Result<Session, Error> {
    let intra_threads = ort_intra_threads(provider);
    let inter_threads = ort_inter_threads(provider);
    let mut builder = Session::builder()?
        .with_parallel_execution(inter_threads > 1)?
        .with_intra_threads(intra_threads)?
        .with_inter_threads(inter_threads)?
        .with_memory_pattern(true)?
        .with_optimization_level(GraphOptimizationLevel::All)?;
    builder = match provider {
        ActiveExecutionProvider::Cuda => builder.with_execution_providers([cuda_provider()])?,
        ActiveExecutionProvider::DirectMl => builder.with_execution_providers([ep::DirectML::default()
            .with_device_id(0)
            .build()])?,
        ActiveExecutionProvider::Cpu => builder.with_execution_providers([cpu_provider()])?,
    };
    builder.commit_from_file(path)
}

fn cuda_provider() -> ort::execution_providers::ExecutionProviderDispatch {
    ep::CUDA::default()
        .with_device_id(0)
        .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
        .with_conv_max_workspace(true)
        .build()
}

fn cpu_provider() -> ort::execution_providers::ExecutionProviderDispatch {
    ep::CPU::default().with_arena_allocator(true).build()
}

fn ort_intra_threads(provider: ActiveExecutionProvider) -> usize {
    std::env::var("ENGIN_ORT_INTRA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| default_ort_intra_threads(provider))
}

fn ort_inter_threads(provider: ActiveExecutionProvider) -> usize {
    std::env::var("ENGIN_ORT_INTER_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| default_ort_inter_threads(provider))
}

fn default_ort_intra_threads(provider: ActiveExecutionProvider) -> usize {
    if !matches!(provider, ActiveExecutionProvider::Cpu) {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4)
}

fn default_ort_inter_threads(provider: ActiveExecutionProvider) -> usize {
    if !matches!(provider, ActiveExecutionProvider::Cpu) {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 4))
        .unwrap_or(2)
}

impl ProviderMode {
    fn from_env() -> Self {
        if let Ok(value) = std::env::var("ENGIN_EP") {
            return match value.to_ascii_lowercase().as_str() {
                "auto" => Self::Auto,
                "cuda" | "cuda_cpu" => Self::CudaThenCpu,
                "cuda_dml" | "cuda_dml_cpu" => Self::CudaThenDmlThenCpu,
                "dml" | "dml_cpu" => Self::DmlThenCpu,
                "cpu" => Self::CpuOnly,
                _ => Self::default_mode(),
            };
        }
        if let Ok(value) = std::env::var("ENGIN_ENABLE_DML") {
            return if matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON") {
                Self::CudaThenDmlThenCpu
            } else {
                Self::CudaThenCpu
            };
        }
        Self::default_mode()
    }

    fn default_mode() -> Self {
        if cfg!(all(target_os = "windows", not(test))) {
            Self::Auto
        } else {
            Self::CudaThenCpu
        }
    }

    fn candidate_providers(self) -> &'static [ActiveExecutionProvider] {
        use ActiveExecutionProvider as P;
        match self {
            Self::Auto | Self::CudaThenDmlThenCpu => &[P::Cuda, P::DirectMl, P::Cpu],
            Self::CudaThenCpu => &[P::Cuda, P::Cpu],
            Self::DmlThenCpu => &[P::DirectMl, P::Cpu],
            Self::CpuOnly => &[P::Cpu],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PositionHistory;
    use ndarray::s;
    use std::path::PathBuf;
    use xiangqi_core::uci_to_move;

    #[test]
    fn backend_attributes_cpu_recommended_batch_matches_lc0_onnx() {
        let attrs = BackendAttributes::from_active_provider(ActiveExecutionProvider::Cpu);
        assert_eq!(attrs.recommended_batch_size, LC0_NETWORK_DEFAULT_MINIBATCH);
        assert_eq!(attrs.maximum_batch_size, LC0_ONNX_MAX_BATCH_SIZE);
    }

    #[test]
    fn backend_attributes_dml_recommended_batch_matches_lc0_onnx() {
        let attrs = BackendAttributes::from_active_provider(ActiveExecutionProvider::DirectMl);
        assert!(!attrs.runs_on_cpu);
        assert_eq!(attrs.recommended_batch_size, 64);
        assert_eq!(attrs.maximum_batch_size, LC0_ONNX_MAX_BATCH_SIZE);
    }

    #[test]
    fn backend_attributes_follow_actual_cpu_provider() {
        let attrs = BackendAttributes::from_active_provider(ActiveExecutionProvider::Cpu);
        assert!(attrs.runs_on_cpu);
        assert!(attrs.suggested_num_search_threads >= 1);
        assert_eq!(resolved_search_threads(0, &attrs), attrs.suggested_num_search_threads);
    }

    #[test]
    fn backend_attributes_follow_actual_gpu_provider() {
        let attrs = BackendAttributes::from_active_provider(ActiveExecutionProvider::Cuda);
        assert!(!attrs.runs_on_cpu);
        assert_eq!(attrs.suggested_num_search_threads, 1);
        assert_eq!(resolved_search_threads(0, &attrs), 2);
    }

    fn candidate_onnx_paths() -> [PathBuf; 2] {
        [
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/checkpoints/x7.onnx"),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/x7.onnx"),
        ]
    }

    #[test]
    fn eval_startpos_if_policy_onnx_present() {
        let Some(path) = candidate_onnx_paths().into_iter().find(|p| p.is_file()) else {
            eprintln!("skip: no candidate onnx");
            return;
        };
        if !path.is_file() {
            eprintln!("skip: {}", path.display());
            return;
        }
        let mut p = PolicyOnnx::from_file(&path).expect("load onnx");
        let out = match p.eval_fen(crate::START_FEN) {
            Ok(out) => out,
            Err(err) if err.to_string().contains("value 形状期望 [1,3]") => {
                eprintln!("skip old scalar x7.onnx: {err}");
                return;
            }
            Err(err) => panic!("infer: {err}"),
        };
        assert!(!out.logits.is_empty());
        eprintln!("loaded onnx provider: {}", p.provider_chain());
    }

    #[test]
    fn eval_batch_if_policy_onnx_present() {
        let Some(path) = candidate_onnx_paths().into_iter().find(|p| p.is_file()) else {
            eprintln!("skip: no candidate onnx");
            return;
        };
        let mut net = PolicyOnnx::from_file(&path).expect("load onnx");

        let history_a = PositionHistory::new_startpos();
        let mut history_b = PositionHistory::new_startpos();
        let mv = uci_to_move(history_b.current(), "h2e2").expect("legal move");
        history_b.push_move(mv);

        let mut batch = Array4::<f32>::zeros((2, crate::fen_tensor::PX0_INPUT_SHAPE.1, 10, 9));
        let mut single = Array4::<f32>::zeros(crate::fen_tensor::PX0_INPUT_SHAPE);
        crate::fen_tensor::history_to_planes_into(&history_a, &mut single).expect("planes a");
        batch.slice_mut(s![0..1, .., .., ..]).assign(&single);
        crate::fen_tensor::history_to_planes_into(&history_b, &mut single).expect("planes b");
        batch.slice_mut(s![1..2, .., .., ..]).assign(&single);

        let out = match net.eval_boards(&batch) {
            Ok(out) => out,
            Err(err) if err.to_string().contains("value 形状期望 [2,3]") => {
                eprintln!("skip non-batched or old-scalar onnx: {err}");
                return;
            }
            Err(err) => panic!("infer batch: {err}"),
        };
        assert_eq!(out.logits.len(), 2);
        assert!(!out.logits[0].is_empty());
        assert!(!out.logits[1].is_empty());
    }

    #[test]
    fn pool_can_grow_and_shrink_if_policy_onnx_present() {
        let Some(path) = candidate_onnx_paths().into_iter().find(|p| p.is_file()) else {
            eprintln!("skip: no candidate onnx");
            return;
        };
        let pool = PolicySessionPool::from_file(&path).expect("load pool");
        let attrs = pool.backend_attributes();
        if pool.provider_chain() == "CPU" {
            assert!(attrs.runs_on_cpu);
        }
        let grown = pool.resize_sessions(3).expect("grow");
        assert_eq!(grown.len(), 3);
        let shrunk = pool.resize_sessions(1).expect("shrink");
        assert_eq!(shrunk.len(), 1);
    }

    #[test]
    #[ignore = "slow onnx parallel smoke; run manually"]
    fn parallel_mcts_reaches_nodes_budget_with_onnx() {
        use std::sync::Arc;
        use std::time::Duration;

        use crate::history::PositionHistory;
        use crate::mcts::{MctsBudget, MctsConfig, MctsEngine, OnnxPolicyValueEval};

        let Some(path) = candidate_onnx_paths().into_iter().find(|p| p.is_file()) else {
            eprintln!("skip: no candidate onnx");
            return;
        };
        let pool = PolicySessionPool::from_file(&path).expect("load pool");
        let policy = Arc::new(pool);
        let mut engine = MctsEngine::new(
            MctsConfig {
                minibatch_size: 256,
                ..MctsConfig::default()
            },
            OnnxPolicyValueEval::new(Some(policy), MctsConfig::default().nn_cache_size),
        );
        let history = PositionHistory::new_startpos();
        let result = engine
            .search_root_history_parallel_with_progress(
                &history,
                MctsBudget {
                    max_playouts: None,
                    max_nodes: Some(2048),
                    max_depth: None,
                    deadline: None,
                    stop: None,
                },
                8,
                Duration::ZERO,
                |_| {},
            )
            .expect("parallel onnx search must finish");
        assert_eq!(result.nodes, 2048);
        assert!(result.best_move.is_some());
        assert!(
            result.seldepth >= 10,
            "seldepth should grow with search: got {}",
            result.seldepth
        );
        let nps = crate::mcts::SearchStats::playouts_per_second(result.playouts, result.nps_elapsed_ms);
        eprintln!(
            "parallel_mcts: nodes={} playouts={} seldepth={} pv_len={} nps={} time_ms~{}",
            result.nodes,
            result.playouts,
            result.seldepth,
            result.pv.len(),
            nps,
            result.nps_elapsed_ms
        );
    }
}
