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

/// 封装 ORT [`Session`]，batch 固定为 1。
pub struct PolicyOnnx {
    session: Session,
    model_path: PathBuf,
    provider_chain: &'static str,
}

pub struct PolicySessionPool {
    model_path: PathBuf,
    sessions: Mutex<Vec<Arc<Mutex<PolicyOnnx>>>>,
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
    /// 当前优先级与 `px0/lc0` 风格对齐到：
    /// - CUDA
    /// - DirectML（Windows fallback）
    /// - CPU
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let provider_mode = ProviderMode::from_env();
        let intra_threads = ort_intra_threads(provider_mode);
        let inter_threads = ort_inter_threads(provider_mode);
        let mut builder = Session::builder()?
            .with_parallel_execution(inter_threads > 1)?
            .with_intra_threads(intra_threads)?
            .with_inter_threads(inter_threads)?
            .with_memory_pattern(true)?
            .with_optimization_level(GraphOptimizationLevel::All)?;
        builder = match provider_mode {
            ProviderMode::Auto | ProviderMode::CudaThenDmlThenCpu => builder.with_execution_providers([
                cuda_provider(),
                ep::DirectML::default().with_device_id(0).build(),
                cpu_provider(),
            ])?,
            ProviderMode::CudaThenCpu => builder.with_execution_providers([cuda_provider(), cpu_provider()])?,
            ProviderMode::DmlThenCpu => {
                builder.with_execution_providers([ep::DirectML::default().with_device_id(0).build(), cpu_provider()])?
            }
            ProviderMode::CpuOnly => builder.with_execution_providers([cpu_provider()])?,
        };
        let session = builder.commit_from_file(path)?;
        Self::check_io(&session)?;
        Ok(Self {
            session,
            model_path: path.to_path_buf(),
            provider_chain: provider_mode.provider_chain(),
        })
    }

    pub fn clone_session(&self) -> Result<Self, Error> {
        Self::from_file(&self.model_path)
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

    pub fn provider_chain(&self) -> &'static str {
        self.provider_chain
    }
}

impl PolicySessionPool {
    pub fn new(net: PolicyOnnx) -> Self {
        let model_path = net.model_path.clone();
        Self {
            model_path,
            sessions: Mutex::new(vec![Arc::new(Mutex::new(net))]),
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
        let session = self.primary();
        let chain = session.lock().unwrap_or_else(|e| e.into_inner()).provider_chain();
        chain
    }

    pub fn resize_sessions(&self, count: usize) -> Result<Vec<Arc<Mutex<PolicyOnnx>>>, Error> {
        let target = count.max(1);
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        while sessions.len() < target {
            sessions.push(Arc::new(Mutex::new(PolicyOnnx::from_file(&self.model_path)?)));
        }
        sessions.truncate(target);
        Ok(sessions.iter().take(target).cloned().collect())
    }
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

fn ort_intra_threads(provider_mode: ProviderMode) -> usize {
    std::env::var("ENGIN_ORT_INTRA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| default_ort_intra_threads(provider_mode))
}

fn ort_inter_threads(provider_mode: ProviderMode) -> usize {
    std::env::var("ENGIN_ORT_INTER_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| default_ort_inter_threads(provider_mode))
}

fn default_ort_intra_threads(provider_mode: ProviderMode) -> usize {
    if provider_mode.prefers_gpu() {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4)
}

fn default_ort_inter_threads(provider_mode: ProviderMode) -> usize {
    if provider_mode.prefers_gpu() {
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

    fn provider_chain(self) -> &'static str {
        match self {
            Self::Auto | Self::CudaThenDmlThenCpu => "CUDA -> DirectML -> CPU",
            Self::CudaThenCpu => "CUDA -> CPU",
            Self::DmlThenCpu => "DirectML -> CPU",
            Self::CpuOnly => "CPU",
        }
    }

    fn prefers_gpu(self) -> bool {
        !matches!(self, Self::CpuOnly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PositionHistory;
    use ndarray::s;
    use std::path::PathBuf;
    use xiangqi_core::uci_to_move;

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
        let grown = pool.resize_sessions(3).expect("grow");
        assert_eq!(grown.len(), 3);
        let shrunk = pool.resize_sessions(1).expect("shrink");
        assert_eq!(shrunk.len(), 1);
    }
}
