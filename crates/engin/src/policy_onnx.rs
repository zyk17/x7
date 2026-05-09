//! ONNX Runtime 加载 `policy.onnx`（`board` → `logits` + 可选 `attack` / `danger` / `tactical`）。

use std::path::Path;

use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use ort::Error;

/// 单次推理结果（与 `export_onnx.py` 输出名一致）。
#[derive(Debug, Clone)]
pub struct PolicyOutputs {
    pub logits: Vec<f32>,
    pub attack: Option<f32>,
    pub danger: Option<f32>,
    pub tactical: Option<f32>,
    /// 局面价值，ONNX 图中一般为 **tanh**，约 **[-1,1]**（与 `export_onnx.py` 一致）。
    pub value: Option<f32>,
}

/// 封装 ORT [`Session`]，batch 固定为 1。
pub struct PolicyOnnx {
    session: Session,
}

impl PolicyOnnx {
    /// 从 `.onnx` 路径构建会话（CPU EP；`ort` 默认配置）。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let session = Session::builder()?.commit_from_file(path)?;
        Self::check_io(&session)?;
        Ok(Self { session })
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

    /// 对局面张量推理（形状 `float32[1,15,10,9]`）。
    pub fn eval_board(&mut self, board: &Array4<f32>) -> Result<PolicyOutputs, Error> {
        let tensor_ref = TensorRef::from_array_view(board)?;
        let outputs = self.session.run(ort::inputs!["board" => tensor_ref])?;

        let logits_val = outputs
            .get("logits")
            .ok_or_else(|| Error::new("ONNX 输出缺少 logits"))?;
        let (log_shape, log_slice) = logits_val.try_extract_tensor::<f32>()?;
        if log_shape.len() != 2 || log_shape[0] != 1i64 {
            return Err(Error::new(format!("logits 形状期望 [1,V]，得到 {:?}", log_shape)));
        }
        let logits = log_slice.to_vec();

        let f1 = |name: &str| -> Result<Option<f32>, Error> {
            let Some(v) = outputs.get(name) else {
                return Ok(None);
            };
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            if shape.num_elements() != 1 {
                return Err(Error::new(format!("{name} 期望标量张量，形状 {:?}", shape)));
            }
            Ok(Some(data[0]))
        };

        Ok(PolicyOutputs {
            logits,
            attack: f1("attack")?,
            danger: f1("danger")?,
            tactical: f1("tactical")?,
            value: f1("value")?,
        })
    }

    /// FEN → 平面 → 推理。
    pub fn eval_fen(&mut self, fen: &str) -> Result<PolicyOutputs, Error> {
        let board = crate::fen_to_planes(fen).map_err(Error::new)?;
        self.eval_board(&board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_startpos_if_policy_onnx_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/policy.onnx");
        if !path.is_file() {
            eprintln!("skip: {}", path.display());
            return;
        }
        let mut p = PolicyOnnx::from_file(&path).expect("load onnx");
        let out = p.eval_fen(crate::START_FEN).expect("infer");
        assert!(!out.logits.is_empty());
        assert!(out.attack.is_some());
        assert!(out.danger.is_some());
        assert!(out.tactical.is_some());
    }
}
