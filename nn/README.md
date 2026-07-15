# nn

当前 Python 训练栈主线已经固定为：

- `PX0 v6 chunks`
- 小型 `policy + value`
- 输入 `124 x 10 x 9`
- 输出 `2062 + WDL`

## 当前目录职责

- `src/nn/px0_record.py`
  解析最小 `px0 v6 classical` record
- `src/nn/dataset_px0.py`
  流式读取 `px0` chunks
- `src/nn/px0_kaggle.py`
  管理 `Kaggle -> C:\work\px0data\{version}\ -> manifest`
- `scripts/train/train_px0.py`
  单一 YAML 主线训练入口
- `scripts/export/export_onnx.py`
  导出 ONNX

## 当前原则

- 默认先纯 `px0`
- 不预设人类数据混入
- value 主语义为 `WDL + qMix`
- `q_ratio` 采用 `px0` 风格固定标量，并允许分阶段切换
- 默认 baseline 容量为 `10x160`（由配置明确记录）
- 训练入口只接受 `--config`
- 本地目录约定为 `C:\work\px0data\{version}\`
- `nn` 包内自带 px0 的 2062 policy 着法表；训练和 ONNX 导出不依赖 `crates/` 或 Rust 工程。

## 独立训练

将整个 `nn/` 目录复制到任意机器后，仅需 Python 环境：

```powershell
cd nn
python -m pip install -e ".[train,dev]"
Copy-Item configs/example.yaml configs/x7_qmix_075_01.yaml
python scripts/train/train_px0.py --config configs/x7_qmix_075_01.yaml
```

训练只接受一个 YAML 配置，布局参考 pxzero-training 的 `dataset / model / training`：

- `dataset.px0_version`：Kaggle px0data 版本；本地 `C:\work\px0data\{version}` 已完整准备时直接复用。
- `model.width`、`model.blocks`：纯 CNN trunk 容量。
- `training`：checkpoint、步数、batch、优化器和固定 `q_ratio`。

输入 `124x10x9`、policy `2062`、WDL value、辅助 moves-left head 和现有 loss 都不是可配置项，避免训练契约漂移。配置内相对路径按启动目录解析。
完整字段、默认值和注释见 [configs/example.yaml](C:/projects/77xiangqi_engine/nn/configs/example.yaml)；实际实验应复制
该文件后改名，而不是直接改示例文件。除 `example.yaml` 外的 `configs/*.yaml` 都是本地实验文件，不进入 Git。

导出 ONNX 同样只依赖该 Python 包：

```powershell
python scripts/export/export_onnx.py --checkpoint data/checkpoints/x7.pt --out data/checkpoints/x7.onnx
```
