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
  只读取已准备数据的训练入口
- `scripts/data/prepare_px0.py`
  一次性下载、解压、文件切分与固定验证 manifest 准备入口
- `scripts/export/export_onnx.py`
  导出 ONNX

## 当前原则

- 默认先纯 `px0`
- 不预设人类数据混入
- value 主语义为 `WDL + qMix`
- `q_ratio` 采用 `px0` 风格固定标量，并允许分阶段切换
- x7 v2 trunk 为 PreAct bottleneck + 两次 mean/max Global Broadcast；默认基准是
  `stem 124->256 + 12x(256->112->112->256)`，实验可调整 width/blocks/bottleneck_channels
- 训练入口只接受 `--config`
- 本地目录约定为 `C:\work\px0data\{version}\`
- `nn` 包内自带 px0 的 2062 policy 着法表；训练和 ONNX 导出不依赖 `crates/` 或 Rust 工程。

## 独立训练

将整个 `nn/` 目录复制到任意机器后，仅需 Python 环境：

```powershell
cd nn
python -m pip install -e ".[train,dev]"
Copy-Item configs/example.yaml configs/x7_qmix_075_01.yaml
python scripts/data/prepare_px0.py --config configs/x7_qmix_075_01.yaml
python scripts/train/train_px0.py --config configs/x7_qmix_075_01.yaml
```

训练只接受一个 YAML 配置，布局参考 pxzero-training 的 `dataset / model / training`：

- `dataset.px0_version`：Kaggle px0data 版本；本地 `C:\work\px0data\{version}` 由准备脚本一次性建立。
- `model.width`、`model.blocks`、`model.bottleneck_channels`：默认基准为 `256`、`12`、`112`，可在同一 v2 结构族内实验。两个
  Global Broadcast 固定分布在三个 trunk stage 之间；同一 checkpoint 的续训或 `init_from` 必须使用
  完全相同的三项模型尺寸。省略 `bottleneck_channels` 时使用原始默认 `width * 7 // 16`。
- value 与 moves-left 共享轻量 global readout：`1x1 conv -> mean/max pool -> FC`，再分别输出 WDL 与
  剩余步数；不再使用展平棋盘的大型全连接 head。
- `dataset.validation_samples`：固定验证局面数。首次运行会按确定性随机 chunk 顺序建立 record-level
  manifest，并按当前盘面子力分为开局/中局/残局三档平衡抽样；`validation_source_files: 0` 会在达到三档
  配额后停止扫描，正数则限制扫描文件数。它不是不可获得的真实对局 ply 阶段，但能避免只读验证文件固定前缀。
- `training.shuffle_size`：训练 record 的有界 replacement shuffle 总大小，按 DataLoader worker 均分。
  这对应 pxzero 的 shuffle buffer，避免顺序读取同一对局的相邻局面；默认 `4096`，约占 200 MB 主存。
- `training`：checkpoint、步数、batch、分阶段学习率和固定 `q_ratio`。学习率采用 pxzero-training 风格
  的 `warmup_steps + lr_values + lr_boundaries`，优化器固定为 `SGD(momentum=0.9, nesterov=true)`；边界相对当前 qMix phase 计算；切换 qMix 会保留模型、
  重置优化器并从新的 phase-local warmup 开始。

输入 `124x10x9`、policy `2062`、WDL value、辅助 moves-left head 和现有 loss 都不是可配置项，避免训练契约漂移。配置内相对路径按启动目录解析。
完整字段、默认值和注释见 [configs/example.yaml](C:/projects/77xiangqi_engine/nn/configs/example.yaml)；实际实验应复制
该文件后改名，而不是直接改示例文件。除 `example.yaml` 外的 `configs/*.yaml` 都是本地实验文件，不进入 Git。

## 数据准备与训练

先运行一次准备命令。它是唯一可能下载、解压、遍历 chunk 和构建固定 validation manifest 的入口：

```powershell
python scripts/data/prepare_px0.py --config configs/x7_qmix_075_01.yaml
```

之后每次训练只读取 `train.json`、`val.json` 和已有 validation manifest，不会再下载、解压或扫描数据。若数据未准备，
训练会立即失败并提示上述准备命令：

```powershell
python scripts/train/train_px0.py --config configs/x7_qmix_075_01.yaml
```

导出 ONNX 同样只依赖该 Python 包：

```powershell
python scripts/export/export_onnx.py --checkpoint data/checkpoints/x7.pt --out data/checkpoints/x7.onnx
```
