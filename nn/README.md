# nn

当前 Python 训练栈主线已经固定为：

- `PX0 v6 chunks`
- `b15c384bt192` 正式网络，训练期含两个辅助 head
- 输入 `124 x 10 x 9`
- 输出 `2062 policy + WDL + moves-left`

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
- 正式 value head 只学习最终 WDL；独立 root-WDL 辅助 head 学当前 root search target，不再 qMix
- Auxiliary Soft Policy 使用 `T=4`，两个辅助 head 均不导出 ONNX
- 当前不做棋盘镜像增强：KataGo 的 8 种方形棋盘对称不能直接用于 `10x9` 象棋。若以后加入水平镜像，必须同时严格变换
  `124` 个输入平面和 `2062` 个 policy target，不能只改 FEN 或 UCI 字符串。
- x7 v2 trunk 为 PreAct bottleneck + 两次 mean/max Global Broadcast；当前基准是
  `stem 124->384 + 15x(384->192->192->384)`
- CUDA 训练为 FP16 trunk autocast、FP32 heads/loss；ONNX 默认是 FP16 trunk、FP32 input/heads/outputs
- 训练入口只接受 `--config`
- 本地目录约定为 `C:\work\px0data\{version}\`
- `nn` 包内自带 px0 的 2062 policy 着法表；训练和 ONNX 导出不依赖 `crates/` 或 Rust 工程。

## 独立训练

将整个 `nn/` 目录复制到任意机器后，仅需 Python 环境：

```powershell
cd nn
python -m pip install -e ".[train,dev]"
Copy-Item configs/example.yaml configs/x7_v3_01.yaml
python scripts/data/prepare_px0.py --config configs/x7_v3_01.yaml
python scripts/train/train_px0.py --config configs/x7_v3_01.yaml
```

训练只接受一个 YAML 配置，布局参考 pxzero-training 的 `dataset / model / training`：

- `dataset.px0_version`：Kaggle px0data 版本；本地 `C:\work\px0data\{version}` 由准备脚本一次性建立。
- `model.width`、`model.blocks`、`model.bottleneck_channels`：当前基准为 `384`、`15`、`192`。两个
  Global Broadcast 固定分布在三个 trunk stage 之间；同一 checkpoint 的续训或 `init_from` 必须使用
  完全相同的三项模型尺寸。省略 `bottleneck_channels` 时使用 `width / 2`。
- value 与 moves-left 共享轻量 global readout：`1x1 conv -> mean/max pool -> FC`，再分别输出 WDL 与
  剩余步数；不再使用展平棋盘的大型全连接 head。
- `dataset.validation_samples`：固定验证局面数。首次运行会按确定性随机 chunk 顺序建立 record-level
  manifest，并按当前盘面子力分为开局/中局/残局三档平衡抽样；`validation_source_files: 0` 会在达到三档
  配额后停止扫描，正数则限制扫描文件数。它不是不可获得的真实对局 ply 阶段，但能避免只读验证文件固定前缀。
- `training.shuffle_size`：训练 record 的有界 replacement shuffle 总大小，按 DataLoader worker 均分。
  这对应 pxzero 的 shuffle buffer，避免顺序读取同一对局的相邻局面；默认 `4096`，约占 200 MB 主存。
- `training.final_value_loss_weight + root_wdl_loss_weight + moves_left_loss_weight`：三项都服务 value 表示。前两项分别监督最终 WDL 与 PX0 当前 root WDL；moves-left 是间接辅助。当前小网络对照的暂定基线为 `0.6 / 0.6 / 0.5`，它们不是可按 loss 数值直接比较的比例。root WDL 不是 KataGo 的未来时间平均 target。
- `training.soft_policy_weight + soft_policy_temperature`：训练期 Soft Policy 辅助头，默认 `8.0` / `4.0`。
- `training.lr + warmup_steps + min_lr_scale`：线性 warmup 后 cosine decay。首次训练会把 cosine horizon 写入 checkpoint；后续仅延长 `steps` 时保持该 horizon，并在 floor LR 继续。优化器固定为 AdamW；Conv/Linear 权重做 decoupled decay，BatchNorm 与 bias 不 decay。

## 当前验证与下一步

在 PX0 v677 上，`b10c64bt32` 的 1000-step 对照确认 `lr=3e-4` 可稳定训练；`moves-left=0.5` 在该小网络上优于 `1.0`。这只是超参数筛选，不代表 b15c384bt192 的最终结论，正式训练仍须复验固定 validation 的 policy、final WDL 与 root WDL。

`root_m` 已确认与真实 `plies_left` 相关但不等价，因此暂不加入模型。若后续实验，必须是训练期独立的 root-moves-left 辅助 head；只在它稳定改善 final/root WDL 时保留，不能用自身 loss 作为保留依据。

输入 `124x10x9`、policy `2062`、WDL value、辅助 moves-left head 和现有 loss 都不是可配置项，避免训练契约漂移。配置内相对路径按启动目录解析。
完整字段、默认值和注释见 [configs/example.yaml](C:/projects/77xiangqi_engine/nn/configs/example.yaml)；实际实验应复制
该文件后改名，而不是直接改示例文件。除 `example.yaml` 外的 `configs/*.yaml` 都是本地实验文件，不进入 Git。

## 数据准备与训练

先运行一次准备命令。它是唯一可能下载、解压、遍历 chunk 和构建固定 validation manifest 的入口：

```powershell
python scripts/data/prepare_px0.py --config configs/x7_v3_01.yaml
```

之后每次训练只读取 `train.json`、`val.json` 和已有 validation manifest，不会再下载、解压或扫描数据。若数据未准备，
训练会立即失败并提示上述准备命令：

```powershell
python scripts/train/train_px0.py --config configs/x7_v3_01.yaml
```

导出 ONNX 同样只依赖该 Python 包：

```powershell
python scripts/export/export_onnx.py --checkpoint data/checkpoints/x7.pt --out data/checkpoints/x7.onnx --precision mixed-fp16
```
