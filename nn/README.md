# nn

当前 Python 训练栈主线已经固定为：

- `PX0 v6 chunks`
- `x7_v3_attentionbody` 正式实验网络，训练期含两个辅助 head；`b15c384bt192` CNN 保留为对照基线
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

## Network Vision

**使命：**构建一个足够强、持续跟进业界最佳实践的 **Knowledge Model**，在单机训练和消费级 GPU
推理的预算内提供高质量 Prediction，并最终以固定时间 Elo 验证价值。网络是 Knowledge 的载体，
不是项目的主要研究对象。

- **Adopt, don't invent**：优先吸收 Lc0、KataGo 已验证有效的 trunk、位置编码、normalization 与
  attention 进展；不把自行发明 block、attention、activation 或 loss 作为长期主线。
- **简单推理接口**：正式 ONNX 保持 policy、WDL 与 moves-left。moves-left 是 value 的辅助尺度输出；
  Auxiliary Soft Policy 与 root-WDL 只用于塑造共享 representation，不进入推理。
- **Representation over heads**：优先让共享 trunk 更好地表达局面关系，不以增加 head 数量作为能力来源。
- **预算是边界而非目标**：必须单机可训练、消费级 GPU 可运行、FP16/batch 友好且工程复杂度可控；
  吞吐、显存和参数量用于确认边界，不是架构排名。
- **研究焦点在协同**：网络稳定演进以维持强 Knowledge；项目真正研究 Knowledge 如何与 Proof 协同，
  以及二者如何在固定时间内共同提高 Elo。
- **系统最终裁决**：loss、固定局面预测、policy 合理性与 value 稳定性用于诊断 Knowledge；是否保留网络
  只由固定时间 Elo 决定，并观察它是否让搜索更容易得到有效 Evidence。

因此 v3 Transformer 是当前应当跟进的 Knowledge 载体，不是项目的网络研究课题；它只需证明能比
现有 CNN 更好地学习象棋关系并提高固定时间 Elo。后续主要问题始终是更强 Knowledge 如何与 Proof 协同。

- 默认先纯 `px0`
- 不预设人类数据混入
- 正式 value head 只学习最终 WDL；独立 root-WDL 辅助 head 学当前 root search target，不再 qMix
- Auxiliary Soft Policy 使用 `T=4`，两个辅助 head 均不导出 ONNX
- 当前不做棋盘镜像增强：KataGo 的 8 种方形棋盘对称不能直接用于 `10x9` 象棋。若以后加入水平镜像，必须同时严格变换
  `124` 个输入平面和 `2062` 个 policy target，不能只改 FEN 或 UCI 字符串。
- x7 v3 是 PX0/Lc0 AttentionBody 90 token encoder：MHA、Smolgen attention bias、LayerNorm 与 FFN；policy 以
  from-to pair score 收集到原 2062 policy 顺序。第一版不含 Smolgen、GQA 或可学习位置表。
- x7 v2 CNN 仍是有效对照，使用 `momentum=0.001` BatchNorm；v3 Transformer 不使用 BatchNorm。
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
- `model.kind`：`x7_v3_attentionbody`（默认）或对照用的 `x7_v2_bottleneck_gbroadcast`。两者 checkpoint 不兼容。
- v3 的 `model.width`、`blocks`、`heads`、`ffn_channels` 当前默认 `512/12/16/768`；`width / heads`
  的 head dimension 必须为整数；默认每个 head 为 32 channels。
- v3 逐结构对齐 PX0/Lc0 AttentionBody：90 个 token、attention-policy-map positional encoding、input MA gate、
  MHA + Smolgen + DeepNorm + LayerNorm + ReLU FFN；policy/value/moves-left 均使用 PX0 attention-body head。
- `dataset.val_ratio`：以固定 seed 在 chunk 级别切出完整 held-out 验证流；不再建立固定 record-level 子集。
- `training.validation_batches`：PX0-style 常规验证从 held-out 流 shuffle 后读取的 batch 数；完整验证集不在每次 eval 扫完。
- `training.shuffle_size`：训练 record 的有界 replacement shuffle 总大小，按 DataLoader worker 均分。
  这对应 pxzero 的 shuffle buffer，避免顺序读取同一对局的相邻局面；默认 `4096`，约占 200 MB 主存。
- `training.final_value_loss_weight + root_wdl_loss_weight + moves_left_loss_weight`：三项都服务 value 表示。前两项分别监督最终 WDL 与 PX0 当前 root WDL；moves-left 是间接辅助。当前小网络对照的暂定基线为 `0.6 / 0.6 / 0.5`，它们不是可按 loss 数值直接比较的比例。root WDL 不是 KataGo 的未来时间平均 target。
- `training.soft_policy_weight + soft_policy_temperature`：训练期 Soft Policy 辅助头，默认 `8.0` / `4.0`。日志中的
  `soft_kl` 是该 soft target 相对预测分布的 KL（最优为 `0`）；训练 total 仍使用交叉熵，因而不受日志表示变化影响。
- `training.lr + warmup_steps + min_lr_scale`：线性 warmup 后 cosine decay。首次训练会把 cosine horizon 写入 checkpoint；后续仅延长 `steps` 时保持该 horizon，并在 floor LR 继续。优化器固定为 AdamW；Conv/Linear 权重做 decoupled decay，BatchNorm 与 bias 不 decay。`init_from` 仅载入模型权重并从 step 0 建立新的优化器和学习率日程；要保留 step、优化器与余弦进度续训，请先复制 checkpoint 到新的 `training.out`。

## 当前验证与下一步

在 PX0 v677 上，`b10c64bt32` 的 1000-step CNN 对照确认 `lr=3e-4` 可稳定训练；`moves-left=0.5` 在该小网络上优于 `1.0`。v3 Transformer 必须先以固定 validation 完成短训稳定性检查，再进入固定时间 Elo 对比。

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
