# nn

`nn/` 是 PX0 网络和训练格式的 Python 重写：读取 PX0 v6 classical chunks，训练 Knowledge Model，导出
Engine 使用的 ONNX。它不包含规则实现，也不承担搜索语义。

正式契约固定为：

```text
124 x 10 x 9 -> 2062 policy + WDL + moves-left
```

模型是 90-token attention encoder：MHA、Smolgen attention bias、DeepNorm、LayerNorm 与 ReLU FFN。
正式 ONNX 只含 policy、WDL 和 moves-left；训练期 Soft Policy、root-WDL 两个辅助 head 不导出。

## 目录

- `src/nn/px0_record.py`：PX0 v6 classical record 解码
- `src/nn/dataset_px0.py`：chunk 流式数据集
- `src/nn/px0_kaggle.py`：下载、解压与固定 train/validation manifest
- `scripts/prepare.py`：一次性准备数据
- `scripts/train.py`：训练与续训
- `scripts/export.py`：ONNX 导出

## 训练

```powershell
cd nn
python -m pip install -e ".[train,dev]"
Copy-Item configs/example.yaml configs/run_01.yaml
python scripts/prepare.py --config configs/run_01.yaml
python scripts/train.py --config configs/run_01.yaml
```

`prepare.py` 是唯一会下载、解压、扫描 chunk 和建立固定验证集的入口。训练只读取已准备的
manifest，不会重复执行这些工作。

YAML 只有三个 section：

- `dataset`：PX0 数据版本、本地目录、固定文件级验证切分。
- `model`：唯一模型的 `width / blocks / heads / ffn_channels`。
- `training`：checkpoint、batch、学习率、loss 权重和设备。

配置字段、默认值和说明见 [configs/example.yaml](C:/projects/77xiangqi_engine/nn/configs/example.yaml)。除示例外的
`configs/*.yaml` 是本地实验文件，不进入 Git。

## 导出

```powershell
python scripts/export.py --checkpoint data/checkpoints/run_01.pt --out data/x7.onnx --precision mixed-fp16
```

默认 mixed-FP16 导出使用 FP16 encoder、FP32 input/heads/outputs。模型通过固定 validation 与固定时间 Elo
评估；loss 只用于训练诊断，不单独决定是否保留权重。
