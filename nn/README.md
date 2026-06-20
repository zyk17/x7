# nn

Python 训练栈只保留 `policy + value` 主线。

## 安装

```bash
cd nn
pip install -e .
pip install -e ".[train]"
```

## 当前能力

- 读取 `XRSH v5`
- 训练 `policy`
- 可选训练 `value`
- 可选蒸馏搜索 visit 分布
- 导出 `logits` 或 `logits + value` ONNX

## 训练

```bash
python scripts/train/train_policy.py \
  --train-dir ../data/xrsh_train \
  --val-dir ../data/xrsh_val \
  --vocab ../data/move_vocab.json \
  --out ../data/checkpoints/policy.pt \
  --value-head \
  --device cuda
```

常用选项：

- `--value-head`：启用 value
- `--search-policy-weight 0.2`：对根 visit 分布做额外蒸馏

## 导出

```bash
python scripts/export/export_onnx.py \
  --checkpoint ../data/checkpoints/policy.pt \
  --out ../data/policy.onnx
```

ONNX 契约：

- 输入：`board`，形状 `[1, 15, 10, 9]`
- 输出：`logits`
- 若 checkpoint 含 value：额外输出 `value`

## 测试

```bash
pytest
```
