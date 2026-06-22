# nn

Python 训练栈只保留 `policy + value` 主线。

## 虚拟环境（重要）

**训练、迁移、pytest 一律用 `nn/.venv`，不要用仓库根目录的 `.venv`。**

| 路径 | 用途 | 本仓库训练 |
|------|------|------------|
| `nn/.venv` | 本模块专用，torch **cu128** | 是 |
| `../.venv` | 根目录杂项环境，torch cu124 等 | 否 |

PowerShell：

```powershell
cd nn
. .\activate.ps1          # 推荐：激活并打印 torch 版本
# 或不用激活，直接：
.\.venv\Scripts\python.exe scripts/train/train_policy.py ...
```

激活后应看到类似：`nn/.venv | torch 2.11.0+cu128 | cuda True`  
若仍是 `cu124`，说明激活错了环境。

## 安装

```powershell
cd nn
python -m venv .venv
.\.venv\Scripts\pip install -e ".[train,dev]"
# RTX 50 系需 cu128 时，按你平时的 torch 源安装，例如：
# .\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 当前能力

- 读取 `XRSH v5`
- 训练 `policy`
- 可选训练 `value`
- 可选蒸馏搜索 visit 分布
- 导出 `logits` 或 `logits + value` ONNX

## 训练

```powershell
cd nn
. .\activate.ps1
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/policy.pt `
  --value-head `
  --device cuda
```

常用选项：

- `--value-head`：启用 value
- `--search-policy-weight 0.2`：对根 visit 分布做额外蒸馏

性能默认值说明：

- 训练默认会自动给 `DataLoader` 分配 worker；Windows + CUDA 不再默认 `0`
- 如果显存够但 GPU 利用率仍低，优先确认输出里的 `train_workers`
- 若机器内存紧张，可手动降到 `--train-num-workers 2` 或 `0`

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

## Baseline 与数据工具

详见 [BASELINES.md](BASELINES.md)。

- `scripts/data/migrate_xrsh_v3_to_v5.py`：`stage` → `verify` → `finalize`（finalize 前勿训练）
- `scripts/data/report_data_mix.py`：数据源占比
- `scripts/data/select_search_positions.py`：冷门局面筛选
- `scripts/eval/eval_value.py`：value 评估 CSV
- `--train-mix`：读取 `data/rounds/*.json` 做受控混合训练
