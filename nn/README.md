# xiangqi-review-nn（Python 训练栈）

仓库根目录 **`README.MD`** 描述 Monorepo 全貌；本节为 **`nn/`** 的安装与命令。

产品边界、方法论与契约见根目录 **`ARCHITECTURE.md`**。

## 依赖

```bash
cd nn
pip install -e .
pip install -e ".[train]"   # PyTorch 训练与 ONNX 导出
```

核心运行时：`pyffish`（规则与合法着）、`tqdm`。训练额外需要 `torch`。

### 小规模试跑（smock）

已有 `data/smock_train.jsonl` / `data/smock_val.jsonl` 与 `data/smock_vocab.json` 时：

```bash
python scripts/train/train_policy.py --train-jsonl data/smock_train.jsonl --val-jsonl data/smock_val.jsonl --vocab data/smock_vocab.json --out data/checkpoints/smock_policy.pt --width 64 --blocks 4 --batch-size 256 --epochs 2
python scripts/export/export_onnx.py --checkpoint data/checkpoints/smock_policy.pt --out data/smock_policy.onnx
```

（`train_policy` 的 `--epochs` 默认值为 `10`；试跑可显式写 `--epochs 2` 等。）

默认 `--device cuda`；仅 CPU 环境或无 CUDA 版 PyTorch 时再显式加 `--device cpu`（脚本也会自动退回 CPU）。

## 数据管线

### A. 按语料划分 train / val

**东萍 dpxq 全量作训练、WXF 全量作验证**：两库**无重叠棋局**。

```bash
python scripts/data_pgn/extract_rows.py --pgn pgns/dpxq-99813games.pgns --out data/train.jsonl
python scripts/data_pgn/extract_rows.py --pgn pgns/WXF-41743games.pgns --out data/val.jsonl
```

**词表**（建议 train∪val）：

```bash
python scripts/data_pgn/build_vocab.py --jsonl data/train.jsonl --jsonl data/val.jsonl --out data/move_vocab.json
```

### B. 单库或合并后再按比例划分

```bash
python scripts/data_pgn/extract_rows.py --pgn pgns/dpxq-99813games.pgns --pgn pgns/WXF-41743games.pgns --out data/all.jsonl
python scripts/data_pgn/split_jsonl_by_game.py --in data/all.jsonl --train-out data/train.jsonl --val-out data/val.jsonl --val-ratio 0.05
```

## 训练

```bash
python scripts/train/train_policy.py --train-jsonl data/train.jsonl --val-jsonl data/val.jsonl --train-pack-dir data/pack_train --val-pack-dir data/pack_val --vocab data/move_vocab.json --out data/checkpoints/policy.pt --device cuda --epochs 30
```

**数 GB 级 JSONL（推荐 mmap）**：

```bash
python scripts/data_pgn/build_jsonl_index.py --jsonl data/train.jsonl --vocab data/move_vocab.json --out-dir data/index_train --weight-by-fen
python scripts/data_pgn/build_jsonl_index.py --jsonl data/val.jsonl --vocab data/move_vocab.json --out-dir data/index_val
python scripts/train/train_policy.py --train-jsonl data/train.jsonl --val-jsonl data/val.jsonl --train-index-dir data/index_train --val-index-dir data/index_val --vocab data/move_vocab.json --out data/checkpoints/policy.pt --device cuda --epochs 30
```

验证集索引**不要**加 `--weight-by-fen`。

### 离线 policy 包（千万级推荐）

```bash
python scripts/data_pgn/materialize_policy_pack.py --jsonl data/train.jsonl --index-dir data/index_train --vocab data/move_vocab.json --out-dir data/pack_train
python scripts/data_pgn/materialize_policy_pack.py --jsonl data/val.jsonl --index-dir data/index_val --vocab data/move_vocab.json --out-dir data/pack_val
python scripts/train/train_policy.py --train-jsonl data/train.jsonl --val-jsonl data/val.jsonl --train-pack-dir data/pack_train --val-pack-dir data/pack_val --vocab data/move_vocab.json --out data/checkpoints/policy.pt --device cuda
```

## 导出 ONNX

```bash
python scripts/export/export_onnx.py --checkpoint data/checkpoints/policy.pt --out data/policy.onnx
```

静态 **batch=1**，输入 `board`：`float32[1,15,10,9]`，输出 `logits`：`float32[1,V]`。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

未安装 `torch` 时会跳过 `tests/test_nn_smoke.py`。
