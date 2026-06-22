# Baseline 训练配方

命名约定：`baseline_<task>_<data>_v<N>.pt`，checkpoint 路径即实验标识。

## 操作顺序（必须按序，finalize 前禁止训练）

```text
1. stage    写入 .xrsh_v5_staging（不改动原 v3 分片）
2. verify   逐分片核对 policy 字段 + search 字段置零
3. review   人工/专项 review 通过后再 finalize
4. finalize 提交 v5（--in-place）
5. 训练     仅在上一步完成后
```

```bash
cd nn

# 1. stage（中断后续传）
python scripts/data/migrate_xrsh_v3_to_v5.py stage --resume ../data/xrsh_train
python scripts/data/migrate_xrsh_v3_to_v5.py stage --resume ../data/xrsh_val

# 2. verify（可先抽查：--max-shards 5）
python scripts/data/migrate_xrsh_v3_to_v5.py verify ../data/xrsh_train
python scripts/data/migrate_xrsh_v3_to_v5.py verify ../data/xrsh_val

# 3. review 通过后

# 4. finalize
python scripts/data/migrate_xrsh_v3_to_v5.py finalize --in-place ../data/xrsh_train
python scripts/data/migrate_xrsh_v3_to_v5.py finalize --in-place ../data/xrsh_val
```

**当前阶段无 PGN**：不跑 `search-label-pgn`；policy 只用现有大师 XRSH。

---

## 1. 纯人类 policy baseline（finalize 之后）

```powershell
cd nn
. .\activate.ps1
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/baseline_policy_human_v1.pt `
  --epochs 10 `
  --device cuda
```

默认会自动带上适合本机的 `train_workers / val_workers`。若日志里 `train_workers=0`，或 GPU 利用率明显偏低，可手动试：

```powershell
--train-num-workers 4 --val-num-workers 2
```

## 2. value(search_q) — 留到自对弈 round

无搜索标注时不训 value。round_1 用 MCTS 自对弈产出 XRSH 后再做。

## 3. search_counts 蒸馏 — 第一轮默认不做

## 4. round_0 混合（当前等同纯人类）

见 `data/rounds/round_0.json`。

## 后续：自对弈 + 人类混合（round_1）

1. 自对弈导出 `data/xrsh_selfplay_r1`
2. `train_mix`：人类 + 自对弈
3. `--value-head --value-min-visits 1`

## 导出 ONNX（policy baseline 稳定后）

```bash
python scripts/export/export_onnx.py \
  --checkpoint ../data/checkpoints/baseline_policy_human_v1.pt \
  --out ../data/checkpoints/baseline_policy_human_v1.onnx
```

## 辅助脚本

| 脚本 | 作用 |
|------|------|
| `migrate_xrsh_v3_to_v5.py` | stage / verify / finalize |
| `report_data_mix.py` | 数据源占比 |
| `eval_value.py` | value 评估 CSV |
