我已经把代码直接改了，改动很小，但足够支撑你现在要做的低成本筛选实验。

这次加了 4 个能力：

- 可以冻结 trunk：`--freeze-trunk`
- 可以冻结 policy head：`--freeze-policy-head`
- 可以把 aux/value 头从单层线性改成小 MLP：
  - `--aux-head-hidden-dim`
  - `--value-head-hidden-dim`
- 可以按头分别设权重：
  - `--attack-loss-weight`
  - `--danger-loss-weight`
  - `--tactical-loss-weight`
  - `--value-loss-weight` 仍保留

相关文件是：

- [nn/scripts/train/train_policy.py](/C:/projects/77xiangqi_engine/nn/scripts/train/train_policy.py)
- [nn/src/nn/model.py](/C:/projects/77xiangqi_engine/nn/src/nn/model.py)

我已经验证过：

- `--help` 正常
- `frozen trunk + frozen policy + 单训 danger + aux MLP` 的 dry run 正常
- 现有 `tests/test_aux_heads.py` 通过

---

**你现在可以直接跑的实验**

我建议你不要撒太大网，先做 3 组，每组都只跑 `5 epoch`，并且 `val_every=1`。

## 1. `danger` 读出实验
目标：验证 policy trunk 里是否已经有“危险感”表征。

命令建议：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/exp_danger_probe.pt `
  --batch-size 512 `
  --epochs 5 `
  --val-every 1 `
  --device cuda `
  --freeze-trunk `
  --freeze-policy-head `
  --attack-loss-weight 0 `
  --danger-loss-weight 0.2 `
  --tactical-loss-weight 0 `
  --value-loss-weight 0 `
  --aux-head-hidden-dim 64 `
  --lr 5e-4
```

关注点：

- `dan` 是否明显优于你现在联合训练时的水平
- `atk/tac/value` 不重要，因为这轮本来就不训
- `policy` 指标可以忽略，不是这轮目标

成功标准：
- `dan` 明显下降
- 并且 5 epoch 内下降趋势稳定

---

## 2. `value` 读出实验
目标：验证 trunk 里是否已经有结果趋势表征。

命令建议：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/exp_value_probe.pt `
  --batch-size 512 `
  --epochs 5 `
  --val-every 1 `
  --device cuda `
  --freeze-trunk `
  --freeze-policy-head `
  --no-aux-heads `
  --value-loss-weight 0.5 `
  --value-head-hidden-dim 64 `
  --lr 5e-4
```

关注点：

- `val_value_mse` 是否稳定低于零基线 `0.0648`
- 是否能明显好过你现在联合训练里的 `0.0594~0.0598`

成功标准：
- 快速掉到比当前联合训练更低
- 至少稳定优于 `0.0594`

---

## 3. `danger + value` 联合读出实验
目标：看这两个最有希望服务复盘的头，能否一起从固定 trunk 中读出来。

命令建议：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/exp_danger_value_probe.pt `
  --batch-size 512 `
  --epochs 5 `
  --val-every 1 `
  --device cuda `
  --freeze-trunk `
  --freeze-policy-head `
  --attack-loss-weight 0 `
  --danger-loss-weight 0.2 `
  --tactical-loss-weight 0 `
  --value-loss-weight 0.5 `
  --aux-head-hidden-dim 64 `
  --value-head-hidden-dim 64 `
  --lr 5e-4
```

关注点：

- `dan`
- `val_value_mse`

成功标准：
- 两个都比当前联合训练更好
- 没有明显互相拖累

---

**你现在最该盯的不是总 `val loss`**
在这些 probe 实验里，重点不是 policy CE 了，而是：

- `dan`
- `val_value_mse`

如果你后面要测 `attack/tactical`，再看：

- `atk`
- `tac`

但短期复盘系统目标下，我建议先别把资源砸在 `attack/tactical` 上。

---

**我对结果的解读标准**

如果 `danger` frozen probe 很快变好：
- 说明 trunk 里已经有危险语义
- 当前问题主要在 head capacity 或多任务干扰

如果 `value` frozen probe 很快变好：
- 说明 trunk 已经有结果趋势信息
- 当前联合训练方式在压制 value 读出

如果 `danger/value` frozen probe 都没明显变好：
- 说明问题更偏标签定义或 trunk 本身没学到
- 这时才值得回头继续改 loss

---

**你当前最值得优先跑的顺序**

1. `danger` probe
2. `value` probe
3. `danger + value` probe

跑完这 3 个，你就能知道下一步是：

- 走“冻结 trunk + 单独小 head”路线
- 还是继续改联合训练 loss

如果你愿意，等你跑完这 3 组，把日志贴给我，我可以继续帮你把结果整理成一张对比表，并直接给出下一步该不该动 `attack/tactical`。