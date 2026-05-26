# 训练命令清单

本文档给出当前主线下可直接复用的训练命令。

目标不是覆盖所有参数组合，而是给出**当前推荐命令模板**。

相关文档：

- [docs/review-system.md](/C:/projects/77xiangqi_engine/docs/review-system.md)
- [nn/README.md](/C:/projects/77xiangqi_engine/nn/README.md)
- [ARCHITECTURE.md](/C:/projects/77xiangqi_engine/ARCHITECTURE.md)

---

## 0. 当前推荐顺序

当前推荐顺序是：

1. 先做大一轮 `policy + trunk`
2. trunk 接近平台后，冻结 trunk 单独训练 `value`
3. 再冻结 trunk 单独训练 `danger`
4. 再冻结 trunk 单独训练 `attack`
5. `tactical` 作为第二波增强头后补

当前复盘 MVP 目标是：

- `policy`
- `value`
- `danger`
- `attack`

---

## 1. trunk 主训练

适用场景：

- 当前还在继续做强 `policy + trunk`
- 暂时不追求把 aux / value 一起训到最好

建议命令：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/policy_trunk_main.pt `
  --batch-size 512 `
  --epochs 20 `
  --val-every 5 `
  --device cuda `
  --no-aux-heads `
  --no-value-head `
  --lr 1e-3 `
  --train-dataset-mode eager `
  --val-dataset-mode eager `
  --train-num-workers 4 `
  --val-num-workers 4
```

关注指标：

- `val loss`
- `top1`
- `top3`

适合继续训练的信号：

- `val loss` 还在下降
- `top1 / top3` 还在涨

---

## 2. `policy + value` 联合训练

适用场景：

- 想在 trunk 主训练阶段保留一个可用的 value 头
- 但不以“把 value 一起训到最优”为目标

建议命令：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/policy_value_main.pt `
  --batch-size 512 `
  --epochs 20 `
  --val-every 5 `
  --device cuda `
  --no-aux-heads `
  --value-loss-weight 0.5 `
  --value-head-hidden-dim 64 `
  --lr 1e-3 `
  --train-dataset-mode eager `
  --val-dataset-mode eager `
  --train-num-workers 4 `
  --val-num-workers 4
```

关注指标：

- `val loss`
- `top1`
- `top3`
- `val_value_mse`

解释方式：

- `policy / trunk` 是主指标
- `val_value_mse` 只要求保持可用，不要求和 trunk 一起一直同步变好

---

## 3. 冻结 trunk 单独训练 `value`

适用场景：

- trunk 已基本接近平台
- 要正式进入复盘头读出阶段

建议命令：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/value_probe.pt `
  --batch-size 512 `
  --epochs 10 `
  --val-every 1 `
  --device cuda `
  --freeze-trunk `
  --freeze-policy-head `
  --no-aux-heads `
  --value-loss-weight 0.5 `
  --value-head-hidden-dim 64 `
  --lr 5e-4 `
  --train-dataset-mode eager `
  --val-dataset-mode eager `
  --train-num-workers 4 `
  --val-num-workers 4
```

关注指标：

- `val_value_mse`

当前经验：

- 小 MLP value head 比单层线性更值得优先尝试
- 重点看是否稳定优于零基线和旧联合训练值

---

## 4. 冻结 trunk 单独训练 `danger`

适用场景：

- `value` 已经做完
- 开始补复盘风险提示

建议命令：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/danger_probe.pt `
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
  --lr 5e-4 `
  --train-dataset-mode eager `
  --val-dataset-mode eager `
  --train-num-workers 4 `
  --val-num-workers 4
```

关注指标：

- `dan`

备注：

- 当前 `danger` 还没有像 `value` 那样给出特别强的正信号
- 这条命令更像正式 probe 模板

---

## 5. 冻结 trunk 单独训练 `attack`

适用场景：

- `danger` 做完后，开始补复盘攻势提示

建议命令：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/attack_probe.pt `
  --batch-size 512 `
  --epochs 5 `
  --val-every 1 `
  --device cuda `
  --freeze-trunk `
  --freeze-policy-head `
  --attack-loss-weight 0.2 `
  --danger-loss-weight 0 `
  --tactical-loss-weight 0 `
  --value-loss-weight 0 `
  --aux-head-hidden-dim 64 `
  --lr 5e-4 `
  --train-dataset-mode eager `
  --val-dataset-mode eager `
  --train-num-workers 4 `
  --val-num-workers 4
```

关注指标：

- `atk`

备注：

- `attack` 正样本更稀，通常比 `danger` 更难学
- 这一头更要结合人工复盘样例看解释质量

---

## 6. 冻结 trunk，只训练 `attack + danger`

适用场景：

- 想测试风险与攻势是否能一起读出
- 不想把 `value` 也混进来

建议命令：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/attack_danger_probe.pt `
  --batch-size 512 `
  --epochs 5 `
  --val-every 1 `
  --device cuda `
  --freeze-trunk `
  --freeze-policy-head `
  --attack-loss-weight 0.2 `
  --danger-loss-weight 0.2 `
  --tactical-loss-weight 0 `
  --value-loss-weight 0 `
  --aux-head-hidden-dim 64 `
  --lr 5e-4 `
  --train-dataset-mode eager `
  --val-dataset-mode eager `
  --train-num-workers 4 `
  --val-num-workers 4
```

关注指标：

- `atk`
- `dan`

---

## 7. 冻结 `policy + value`，后续单训其它头

适用场景：

- 后面已经确定 `policy + value` checkpoint 可用
- 想把 value 也固定住，再训练其它语义头

建议在上述 `danger` / `attack` 命令基础上追加：

```powershell
--freeze-value-head
```

例如：

```powershell
python scripts/train/train_policy.py `
  --train-dir ../data/xrsh_train `
  --val-dir ../data/xrsh_val `
  --vocab ../data/move_vocab.json `
  --out ../data/checkpoints/danger_after_value.pt `
  --batch-size 512 `
  --epochs 5 `
  --val-every 1 `
  --device cuda `
  --freeze-trunk `
  --freeze-policy-head `
  --freeze-value-head `
  --attack-loss-weight 0 `
  --danger-loss-weight 0.2 `
  --tactical-loss-weight 0 `
  --value-loss-weight 0 `
  --aux-head-hidden-dim 64 `
  --lr 5e-4 `
  --train-dataset-mode eager `
  --val-dataset-mode eager `
  --train-num-workers 4 `
  --val-num-workers 4
```

---

## 8. 一些当前约定

### 8.1 数据模式

当前你已经明确接受 probe 命令使用：

- `--train-dataset-mode eager`
- `--val-dataset-mode eager`
- `--train-num-workers 4`
- `--val-num-workers 4`

所以本文档统一按这组给命令。

### 8.2 验证频率

- trunk 主训练：建议 `--val-every 5`
- 单头 probe：建议 `--val-every 1`

### 8.3 当前最该优先跑什么

如果只看当前主线，优先级是：

1. `trunk` 主训练
2. `value` 单头
3. `danger` 单头
4. `attack` 单头

---

## 9. 什么时候切阶段

### 从 trunk 切到 value

当这些信号同时出现时可以考虑切：

- `val loss` 基本平台
- `top1 / top3` 提升明显放缓
- 再继续训练 trunk 的边际收益开始变小

### 从 value 切到 danger / attack

当这些信号出现时可以继续往后排：

- `value` 已经能稳定给出可用趋势信号
- 复盘系统已经不再只缺“趋势”，而开始更缺“风险 / 攻势”解释

---

## 10. 说明

本文档只整理**当前推荐命令**。

如果训练脚本参数、复盘 MVP、或阶段顺序发生变化，需要同步更新：

- [docs/review-system.md](/C:/projects/77xiangqi_engine/docs/review-system.md)
- [NEXT_STEPS.md](/C:/projects/77xiangqi_engine/NEXT_STEPS.md)
- [README.MD](/C:/projects/77xiangqi_engine/README.MD)
