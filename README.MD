# 77xiangqi_engine

一个重新收敛后的象棋工程，只保留第一阶段地基：

- `xiangqi_core`：规则、合法着、局面基础设施
- `engin`：MCTS 搜索、ONNX policy/value 消费、UCI 入口
- `xiangqi_dataset`：PGN -> XRSH、对人类局面做搜索标注
- `nn/`：小型 `policy + value` 网络训练与 ONNX 导出

当前不再保留 Alpha-Beta 路线，也不再保留 `danger / attack / tactical` 等辅助头。

## 当前共识

- 搜索主产品路线是 **MCTS**
- 网络只做 **policy + value**
- `policy` 主要学大师数据
- `value` 默认优先学 **搜索 `q`**
- 自对弈只做**小批量、分轮次、受控混合**
- 长期正式训练格式只保留 **XRSH v5**

## 快速结构

```text
crates/
  xiangqi_core      规则内核
  engin             MCTS 引擎
  xiangqi_dataset   数据集生产
nn/
  src/nn            Python 训练基础设施
  scripts/train     训练入口
  scripts/export    ONNX 导出
```

## 数据格式

当前统一训练格式是 **XRSH v5**。单条样本只保留主线字段：

- `fen`
- `target_idx`
- `legal_idx`
- `game_result_red`
- `ply_total`
- `search_q`
- `search_visits`
- `search_counts`

这意味着：

- 人类样本可以只带 `policy target`
- 搜索标注样本可以额外带 `search_q + root visits`
- 同一个格式同时服务 policy、value、MCTS 标注闭环

## 最小工作流

1. 用 `xiangqi_dataset` 生成 canonical `move_vocab.json`
2. 用 `xiangqi_dataset` 把 PGN 转成 `XRSH v5`
3. 可选：对人类局面跑 MCTS，把搜索标注继续写回 `XRSH v5`
4. 用 `nn/scripts/train/train_policy.py` 训练 `policy` 或 `policy + value`
5. 用 `nn/scripts/export/export_onnx.py` 导出 `policy.onnx`
6. 用 `engin` 直接消费 ONNX 做 MCTS 搜索

更详细的边界见 [ARCHITECTURE.md](/C:/projects/77xiangqi_engine/ARCHITECTURE.md)。
