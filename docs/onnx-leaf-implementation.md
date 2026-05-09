# ONNX Leaf Value 实现方案

本文件只定义一条近期实现路线：

**保留 ONNX 推理，但收紧 `nn-leaf` 的调用范围与调用成本。**

适用前提：

- 当前 `engin` 已有 `--bench`
- `UsePolicyOrdering` / `UseNNLeaf` 已可消融
- bench 已确认主要瓶颈在 `nn-leaf`

不在本方案范围内的内容：

- 改写为 Rust 原生前向推理
- 改写为增量式 NNUE 评估器
- 引入 batching / 推理服务化
- 让 `attack / danger / tactical` 进入搜索树

---

## 1. 问题定义

当前慢的根因不是 ONNX session 是否复用，而是：

1. `UseNNLeaf=true` 时，叶子评估调用次数过多
2. 当前叶子评估路径过重：
   - `Position -> FEN`
   - `FEN -> planes`
   - ONNX Runtime 推理
3. 静止搜索会进一步放大调用次数

因此当前目标不是“换推理器”，而是：

- **先减少调用次数**
- **再降低单次调用成本**

---

## 2. 总体策略

近期只做三件事：

1. 把 `nn-leaf` 从“全叶子默认开启”改成“受控模式”
2. 给 value 推理增加缓存
3. 去掉 `Position -> FEN -> planes` 的中转

优先级必须严格按这个顺序执行。

原因：

- 第 1 步能最快止血
- 第 2 步能显著减少重复推理
- 第 3 步能降低单次推理成本

---

## 3. 阶段一：收紧 `nn-leaf` 模式

### 3.1 目标

不要再用一个布尔 `UseNNLeaf=true/false` 表达所有叶子评估策略。

应改成 3 档：

- `Off`
  - 不使用 NN value
  - 叶子与静止评估均使用 `material_stm`
- `MainLeafOnly`
  - 仅在主搜索叶子使用 NN value
  - `qsearch` / 静止评估不用 NN
- `AllLeaf`
  - 保留当前行为
  - 仅用于实验与对照

### 3.2 默认值

默认值应改为：

- **`MainLeafOnly`**

原因：

- `Off` 无法验证 value 的搜索收益
- `AllLeaf` 成本过高
- `MainLeafOnly` 是当前最合理折中

### 3.3 实现要求

- 搜索上下文中不再只存 `bool nn_leaf_eval`
- 改成显式的 `NNLeafMode`
- `static_eval()` 或其上层调用点能区分：
  - 主搜索叶子
  - 静止搜索节点

### 3.4 验收标准

至少能稳定比较以下 4 组：

1. `policy on + nn-leaf off`
2. `policy on + nn-leaf main-leaf-only`
3. `policy on + nn-leaf all-leaf`
4. `policy off + nn-leaf off`

---

## 4. 阶段二：增加 value cache

### 4.1 目标

避免同一搜索内对相同局面重复调用 ONNX value。

### 4.2 最小实现

- key：`pos.key()`
- value：`i32 score_cp`
- 生命周期：**单次搜索内**
- 容器：先用 `HashMap<u64, i32>`

不要求第一版就做：

- 全局跨搜索缓存
- 线程共享缓存
- 与 TT 深度/边界绑定

### 4.3 放置位置

建议放在搜索上下文中，而不是挂在全局推理器里。

原因：

- 生命周期更清晰
- 不会和 TT / 线程模型缠在一起
- 容易统计单次搜索收益

### 4.4 统计项

benchmark 输出应新增：

- `nn_eval_calls`
- `nn_eval_cache_hits`
- `nn_eval_cache_misses`

如果后面还需要更细：

- `nn_eval_main_leaf_calls`
- `nn_eval_qsearch_calls`

### 4.5 验收标准

在 `MainLeafOnly` 和 `AllLeaf` 模式下：

- cache hit 不是 0
- 总 ONNX 调用数明显下降
- NPS 明显改善

---

## 5. 阶段三：去掉 FEN 中转

### 5.1 目标

当前叶子评估路径不应继续依赖：

- `pos.fen()`
- `fen_to_planes(fen)`

应改成：

- `Position -> planes`

### 5.2 目标接口

建议新增类似接口：

```rust
fn position_to_planes(pos: &Position) -> Result<Array4<f32>, String>
```

然后让 value 推理直接消费 `Position`。

### 5.3 预期收益

这一步不会像“减少调用次数”那样立刻带来数量级收益，但会减少：

- 字符串分配
- FEN 格式化
- FEN 解析

这是后续继续保留 ONNX 路线时必须做的基础优化。

### 5.4 验收标准

在相同 `NNLeafMode` 和相同 benchmark 下：

- 节点数不变或近似不变
- NPS 上升
- `bestmove` 不漂移

---

## 6. benchmark 口径

本方案的验收必须依赖固定 benchmark，而不是只看体感。

至少记录：

- `bestmove`
- `score_cp`
- `depth`
- `seldepth`
- `nodes`
- `time_ms`
- `nps`
- `nn_leaf_mode`
- `policy_ordering`
- `nn_eval_calls`
- `nn_eval_cache_hits`
- `nn_eval_cache_misses`

近期判断价值时，优先看：

1. 性能是否改善
2. 行为是否稳定
3. `MainLeafOnly` 是否比 `Off` 更值得保留

---

## 7. 推荐实现顺序

严格按以下顺序：

1. 引入 `NNLeafMode`
2. 默认改为 `MainLeafOnly`
3. benchmark 输出 `nn_leaf_mode`
4. 搜索上下文增加 value cache
5. benchmark 输出 `nn_eval_calls/hits/misses`
6. 实现 `Position -> planes`
7. 再次比较 `Off / MainLeafOnly / AllLeaf`

不要颠倒顺序。

---

## 8. 暂不做

在本方案完成前，以下事项暂不进入主线：

- Rust 原生前向推理
- batching
- 推理 worker / 服务化
- value 与 TT 深度绑定缓存
- attack/danger/tactical 进入叶子评估
- 更大模型或更多 head

---

## 9. 最终目标

本方案不是为了让 ONNX 成为最终形态，而是为了回答两个更关键的问题：

1. **在合理调用方式下，value head 是否真的值得进入搜索主线**
2. **当前 ONNX 路线是否已经足够支撑 P3/P4 阶段研发**

如果答案是：

- 值得，并且性能可接受
  - 继续沿这条线推进
- 值得，但性能仍不够
  - 再讨论 Rust 原生前向
- 不值得
  - 收缩 value 在搜索中的角色

一句话：

**先把 ONNX 用对，再决定要不要摆脱 ONNX。**
