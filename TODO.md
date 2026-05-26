# 执行清单（阶段索引）

本文件只保留**阶段入口**与**近期主线**。

- 阶段定义：`MILESTONES.md`
- 目标原则：`工程目标.md`
- 近期执行：`NEXT_STEPS.md`
- 架构与契约：`ARCHITECTURE.md`

---

## 产品与方法论（全员对齐）

- **定位**：**人类认知驱动的搜索**，但短期先落到**复盘系统**
- **双主线**：短期主产品是复盘系统；长期路线是搜索引擎
- **复盘 MVP**：`policy + value + danger + attack`
- **复盘系统允许结合**：真理引擎、滑动窗口前后文
- **分工**：机器侧战术与验证 → `engin` / `xiangqi_core`；人类风格与局面语义 → `nn/`

---

## 近期主线

当前只做三件事：

1. 持续做强 `policy + trunk`
2. 将复盘 MVP 收敛为 `policy + value + danger + attack`
3. 用真理引擎 + 滑动窗口上下文把这些输出组织成可解释复盘

执行方式：

- 写代码时：优先做复盘消费链路、冻结 trunk 后的单头训练能力、真理引擎/滑动窗口整合
- 不写代码时：优先跑 trunk 训练与 `value / danger / attack` 单头实验

---

## P0 — `xiangqi_core`：规则 + 合法 UCI

- [x] 对照 `pikafish-rust` 划定模块边界
- [x] 局面表示、perft、do/undo 回归
- [x] 合法着 UCI：`legal_moves_uci`、`parse_move_uci`
- [x] 与 `pyffish` / Pikafish 基本对拍工具

---

## P1 — `xiangqi_dataset`：PGN → XRSH

- [x] `vocab-enum` 固定 canonical 词表
- [x] `pgn-shards` 生成 XRSH
- [x] XRSH v3 包含 `aux_* + game_result_red + ply_total`
- [x] Python `PolicyXrshDataset` 仅读 XRSH

---

## P2 — `nn/`：训练与 ONNX

- [x] XRSH only 训练路径
- [x] shared trunk + policy + aux 头
- [x] value 头训练支持
- [x] ONNX 导出
- [x] 冻结训练基础能力：`freeze-trunk` / `freeze-policy-head` / `freeze-value-head`

---

## P3 — `engin`：搜索 + UCI

- [x] UCI 最小闭环
- [x] Alpha-Beta + qsearch + TT + ordering
- [x] benchmark / ablation 基础设施
- [x] ONNX 推理接线
- [ ] 维持为可用实验平台

说明：

- `engin` 仍然重要
- 但不是当前最急主线

---

## P4 — 复盘 MVP 头

- [ ] 先做大一轮 `policy + trunk`
- [ ] trunk 接近平台后，冻结 trunk 单独训练 `value`
- [ ] 再冻结 trunk 单独训练 `danger`
- [ ] 再冻结 trunk 单独训练 `attack`
- [ ] `tactical` 暂列第二波增强头

---

## P5 — 复盘编排层

- [ ] 固定候选着、趋势、风险、攻势的输出契约
- [ ] 将模型头、真理引擎、滑动窗口整合成完整复盘输出
- [ ] 形成小规模人工样例集，验证解释是否像人会说的话

---

## P6 / P7 — 长期搜索线

- [ ] 将已验证有效的 head 逐步接入搜索
- [ ] 做搜索收益 benchmark
- [ ] 只保留对搜索真正有收益的头
- [ ] 后续再考虑蒸馏与动态搜索
