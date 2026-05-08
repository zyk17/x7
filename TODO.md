# 执行清单（随进度勾选）

权威优先级见 **`MILESTONES.md`**；架构上下文见 **`ARCHITECTURE.md`**。

---

## P0 — `xiangqi_core`：规则 + 合法 UCI（对齐 pikafish-rust）

**当前阶段：核心已落地，可选增强见未勾项**

- [x] 对照 `pikafish-rust` 划定模块边界：`types` / `misc` / `board` / `movegen`
- [x] 局面表示：`from_fen` / `set_fen`；perft + do/undo 回归
- [x] 各子力走法与阻挡、将/帅照面、过河等（移植自 pikafish-rust）
- [x] 将军 / 应将 / 合法性过滤（`GenType::Legal` + `Position::legal`）
- [x] **合法着 UCI**：`legal_moves_uci`、`parse_pyffish_uci`（纵坐标 **1～10**，与 pyffish 一致）
- [ ] **一致性测试**：抽样局面与 **pyffish** `legal_moves` 对比（CI / 脚本）
- [x] 文档：`crates/xiangqi_core/README.md`（来源、API、测试）

---

## P1 — `xiangqi_dataset`：PGN → 二进制 shards（按局并行）

**当前阶段：MVP 已可用**

- [x] 输入：**PGN**（ICCS / UCI，与 `dataset_pgn` 对齐）+ **`jsonl-shards`** 读 JSONL
- [x] 输出：**`XQB` v1**（`shard_NNNNN.xqb`）+ `pack_meta.json`
- [x] `vocab_sha256` 与 Python 词表指纹一致；`format: xqb_v1`
- [x] CLI：`pgn-shards` / `jsonl-shards`，`--jobs`、`--games-per-shard`
- [ ] Python / PyTorch **读取 XQB** 的 Dataset（并入 P2）；可选 Rust **读回校验** 单测

---

## P2 — `nn/`：二进制训练包 + 多头网络

- [ ] `Dataset` / DataLoader 读取 P1 二进制（mmap）
- [ ] 模型：shared trunk + **policy** + **attack / danger / tactical**（头数与标签定义见 ARCHITECTURE）
- [ ] 损失与权重策略；验证指标
- [ ] **ONNX 导出**：多输出契约文档化
- [ ] 与现有 `policy_pack_v2` 迁移/共存策略（若仍保留 JSONL 路径）

---

## P3 — `engin`：搜索 + UCI

- [ ] UCI：`uci` / `isready` / `position` / `go` / `stop` / `quit` 等最小闭环
- [ ] **Alpha-Beta** + **置换表 TT**
- [ ] **Move ordering**（静态启发 + 日后 policy 排序接口）
- [ ] ONNX Runtime（或约定后端）加载 P2 模型；节点上推理接口
- [ ] 与 `xiangqi_core` 走子、合法性、终局判定联调

---

## 已完成（归档区）

- **2026-05**：P0 主体 — 自 pikafish-rust 并入 `types` / `misc` / `board` / `movegen`；`Position::from_fen`、`global_zobrist`、`legal_moves_uci`；`tests/perft.rs`（depth 1–3 = 44 / 1926 / 80069）。
- **2026-05**：P1 MVP — `xiangqi_dataset`：`pgn-shards` / `jsonl-shards`，`XQB` v1，`pack_meta.json`；`uci_format` 与 pyffish 纵坐标 1～10 对齐。
