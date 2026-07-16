# AGENTS

面向在本仓库内工作的自动化助手。

## 开始前先读

1. `README.MD`
2. `ARCHITECTURE.md`
3. `AGENTS.md`
4. `NextStep.md`
5. `TODO.md`
6. `temp.md`

## 当前共识

- 这是全新 Rust 实现；旧 `xiangqi_core` 与旧 MCTS 不是兼容目标。
- 当前唯一工程参考是：
  - `C:\Users\Administrator\projects\px0`
  - `C:\Users\Administrator\projects\pxzero-training`
- 当前工作是逐函数 1:1 翻译 px0；翻译完成前不吸收 lc0/KataGo 的结构或性能细节。
- 搜索主路线只有 MCTS；重建完成前 `go` 必须明确返回不可搜索状态，不能返回 heuristic 或旧结果。
- 正式模型契约保持 `124x10x9 -> 2062 + WDL`。

## 模块边界

- `crates/xiangqi_core`：翻译 px0 `src/chess`。
- `crates/engin`：翻译 px0 UCI/controller、网络外围与搜索主线；P2/P3 已完成，P4 的 ONNX、
  单 worker、minibatch、MemCache、prefetch、collision、shared-tree 与 watchdog 主线已接入。GPU
  task-worker 仍未完成，当前必须保持 `task_workers=0`。
  正式 ONNX 必须经 `CachingBackend`，其 key/collision guard/回填时序只能对照
  `px0/src/neural/memcache.cc:38-190` 修改。
  已删除的 `TaskTreeBridge` / `TaskWorkerRunner` 不是可用实现；后续只能以无别名的数据所有权重译
  px0 task 生命周期。唯一准确状态以 `NextStep.md`、`TODO.md` 为准。
- `nn/`：对齐 pxzero-training 的数据、训练和导出。

不要引入：

- 多套正式训练格式
- 未经 px0 对照的抽象、参数或启发式

禁止为 P4 task-worker 使用 raw pointer、`unsafe impl Send` 或跨线程共享 `&mut SearchWorker`。
此前这条翻译在真实 ONNX 下重复 `ExtendNode`，已停用。只有先在 `NextStep.md`/`TODO.md` 记录与
px0 `src/search/classic/search.h:205-244,348-445`、`search.cc:1069-1508` 对应的无别名所有权方案，
并补齐真实 ONNX 回归后，才可讨论最小必要的 `unsafe`。

## 依赖与关键路径

- 不重复造轮子：成熟、通用且不承载象棋/搜索语义的能力，优先评估并复用 Rust 生态现有 crate。
- 不为“零依赖”手写成熟的同步、容器、解析或工具能力；引入依赖时说明它替代的通用职责。
- 关键路径保留可控实现：象棋规则、MCTS selection/backup/task 生命周期、模型输入输出与数据目标必须能逐行对照 px0，不能被通用库或黑盒抽象掩盖语义。
- 判断标准是职责而非实现位置：crate 可以承载通用机制，项目代码必须保有决定棋力和搜索行为的语义。

## 参考纪律

- 每个新 Rust 函数必须在注释、变更说明、`NextStep.md`、`TODO.md` 或 review 记录中标出 px0 文件路径和连续行区间。
- 对规则层优先参考：
  - `px0/src/chess/types.h`
  - `px0/src/chess/bitboard.h`
  - `px0/src/chess/board.h/.cc`
  - `px0/src/chess/position.h/.cc`
  - `px0/src/chess/board_test.cc`
  - `px0/src/chess/position_test.cc`
- 对搜索层优先参考 `px0/src/search`；对 UCI 优先参考 `px0/src/chess/uciloop.cc` 和 `px0/src/engine.cc`。
- 找不到参考位置时，先记录缺口，不实现。
- `position ... moves ...` 必须保留完整历史，不能退化成最终局面。

## 文档规则

稳定文档：`README.MD`、`ARCHITECTURE.md`、`AGENTS.md`。

临时文档：`NextStep.md`、`TODO.md`、`temp.md`。
