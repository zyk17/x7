# TODO

> 只做 px0 1:1 Rust 翻译。没有连续 px0 参考位置，不实现。

## 已完成

- [x] P0/P1：`src/chess` 的 board、legal move、FEN、Position、history、RuleJudge 与 release 对拍。
- [x] P2：`GameState`、UCI/controller、`position ... moves ...` 的完整 history、`Abort + Wait` 生命周期。
- [x] P3：classic tree、UCT、selection、extend、backup、tree reuse、WDL display。
- [x] P4 安全主线：ONNX、`CachingBackend`、minibatch、prefetch、collision、OOO、watchdog、legacy clock manager。
- [x] P4 生命周期：每次 `Abort + Wait` 后按 `src/search/classic/search.cc:1027-1064` 清理所有 shared
  collision virtual visit，防止长生命周期 Rust wrapper 把 reservation 带入下一条 `go`。
- [x] P4 UCI 构造期：按 `src/search/classic/search.cc:156-170` 在无限搜索禁用 `play` contempt，并仅在
  WDL rescale diff 非零时输出 warning；删除无 px0 对应且无人读取的旧 bridge 输出状态。
- [x] P4 watchdog：按 `src/search/classic/search.cc:351-389,981-1017` 在 stop 后仍输出最终 info；没有
  `bestmove` 响应资格的停止搜索输出 px0 的无进展 warning。
- [x] P4 DirectML 打包：静态 ORT + `DirectML.dll`，bundle UCI 冒烟不得回退 CPU。

## P4：常驻 Task Worker

- [x] 按 `src/search/classic/search.h:205-244,348-445` 与
  `search.cc:1069-1140,1322-1362,1423-1462` 定义 processing 的安全所有权边界：task 仅消费 owned
  extension input/history 并产出 result；owner 在 `WaitForTasks` 后独占写 tree、提交 backend input。
- [x] 翻译 processing 的常驻 `task_threads_`、`task_workspaces_`、`RunTasks`、`WaitForTasks`；不使用 raw
  pointer、`unsafe impl Send`、共享 `&mut SearchWorker`、`NodeTree` 或整树锁串行化。
- [ ] 只有给出 owned selection delta 后，翻译 px0 gathering task 的并发版本
  (`search.cc:1551-1897`)；当前 owner 同步 gathering 是有意的安全边界。
- [ ] 补真实 x7 ONNX/DirectML 回归：固定 nodes、长 `movetime`、`stop -> wait`、
  `position ... moves ...` 与所有节点 `NInFlight == 0`。

## 约束

- `UniformBackend` 只用于单元测试与对拍，正式 UCI 不得使用。
- P4 gathering task 未完成前，不吸收 lc0/KataGo 的搜索结构或性能优化。
