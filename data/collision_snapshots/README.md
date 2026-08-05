# Collision tree snapshots

固定条件：`data/x7.onnx`、fresh tree、默认 `SearchConfig`（Gather/Eval/Backprop = `4/4/1`、
backend 默认 batch）、1000 completed playouts、无 virtual loss、`tree_shape --depth 3 --top 3`。

这些文件是一次诊断快照，不是可重复的性能基准：并发调度会改变确切的 collision 数与边界 N。
它们用于保留“高 collision 时树长什么样”的研究样本。

初步观察：三个高样本都存在访问漏斗，而非从根到叶都完全窄。

- `evasion_01.txt`：root 的 `c9b7` 得到 586/1000，之后 `e1d0` 得到 535/586，形成两层主干。
- `middle_38.txt`：M 很低（约 20–30 ply），多个 root 分支后都迅速汇聚到强应手 `e0d0`。
- `middle_30.txt`：低先验 `g6g9`（P=0.052）被 Q 推到 597/1000，随后 `e2g0` 是 P=1 的强制节点。

对照：同条件的初始局面得到约 197k collision；root 前三支为 341/275/125，前三层仍持续分叉。
三个快照分别约为 470k、532k、344k collision。固定 completed-N 下，绝对 collision 数可作粗略
比较；rate 在无 virtual loss 的当前流水线中几乎总在 99.5% 以上，不适合作为二元判据。

因此当前证据支持“访问集中 + 强制/有效窄主干会抬高 collision”，但还不能证明这是全部原因；
NN 延迟、batch 与 worker 并发仍会放大同一树形的 collision 数。
