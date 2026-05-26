# 仓库 Agent 指南

面向在本仓库内工作的自动化助手 / CI Agent。

## 必读

1. **`ARCHITECTURE.md`**
   - 产品边界
   - Rust / Python 分层
   - 数据与 ONNX 契约
   - 双主线：复盘系统（短期）+ 搜索引擎（长期）
2. **`docs/review-system.md`**
   - 复盘 MVP
   - 真理引擎与滑动窗口的角色
   - 当前训练顺序
3. **`.cursorrules`**
   - 沟通语言
   - 文档同步规则
   - 当前执行主线

## 当前共识

- 本项目走 **人类认知驱动的搜索**，不是“把网络做成引擎静态评估的唯一真理”。
- **短期主产品**是复盘系统：模型输出需要**可解释**。
- **长期路线**是搜索引擎：搜索负责验证这些语义是否真的有用。
- 当前复盘 MVP 目标是：**`policy + value + danger + attack`**。
- 复盘系统允许结合：**真理引擎** 与 **滑动窗口前后文**。
- 当前主线不是继续加 head，也不是优先把所有 head 接进搜索，而是：
  - 先做强 `policy + trunk`
  - 再冻结 trunk 分别训练 `value / danger / attack`
  - 把这些输出整合进复盘系统

## 目录约定

- **Python 包**：在 `nn/` 下开发与安装；虚拟环境放在 `nn/.venv`（或用户自定，勿提交 venv）。
- **Rust workspace**：仓库根 `Cargo.toml`。
- **规则库**：`crates/xiangqi_core`
- **用户 UCI 引擎**：`crates/engin`
- **维护者数据工具**：`crates/xiangqi_dataset`

## 修改契约时

若变更以下任一内容：

- ONNX 输入输出
- policy pack 格式
- XRSH / `pack_meta` 等数据契约字段
- Rust 二进制 dataset 头格式

必须同步：

- `ARCHITECTURE.md`
- 根目录 `README.MD`
- `nn/README.md`（若命令或训练入口受影响）

若变更以下任一内容：

- 产品定位
- 双主线表述
- 复盘 MVP
- 里程碑顺序
- 近期执行主线

必须同步：

- `ARCHITECTURE.md`
- `docs/review-system.md`
- `工程目标.md`
- `NEXT_STEPS.md`
- `TODO.md`
- `README.MD`
- `.cursorrules`

## 实现优先级

1. **象棋规则与合法着（Rust）**
2. **数据生成 / 标注加速（Rust）**
3. **Python 训练与复盘语义头**
4. **复盘系统消费链路**
5. **搜索消费**

## 勿做

- 未被任务要求时，不做无关大重构
- 不将训练代码与搜索引擎绑死在同一进程
- 不把数据标注、二进制 dataset 生成塞进 `engin`
- 不默认把“更像 Pikafish 静态评估”当成唯一优化方向
- 在现有 head 未完成收益归因前，不新增 `style / sacrifice / initiative / psychological`
