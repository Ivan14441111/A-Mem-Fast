# Joint：独立会话实验方案

## 方案记录

- 2026-10-17T17:29:33Z：在任何 joint 实时输出前确定初始方案，排在主实验之后执行。
- 2026-10-17T17:45:53Z：严格标签上限实验因 18-tag 响应中止，共同接受规则改为 `observed-tags-v1`。原 16-tag 指令、RP=1.05 和 max_tokens=2000 保持不变，超限改为记录并保留全文；缓存区分两种接受规则。
- 2026-10-17T18:07:50Z：fused/joint 增加 JSON schema 生成约束，其余组沿用原文本格式。三项[固定格式检查](structured_policy_gate_protocol.md)全部通过后运行，执行顺序排在完整会话案例之后。

样本、问题和四组顺序在上述修订中保持不变。各版本配置及失败记录分别保存。

## 数据与运行顺序

使用主实验以外的 sample 3 / `conv-42` 和 sample 4 / `conv-43`，各取时间顺序前 64 条记录。问题须有完整标注依据、属于类别 1–4，并按类别轮转选取，最多 24 题；实际有 17 和 14 题，全部纳入。每个会话重复两次，共 4 轮、31 道独立问题。保存问题原始下标和子集校验值。

A=baseline，B=exact，C=fused，D=joint。采用四行 Williams 顺序，同一会话的两次顺序互逆：

| 轮次 | 运行顺序 |
| --- | --- |
| sample 3 / repeat 1 | A → B → D → C |
| sample 3 / repeat 2 | C → D → B → A |
| sample 4 / repeat 1 | B → C → A → D |
| sample 4 / repeat 2 | D → A → C → B |

每组重新构建并完成 QA，使用独立空响应缓存，运行前重置前缀缓存。每轮结束后，用 baseline 缓存驱动 exact，并对 exact、fused、joint 分别做自身回放。回放要求零 HTTP，计时单列。

## 共同配置

Qwen3-8B BF16、固定版本 MiniLM、temperature=0、seed=42、max_tokens=2000、repetition_penalty=1.05、tag_budget=16、tag_budget_policy="observe"，以及共同的 16-tag system 指令（`observed-tags-v1`）。retrieve-k=10、evo-threshold=100、workers=4、prefetch-window=16。四组 QA 都使用原上下文，保留重复条目。

joint 每次调用同时生成新记忆信息与旧记忆更新，顺序执行，不启动笔记信息预取。fused/joint 的 JSON 请求加入固定 schema。执行前保存源码快照、模型版本、配置和命令。

## 失败处理与统计

超过 16 个标签时保留完整响应并记录实际长度，分别统计实时生成和缓存读取。HTTP、截断、字段格式、候选引用和回放错误仍判为失败。

四轮各尝试一次。普通推理失败保存诊断后继续下一轮；源码、配置或回放不一致时停止。报告各组计划、尝试、完成、失败和未运行次数，速度及质量采用两组均完成的配对。

记录构建和构建+QA 时间、实际 HTTP、输入/输出 tokens、逐题 EM/F1、上下文规模，以及演化分支、连接和邻居更新数量。先汇总同一会话的重复，再等权汇总会话；同时展示原始配对结果。

保存 GPU2 监控，报告全部运行及显存稳定子集。筛选要求：显存范围 ≤1,024 MiB、至少 10 个样本、相邻采样间隔 ≤10 秒。共享计算负载和两个会话的小样本规模是本实验的主要限制。

## 相关实验

完整长会话采用 baseline/exact 两组，检查默认工程优化和多次索引重建，详见 [full 会话协议](full_conversation_protocol.md)。

同两个会话的 128 条扩展也在 joint 输出前确定：各选 24 题、重复两次，按[扩展协议](joint128_extension_protocol.md)的固定时间条件启动。两个长度共享题目，合计 48 道独立题，其中 31 道来自 64 条实验。
