# 实验结果

[实验报告](../docs/experiments_v2.md)解释各项结果。这里按实验列出汇总、图表和运行记录，便于查找。JSON 文件包含逐题答案、耗时、token 用量、运行配置和错误信息。

## 主要实验

| 实验 | 结果与运行情况 | 查看 |
| --- | --- | --- |
| 默认流程，128 条记忆 | 三个会话的完成配对，构建速度 1.143×；共完成原版 3 次、优化版 6 次。九次可选 fused 均因响应格式或字段错误中止 | [汇总](suite128-primary-observed-v1-20261018/derived.json) · [配对分析](suite128-primary-observed-v1-20261018/analysis_views.json) · [运行记录](suite128-primary-observed-v1-20261018/suite.json) |
| 回答材料去重 | 三个固定图、36 道不同问题，原版与去重各生成 36 个答案；答案输入减少 36.98% | [结果](context128-observed-v1-20261018/derived.json) · [实验配置](context128-observed-v1-20261018/protocol.json) |
| 联合策略，64 条记忆 | 两个会话各重复两次，比较四种方案；计划 16 次，执行 13 次，完整完成 11 次，joint 关键词生成失败 2 次，其余 3 次随所在组中止 | [配对结果](joint64-heldout-structured-v1-20261018/paired_metrics.md) · [汇总](joint64-heldout-structured-v1-20261018/derived.json) · [运行记录](joint64-heldout-structured-v1-20261018/suite.json) |
| 完整会话，419 条记忆 | 优化版完成构建与 48 道问答，构建 1,067.105 秒，问答 67.610 秒。原版在第 35 道问答截断，完整会话按单次运行结果展示 | [汇总](full419-default-observed-v1-20261018/derived.json) · [运行记录](full419-default-observed-v1-20261018/suite.json) |
| 索引重建 | 1,024 文档、真实 MiniLM，批量重建速度约 2.52× | [索引测量](index_benchmark.json) |

64 条联合策略的配对范围：默认优化为 3 对、2 个会话、31 道不同问题；fused 和 joint 各为 2 对、1 个会话、14 道不同问题。速度按每个会话内的配对比值汇总，再对会话等权计算。

图表：[默认流程](joint64-heldout-structured-v1-20261018/default_pairs.svg) · [fused](joint64-heldout-structured-v1-20261018/fused_pairs.svg) · [joint](joint64-heldout-structured-v1-20261018/joint_pairs.svg) · [同一子集的四种方案](joint64-heldout-structured-v1-20261018/four_arm_pairs.svg)。

## 回放与异常记录

| 内容 | 记录 |
| --- | --- |
| 128 条默认流程 | [九次完成运行的响应回放](suite128-primary-observed-v1-20261018/replay_recovery.json)，逐步状态、请求和问答输出一致 |
| 64 条策略比较 | [十一次完成运行的回放](joint64-heldout-structured-v1-20261018/replay_recovery.json)，各方案使用自己的录制响应重建；[资源记录](joint64-heldout-structured-v1-20261018/analysis_views.json) |
| 419 条完整会话 | [优化版自身回放](full419-default-observed-v1-20261018/replay_recovery.json)匹配全部状态、请求和 48 个答案；[原逻辑与优化实现](full419-default-observed-v1-20261018/build_only_reconstruction.json)在同一份优化版响应记录下匹配 419 步构建，覆盖 4 次索引重建 |
| 419 条原版问答失败 | 原版构建完成，问答第 35 题截断；旧版程序尚待问答结束后保存缓存，构建计时和录制响应随退出丢失。详见[运行说明](../docs/full419_evidence_audit.json)与[后续优化版运行](full419-default-observed-v1-20261018/independent_continuation.json) |
| 失败请求的用量 | [128 条实验](suite128-primary-observed-v1-20261018/failed_response_usage.json)与[64 条实验](joint64-heldout-structured-v1-20261018/failed_response_usage.json)统计已保存响应的 token 用量 |

这些回放均从本地读取模型响应。原版响应驱动优化实现，用于检查实现是否一致；各方案自己的响应驱动自己，用于重复检查运行结果。答案质量由真实问答实验单独评分。

## 生成问题排查

| 实验 | 记录与发现 |
| --- | --- |
| 最初的 1,000-token 实验 | [一组完成、一组失败、七组待运行](suite128-three-arm-20261018/suite.json)；完成组运行时发生 GPU 显存分配变化，单独保留 |
| 严格标签上限 | [优化版生成 18 个标签，超过设定的 16 个](suite128-primary-bounded-v1-20261018/suite.json)；第一组停止，另八组待运行 |
| 重复生成 | [同一请求增加到 2,000 tokens 后仍截断](diagnostics/strengthen_cap2000.json) |
| 重复惩罚 1.05 | [十个固定请求中九个通过](diagnostics/rp105-fixed-controls/results.json)，另一个超过标签数量上限 |
| 增加标签数量提示 | [相同十个请求通过](diagnostics/bounded-tags-v1-fixed-controls/results.json)；后续长链实验仍出现超限 |
| JSON 结构约束 | [三个固定输入通过格式和记忆引用检查](diagnostics/structured-policy-v1-fixed-controls/results.json) |

## 配置与代码版本

主要实验使用相同生成设置：最多输出 2,000 tokens，重复惩罚 1.05，提示词要求最多 16 个标签，完整的超限列表记录为观察项。64 条策略比较额外为 fused/joint 启用 JSON 结构约束。各组开始前清空响应缓存，模型和结构模板已预热。

| 实验 | 测量时的提交 |
| --- | --- |
| 128 条默认流程、回答材料去重 | `2019a73d4ce7135acff4fc2cf51978ccee087124` |
| JSON 结构检查、419 条完整会话、64 条策略比较 | `21b5b23de2089e6b096a86242a7d51a67cc06d29` |

后续代码补充了构建完成后立即保存响应和报告的处理。回看历史结果时，可通过提交号和各目录中的源码快照找到对应实现。原始记录中的绝对路径指向当时使用的机器；新环境配置见[使用指南](../README_FAST.md)。

## 计划记录

| 计划 | 状态 |
| --- | --- |
| [128 条联合策略扩展](joint128-heldout-structured-v1-20261018/suite.json) | 因时间预算取消。预计需要 121.12 分钟，当时扣除收尾时间后可用 114.67 分钟；[计算记录](joint128-heldout-structured-v1-20261018/scheduling_decision.json)、[执行状态](joint128-heldout-structured-v1-20261018/execution_status.json) |
| [原联合策略计划](joint64-heldout-observed-v1-20261018/suite.json) | 启动前改用 JSON 结构约束方案，见[实验安排](../docs/joint_pilot_protocol.md) |
| [严格标签配置下的去重计划](context128-bounded-v1-20261018/protocol.json)、[联合策略计划](joint64-heldout-bounded-v1-20261018/suite.json) | 配置调整后重新安排，原计划留存 |

## 早期实验与软件测试

| 内容 | 文件 |
| --- | --- |
| 早期 64 条对照 | [locomo64-final](locomo64-final/summary.json)、[locomo64](locomo64/summary.json)，解读见[早期报告](../docs/experiments.md) |
| 小规模联调 | [三种方案](smoke-three-arm-20261018/summary.json)、[8 条记录](smoke8/summary.json) |
| 早期回答材料分析 | [本地回放与材料长度](context-plan-locomo64/manifest.json) |
| 自动测试 | [281 项通过](clean-client-validation-final.json)，独立 CPU 环境，耗时 12.24 秒；[较早版本的 235 项测试](clean-client-validation-structured.json) |
| 安装包检查 | [c95c918 版本](package-validation-c95c918.json)，安装后检查导入、接口、缓存和包内文件；历史版本：[81fcb979](final-package-validation.json)、[21b5b23](package-validation-21b5b23.json) |
| 缓存归档恢复 | [在测量版本重建 128 条记忆与 12 道问答](archive-replay-validation-final.json) |
| 接口示例 | [原示例](readme-example-validation.json)、[带结构约束的示例](readme-example-validation-structured.json)，使用模拟模型和文本向量执行 |

SQLite 响应缓存保存在本机私有目录。报告和图表随仓库提交。
