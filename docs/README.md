# 文档

从[项目首页](../README.md)了解功能和结果，再按需要选择下面的内容。

## 使用与实现

| 文档 | 内容 |
| --- | --- |
| [使用指南](../README_FAST.md) | 安装、模型服务、Python 接口、实验命令和回放 |
| [性能分析](deep_analysis.md) | A-Mem 的主要耗时、并行方法、任务合并和后续方向 |
| [代码阅读](audit.md) | 原始实现的调用流程、重复工作与优化位置 |

## 实验

| 文档 | 内容 |
| --- | --- |
| [实验报告](experiments_v2.md) | 当前各方案的速度、输入输出量、回答质量和运行情况 |
| [结果目录](../results/README.md) | 图表、逐题结果、运行日志与测试记录 |
| [早期实验](experiments.md) | 最初的 64 条记忆对照与索引测量 |

实验开始前记录的配置与安排：

- [回答材料去重](context_protocol_amendment.md)
- [419 条完整会话](full_conversation_protocol.md)
- [JSON 结构检查](structured_policy_gate_protocol.md)
- [64 条联合策略对照](joint_pilot_protocol.md)
- [128 条扩展计划](joint128_extension_protocol.md)

## 项目来源

[上游 README](UPSTREAM_README.md)保留 A-Mem 的原始介绍和论文引用；数据来源与许可见 [LoCoMo 说明](../data/README.md)。
