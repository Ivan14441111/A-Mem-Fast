# A-Mem-Fast 早期实验记录

实验日期：2026-10-18。使用上游 **robust** 分支、本机 Qwen3-8B 和选定 LoCoMo 子集。多会话实验与后续策略见 [当前实验报告](experiments_v2.md)。

## 环境与协议

- 上游：`WujiangXu/A-mem@0c8039f28fdcc08189a23c07a3437d9d2482f9c2`；原 memory layer、提示词、解析器未改。
- 模型：本地 Qwen3-8B BF16，vLLM 0.11.2，GPU2 / A100 80GB，16384 context，最大并发 16，关闭 thinking。
- 两组均 temperature=0、seed=42、max_tokens=1000；共同替换原脚本默认的 temperature=0.7。
- 向量编码：固定版本的 `all-MiniLM-L6-v2`，CPU 4 线程。下载、服务启动与预热不计入 build/QA；初始化单列。
- 数据：`conv-26` 按时间顺序的前 64 条记录。仅选 evidence 非空且全部在这 64 条记录内的问题；排除类别 5，按类别轮转选 12 题。原始问题、答案、证据和数据指纹均在每轮 `subset.json`。
- 检索 k=10、evo_threshold=100；优化版使用 4 个分析线程、16 条预取窗口。记忆写入和演化仍按原顺序执行。
- 每组实时空缓存实验都清空自己的 SQLite 响应缓存，并重置 vLLM KV 前缀缓存；每组独立实际推理。

配置与版本见 [environment.json](environment.json)。最终协议另保存优化源码 SHA256；模型版本由配置、分片索引、tokenizer 和模板的校验值记录。

所有实验请求的真实 LLM 推理均在 build/QA 计时内；排除的是模型下载、服务启动和单独的 READY/编码器预热。配置记录中 `Inference/model download/startup excluded` 的原始简写指启动与预热，实际推理包含在计时内。QA 时间包含极轻量的字符串评分，未单独测量评分时间。

## 第一轮端到端观察

原始数据：[summary.json](../results/locomo64/summary.json)、[baseline.json](../results/locomo64/baseline.json)、[optimized.json](../results/locomo64/optimized.json)。这一轮先运行 baseline；两组均同步录制 SQLite 响应，录制开销计入时间。下一轮将基线响应落盘移到计时后。

| 项目 | 上游 robust | 优化版 |
| --- | ---: | ---: |
| 64 条记忆构建 | 140.449 s | 108.984 s |
| 12 题 QA | 11.720 s | 11.843 s |
| 构建实际 HTTP 请求 | 193 | 193 |
| 构建 + QA 实际请求 | 217 | 217 |
| normalized token F1 | 0.4897 | 0.4342 |
| normalized EM | 0.1667 | 0.0833 |

构建观察到 **1.289×**、build+QA **1.259×**。这是共享机器上的单次结果；两组实时输出有差异，F1 下降 **0.0556**。

基线的 193 次构建请求为：信息提取 64、更新决策 63、建立关联 63、邻居修改 3。对应耗时和 token 均在 JSON 的 `build_llm.by_stage`。LLM 入口累计约 137.65 秒，构建共 140.45 秒；本规模的主瓶颈是 LLM 串行步骤。CPU 编码器构建加 QA 共 139 次、约 2.85 秒。本轮未触发索引重建。

笔记信息预取保留全部必要的首次推理，空缓存请求数保持相同。`logical_calls` 还包括预取与随后读取缓存的入口，应使用 `upstream_calls` 比较原有步骤、`http_requests` 比较实际推理，两次 client 入口可能对应一次实际推理。

## 第二轮对照

原始数据：[summary.json](../results/locomo64-final/summary.json)、[配置记录.json](../results/locomo64-final/manifest.json)、[baseline.json](../results/locomo64-final/baseline.json)、[optimized.json](../results/locomo64-final/optimized.json)。改为先运行 optimized，再运行 baseline，仍逐组重置 KV 缓存。基线仅在 RAM 录制响应，实验结束后落盘的 6.952 秒排除在计时外；优化版真实 SQLite 读写仍在计时内。源码版本由该轮配置记录记录。

| 项目 | 上游 robust | 优化版 |
| --- | ---: | ---: |
| 64 条记忆构建 | 280.839 s | 209.717 s |
| 12 题 QA（含轻量评分） | 20.839 s | 23.277 s |
| build + QA | 301.678 s | 232.994 s |
| 构建实际 HTTP 请求 | 193 | 196 |
| 构建 + QA 实际请求 | 217 | 220 |
| normalized token F1 | 0.4897 | 0.4008 |
| normalized EM | 0.1667 | 0.0833 |

观察到构建 **1.339×**、build+QA **1.295×**；F1 下降 **0.0889**。优化版多了 3 次邻居修改，来自独立 LLM 输出的演化分叉。预取重叠了等待时间，首次推理仍全部执行。

这轮发现其他进程也占用 GPU2，额外约 15.6 GiB 显存。保存的 36 次 GPU 快照均记录到该进程：[gpu-monitor.jsonl](../results/locomo64-final/gpu-monitor.jsonl)。两轮时间和速度比均受共享 GPU 影响；第一轮的 GPU 连续监控不完整。两次观察均约 1.3×，稳定吞吐和质量变化还需更大规模对照。

最终基线构建中真实 HTTP 等待约 278.005/280.839 秒，占 **99.0%**，进一步确认 LLM 主导。优化版构建的 SQLite 写入耗时累计 7.980 秒；并发请求耗时之和可以超过实际耗时，不应与其他包含关系的计时直接相加。

本轮 baseline→replay 与 optimized→warm 均再次通过：64 个逐步状态、原顺序请求键和 12 个完整 QA 结果一致。两个缓存均通过 SQLite integrity_check；请求错误和截断均按失败处理。

## 固定响应重放与热缓存

第一轮 baseline 响应驱动优化实现时，64 个逐步记忆状态指纹、原顺序请求键和 12 题完整 QA 结果均一致；重放计时阶段零 HTTP 请求。用优化版自己的响应重跑也全部一致。CLI 启动仍另有一次 READY 预热请求。UUID 和当前访问时间不参与状态指纹，links/context/tags 等实质字段均参与。

对应 replay build 为 1.726 秒，warm build 为 1.715 秒。这些是固定回答下的程序执行时间。上游支持整库 pickle 缓存；新增请求缓存支持单次响应复用、部分结果保存、完整参数区分和回归验证。

SQLite 记录了相同笔记信息请求返回不同正文的情况，具体见 [audit.md](audit.md)。temperature=0 和固定 seed 下，当前服务仍返回过不同正文。当前 vLLM 0.11.2 的官方 batch-invariance 支持要求 GPU compute capability ≥9.0，本机 A100 为 8.0，未开启这个不在该版本支持范围内的模式。[vLLM 0.11.2 文档](https://docs.vllm.ai/en/v0.11.2/features/batch_invariance/)

## 索引微测

原始数据：[index_benchmark.json](../results/index_benchmark.json)。真实 LoCoMo 文本与真实 MiniLM，CPU 4 线程；单独测索引路径，只测索引，输入使用原始真实文本。初次批量重建与上游逐条重建均不含模型加载，加载耗时另列。

| 文档数 | 上游逐条重建 | 批量重建 | 加速 | encode 调用数 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 1.701 s | 0.664 s | 2.56× | 128 → 2 |
| 512 | 6.697 s | 2.438 s | 2.75× | 512 → 8 |
| 1024 | 12.087 s | 4.800 s | 2.52× | 1024 → 16 |

1024 条中：无变更重建 0.264 毫秒、0 次 encode；更改 11 条文本，重建 49.39 毫秒，只编码这 11 条。原始/修改后共 48 次查询对照的 top-k 排名全部一致。

主要收益来自批量编码与避免重复编码。1024 条上游 encode 自身占 11.881/12.087 秒；已处于 OS 缓存的模型重载约 0.057 秒，向量追加等剩余开销约 0.2 秒。虽然几何扩容解决了反复 vstack 的二次复制，本规模的主要收益来自编码。批量编码减少调用批次，待编码文本数量保持不变。

## 验证与局限

完整测试 **60 passed**；覆盖缓存版本隔离、同请求并发合并、截断不入库、精确排名及边界、重复/变更/重排文档，以及 threshold=100 和 threshold=2 下的顺序演化与重建等价。仅有上游 `ast.Str` 的弃用警告。

评分采用 normalized EM 和 multiset token F1。研究范围为单会话 12 题、类别 1–4，报告保留各题预测。上游 set-F1、论文 Score、BERTScore/ROUGE 使用其他评分方式。

本阶段实现了分阶段计时、有界预取、增量精确索引、请求缓存和逐步回放；真实 MiniLM 在 1024 条规模的索引重建达到约 2.52×。多会话、重复运行、请求合并与上下文去重的后续结果见 [当前实验报告](experiments_v2.md)。
