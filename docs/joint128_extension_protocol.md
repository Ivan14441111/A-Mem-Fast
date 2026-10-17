# Structured held-out 128-turn 扩展：输出前预设

预设时间：2026-10-17T18:18:11Z。在三项结构化门禁、64-turn joint pilot 或本扩展的任何实时输出之前固定。它是同两个 held-out conversation 的更长前缀实验，不替换失败的无结构约束主实验，也不是另一批独立 conversation。

**固定样本。** sample 3 / conv-42 和 sample 4 / conv-43，各取时间顺序前 128 turns。类别 1–4、evidence 非空且全部在前缀范围内，沿用同一轮转规则，各选 24 QA；候选分别 28 和 31。选中类别计数分别为 `{1:4, 2:7, 3:3, 4:10}` 与 `{1:2, 2:5, 3:4, 4:13}`。每个 conversation 两次重复，共四个 cell。128-turn 共 48 道独立题，其中包含 64-turn 的全部 31 道题，因此两种深度合计仍只有 48 道，不是 79 道；新增 17 道。

**固定四组。** A=baseline，B=exact，C=structured fused，D=structured joint。沿用 64-turn 的四行顺序：sample3r1 A-B-D-C，sample3r2 C-D-B-A，sample4r1 B-C-A-D，sample4r2 D-A-C-B。每个 arm 从空图、空独立响应缓存冷运行；prefix cache 重置。每个 cell 记录 baseline→exact，以及 exact/fused/joint 自身的零 HTTP 重放。失败前未执行的重放明确标记，可对完成 arm 单独恢复验证。

模型、prompt、通用 schema、服务后端和生成参数在 64-turn 开始前冻结，128-turn 不按64结果调整：Qwen3-8B、MiniLM固定revision、temperature0、seed42、max2000、RP1.05、tag16/observe；仅 fused/joint JSON 阶段启用同一 schema 输出约束，QA context保持upstream，retrieve-k10、evo-threshold100、workers4、window16。schema 编译门禁已先运行，不能把预热后运行说成包含首次编译冷启动；明确区分空响应缓存和模型/schema已加载。

## 启动门槛，仅依据可用性和时间

必须完成既定优先项：当前九个主实验尝试与恢复验证、保留原测量源码的可用三图 context 消融、结构化实现推广与三项 schema 格式门禁、默认完整419-turn案例、64-turn structured pilot。门禁必须全部通过，且没有未解决的源码/配置/重放完整性错误。

无论 64-turn QA 高低、某策略是否更快或是否出现普通冷运行失败，都不能据此换参数、换样本或挑选128组别。只有下面时间条件允许时才启动；否则完整记录未执行原因：

1. 距 UTC2026-10-17 22:42 截止至少120分钟，即不晚于20:42启动。
2. 保守投影运行时间不超过剩余时间减40分钟，后者留给最终验证、证据汇总、文档和私有简历。

投影在查看任何128输出前保存为 JSON。对四个对应64 cell 的各冷 arm，用 `2×build_seconds + qa_seconds×24/该cell题数 + init_seconds`，另加入该cell实际零HTTP重放耗时的两倍及记录的其他固定开销；全部求和再乘1.20。若某64 arm没有完整计时，不用短暂失败耗时低估工作：其 build 代入所有完成64 arm中的最大值，QA代入最大每题时长乘该cell题数，初始化/重放代入可用记录的保守最大值。若所需计时完全不可获得，投影标为不可估计，本扩展不启动。该规则只是排期估算，不是性能上界或实验结论。

## 运行与报告

四个独立 cell 各尝试一次，普通冷运行失败保留诊断后继续；源码、配置或重放完整性失败立即终止。不得补跑至成功，不产生第五种策略，不做reason-stop或提示词搜索。每个arm、比较与conversation分别报告planned/attempted/completed/failed/not-run；配对速度/质量只基于完成配对，并标注其条件与资源范围。

独立报告128深度的构建及build+QA速度、tokens、HTTP、完成率、指令违例、分支与实际consolidation次数、逐题质量及严格重放。先在conversation内平均重复，再等权汇总两个conversation；速度采用配对log均值的指数。64/128重叠题可以逐题描述变化，但增加前缀同时增加事实、图和检索候选，不能把质量变化单独归因于某一项机制。两个conversation、48题仍不足以宣称统计非劣效。
