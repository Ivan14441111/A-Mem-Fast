# Joint held-out pilot：运行前固定的协议

固定时间：2026-10-17T17:29:33Z。本文写于任何 joint 实时输出之前；只有共同 bounded-tags 配置通过格式诊断、主实验结束后才执行。若时间不足，报告未执行，不用其他配置结果替代。

**数据。** 主实验之外的 sample 3 / `conv-42` 与 sample 4 / `conv-43`，分别取时间顺序前 64 turns。沿用证据完整与类别 1–4 过滤、轮转选取规则，max_questions=24；实际各有 17、14 道符合条件的 QA，全部纳入。每个 conversation 重复两次，共 4 个 cell、31 道独立问题；重复不增加独立样本数。保留 source QA index 与 subset 指纹。

**四组与顺序。** A=baseline，B=exact，C=fused，D=joint。采用四行 Williams 顺序并在每个 conversation 内反转：

| cell | 冷运行顺序 |
| --- | --- |
| sample 3 / repeat 1 | A → B → D → C |
| sample 3 / repeat 2 | C → D → B → A |
| sample 4 / repeat 1 | B → C → A → D |
| sample 4 / repeat 2 | D → A → C → B |

每组都重新进行真实空响应缓存构建与 QA，每组前重置 prefix cache；不能借用其他时段的 baseline 时间。每个 cell 后校验 baseline→exact，以及 exact、fused、joint 各自的响应重放；重放阶段零 HTTP，不作为实时提速数据。

**共同配置。** 与新主实验相同的 Qwen3-8B BF16、MiniLM revision、temperature=0、seed=42、max_tokens=2000、repetition_penalty=1.05、tag_budget=16 和 `bounded-tags-v1` system 指令；retrieve-k=10、evo-threshold=100、workers=4、prefetch-window=16。joint 不实际启动 metadata 预取，它的一次请求同时生成 metadata 与演化；明确作为策略变更。四组上游 QA 上下文均不去重。执行前保存完整源码快照、模型/配置身份与命令；不同源码或参数产生新实验身份。

**终止和诊断。** 保留 HTTP、截断、tag budget、领域 schema、重放或源码完整性失败，报告失败 cell 和未运行 cell；不截断列表以伪造成功，不在结果产生后换 prompt、改参数或按质量补跑。失败时是否终止剩余任务由驱动的固定运行策略记录，不从均值中悄悄抹去失败。

**观察项。** 核心是 joint/fused 的同轮构建与 build+QA 速度、实际 HTTP、输入/输出 tokens、格式/截断失败、逐题 EM/F1 和证据/上下文规模；同时展示 baseline/exact 作为参照。查看各组实际演化分支、连接与邻居更新数量，不把请求数减少直接等同于延迟减少。对同一 conversation 的重复先汇总，再对两个 conversation 等权汇总；展示原始配对分布，不以小 pilot 宣称统计非劣效。

**资源。** 保存 GPU2 监控，并同时报告全部 cell 与事先固定的显存稳定子集：范围 ≤1,024 MiB、至少 10 个样本、最大相邻间隔 ≤10 秒。显存稳定不代表计算独占。这个 pilot 只判断进一步研究价值，不用于根据 QA 得分挑出下一版 prompt。

**可选长会话案例。** 完成主实验、QA 去重消融及上述 pilot 后，时间允许则单独运行 sample 0 完整 419 turns、最多 48 道证据完整类别 1–4 QA，所有候选策略使用相同共同配置和新的同轮 baseline。至少包含 baseline/exact/fused；joint 是否纳入须在该案例任何实时调用前写入 manifest。只做一次时明确为单案例，报告实际 consolidation 次数，不与 128-turn 多会话主结果混合。若启动前判断剩余时间不足以完成全部选定组和最终分析，则不启动该案例。
