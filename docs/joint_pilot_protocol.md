# Joint held-out pilot：运行前固定的协议

固定时间：2026-10-17T17:29:33Z。本文写于任何 joint 实时输出之前；只有共同配置完成验收、主实验结束后才执行。若时间不足，报告未执行，不用其他配置结果替代。

修订时间：2026-10-17T17:45:53Z，仍在任何 joint 实时输出之前。严格 tag 预算主协议因正常结束的 18-tag 响应中止，因此本 pilot 共同接受策略修订为 `observed-tags-v1`；数据、组别、顺序和问题选择不变。模型 HTTP payload 保持原 16-tag 指令、RP=1.05 和 max_tokens=2000，新增观察策略有独立缓存/源码身份。原始协议及失败历史保留于版本记录，不与新运行混合。

第二次修订时间：2026-10-17T18:07:50Z，仍在任何 joint 实时输出之前。可选 fused/joint arm 增加生成期间 JSON schema 约束，其他组保持原文本协议；这是新 structured 策略身份，不与无结构约束主实验混合。只在[固定三项格式门禁](structured_policy_gate_protocol.md)全部通过后运行，schema 与 response_format 必须进入缓存/manifest。样本、问题与 Williams 顺序不变。执行优先级改为默认完整 conversation 验证之后，时间不足则本 pilot 不执行，不删减问题或挑选成功组。

**数据。** 主实验之外的 sample 3 / `conv-42` 与 sample 4 / `conv-43`，分别取时间顺序前 64 turns。沿用证据完整与类别 1–4 过滤、轮转选取规则，max_questions=24；实际各有 17、14 道符合条件的 QA，全部纳入。每个 conversation 重复两次，共 4 个 cell、31 道独立问题；重复不增加独立样本数。保留 source QA index 与 subset 指纹。

**四组与顺序。** A=baseline，B=exact，C=fused，D=joint。采用四行 Williams 顺序并在每个 conversation 内反转：

| cell | 冷运行顺序 |
| --- | --- |
| sample 3 / repeat 1 | A → B → D → C |
| sample 3 / repeat 2 | C → D → B → A |
| sample 4 / repeat 1 | B → C → A → D |
| sample 4 / repeat 2 | D → A → C → B |

每组都重新进行真实空响应缓存构建与 QA，每组前重置 prefix cache；不能借用其他时段的 baseline 时间。每个 cell 后校验 baseline→exact，以及 exact、fused、joint 各自的响应重放；重放阶段零 HTTP，不作为实时提速数据。

**共同配置。** 与新主实验相同的 Qwen3-8B BF16、MiniLM revision、temperature=0、seed=42、max_tokens=2000、repetition_penalty=1.05、tag_budget=16、tag_budget_policy="observe" 和原共同 16-tag system 指令（`observed-tags-v1`）；retrieve-k=10、evo-threshold=100、workers=4、prefetch-window=16。joint 不实际启动 metadata 预取，它的一次请求同时生成 metadata 与演化；明确作为策略变更。四组上游 QA 上下文均不去重。执行前保存完整源码快照、模型/配置身份与命令；不同源码或参数产生新实验身份。

**终止和诊断。** tag 数量超 16 是记录并保留全文的指令违例，不视为硬错误；分开统计 live 生成与缓存观察，报告每阶段违例率和最大实际长度。HTTP、截断、领域 schema、引用和重放仍严格失败。四个独立 cell 各尝试一次，普通冷运行 cell 失败保留记录后继续，源码、配置或重放完整性失败立即终止；不截断列表以伪造成功，不在结果产生后换 prompt、改参数或按质量补跑。若有失败，速度/质量描述限于完成配对，明确 planned/attempted/completed/failed/not-run 数量及 conversation 完成数。

**观察项。** 核心是 joint/fused 的同轮构建与 build+QA 速度、实际 HTTP、输入/输出 tokens、格式/截断失败、逐题 EM/F1 和证据/上下文规模；同时展示 baseline/exact 作为参照。查看各组实际演化分支、连接与邻居更新数量，不把请求数减少直接等同于延迟减少。对同一 conversation 的重复先汇总，再对两个 conversation 等权汇总；展示原始配对分布，不以小 pilot 宣称统计非劣效。

**资源。** 保存 GPU2 监控，并同时报告全部 cell 与事先固定的显存稳定子集：范围 ≤1,024 MiB、至少 10 个样本、最大相邻间隔 ≤10 秒。显存稳定不代表计算独占。这个 pilot 只判断进一步研究价值，不用于根据 QA 得分挑出下一版 prompt。

**完整长会话案例的修订。** 原先至少 baseline/exact/fused 的可选计划已在任何完整案例调用前修订为 baseline/exact 两组，优先隔离默认工程路径并验证多次 consolidation；不再要求格式尚待验证的可选策略先完成。具体数据、顺序与失败规则见 [full conversation 协议](full_conversation_protocol.md)。本 held-out pilot 若执行，仍保留原四组和四个 cell，不把长会话结果混入其中。

**预设128-turn后续。** 同两个conversation、每个128turn/24QA、两次重复的扩展已在任何joint输出前独立预设，见 [扩展协议](joint128_extension_protocol.md)。启动只依赖固定门禁、完整性和剩余时间，不能依64-turn QA或速度好坏改配置。两种深度题目重叠，合计48独立题而非31+48。
