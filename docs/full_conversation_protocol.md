# 完整 conversation 的默认工程路径验证

预设时间：2026-10-17T18:07:50Z；任何本案例实时输出之前固定。

本协议修订此前“至少 baseline/exact/fused”的可选长会话计划，明确只运行 baseline 与 exact。原因是隔离默认工程优化并覆盖重复 consolidation；当前可选 JSON 策略已暴露格式可靠性问题，不应阻止默认路径的长链验证。这不是根据性能结果挑选更有利的组，也不替换九个 cell 的主实验。fused/joint 的结构化方案另按 held-out 协议研究。

**数据。** LoCoMo sample 0 / conv-26 的完整 chronological conversation，预计 419 turns；启动前核对实际 turn 数与完整内容指纹。使用同一 evidence 非空且全部在范围内、类别 1–4 的筛选和轮转规则，max_questions=48，保留实际选择、原始 QA 索引与类别分布；若实际 eligible 不足则全取并报告。只有一个 conversation 和一次重复，不能作为新的多会话总体估计。

**运行顺序与配置。** baseline → exact 各一次真实冷运行，均从空独立响应缓存和空记忆图开始，每组前重置 prefix cache。Qwen3-8B BF16、MiniLM revision、temperature=0、seed=42、max_tokens=2000、RP=1.05、tag_budget=16、tag_budget_policy=observe；QA context=upstream，retrieve-k=10，evo-threshold=100，metadata workers=4、prefetch-window=16。结构化请求开关关闭；不加入 fused/joint，不复用数小时前的 baseline 时间。保存最终源码快照、命令和模型身份。

**验证与结果。** 冷运行后用 baseline 缓存做 baseline→exact 逐条状态/请求键/QA 重放，并对 exact 自身重放；全部重放零 HTTP。记录构建与 QA 耗时、真实请求与 tokens、tag 指令违例、各阶段分布、实际 consolidation 次数/时长和逐题质量。只按这个单案例的配对结果陈述，不与 128-turn 主结果合并，不因负收益而删除。

**失败。** 各冷 arm 至多尝试一次，不修改配置补跑。一个冷 arm 失败时保留进度与诊断，若驱动支持独立继续则尝试另一个 arm；不能把未尝试记作失败或成功。失败后不存在完整配对速度比，仍可报告完成 arm 的描述和可进行的自身零 HTTP 重放。源码、配置或重放完整性失败立即终止。

**资源与时间。** 保存 GPU2 轨迹，使用与主实验相同的显存分配稳定描述规则，但不把显存稳定当成计算独占。执行顺序为当前主实验 → 离线恢复验证 → 可用三图 QA context 消融（保留测量源码，确保重建身份一致）→ 推广已审阅的结构化实现并执行固定三项 schema 门禁 → 本默认长会话案例 → 时间允许时 structured held-out pilot。启动前必须预留完成两组、验证和最终解释的时间；否则明确未执行，不用更短案例冒充完整 419-turn 验证。

**实际启动身份。** 协议于UTC2026-10-17 18:50:52保存后启动，结果目录 [full419-default-observed-v1-20261018](../results/full419-default-observed-v1-20261018/)。419turns，149道eligibleQA中固定选择48题，类别1/2/3/4分别13/12/11/12。源码checkpoint `21b5b23de2089e6b096a86242a7d51a67cc06d29`，fingerprint `604e19a2aa7e2c145ff5636edb2064eb99d974fbc6744cebcd838abbf2ada8f0`。本案例明确structured_policy_outputs=False；schema门禁通过不改变默认两组请求。结果将在两组及验证结束后独立报告。

**独立冷 arm 的执行说明。** 若整体驱动因 baseline 的普通冷运行错误退出，尚未尝试的 exact 可通过同一冻结 `benchmark.py`、相同配置的 `modes=optimized,warm` 独立执行。保留原失败 cell 和单独的 continuation 记录；这只落实上文每个 arm 至多尝试一次的覆盖，不重试失败 baseline，不修复其响应，不换源码/参数。若 exact 已尝试则不得再次运行。源码、配置或重放完整性失败仍全局终止。此执行方式在任何完整案例失败结果出现前明确。
