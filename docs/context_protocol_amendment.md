# QA context 消融：来源选择修订

修订时间：2026-10-17T18:11:13Z，在任何 context 消融实时输出之前。原协议见 [context128 observed protocol](../results/context128-observed-v1-20261018/protocol.json)。此时 sample 0 的 baseline 已完成；sample 1/2 的下述指定 baseline 尚未运行。修订只改变取图 repeat，不改变主实验、问题、答案顺序或推理参数。

原计划三个 conversation 都取 repeat 1 的 baseline 图。但主实验按不同顺序轮转，fused 若先失败，会使后置 baseline 根本未被尝试。为避免按“第一个成功结果”筛选图，改为对每个 conversation **固定选择预设冷运行顺序中 baseline 位于第一的 cell**，可直接由原 suite 顺序机械确定：

| sample / conversation | 唯一指定源 cell | context 答案调用顺序 |
| --- | --- | --- |
| 0 / conv-26 | turns128-sample0-upstream-r1 | upstream → deduplicated |
| 1 / conv-30 | turns128-sample1-upstream-r3 | deduplicated → upstream |
| 2 / conv-41 | turns128-sample2-upstream-r2 | upstream → deduplicated |

这一规则不读取图质量、QA 分数或速度。若指定 baseline 失败、未完成或无法通过严格零 HTTP 重建，该 case 标为 unavailable，不改用另一次成功 repeat，也不补跑构图。各 case 的固定 12 题全部保留，图和关键词均从该指定 baseline 严格缓存重放；两种 context 只重新调用答案模型，使用相同固定图与关键词。

沿用 observed-tags-v1、max_tokens=2000、RP=1.05、tag16/observe、temperature=0、seed42。所有九个主实验 cell 完成尝试后才允许 context 答案调用。每组独立冷答案缓存和 prefix reset 的原约定不变；保存原始与去重行数、prompt/output tokens、答案耗时、逐题 EM/F1 及配置身份。失败/unavailable 进入计划三图的分母，不选择性隐藏，也不把同一个图上的两种 context 视作独立会话样本。

本修订是看到可选策略失败后、context 输出之前的显式协议调整，并非声称原始协议从未变化。它隔离源图可用性与第三个策略的运行先后，不保证选定图或答案质量更高。
