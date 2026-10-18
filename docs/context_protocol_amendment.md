# QA 上下文去重：固定源图的选择

修订时间为 2026-10-17T18:11:13Z，早于本实验的答案生成。原方案见 [context128 observed protocol](../results/context128-observed-v1-20261018/protocol.json)。当时 sample 0 的 baseline 已完成，sample 1/2 的指定 baseline 尚未运行。本次只调整取用哪次重复的记忆图，问题、答案生成顺序和推理参数沿用原方案。

主实验轮换三组顺序，前面的 fused 失败可能使后面的 baseline 未被执行。因此，将原定“都取 repeat 1”改为“各会话固定取 baseline 排在第一的那次运行”：

| sample / 会话 | 指定来源 | 答案生成顺序 |
| --- | --- | --- |
| 0 / conv-26 | turns128-sample0-upstream-r1 | upstream → deduplicated |
| 1 / conv-30 | turns128-sample1-upstream-r3 | deduplicated → upstream |
| 2 / conv-41 | turns128-sample2-upstream-r2 | upstream → deduplicated |

来源直接按主实验顺序确定。若指定 baseline 失败、未完成或无法零 HTTP 回放，该案例记为不可用。每图保留固定的 12 题；记忆图和问题关键词从指定缓存重建，两种上下文只重新生成答案。

沿用 observed-tags-v1、max_tokens=2000、RP=1.05、tag16/observe、temperature=0、seed42。九次主实验尝试结束后再生成答案。每组使用独立空答案缓存，重置前缀缓存，记录去重前后行数、输入/输出 tokens、答案时间和逐题 EM/F1。

这是一次在观察到 fused 失败后、context 输出前作出的方案修订。全部三个计划案例均保留状态；同一张图上的两种上下文共享一组问题，按配对比较。
