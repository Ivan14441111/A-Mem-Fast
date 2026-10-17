# A-Mem-Fast 多会话对照实验

> 工作稿：严格 16-tag 协议因一个正常结束的 18-tag 响应中止。下一主协议为 `observed-tags-v1`：模型请求不变，记录 tag 指令违例而不把它当作应用错误。主实验尚待离线验收与新源码快照后启动；结果表不混入此前失败协议。

本轮研究当前上游 robust 实现的长记忆工程效率，比较顺序基线、保持成功路径请求语义的 exact 优化，以及改变演化任务的 fused 策略。范围为三个 LoCoMo conversation、每个前 128 turns、三次重复；不是原论文全量复现。固定上游版本为 `WujiangXu/A-mem@0c8039f28fdcc08189a23c07a3437d9d2482f9c2`。瓶颈推导、具体代码语义与后续设计见 [深度分析](deep_analysis.md)。

## 1. 共同推理配置与比较对象

所有组使用本地 Qwen3-8B BF16 / vLLM 0.11.2 / A100 80GB，关闭 thinking，temperature=0、seed=42；新协议统一 max_tokens=2000、repetition_penalty=1.05、tag_budget=16、tag_budget_policy="observe"，并在原 system message 后附加相同的指令：

> Return no more than 16 items in each TAGS/tags list, including lists for updated neighbors. Follow the requested format and stop after completing its required fields.

这是显式配置后的 **configured-robust 基线**，不是上游默认 temperature=0.7 配置。共同重复惩罚和 system message 会改变模型输出，不能把它们的收益归给 exact 实现。上游 user prompt 模板保持原文；默认 API 不启用新增选项，只有实验显式设置时才改变请求。完整推理参数和 system message 进入缓存身份与 manifest。

客户端在缓存写入前及读取后观察识别出的 tags 列表长度。超过 16 记录指令违例与实际长度，保留完整响应，不裁剪、不去重；它是生成偏好，不是强制 schema 上限。最大生成 tokens 仍由 2000 控制，触及截断、HTTP 错误或缺少正文仍明确失败；JSON 领域 schema 与候选引用仍严格校验。live 生成违例和缓存读取观察分别统计，避免同一响应预取后再次消费被算成两次模型违例。`observed-tags-v1` 与严格 `bounded-tags-v1` 的 HTTP payload 相同，但接受策略、缓存身份与实验身份不同。默认 API 仍保留严格策略，实验显式选择 observe。

| 组别 | 冷构建流程 | 是否改变模型任务 | 主要检查 |
| --- | --- | --- | --- |
| baseline | 上游 metadata、decision、条件 strengthen/update 顺序执行 | 共同配置后的对照组 | 录制真实响应 |
| exact / `optimized` | 滚动有界 metadata 预取、增量精确索引、变化文本重建、请求缓存 | 成功路径保持共同配置下的请求内容与处理次序 | baseline→exact 严格重放 |
| fused | metadata 预取后，一次请求输出决策、连接、tags、稀疏邻居更新 | 是；同时明确全局 ID、限制候选、删除 reason 和无变化邻居重述 | 独立实时质量对照与自身重放 |

“exact”不保证两次实时推理逐字相同。不同批次和调度仍可能产生不同模型输出；它保持的是固定响应下的成功路径语义。错误处理更严格，向量数值验证也限定于固定 encoder、dtype 和测试输入。fused 是整体策略消融，其收益不能单独归因于合并 HTTP 请求。本轮三组 QA 均采用 upstream 上下文策略，不同时打开去重。

## 2. 数据、环境与测量

| sample | conversation | turns | 证据完整且类别 1–4 的候选 QA | 实际 QA | 实际类别分布 |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | conv-26 | 128 | 52 | 12 | 1:3，2:3，3:3，4:3 |
| 1 | conv-30 | 128 | 34 | 12 | 1:4，2:4，4:4 |
| 2 | conv-41 | 128 | 19 | 12 | 1:4，2:4，3:1，4:3 |

每个 conversation 取时间顺序前 128 turns；只选 evidence 非空且全部已载入的问题，从类别 1–4 轮转抽取 12 题。类别 5 未纳入。独立问题数为 **36**；三次重复仍回答同一题集，不能称作 108 道独立问题。subset 保存原始 QA 索引、证据和指纹。

MiniLM 使用固定 revision、CPU 4 线程；retrieve-k=10、evo-threshold=100、metadata workers=4、prefetch-window=16。基础环境见 [environment.json](environment.json)；最终实验以各自 manifest 和代码快照为准。

三组每轮均从空响应缓存运行，每组前重置 vLLM prefix cache，并在三个重复中轮换位置。模型已加载；下载、服务启动、READY/encoder 预热不计入 build/QA，初始化单列。**真实请求的推理等待全部计入 build/QA**。baseline 为对照录制而新增的 SQLite flush 单列排除，优化组实际缓存读写计入时间。QA 包含轻量字符串评分。

每个 cell 保存完整预测、阶段请求/tokens、live 事件、逐步记忆指纹和源文件哈希。测量源码独立归档；测量结束后的工程修订不能追溯成旧运行已使用的代码。新 observed 主实验的目录、源码 checkpoint 与指纹将在离线验收后启动时填写。此前 [bounded 主协议](../results/suite128-primary-bounded-v1-20261018/protocol.json) 和其 `81fcb979867b034111250b19edbd3023fa94476e` 源码快照作为失败协议保留，不能视作新协议已经使用的源码。

## 3. 结果与聚合口径

| 数据范围 | cell / conversation | exact 构建 / 总流程速度比 | fused 构建 / 总流程速度比 | baseline / exact / fused F1 |
| --- | --- | --- | --- | --- |
| 新主实验全部预设对照 | 待完成 | 待机器汇总 | 待机器汇总 | 待机器汇总 |
| 显存分配稳定子集 | 待资源审计 | 待机器汇总 | 待机器汇总 | 待机器汇总 |

速度首先按同一 cell 内 baseline/对应组的配对比值计算。汇总时先对同一 conversation 的 log speedup 取均值，再在 conversation 间等权平均并取指数。F1 先平均同一 conversation 的重复运行，再对 conversation 等权平均；次数不等时不隐式降低某个 conversation 的权重。逐 cell、逐 conversation、范围和失败均保留，不把少量重复测量当作总体置信区间。新协议的 9 个预设 cell 各尝试一次；普通推理/策略/重放失败保留记录后继续下一独立 cell，源码完整性失败仍立即终止。不会按结果修改参数或补跑到成功。若有失败，速度和 QA 汇总明确为“完成配对条件下”的描述，同时列出 planned/attempted/completed/failed/not-run 分母及每个 conversation 的完成数；不能以完成子集冒充全部九次结果。

质量为本 harness 的 normalized EM 与 multiset token F1，不是论文综合 Score，也不等于上游 set-F1。最终解释同时报告构建 HTTP、输入/输出 tokens、fused 决策分布、连接/邻居更新、consolidation 次数，以及每阶段 live tag 指令违例率和实际最大列表长度。少调用不代表同比例省时间；不同演化率也会改变重建与 QA 上下文。

推理时延分位数使用 `http_requests > 0` 的 live 事件。原始 client 入口分位数包含 cache hit，不能把优化组 metadata 的入口 p50 当作纯模型推理 p50。请求耗时之和在并发组可能超过 wall time，嵌套计时也不能直接相加。

## 4. 重放与共享资源边界

每个 cell 做 baseline→exact、exact→自身、fused→自身三项重放，校验逐条记忆状态、原序请求键及完整 QA，且重放阶段零 HTTP。baseline→exact 隔离实时服务输出差异；fused 自身重放只能证明能复现该策略的记录，不证明与 baseline 等价。最终通过数量和重建路径覆盖将在完整结果产生后填写。

所有预设 cell 都进入完整报告。另按新实验启动前固定的规则做描述性敏感性分析：GPU2 整个 cell 的显存最大值与最小值差不超过 1,024 MiB；至少 10 个样本；相邻采样间隔不超过 10 秒。监控不足不进入子集。这是“显存分配稳定”，不是“独占 GPU”或“计算负载恒定”；显存不变仍可能存在计算竞争。规则只按资源监控决定，不按速度或分数筛选。若发生新的资源切换，保留原始结果并说明，不静默替换。

## 5. Pilot 记录与共同配置的由来

最初 max_tokens=1000 的 [pilot suite](../results/suite128-three-arm-20261018/suite.json) 为 1 个完成、1 个失败、7 个未运行。第 116 条 `D7:8` 的 strengthen 触及输出上限，strict client 拒绝截断响应。用前序 decision 的文本、keywords 和邻居快照可复原此前 114 条 strengthen 请求键，唯一缺失键对应这次失败。旧失败正文未保存；[失败记录](../results/suite128-three-arm-20261018/failure_report.json) 保留了可验证范围。

同一 prompt 的 [2000-token 诊断](../results/diagnostics/strengthen_cap2000.json) 仍截断，生成 15,001 字符，`Transformational` 重复 683 次。这是 Qwen3-8B、temperature=0 配置下的实证，不是上游默认配置或所有 A-Mem 模型的普遍行为。随后固定 RP=1.05、对预先选择的 10 个请求做格式诊断：9 个正常控制通过，失败输入变为 78 tokens、1.11 秒正常停止，但生成 25 个 tags，未满足预设 ≤16 条，因此 [RP-only 诊断](../results/diagnostics/rp105-fixed-controls/results.json) 仍记为失败，没有事后放宽门槛。

随后仅加入第 1 节共同的 16-tag system 约束，保持 RP=1.05、同一批 10 个输入、同一接受标准，检查停止、明确格式与完整邻居块，不依据 QA 得分搜索参数。[bounded-tags-v1 诊断](../results/diagnostics/bounded-tags-v1-fixed-controls/results.json) 10/10 通过，实际执行 10 次 HTTP，未写入响应缓存。原失败输入正常停止于 58 output tokens，tags 不超过 16；两个五邻居 update 控制均保留完整块，分别为 326、391 tokens。这证明固定控制输入满足接受条件，不保证任意新输入不会失败。之前的失败与源码快照均保留，pilot 不与新主实验混合汇总。

严格 `bounded-tags-v1` 随后仍在首个 cell 的 exact 组第 83 条 `D5:7` 失败：strengthen 正常停止于 68 output tokens、271 字符，却给出 18 个 tags，超过固定 16 门槛。完整诊断见 [optimized-failure.json](../results/suite128-primary-bounded-v1-20261018/turns128-sample0-upstream-r1/optimized-failure.json)。这是指令遵循偏差，和前述 2000-token 重复长尾不同；十个固定控制通过并不保证运行时所有输入满足软提示。该失败判定不追溯更改。

新的 `observed-tags-v1` 不改模型请求或继续搜索惩罚系数，仅把非语义必需的 tag 数量上限改成显式观察项；严格截断及领域校验仍保留。验收使用先前十个成功响应与这条 18-tag 响应做离线回归，不重新发起十次实时诊断，也不按 QA 分数选择参数。新协议九个 cell 重新从空缓存执行，与此前协议不混合。任何后续失败按固定继续策略报告，不再因正常 cell 失败而调整参数重启研究。

pilot 的唯一完整 cell 还遇到 GPU2 分配下降 15,995 MiB，导致顺序基线与优化组处于不同资源状态；其约 2.76× exact 构建比值不能作为纯实现收益。定位证据见 [resource_audit.json](resource_audit.json)。同一 cell 的 fused 上下文更稀疏，原始标注证据平均覆盖从 baseline 的 0.750 降至 0.542，见 [逐题诊断](first_cell_quality_audit.json)。它支持证据覆盖减少的线索，但连接密度、metadata、检索与重建同时变化，尚不能归因于某一条提示词。证据匹配也不是官方检索指标。

## 6. 独立消融与项目结论

QA 去重、joint 单请求构建及完整 419-turn 案例如开展，使用独立配置和结果目录，不混入三组主结果。joint 会改变 metadata 的生成条件并取消独立预取，须用新的同轮 baseline 对照，不能借用数小时前的时间。联合策略的固定样本、顺序与比较口径见 [joint pilot 协议](joint_pilot_protocol.md)。

推荐配置和项目介绍中的实测数字将在有效主结果与资源审计完成后确定。项目贡献以长记忆处理的瓶颈定位、可验证调度与索引优化、缓存/失败可观测性、实验可复现性为核心；fused/joint 作为明确改变行为的策略研究，不预先宣称质量保持或原创记忆算法。
