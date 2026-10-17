# A-Mem-Fast 多会话对照实验

> 最终分协议报告：128-turn 主协议得到三个 baseline/exact 完成配对，九个无 schema 约束 fused arm 均失败；固定图 context 实验完成三个会话、36 道 QA。419-turn 案例仅 exact 完整完成，baseline 在 QA 失败且历史证据缺失，不形成速度配对。独立 64-turn structured pilot 完成四次预设尝试：11 个完整冷 arm 均通过零 HTTP 重放，两个 joint arm 在关键词阶段失败；128-turn 扩展因预设时间门槛未执行。各协议单列分母与配对范围，不合并成总体加速。

本轮研究当前上游 robust 实现的长记忆工程效率，比较顺序基线、保持成功路径请求语义的 exact 优化，以及改变演化任务的 fused 策略。范围为三个 LoCoMo conversation、每个前 128 turns、三次重复；不是原论文全量复现。固定上游版本为 `WujiangXu/A-mem@0c8039f28fdcc08189a23c07a3437d9d2482f9c2`。瓶颈推导、具体代码语义与后续设计见 [深度分析](deep_analysis.md)。

## 1. 共同推理配置与比较对象

所有组使用本地 Qwen3-8B BF16 / vLLM 0.11.2 / A100 80GB，关闭 thinking，temperature=0、seed=42；新协议统一 max_tokens=2000、repetition_penalty=1.05、tag_budget=16、tag_budget_policy="observe"，并在原 system message 后附加相同的指令：

> Return no more than 16 items in each TAGS/tags list, including lists for updated neighbors. Follow the requested format and stop after completing its required fields.

这是显式配置后的 **configured-robust 基线**，不是上游默认 temperature=0.7 配置。共同重复惩罚和 system message 会改变模型输出，不能把它们的收益归给 exact 实现。上游 user prompt 模板保持原文；默认 API 不启用新增选项，只有实验显式设置时才改变请求。完整推理参数和 system message 进入缓存身份与 manifest。

Qwen3-8B 官方模型卡对非 thinking 模式建议 temperature=0.7、top_p=0.8、top_k=20、min_p=0；本轮 temperature=0 是实验选择，并非采用厂商推荐解码。模型卡的明确 greedy 重复警告位于 thinking 模式条目，不能直接作为当前非 thinking 失败的因果证明。本轮不因长尾结果再调参，未来可预设推荐解码的可靠性对照，并同时衡量质量、tokens 和完成率。[官方模型卡](https://huggingface.co/Qwen/Qwen3-8B#best-practices)

客户端在缓存写入前及读取后观察识别出的 tags 列表长度。超过 16 记录指令违例与实际长度，保留完整响应，不裁剪、不去重；它是生成偏好，不是强制 schema 上限。最大生成 tokens 仍由 2000 控制，触及截断、HTTP 错误或缺少正文仍明确失败；JSON 领域 schema 与候选引用仍严格校验。live 生成违例和缓存读取观察分别统计，避免同一响应预取后再次消费被算成两次模型违例。`observed-tags-v1` 与严格 `bounded-tags-v1` 的 HTTP payload 相同，但接受策略、缓存身份与实验身份不同。默认 API 仍保留严格策略，实验显式选择 observe。

| 组别 | 冷构建流程 | 是否改变模型任务 | 主要检查 |
| --- | --- | --- | --- |
| baseline | 上游 metadata、decision、条件 strengthen/update 顺序执行 | 共同配置后的对照组 | 录制真实响应 |
| exact / `optimized` | 滚动有界 metadata 预取、增量精确索引、变化文本重建、请求缓存 | 成功路径保持共同配置下的请求内容与处理次序 | baseline→exact 严格重放 |
| fused（仅提示词要求 JSON） | metadata 预取后，一次请求输出决策、连接、tags、稀疏邻居更新 | 是；同时明确全局 ID、限制候选、删除 reason 和无变化邻居重述 | 独立实时质量对照与自身重放 |

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

每个 cell 保存完整预测、阶段请求/tokens、live 事件、逐步记忆指纹和源文件哈希。测量源码独立归档；测量结束后的工程修订不能追溯成旧运行已使用的代码。新主实验位于 [suite128-primary-observed-v1-20261018](../results/suite128-primary-observed-v1-20261018/)，[协议](../results/suite128-primary-observed-v1-20261018/protocol.json) 固定于 UTC 2026-10-17 17:50:57；源码 checkpoint 为 `2019a73d4ce7135acff4fc2cf51978ccee087124`，源文件指纹为 `4d475aeb3218110018ab0d03505c0b0b95f27fa2c8b89012eaf71fee544fdd4b`。213 项测试通过后冻结源码，源归档 SHA-256 为 `a92b9787e52baa468c68cba7392719313181af1575abfcb5046e686c05486329`。此前 [bounded 主协议](../results/suite128-primary-bounded-v1-20261018/protocol.json) 和其 `81fcb979867b034111250b19edbd3023fa94476e` 源码快照作为失败协议保留，不能视作新协议已经使用的源码。

## 3. 结果与聚合口径

九个预设 cell 均尝试一次，但没有一个完成全部三组：中止点均为 fused。不能把“9 个 cell 失败”解释成默认实现失败；按实际冷 arm 区分如下。

| 冷 arm | 计划 | 实际尝试 | 完成 | 失败 | 因前组失败未运行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 9 | 3 | 3 | 0 | 6 |
| exact | 9 | 6 | 6 | 0 | 3 |
| fused | 9 | 9 | 0 | 9 | 0 |

三个 conversation 各有一对 baseline/exact；其余三个完整 exact arm 仍保留独立描述，不混入配对均值。无完整 fused 构建/QA，因此不报告其完成耗时加速比或 F1。全部结局和源文件哈希见 [失败分类审计](primary_failure_audit.json)、[机器汇总](../results/suite128-primary-observed-v1-20261018/analysis_views.json)。

| 数据范围 | cell / conversation | exact 构建 / 总流程速度比 | fused 构建 / 总流程速度比 | baseline / exact / fused F1 |
| --- | --- | --- | --- | --- |
| 已完成 baseline/exact 配对 | 3 对 / 3 conversations | 1.143× / 1.131× | 无完成配对 | 0.3981 / 0.4039 / — |
| 显存分配稳定的完成配对 | 2 对 / 2 conversations | 1.075× / 1.068× | 无完成配对 | 0.4129 / 0.3930 / — |

三个保留下来的默认配对均为 baseline-first（sample0/r1、sample1/r3、sample2/r2），因此完成子集已不再保留原计划的位置平衡。上表 1.143× 与 stable 1.075× 都受这一顺序边界及共享资源影响，不能作为消除了顺序因素的因果估计。

同一完成配对范围的构建资源均值如下；先在 conversation 内平均重复，再等权汇总，不能混入未配对的完整 arm：

| 配对范围 | baseline → exact HTTP | input tokens | output tokens |
| --- | ---: | ---: | ---: |
| 全部三对 | 390.00 → 392.67 | 313,779 → 317,631 | 23,337.67 → 24,843.00 |
| stable 两对 | 390.50 → 396.00 | 319,124.50 → 326,861.00 | 23,450.50 → 26,381.50 |

速度首先按同一 cell 内 baseline/对应组的配对比值计算。汇总时先对同一 conversation 的 log speedup 取均值，再在 conversation 间等权平均并取指数。F1 先平均同一 conversation 的重复运行，再对 conversation 等权平均；次数不等时不隐式降低某个 conversation 的权重。逐 cell、逐 conversation、范围和失败均保留，不把少量重复测量当作总体置信区间。新协议的 9 个预设 cell 各尝试一次；普通冷运行推理/策略失败保留记录后继续下一独立 cell；源码、配置或重放完整性失败立即终止。不会按结果修改参数或补跑到成功。若有失败，速度和 QA 汇总明确为“完成配对条件下”的描述，同时列出 planned/attempted/completed/failed/not-run 分母及每个 conversation 的完成数；不能以完成子集冒充全部九次结果。

配对分析按 arm 完成状态判断，而不是要求同一 cell 的三个 arm 全部成功：baseline 与 exact 完成而 fused 失败时，前两者仍构成一个已完成配对，fused 保留失败。这个分析口径在首个 fused 失败后明确，未改变实时运行或增加模型调用。每个比较单列有效配对数、失败与未尝试 arm；不把其他策略失败的记录删除。若失败发生在原计划重放之前，明确标记“未执行”，可在 suite 后仅使用已有缓存补做零 HTTP 验证，并单列其源码与执行身份。

质量为本 harness 的 normalized EM 与 multiset token F1，不是论文综合 Score，也不等于上游 set-F1。最终解释同时报告构建 HTTP、输入/输出 tokens、fused 决策分布、连接/邻居更新、consolidation 次数，以及每阶段 live tag 指令违例率和实际最大列表长度。少调用不代表同比例省时间；不同演化率也会改变重建与 QA 上下文。

推理时延分位数使用 `http_requests > 0` 的 live 事件。原始 client 入口分位数包含 cache hit，不能把优化组 metadata 的入口 p50 当作纯模型推理 p50。请求耗时之和在并发组可能超过 wall time，嵌套计时也不能直接相加。

默认路径三对均来自预设 baseline-first cell（sample0/r1、sample1/r3、sample2/r2），其他完整 exact arm 不混入配对均值。三对汇总的 EM 两组均为 0.1111；stable 子集为 baseline 0.1667、exact 0.1250。实时质量有波动，不能凭三个 conversation 的 F1 均值接近就宣称非劣效或保持质量。数值和来源哈希见 [独立审计](primary_default_pair_audit.json)。

### 已完成默认路径配对的解释示例

在最终聚合之前，三条已完成 baseline/exact 配对说明为什么预取并不保证每次实时运行更快。其所属 cell 随后在 fused 失败，下面只描述有效的默认路径 arm，不把它们当作完整三组成功。

| 配对 | baseline → exact 构建 | update 请求 | 构建输出 tokens | 串行演化客户端耗时增量 | 构建减串行演化的残差变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sample0/r1 | 344.09 → 358.46 s | 13 → 23 | 25,787 → 31,422 | +79.65 s | −65.28 s |
| sample1/r3 | 285.57 → 236.99 s | 4 → 5 | 21,114 → 21,341 | +14.81 s | −63.39 s |
| sample2/r2 | 309.71 → 240.01 s | 7 → 4 | 23,112 → 21,766 | −5.63 s | −64.06 s |

sample0 的新增 5,635 output tokens 中，5,331 来自更多/更长的邻居 update，占净增量约 94.6%。尽管串行演化之外的暴露时间减少，更多演化工作仍使总构建变慢；sample1 的生成工作量更接近，减少的残差则超过演化增加量。这里的残差只是 `build wall − decision/strengthen/update 客户端时长之和`，还包含 embedding、index、调度等，**不是纯 metadata 时间，也不是隐藏时长的直接测量**。并发 metadata 与演化计时不能直接相加得到 wall time。

对相同完整请求身份做只读缓存对比，每个配对均有 128 个相同 metadata 请求，其中分别有 34、18、23 个返回不同正文。即使输出 token 总数几乎相同，metadata 也不必相同，后续图与分支可能沿不同路径发展。这些记录证明实时响应存在差异；没有隔离实验，不能断言并发本身导致这些语义变化。Amdahl 估算假定剩余工作不变，实际 LLM 分支和 tokens 不固定正是它的解释边界。固定响应重放另行检验实现是否改变同一响应下的处理语义。原始文件哈希、计时代数、请求键与响应哈希见 [默认路径配对审计](primary_default_pair_audit.json)。

## 4. 重放与共享资源边界

原计划每个 cell 做 baseline→exact、exact→自身、fused→自身三项重放，但 fused 中止使这些后置验证未被执行。suite 后对三个完整 baseline 和六个完整 exact arm 独立恢复零 HTTP 重放，校验逐条记忆状态、原序请求键及完整 QA；fused 没有完整可验证 arm。baseline→exact 隔离实时服务输出差异；fused 自身重放只能证明能复现该策略的记录，不证明与 baseline 等价。恢复验证于 UTC18:43:04 完成：3 次 baseline→exact 与 6 次 exact 自身重放全部通过，逐条规范化 memory trace、原序请求键和完整 QA 一致，新增 HTTP 与 client errors 均为 0。九次验证共覆盖 1,152 个逐条状态比较和 9 次 consolidation；这是重复运行的检查数量，不是 1,152 条独立实验样本。测量源码保持原冻结身份。证据见 [replay_recovery.json](../results/suite128-primary-observed-v1-20261018/replay_recovery.json) 与 [独立原始报告复核](primary_default_pair_audit.json)。原始九个 cell 的 failed 状态不因后置默认路径验证通过而改写。

所有预设 cell 都进入完整报告。另按新实验启动前固定的规则做描述性敏感性分析：GPU2 整个 cell 的显存最大值与最小值差不超过 1,024 MiB；至少 10 个样本；相邻采样间隔不超过 10 秒。监控不足不进入子集。这是“显存分配稳定”，不是“独占 GPU”或“计算负载恒定”；显存不变仍可能存在计算竞争。规则只按资源监控决定，不按速度或分数筛选。若发生新的资源切换，保留原始结果并说明，不静默替换。

实际资源审计中，sample0/r1 和 sample1/r3 的 GPU2 全 cell 显存均保持 44,483 MiB；sample2/r2 在 UTC18:29:10 至18:29:24 间短暂上升，峰值80,216 MiB后回到44,483 MiB，发生在 baseline 构建内部。按预设规则，该配对仍保留在全部完成配对结果，但不进入 stable 子集。不能由显存轨迹量化未测到的计算竞争开销。stable 结果因此只覆盖两个 conversation，不是原三个 conversation 的均衡重复估计；显存稳定本身也不保证计算独占。

无 schema 约束的 fused 在第 4–31 条之间失败：4 次完整正文 JSON 解析发现一个多余末尾 `}`；5 次 JSON 可解析，但邻居更新项缺少必需的 `context` 字段。九次响应均正常 `stop`，不是 token 截断。当前协议允许用空字符串表示 context 不变，但不允许省略必需字段；没有事后补空值或剪括号以伪造成功。这说明 prompt-only JSON 在长链上的完成率需要作为一等指标。后续固定 JSON schema 同时约束正确语法和 required 字段，并继续保留全局 ID、分支与领域校验；它是独立的新实验身份，不是对这九次失败的追溯修复。 九个失败 arm 的缓存还保留 419 个不同请求响应，usage 合计至少 190,802 input tokens、21,183 output tokens；独立只读复算一致。这个下界包括失败前已发出的 metadata 预取，不等于已提交记忆的工作量，也不包含所有可能重试，不能据此推算失败 arm 的完整速度或质量。见 [失败响应成本记录](../results/suite128-primary-observed-v1-20261018/failed_response_usage.json)。

## 5. Pilot 记录与共同配置的由来

最初 max_tokens=1000 的 [pilot suite](../results/suite128-three-arm-20261018/suite.json) 为 1 个完成、1 个失败、7 个未运行。第 116 条 `D7:8` 的 strengthen 触及输出上限，strict client 拒绝截断响应。用前序 decision 的文本、keywords 和邻居快照可复原此前 114 条 strengthen 请求键，唯一缺失键对应这次失败。旧失败正文未保存；[失败记录](../results/suite128-three-arm-20261018/failure_report.json) 保留了可验证范围。

同一 prompt 的 [2000-token 诊断](../results/diagnostics/strengthen_cap2000.json) 仍截断，生成 15,001 字符，`Transformational` 重复 683 次。这是 Qwen3-8B、temperature=0 配置下的实证，不是上游默认配置或所有 A-Mem 模型的普遍行为。随后固定 RP=1.05、对预先选择的 10 个请求做格式诊断：9 个正常控制通过，失败输入变为 78 tokens、1.11 秒正常停止，但生成 25 个 tags，未满足预设 ≤16 条，因此 [RP-only 诊断](../results/diagnostics/rp105-fixed-controls/results.json) 仍记为失败，没有事后放宽门槛。

随后仅加入第 1 节共同的 16-tag system 约束，保持 RP=1.05、同一批 10 个输入、同一接受标准，检查停止、明确格式与完整邻居块，不依据 QA 得分搜索参数。[bounded-tags-v1 诊断](../results/diagnostics/bounded-tags-v1-fixed-controls/results.json) 10/10 通过，实际执行 10 次 HTTP，未写入响应缓存。原失败输入正常停止于 58 output tokens，tags 不超过 16；两个五邻居 update 控制均保留完整块，分别为 326、391 tokens。这证明固定控制输入满足接受条件，不保证任意新输入不会失败。之前的失败与源码快照均保留，pilot 不与新主实验混合汇总。

严格 `bounded-tags-v1` 随后仍在首个 cell 的 exact 组第 83 条 `D5:7` 失败：strengthen 正常停止于 68 output tokens、271 字符，却给出 18 个 tags，超过固定 16 门槛。完整诊断见 [optimized-failure.json](../results/suite128-primary-bounded-v1-20261018/turns128-sample0-upstream-r1/optimized-failure.json)。这是指令遵循偏差，和前述 2000-token 重复长尾不同；十个固定控制通过并不保证运行时所有输入满足软提示。该失败判定不追溯更改。

新的 `observed-tags-v1` 不改模型请求或继续搜索惩罚系数，仅把非语义必需的 tag 数量上限改成显式观察项；严格截断及领域校验仍保留。observe 接受策略通过 213 项离线测试，包括 HTTP payload 不变、缓存身份隔离与已知超限 fixture；没有将十个先前控制全部再次离线重放，也没有重发十次实时诊断。先前十个 live 控制通过是相同请求配置的独立历史证据，不夸大成新接受策略的十项回归。未按 QA 分数选择参数。新协议九个 cell 重新从空缓存执行，与此前协议不混合。任何后续失败按固定继续策略报告，不再因正常 cell 失败而调整参数重启研究。

pilot 的唯一完整 cell 还遇到 GPU2 分配下降 15,995 MiB，导致顺序基线与优化组处于不同资源状态；其约 2.76× exact 构建比值不能作为纯实现收益。定位证据见 [resource_audit.json](resource_audit.json)。同一 cell 的 fused 上下文更稀疏，原始标注证据平均覆盖从 baseline 的 0.750 降至 0.542，见 [逐题诊断](first_cell_quality_audit.json)。它支持证据覆盖减少的线索，但连接密度、metadata、检索与重建同时变化，尚不能归因于某一条提示词。证据匹配也不是官方检索指标。

## 6. 固定图 QA context 去重：已完成的独立消融

按输出前修订的来源规则，使用 sample0/r1、sample1/r3、sample2/r2 的三个 baseline 图，36 道固定 QA。图与关键词从原缓存严格重放，新模型调用只生成答案。独立核对了源图和两种上下文的 memory digest，从 baseline SQLite 重新计算固定关键词 digest，并检查每题 context 与预设计划的 hash、去重行数和分数；全部一致。两组各 36 次真实答案请求，无 client error；图/关键词重建 0 HTTP，另有每图一次不计时 READY 预热。

| sample | 答案请求输入 tokens：upstream → 去重 | 输入减少 | 答案阶段速度比 | F1：upstream → 去重 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 89,040 → 53,383 | 40.05% | 1.534× | 0.4360 → 0.4318 |
| 1 | 78,476 → 49,245 | 37.25% | 1.405× | 0.3771 → 0.3804 |
| 2 | 75,463 → 50,492 | 33.09% | 1.329× | 0.4222 → 0.4476 |
| 三图汇总 | 242,979 → 153,120 | 36.98% | 1.420×（配对几何均值） | 0.4118 → 0.4199 |

context UTF-8 字节由 1,025,519 降至 636,340（−37.95%），渲染记忆行数由 1,447 降至 912（−36.97%）。两组 EM 均为 0.1111；36 题中 18 个预测正文相同，F1 提高 7 题、下降 3 题、其余 26 题相同。总体分数接近不等于去重无损或统计非劣效；sample0 F1 略降，另两个 conversation 略升。

计时的答案阶段合计 29.435 → 20.717 秒，包含检索/拼接、字符串评分与 SQLite 记录，**排除记忆构建、图重放和关键词 LLM 调用**；不能把 1.420× 称为整体记忆系统或完整 QA 流程加速。本阶段没有独立 GPU 监控工件，不沿用主实验的 allocation-stable 标签，时延只作为共享服务环境的观测。输入 tokens 和上下文字节减少则由真实请求 usage 与逐题内容直接核对。

历史 64-turn 的约 40.5% 是离线按完整文本去重的字符估算；这里是三个 128-turn 固定图中按记忆行号去重后的实测 tokens、UTF-8 bytes 和真实答案，不混用两种百分比。原始结果见 [context 实验](../results/context128-observed-v1-20261018/derived.json)，独立校验与逐题变化见 [context_live_audit.json](context_live_audit.json)。

## 7. 其他独立消融与项目结论

QA 去重、structured fused/joint 及完整 419-turn 案例均使用独立配置和结果目录，不混入三组主结果。完整案例已在调用前固定为 baseline/exact 两组，优先验证默认路径的多次 consolidation，见 [完整 conversation 协议](full_conversation_protocol.md)。可选 structured 策略须先通过 [固定三项格式门禁](structured_policy_gate_protocol.md)，不得修剪已有失败 JSON 来伪造成功。joint 会改变 metadata 的生成条件并取消独立预取，须用新的同轮 baseline 对照，不能借用数小时前的时间。联合策略的固定样本、顺序与比较口径见 [joint pilot 协议](joint_pilot_protocol.md)；另在任何 joint 输出前预设 [128-turn 扩展](joint128_extension_protocol.md)，只按固定时间/可用性门槛决定是否启动，不按64-turn质量挑选。

固定三项 schema 门禁已于 UTC18:50 完成，三项全部通过，恰好三次 POST；完整正文 JSON、固定 schema、候选引用与首条语义均通过独立校验。output tokens 分别170/88/100，全部正常stop。原失败 fused 请求的system/user/payload仅增加response_format；joint有邻居控制使用冻结的原baseline metadata。具体来源与哈希见 [structured_gate_audit.json](structured_gate_audit.json)。该小门禁只验证固定输入的格式与领域约束，不能替代长链完成率和QA对照。原无schema九次失败仍单列保留。

完整 419-turn 默认两组案例使用相同 observed 配置、关闭结构约束，预设 48 QA。baseline 日志确认完成 419/419 条构建、1,286 次 build HTTP，但在第 35 道 QA（原索引 69）的答案生成触及 2,000-token 上限；诊断显示重复短答案列表，响应被严格拒绝。冻结 harness 直到全部 QA 完成后才 flush baseline 录制缓存，因此 `baseline.sqlite` 为 0 条，原 live trace、build 计时和完整 QA 报告没有保存。不能据日志恢复这些指标，也不能宣称 baseline 零 HTTP 回放通过。已按预设规则独立尝试尚未运行的 exact arm；不重试 baseline、不更换参数。原始证据和可用性规则见 [full419_evidence_audit.json](full419_evidence_audit.json)。

未尝试过的 exact arm 随后独立完成：419 条构建 1,067.105 秒，48 QA 67.610 秒，F1 0.30086、EM 0.04167；构建产生 1,314 次 HTTP、1,117,250 input tokens、93,730 output tokens，实际触发 4 次 consolidation。完整 exact 自身回放的逐条状态、逻辑请求序列和 48 个 QA 与原 live 报告全部一致，0 HTTP、0 client errors。这是一个完整默认路径规模案例，缺失完整 baseline，**没有配对速度或质量提升结论**。continuation GPU 监控在构建开始约 3 分钟后才启动，不能给整组分配稳定标签。

预先固定的共同缓存验证也已完成：reference 与 Fast 均从 exact 的 1,410 条持久响应中重建全部 419 条，逐条状态、原序请求、最终图和演化次数一致；在第 101、201、301、401 条执行四次 consolidation，两次重建还都与原 live exact trace 一致，0 HTTP、0 errors。独立核对了原始报告、缓存 digest、全部 419 步、48 题评分及源码哈希，见 [审计](full419_evidence_audit.json)。这证明共同响应下的构建等价，**并未恢复丢失的原 baseline trace**；自回放的完整 QA 证明也仅属于 exact。全部实时与重放实验结束后，最终代码加入构建计时结束后的录制 flush 和原子 build checkpoint，以及失败时可用证据保存；原生成异常与持久化错误分别保留。该工程修复通过 [281 项离线测试](../results/clean-client-validation-final.json)，不追溯改变旧实验，也不等于生产向量索引恢复。

项目贡献包括可核验的默认路径调度/索引优化、固定图 QA token 压缩和可选联合策略的有条件速度收益；同时保留失败分母、质量变化与共享资源边界。fused/joint 改变模型任务，不归入无损工程优化，也不声称原创 A-Mem 算法或论文全量复现。


### held-out 的关键词长尾：完整应用链与图语义分开

64-turn sample3/r1 的 joint 已完成全部构建，却在第一题关键词生成截断。该请求身份与同轮已完成 baseline、exact 的关键词请求完全相同：后二者均 87 input / 41 output tokens、正常停止且正文一致；joint 同一身份生成到 2,000 tokens，strict client 拒绝并未写缓存。问题为 Joanna 的 hobbies；关键词请求只含问题文本，在检索前发出，不携带 joint 图内容，也没有 schema。这个观察说明相同请求身份不保证实时返回相同，不能归因于 joint 图语义；具体服务/数值原因没有隔离。完整 payload、响应哈希与代码依赖见 [关键词失败审计](heldout_keyword_failure_audit.json)。原 cell 保留失败，后续预设 cell 继续，不改参数或补跑。
同 conversation 的第二次重复也在 joint 的首个关键词请求失败：request key、87/2,000 tokens、8,949 字符及完整正文 hash 均与第一次一致。该轮 fused 的相同请求正常停止，41-token 正文又与首轮 baseline/exact 相同。因此当前记录中是三次同一正常正文、两次同一截断正文；这是可复核的重复模式，不是已隔离的原因。没有额外重发、prefix-reset 搜索或参数调整。


## 8. 独立 64-turn structured pilot 与未执行的 128-turn 扩展

sample3/conv-42 与 sample4/conv-43 各取前 64 turns、重复两次，分别有 17 和 14 道证据完整 QA，共 31 道独立题。四组采用预设 Williams 顺序；仅 fused/joint 的记忆 JSON 阶段启用固定通用 schema，其他模型、解码和 QA 上下文设置保持预设值。源码仍为测量 checkpoint `21b5b23`，完整记录见 [pilot](../results/joint64-heldout-structured-v1-20261018/suite.json)。

四个 cell 全部尝试：两个完整、两个因 joint 首题关键词截断而失败。13 个实际尝试的构建均完成 64 条记忆，但只有 11 个 arm 完成构建与全部 QA，不能把两种完成率混用：

| arm | 计划 | 尝试 | 构建完成 | 构建+QA 完成 | 失败 | 未到达 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4 | 3 | 3 | 3 | 0 | 1 |
| exact | 4 | 3 | 3 | 3 | 0 | 1 |
| structured fused | 4 | 3 | 3 | 3 | 0 | 1 |
| structured joint | 4 | 4 | 4 | 2 | 2 | 0 |

两个失败 joint 的 live 构建计时未保存，不从日志补造速度。其关键词请求没有读取图，并与 baseline/exact/fused 正常完成的请求身份相同；这仍是整条 joint arm 的失败，但不能归因于 joint 图。所有 11 个完整 arm 已逐条核对图、请求序列和完整 QA，自身/对照重放新增 HTTP 与 client errors 均为零，见 [恢复验证](../results/joint64-heldout-structured-v1-20261018/replay_recovery.json)。

以下均只用同一 cell 的完成配对；先对 conversation 内重复取均值（速度用 log），再在 conversation 间等权平均。各行的 baseline、QA 与资源范围不同，不能横向当作共同全样本排名：

| 对照 | 配对 / conversation / 独立 QA | 构建 / build+QA 速度比 | baseline → candidate F1 | baseline → candidate EM |
| --- | --- | --- | --- | --- |
| exact | 3 / 2 / 31 | 1.383× / 1.323× | 0.4449 → 0.4367 | 0.2143 → 0.1786 |
| structured fused | 2 / 1 / 14 | 1.970× / 1.852× | 0.5475 → 0.5793 | 0.4286 → 0.3929 |
| structured joint | 2 / 1 / 14 | 2.301× / 2.159× | 0.5475 → 0.6095 | 0.4286 → 0.4286 |

| 同范围构建资源 | baseline → candidate HTTP | input tokens | output tokens |
| --- | ---: | ---: | ---: |
| exact | 198.50 → 194.25 | 151,700 → 144,832.75 | 13,636 → 11,439.75 |
| structured fused | 206 → 127 | 166,671 → 81,758 | 17,199 → 11,056 |
| structured joint | 206 → 64 | 166,671 → 68,619 | 17,199 → 7,250 |

joint 的积极结果明确限于 sample4 的两个完成配对：baseline 构建 224.221/223.264 秒，joint 97.407/97.061 秒，配对几何速度比 2.301×，构建请求 206→64。平均 F1 提高，但同一 14 题中有 2 题提高、1 题下降、11 题不变，两次重复的变化计数相同；不是无损或统计非劣效证明。构建任务、输出 tokens、演化分支和图同时改变，不能把收益单独归因于减少 HTTP 或预取。64-turn 各 arm 均未触发 consolidation；多轮重建的正确性证据来自独立 419-turn 案例。

原先四行平衡顺序因两个 cell 中止而未在完整子集保留；sample4 的两个完整顺序互逆仍不足以排除共享 GPU 竞争。更不能把 samples0–2、128 turns 的九次无约束失败与 samples3–4、64 turns 的 structured 构建完成作受控可靠性提升对照。固定三项门禁只在它的相同输入上隔离增加 response_format 的变化。原始计数、逐题重算、资源加总与官方配对汇总已独立核对，见 [heldout64_audit.json](heldout64_audit.json) 和 [机器分析](../results/joint64-heldout-structured-v1-20261018/analysis_views.json)。

128-turn 扩展没有执行：预设公式得到 7,266.976 秒（121.12 分钟），当时剩余 9,280.230 秒，扣除 40 分钟收尾预算后仅有 6,880.230 秒，未满足启动门槛；四个计划 cell 均为零尝试。没有依据 64-turn 分数、策略速度或失败结果更换样本/参数，也没有追加模型调用。见 [固定排期决定](../results/joint128-heldout-structured-v1-20261018/scheduling_decision.json)。
