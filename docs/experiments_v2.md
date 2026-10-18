# A-Mem-Fast 多会话对照实验

> 本报告包含四项结果：128 条记录的主实验、固定图上下文去重、完整 419 条案例，以及独立会话的 64 条 structured 实验。128 条 structured 扩展因剩余时间不足取消。各节列出计划与实际完成情况，速度按同轮完成配对计算。

本轮以 `WujiangXu/A-mem@0c8039f28fdcc08189a23c07a3437d9d2482f9c2` 的 robust 实现为对照，研究记忆构建速度和问答（QA）变化。主实验使用三个 LoCoMo 会话、各前 128 条记录、三次重复；可选 fused/joint 策略分别测量。实现分析见 [深度分析](deep_analysis.md)。

## 1. 共同推理配置与比较对象

所有组使用本地 Qwen3-8B BF16 / vLLM 0.11.2 / A100 80GB，关闭 thinking，temperature=0、seed=42；本方案统一 max_tokens=2000、repetition_penalty=1.05、tag_budget=16、tag_budget_policy="observe"，并在原 system message 后附加相同的指令：

> Return no more than 16 items in each TAGS/tags list, including lists for updated neighbors. Follow the requested format and stop after completing its required fields.

各组统一使用上述配置，上游默认 temperature=0.7。原 user 提示词保持不变，重复惩罚和 system 指令由实验显式开启；API 默认保留原设置。完整参数保存在配置记录并参与缓存键计算。

Qwen3-8B 对非 thinking 模式建议 temperature=0.7、top_p=0.8、top_k=20、min_p=0；本轮固定 temperature=0。官方模型卡的 greedy 重复警告写在 thinking 条目，当前重复问题的具体原因还需解码对照。[官方模型卡](https://huggingface.co/Qwen/Qwen3-8B#best-practices)

客户端在写入和读取缓存时记录 tags 列表长度。超过 16 项时保留全文并计数，最大输出仍为 2000 tokens；HTTP、缺少正文、截断、JSON 字段或候选引用错误仍会失败。实时生成和缓存读取分别统计。`observed-tags-v1` 与 `bounded-tags-v1` 使用相同 HTTP 请求，前者观察超限，后者拒绝超限，缓存分别区分。API 默认使用拒绝规则。

| 方案 | 构建流程 | 与原流程的关系 | 验证 |
| --- | --- | --- | --- |
| baseline | 依次提取关键词与摘要、判断更新（更新决策）、建立关联（建立关联）或修改邻居（邻居修改） | 统一配置的原流程对照 | 保存实际模型回答 |
| exact / `optimized` | 提前提取信息，复用文本向量，按原顺序更新和写入 | 保留请求内容与处理顺序 | 用 baseline 回答逐步回放 |
| fused（仅提示词要求 JSON） | 提前提取信息，一次返回决策、关联和需要修改的邻居 | 合并任务，调整编号与输出方式 | 独立测量并用自身回答回放 |

exact 保留原请求内容和顺序，并在固定模型回答下比较逐步状态。实时输出可能因服务调度等因素变化；向量测试使用固定编码器和 dtype。fused 同时修改任务、格式和邻居更新方式。三组 QA 均使用 upstream 上下文策略，保留重复条目。

## 2. 数据、环境与测量

| 样本 | 会话 | 记录数 | 证据完整且类别 1–4 的候选问题 | 实际问题数 | 实际类别分布 |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | conv-26 | 128 | 52 | 12 | 1:3，2:3，3:3，4:3 |
| 1 | conv-30 | 128 | 34 | 12 | 1:4，2:4，4:4 |
| 2 | conv-41 | 128 | 19 | 12 | 1:4，2:4，3:1，4:3 |

每会话按时间顺序取前 128 条，从类别 1–4 中轮转选 12 题，要求标注依据非空且全部已加载；类别 5 排除。共 **36** 道独立问题。重复运行复用这 36 道题，逐次保存实际完成的回答。subset 保存原始问题下标、标注依据与内容校验值。

MiniLM 使用固定版本、CPU 4 线程；retrieve-k=10、evo-threshold=100、笔记信息 workers=4、prefetch-window=16。基础环境见 [environment.json](environment.json)；最终实验以各自配置记录和代码快照为准。

三组每轮均从空响应缓存运行，每组前重置 vLLM prefix cache，并在三个重复中轮换位置。模型已加载；下载、服务启动、READY/编码器预热不计入 build/QA，初始化单列。**真实请求的推理等待全部计入 build/QA**。baseline 为对照录制而新增的 SQLite flush 单列排除，优化组实际缓存读写计入时间。QA 包含轻量字符串评分。

每轮保存预测、阶段耗时和请求量、逐条记忆状态、源码与配置。[主实验目录](../results/suite128-primary-observed-v1-20261018/)的[方案](../results/suite128-primary-observed-v1-20261018/protocol.json)确定于 UTC 2026-10-17 17:50:57，源码提交为 `2019a73d4ce7135acff4fc2cf51978ccee087124`，冻结前通过 213 项测试。文件与归档校验值见目录内记录。此前 [bounded 方案](../results/suite128-primary-bounded-v1-20261018/protocol.json)使用 `81fcb979867b034111250b19edbd3023fa94476e`，结果单独保存。

## 3. 主实验结果与统计方法

九轮各尝试一次，都在 fused 阶段中止。baseline、exact 已完成的构建与 QA 仍单独保留，状态如下。

| 方案 | 计划 | 实际尝试 | 完成 | 失败 | 因前组失败未运行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 9 | 3 | 3 | 0 | 6 |
| exact | 9 | 6 | 6 | 0 | 3 |
| fused | 9 | 9 | 0 | 9 | 0 |

三个会话各有一对完整 baseline/exact，其他三个完整 exact 单独描述。fused 九次均未完成构建和 QA。全部状态见 [失败记录](primary_failure_audit.json)及[结果汇总](../results/suite128-primary-observed-v1-20261018/analysis_views.json)。

| 数据范围 | 配对数 / 会话数 | exact 构建 / 总流程速度比 | fused 构建 / 总流程速度比 | baseline / exact / fused F1 |
| --- | --- | --- | --- | --- |
| 已完成 baseline/exact 配对 | 3 对 / 3 个会话 | 1.143× / 1.131× | 无完成配对 | 0.3981 / 0.4039 / — |
| 显存分配稳定的完成配对 | 2 对 / 2 个会话 | 1.075× / 1.068× | 无完成配对 | 0.4129 / 0.3930 / — |

三个默认配对都先运行 baseline：sample0/r1、sample1/r3、sample2/r2。因此上表 1.143× 和 stable 1.075× 同时受运行顺序、共享资源影响。

同一完成配对的构建资源如下，先平均会话内重复，再等权汇总会话：

| 配对范围 | baseline → exact HTTP | 输入 tokens | 输出 tokens |
| --- | ---: | ---: | ---: |
| 全部三对 | 390.00 → 392.67 | 313,779 → 317,631 | 23,337.67 → 24,843.00 |
| stable 两对 | 390.50 → 396.00 | 319,124.50 → 326,861.00 | 23,450.50 → 26,381.50 |

速度先计算同轮 baseline/候选组的时间比，再在会话内和会话间等权平均其对数，最后取指数，即配对几何平均。F1 也先平均会话内重复，再等权汇总会话。九轮各执行一次；普通推理失败后保存记录并继续下一轮，源码、配置或回放不一致时停止。表中同时保留计划、尝试、完成、失败和未运行数量。

配对要求两组都完成，例如 baseline 和 exact 完成、fused 失败时，前两组仍可比较。该方法在首个 fused 失败后明确，实时运行方案保持不变。被中止跳过的回放先标为未执行，实验结束后仅使用已有响应补做，并单列记录。

质量采用标准化精确匹配（EM）和按词频重合计算的 token F1，定义见实现分析。上游 set-F1、论文 Score 使用其他评分方式。配套数据还记录演化分支、连接与邻居更新、索引重建次数、各阶段标签超限率和最长列表。请求数、输出量和分支共同影响时间及 QA。

推理时延分位数取 `http_requests > 0` 的实时事件。包含缓存命中的 client 入口时间单独报告；并发请求耗时和嵌套耗时与总墙钟时间分别阅读。

三对结果的 EM 两组均为 0.1111；stable 子集分别为 baseline 0.1667、exact 0.1250。原始预测与计算见 [配对明细](primary_default_pair_audit.json)。样本量、运行顺序和共享 GPU 是这些均值的主要限制。

### 已完成默认路径配对的解释示例

下面展开三个完成配对，说明预取与实际生成工作如何共同影响时间。各轮随后均在 fused 失败。

| 配对 | baseline → exact 构建 | 邻居更新请求 | 构建输出 tokens | 串行更新耗时变化 | 其余构建耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sample0/r1 | 344.09 → 358.46 s | 13 → 23 | 25,787 → 31,422 | +79.65 s | −65.28 s |
| sample1/r3 | 285.57 → 236.99 s | 4 → 5 | 21,114 → 21,341 | +14.81 s | −63.39 s |
| sample2/r2 | 309.71 → 240.01 s | 7 → 4 | 23,112 → 21,766 | −5.63 s | −64.06 s |

sample0 新增 5,635 output tokens，其中 5,331 来自更多或更长的邻居邻居修改，占净增量约 94.6%。额外更新使串行步骤增多，最终构建变慢。sample1 生成量较接近，最终构建缩短。表中残差为 `build wall − decision/strengthen/update 客户端时长之和`，还包含向量编码、索引和调度，用于描述演化之外的耗时。

每个配对都有 128 个相同笔记信息请求，其中分别有 34、18、23 个返回不同正文。即使输出 token 总数接近，内容也可能改变后续图和分支。具体服务端原因尚未定位；固定回答回放用于另行检查程序处理是否一致。来源与计算见 [配对明细](primary_default_pair_audit.json)。

## 4. 回放与 GPU 监控

原计划每轮做 baseline→exact、exact 自身、fused 自身三项回放。fused 中止后，三个 baseline 和六个 exact 于实验结束后独立补做缓存回放；UTC18:43:04 全部完成。3 次 baseline→exact、6 次 exact 自回放的逐条状态、原顺序请求和完整 QA 均一致，新增 HTTP 和 client errors 为 0。共比较 1,152 个状态、执行 9 次索引重建，计数包含重复运行。九轮原失败状态保留。详见 [回放结果](../results/suite128-primary-observed-v1-20261018/replay_recovery.json)和[配对明细](primary_default_pair_audit.json)。

全部预设运行均纳入报告；另按运行前确定的规则筛选显存稳定子集：GPU2 全轮显存最大最小差 ≤1,024 MiB，至少 10 个采样，相邻间隔 ≤10 秒。缺少监控时跳过筛选。该规则只看资源记录，显存稳定时仍可能存在共享计算竞争。

sample0/r1 和 sample1/r3 全轮 GPU2 显存为 44,483 MiB。sample2/r2 在 UTC18:29:10 至18:29:24 的 baseline 构建期间短暂升至 80,216 MiB，随后回到 44,483 MiB。该配对保留在全部结果中，stable 子集仅含另外两个会话。

无 schema 约束的 fused 在第 4–31 条失败：4 次 JSON 末尾多一个 `}`，5 次缺少邻居更新的必需 `context` 字段。九次均正常 `stop`，程序按完整正文和必需字段判定失败。后续 schema 方案另行记录。失败组缓存保留 419 个不同请求响应，usage 至少为 190,802 input tokens、21,183 output tokens，包含已发出的预取；这是已保存响应的成本下限。详见 [失败响应记录](../results/suite128-primary-observed-v1-20261018/failed_response_usage.json)。

## 5. 早期试运行与配置调整

max_tokens=1000 的[早期试运行](../results/suite128-three-arm-20261018/suite.json)共 1 轮完成、1 轮失败、7 轮未运行。第 116 条 `D7:8` 建立关联达到输出上限。利用前序更新决策中的文本、keywords 和邻居快照，复原了此前 114 条建立关联请求键，缺失的键对应这次失败。旧正文未保存，详见 [失败记录](../results/suite128-three-arm-20261018/failure_report.json)。

同一提示词的 [2000-token 诊断](../results/diagnostics/strengthen_cap2000.json)仍截断，生成 15,001 字符，`Transformational` 重复 683 次。随后固定 RP=1.05，检查预选的 10 个请求：9 个正常控制通过；原失败输入在 78 tokens、1.11 秒正常停止，但生成 25 个 tags，超过预设 ≤16 条，故 [RP-only 诊断](../results/diagnostics/rp105-fixed-controls/results.json)仍判失败。

加入第 1 节共同标签指令后，RP=1.05、10 个输入和接受标准保持相同，[bounded-tags-v1 诊断](../results/diagnostics/bounded-tags-v1-fixed-controls/results.json)10/10 通过，实际 10 次 HTTP，诊断输出单独保存。原失败输入输出 58 tokens、tags ≤16；两个五邻居邻居修改分别输出 326、391 tokens，保留全部邻居块。

随后严格 `bounded-tags-v1` 在首轮 exact 第 83 条 `D5:7` 失败：建立关联正常停止，输出 68 tokens、271 字符、18 个 tags，超过 16 项上限。详见 [失败记录](../results/suite128-primary-bounded-v1-20261018/turns128-sample0-upstream-r1/optimized-failure.json)。

`observed-tags-v1` 保持同一请求和惩罚系数，将完整标签超限改为观察项，截断、字段和引用仍严格检查。接受规则通过 213 项离线测试，覆盖请求不变、缓存区分和超限样例；先前 10 个实时控制保留为独立诊断，未再次回放或重发。新方案从空缓存重新执行九轮，参数固定，普通失败按规则继续。

早期试运行唯一完整轮次的 GPU2 显存下降 15,995 MiB，约 2.76× 构建比值受资源切换影响，详见 [资源记录](resource_audit.json)。同轮 fused 上下文较稀疏，标注依据的平均覆盖从 baseline 0.750 降至 0.542，详见 [逐题记录](first_cell_quality_audit.json)。图连接、笔记信息、检索与重建同时变化，此项匹配用于定位答案差异。

## 6. 固定图 QA 上下文去重

按事前修订的来源规则，使用 sample0/r1、sample1/r3、sample2/r2 的三个 baseline 图和 36 道固定问题。图和关键词从原缓存回放，两组只重新生成答案。记忆内容、关键词、上下文校验值和去重行数均核对一致。每组 36 次实时请求，client error 为 0；图与关键词重建为 0 HTTP，每图另有一次不计时 READY 预热。

| sample | 答案请求输入 tokens：upstream → 去重 | 输入减少 | 答案阶段速度比 | F1：upstream → 去重 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 89,040 → 53,383 | 40.05% | 1.534× | 0.4360 → 0.4318 |
| 1 | 78,476 → 49,245 | 37.25% | 1.405× | 0.3771 → 0.3804 |
| 2 | 75,463 → 50,492 | 33.09% | 1.329× | 0.4222 → 0.4476 |
| 三图汇总 | 242,979 → 153,120 | 36.98% | 1.420×（配对几何均值） | 0.4118 → 0.4199 |

上下文 UTF-8 字节从 1,025,519 降至 636,340（−37.95%），记忆行从 1,447 降至 912（−36.97%）。两组 EM 为 0.1111；36 题中有 18 个答案正文相同，F1 提高 7 题、下降 3 题、26 题不变。sample0 均值略降，另外两个会话略升。

答案阶段合计 29.435 → 20.717 秒，包含检索、拼接、评分和 SQLite 记录，构建、图回放及关键词推理另计。因此 1.420× 对应本答案阶段；本阶段缺少独立 GPU 监控，时间受共享服务影响。输入 tokens 和字节数直接由逐题请求统计。

历史 64 条约 40.5% 来自按完整文本去重的离线字符估算。本实验在三个 128 条图中按行号去重，记录真实 tokens、UTF-8 字节和答案。详见 [实验结果](../results/context128-observed-v1-20261018/derived.json)与[逐题比较](context_live_audit.json)。

## 7. 结构化检查与完整会话案例

上下文去重、structured fused/joint 和完整 419 条案例分别保存配置及结果。完整案例按[方案](full_conversation_protocol.md)使用 baseline/exact，检查多次索引重建；可选策略先通过[三项格式检查](structured_policy_gate_protocol.md)，再按 [joint 方案](joint_pilot_protocol.md)做同轮对照。[128 条扩展](joint128_extension_protocol.md)提前固定样本和时间条件。

三项 schema 检查于 UTC18:50 完成，各一次 POST，全部通过完整 JSON、schema、候选引用和首条规则检查，output tokens 为 170/88/100，均正常 stop。已知失败 fused 的原请求仅增加 response_format；joint 有邻居输入使用原 baseline 的固定笔记信息。详见 [检查结果](structured_gate_audit.json)。本检查覆盖三个固定输入，长期完成率和 QA 另测。

完整 419 条案例使用共同 observed 配置，关闭结构约束，选 48 QA。baseline 完成 419/419 条构建、1,286 次 build HTTP，在第 35 题（原索引 69）答案生成时达到 2,000-token 上限，正文重复短答案列表。旧评测脚本在全部 QA 后才保存录制缓存，因而 `baseline.sqlite` 为 0 条，原构建轨迹和时间丢失。随后按既定规则独立运行尚未尝试的 exact。详见 [运行记录](full419_evidence_audit.json)。

exact 完成 419 条构建，耗时 1,067.105 秒；48 QA 为 67.610 秒，F1 0.30086、EM 0.04167。构建有 1,314 次 HTTP、1,117,250 input tokens、93,730 output tokens、4 次索引重建。自身回放的逐条状态、请求和全部 QA 一致，HTTP 与 client errors 为 0。该组完成规模验证；baseline 数据缺失，配对时间与质量无法计算。GPU 监控在构建开始约 3 分钟后启动。

reference 和 Fast 使用 exact 保存的 1,410 个响应重建 419 条记忆，逐条状态、请求、最终图和演化次数一致，并与原实时 exact 轨迹一致。第 101、201、301、401 条发生四次索引重建，HTTP 与 errors 为 0；48 题评分和来源文件也已核对，详见 [检查记录](full419_evidence_audit.json)。这项比较固定相同模型回答，QA 完整自回放属于 exact。全部测量后，最终脚本加入构建结束即保存、异常时保留部分结果的机制，通过 [281 项离线测试](../results/clean-client-validation-final.json)。旧 baseline 丢失的数据仍无法恢复。

主要工作包括有界预取和索引复用、固定图 QA 去重、联合任务策略，以及请求回放和失败保存。默认工程优化与可选任务合并分别测量，质量变化和完成次数随结果一起给出。


### 独立会话中的关键词请求失败

64 条 sample3/r1 的 joint 完成构建后，首题关键词请求截断。问题为 Joanna 的 hobbies，输入只含问题文本，在检索前生成关键词。相同请求在 baseline、exact 均以 87 input / 41 output tokens 正常停止，joint 达到 2,000 tokens。详见 [请求比较](heldout_keyword_failure_audit.json)。

同一会话第二次 joint 重复也在首题失败，请求键、87/2,000 tokens、8,949 字符及正文校验值均相同；该轮 fused 正常返回同一 41-token 正文。共记录三次相同正常正文、两次相同截断正文。失败请求未读取记忆图，具体服务或数值原因尚未定位。


## 8. 独立会话的 64 条实验与取消的 128 条扩展

sample3/conv-42 与 sample4/conv-43 各取前 64 条记录、重复两次，分别有 17 和 14 道标注依据完整的问题，共 31 道独立题。四组按预设顺序轮换位置；fused/joint 的记忆生成使用 JSON Schema 格式约束，其余配置相同。源码提交为 `21b5b23`，完整记录见 [实验记录](../results/joint64-heldout-structured-v1-20261018/suite.json)。

四轮全部启动，两轮完整完成，两轮在 joint 首题关键词阶段失败。实际 13 组构建全部完成 64 条记忆，完整构建+QA 共 11 组：

| 方案 | 计划 | 尝试 | 构建完成 | 构建+QA 完成 | 失败 | 未到达 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4 | 3 | 3 | 3 | 0 | 1 |
| exact | 4 | 3 | 3 | 3 | 0 | 1 |
| structured fused | 4 | 3 | 3 | 3 | 0 | 1 |
| structured joint | 4 | 4 | 4 | 2 | 2 | 0 |

两个失败 joint 缺少保存的构建计时。11 个完整组均通过逐条图、请求和全部 QA 回放，新增 HTTP 与 client errors 为零。详见 [回放结果](../results/joint64-heldout-structured-v1-20261018/replay_recovery.json)。

下表使用同轮完成配对，先平均会话内重复，再等权汇总会话；速度采用配对比值的几何平均。各行的会话和 QA 范围分别列出：

| 对照 | 配对数 / 会话数 / 独立问题 | 构建 / 构建+QA 速度比 | baseline → 对照组 F1 | baseline → 对照组 EM |
| --- | --- | --- | --- | --- |
| exact | 3 / 2 / 31 | 1.383× / 1.323× | 0.4449 → 0.4367 | 0.2143 → 0.1786 |
| structured fused | 2 / 1 / 14 | 1.970× / 1.852× | 0.5475 → 0.5793 | 0.4286 → 0.3929 |
| structured joint | 2 / 1 / 14 | 2.301× / 2.159× | 0.5475 → 0.6095 | 0.4286 → 0.4286 |

| 同范围构建资源 | baseline → 对照组 HTTP | 输入 tokens | 输出 tokens |
| --- | ---: | ---: | ---: |
| exact | 198.50 → 194.25 | 151,700 → 144,832.75 | 13,636 → 11,439.75 |
| structured fused | 206 → 127 | 166,671 → 81,758 | 17,199 → 11,056 |
| structured joint | 206 → 64 | 166,671 → 68,619 | 17,199 → 7,250 |

joint 的完成配对来自 sample4：baseline 构建为 224.221/223.264 秒，joint 为 97.407/97.061 秒，配对速度比 2.301×，请求 206→64。14 题中 2 题 F1 提高、1 题下降、11 题不变，两次重复的变化相同。任务、输出 tokens、分支和记忆图同时变化，质量和耗时需结合阅读。各 64 条组均未触发索引重建，多次重建另由 419 条案例验证。

两轮中止使完成子集失去原计划的完整顺序平衡；sample4 两次顺序互逆，仍受共享 GPU 影响。128 条无结构约束实验与 64 条 structured 实验使用不同样本和长度，分别报告。仅三项固定格式检查在相同输入上增加 response_format。原始计数、评分和资源已核对，详见 [逐项检查](heldout64_audit.json)与[结果汇总](../results/joint64-heldout-structured-v1-20261018/analysis_views.json)。

128 条扩展未执行。预设公式估计 7,266.976 秒（121.12 分钟）；当时剩余 9,280.230 秒，扣除 40 分钟收尾时间后可用 6,880.230 秒，未达到启动条件。四轮均未尝试，详见 [排期记录](../results/joint128-heldout-structured-v1-20261018/scheduling_decision.json)。
