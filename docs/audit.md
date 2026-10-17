# A-Mem 上游性能与实验口径审计

审计对象为 `WujiangXu/A-mem` 的提交 `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`（2026-03-05）。以下行号均指该提交的源码。源码检查与小批量实测应分开理解：本文的复杂度、调用次数公式是从代码推导的；具体耗时与加速比以实验生成的 JSON 和报告为准。

本次优化以 `memory_layer_robust.py` 为基线。它是仓库现有的兼容分支，保留 note、关键词/上下文/标签、邻居检索、连接和邻居演化这些记忆步骤，但它的提示词与调用流程不同于最初 JSON-schema 分支。因此结果应表述为“相对当前上游 robust 实现”，不能直接当作论文 A-Mem 的复现结果，也不能直接与用户截图中的 RealMem 表格比较。当前实验使用 LoCoMo 子集，数据集、模型、硬件及评测范围均与该表不同。

## 1. LLM 调用为何多

| 路径 | 源码位置 | 每条记忆的逻辑调用 |
| --- | --- | --- |
| 原始 metadata 分析 | `memory_layer.py:279–286, 308–365` | 通常 1 次 |
| 原始 evolution | `memory_layer.py:753–819` | 1 次，即使没有邻居也调用 |
| robust metadata 分析 | `memory_layer_robust.py:293–298, 318–345` | 通常 1 次，空关键词分支可能再调用 1 次 |
| robust evolution | `memory_layer_robust.py:461–519` | 有邻居时 decision 1 次，strengthen 和 update 各条件调用 1 次 |
| QA | `test_advanced_robust.py:96–112, 146–153` | query keywords 1 次，answer 1 次 |

robust 在第一条记忆上跳过 evolution；其后若每条都 strengthen + update，N 条记忆的构建调用数为 `N + 3(N-1) = 4N-3`，另加关键词补救和网络重试。原始分支通常为 `2N`。这不是同一请求被网络意外重复发送，而是 robust 为兼容性主动拆分了逻辑步骤。

这些调用存在不同的数据依赖：metadata 只读取当前 turn，可以提前并发；decision 必须读取此前已按顺序演化的记忆；strengthen/update 依赖 decision。因此可以并发预取 metadata，但仍需按原有 turn 顺序执行 evolution 和写入。并发执行整个 `add_note()` 会改变邻居、更新顺序及后续提示词，不能归入严格保持语义的优化。

`retry_llm_call` 为每次逻辑调用最多尝试 3 次，并等待 1 秒、2 秒（`memory_layer_robust.py:44–65`）。vLLM 请求单次超时为 120 秒（`:187–206`）；一次失败可造成超过 6 分钟的停顿。它捕获所有异常，连不可恢复错误也会重试。统计时必须同时记录逻辑调用、实际 HTTP 请求、重试、错误和有效解析，不能仅看最终成功数。`check_connection` 默认关闭（`:253, :364`），并非每次初始化必然多打一条探活请求。

`retrieve_memory_llm()` 虽定义了额外精筛调用，但当前 `answer_question()` 没有调用它（`test_advanced_robust.py:81–94, 109–114`）；不能将这个未执行的方法计入 QA 调用数。

## 2. embedding 与索引开销

实际使用的是 `SimpleEmbeddingRetriever`，不是 BM25/向量混合检索（`memory_layer.py:666–679`、`memory_layer_robust.py:366–367`）。给它接入数据库或 ANN 前，首先应优化当前明确执行的路径。

1. 每条记忆单独编码一次文档，非首条还单独编码一次检索 query；缺少批处理（`memory_layer.py:568–605`）。metadata 未依赖此前状态，原始 turn 文本的 query embedding 也可以预先编码和缓存；最终文档 embedding 则必须等新 note 的演化确定后再生成。
2. 每次插入用 `np.vstack` 复制全部既有 embedding（`:578–584`）。N 次逐条插入累计为 `O(N²d)` 的内存搬运，d 为向量维数。容量按倍数增长的连续缓冲区可以把累计搬运降到 `O(Nd)`，同时保持行号与记忆插入顺序不变。
3. 每次检索对完整矩阵调用 `cosine_similarity`，重复计算既有文档的范数（`:602–607`）。保存归一化向量、以矩阵乘积计算 cosine 可以省下重复归一化。浮点舍入可能影响极近分数；应比对每步 top-k，保留原有排序与并列分数处理。
4. 每次检索重建 `list(self.memories.values())`，邻居更新还重建 key/value 两份列表（`memory_layer_robust.py:416–417, 435–436, 522–523`）。稳定的行号到 note 的列表可消除这些 `O(N)` 分配，但必须与向量行号严格同步。
5. `consolidate_memories()` 重新实例化 SentenceTransformer，然后逐条重新 encode 并反复 vstack（`memory_layer_robust.py:399–409`；原始版 `memory_layer.py:729–751`）。复用已有 encoder，并在相同触发时刻批量重建，是最直接的可比优化。批内浮点差异仍需校验检索结果。

默认 `evo_threshold=100` 统计的是“发生演化的 note 数”，不是所有插入数（`memory_layer_robust.py:393–396`）。64 turn 主实验不会触发默认全量重建。因此不能用这个主实验声称“消除了重建造成的多少实际耗时”；需要单独设置低阈值的压力实验，并同时修改两组阈值。`threshold=8` 的离线重放可以验证重建路径及结构开销，但它不等于真实 LLM 端到端加速比。

## 3. 需要保留或单独报告的语义细节

- 插入的索引文本使用 `content:… context:… keywords:… tags:…`，而 consolidation 和 `load_from_local_memory()` 使用 `content + ' , ' + metadata`（`memory_layer_robust.py:387–392, 407–409`；`memory_layer.py:655–659`）。统一这些格式会改变向量；严格对照应在对应阶段保持上游原格式。
- 邻居的 tags/context 更新后，向量不会立即更新，直到 consolidation（`memory_layer_robust.py:524–534, 393–409`）。立即增量更新虽然是合理改进，却会改变下一条记忆检索的候选集合，必须作为单独消融项，不能暗中并入“等价”版本。
- `RobustMemoryNote.__init__()` 的分析条件包含 `category is None`，但分析不生成 category（`:293–298, 314`）。传入预取的 keywords/context/tags 时，若漏传 category，仍会再次调用 LLM。应明确传 `category='Uncategorized'`，或在已获得完整 metadata 时关闭该次构造的 controller。
- 邻居文本展示全局 memory index（`:421`），连接提示词示例却写 `0, 2, 3`（`llm_text_parsers.py:164–177`）；更新提示词另外要求局部 `0..k-1`（`:189`）。连接直接存储并作为全局列表下标使用，未做范围校验（`memory_layer_robust.py:505, 448–454`）。这是上游歧义及潜在崩溃点。修复时应建立明确 ID 约定并单独评估检索变化。
- 邻居扩展没有去重，而且先追加、再检查 `j >= k`，实际可能每个主记忆追加 `k+1` 个连接（`memory_layer_robust.py:438–458`）。这会膨胀 QA 上下文。同样，去重或修正上限会改变提示词，应独立消融。
- 原始 `MemoryNote.analyze_content()` 调用 `re.sub` 却没有导入 `re`，内部 bare except 又引用未绑定的 `e`（`memory_layer.py:379–395`）。用该函数 AST 和有效固定 JSON 做独立复现，结果为 `keywords=[]、context='General'、tags=[]`。即一次有效 metadata LLM 的结果被错误处理丢弃。必须修复后才能合理评估原始分支，不能把这个缺陷当作算法本身的行为。
- robust 的解析会以启发式补齐空关键词/上下文（`llm_text_parsers.py:230–234, 430–467`）；因此 metadata 的 focused retry 分支通常已被启发式修复绕过。仅统计 API 成功无法说明 metadata 有效，应另记解析回退次数。

## 4. 评测脚本本身的耗时与缓存风险

`--ratio=0.1` 按完整 conversation 数切片；10 个 sample 会留下整个第一个 conversation，共 419 turns，而非 10% turns（`test_advanced_robust.py:191–194`）。这解释了所谓“小样本”仍然产生大量请求。独立 harness 应显式提供 sample、turn、QA 限额。

原评测逐条 QA 调用 `bert_score()`（`utils.py:68–71`），该函数入口没有缓存 `BERTScorer` 对象；还逐题两次单独 encode 预测和参考文本（`:93–104`）。`utils.py` 导入时及 test 文件导入时各加载一次 SentenceTransformer（`utils.py:27–32`、`test_advanced_robust.py:43–48`），每个 memory system 也加载一次。上述下载、模型初始化和指标耗时应与记忆构建、QA 推理时间分开。小批量首先报告透明的 exact match/token F1，重指标批量离线计算；明确说明该 F1 定义与上游或官方评估是否相同。

原缓存仅按 backend/model/sample index 命名（`test_advanced_robust.py:204–219`），不包含数据内容、代码、提示词、阈值、tokenizer 或模型 revision。缓存命中会完整跳过构建（`:221–249`）。因此，上游已经具备“已构建记忆复用后零构建调用”的能力；新响应缓存的热重跑不能当作此前不存在的算法优势。新缓存的额外价值是请求级复用、部分失败后的恢复以及固定响应的语义重放。新的缓存至少应包含完整 prompt、system message、模型身份、生成参数及版本/命名空间；数据选取与源码版本也应写入实验 manifest。

SQLite WAL 可提供可恢复、可查询的响应持久化与跨进程读写，但在几十条记忆规模下，磁盘本身并非已证实的主要瓶颈。不要凭采用了数据库就声称端到端提速。原版只在一个 conversation 构建结束时写 pickle/npy，并非每次 `add_note` 都全量持久化（`:246–249`）。

## 5. 公平的小批量协议

1. 固定同一 upstream commit、Qwen 模型和 revision、embedding 模型、硬件、服务参数、temperature、seed、输出上限、检索 k、阈值及提示词。并发推理即使 seed 相同也不保证逐字一致，应保存输出并做重放校验。
2. 取同一 conversation 的时间顺序前 64 turns；加载图片时保持原 parser 的 caption 拼接规则（`load_dataset.py:59–77`）。只选择非空 evidence 且所有 evidence turn 都出现在截取范围内的 QA。未满足证据覆盖的 QA 不能算作普通错题。sample 0 前 64 turns 有 36 个符合条件的 QA，类别分布为 `{1:4, 2:8, 3:1, 4:13, 5:10}`。当前 harness 明确排除类别 5，剩余 26 道，从类别 1–4 按轮转顺序抽取最多 12 道；这不能外推到对抗题。应保存问题和 evidence，后续建议也保存原始 QA 下标。只取前几题会偏向数据原顺序。
3. 两组均从空响应缓存、空记忆开始，分别运行真实 LLM，报告初始化时间、构建 wall time、metadata/evolution/embedding/rebuild/持久化时间、实际 HTTP 调用与 tokens、重试及失败。初始化已预热时应明确称“模型已加载的空缓存构建”，不要称完整冷启动。
4. 缓存重复运行单独报告为 warm-cache/repeat-run。响应缓存命中可以减少实际 API 请求，但没有减少新输入所需的逻辑 LLM 步骤。不能把 warm-cache 命中相对首次运行的倍数当成冷缓存算法加速。
5. 保存 baseline 的完整响应，以严格禁止 cache miss 的 replay 模式分别驱动两种实现，比较每步 evolution 提示词、检索行号、逻辑记忆字段及 QA 上下文。UUID 与运行时间字段需规范化；不能为了通过比较而忽略 links/context/tags。重放用于确认实现等价和测量 Python/embedding 路径，不用于声称实时推理吞吐。
6. 除 64-turn、threshold=100 主实验，再以同一批数据和 threshold=8 对照重建路径。分开标注生产默认路径、压力测试和离线重放，报告实际触发的 rebuild 次数。
7. QA 阶段固定类别 5 的选项顺序与 RNG；原脚本使用未设 seed 的 `random.random()`（`test_advanced_robust.py:117–127`）。类别 5 使用参考 adversarial answer 组成候选项是上游评测方式，结果应明确是该方式的子集测试；不可宣称为开放式回答的通用能力。
8. 同时报告速度、质量与有效调用率。小样本只能支持工程可行性和特定输入的等价检查，不足以证明 LoCoMo 整体质量不下降。后续可扩大到多个 conversation、至少 3 次重复，并给出方差或置信区间。

## 6. 优化优先级

先做独立 metadata 并发、encoder 复用、embedding 缓存、几何增长的向量缓冲区，以及在原触发时机批量重建。持久化响应可服务反复调 k、故障恢复和实验复现，价值应按这些场景衡量。

第二阶段再评估增量更新邻居向量、稳定 ID 连接、上下文去重、合并 evolution 请求、规则跳过 evolution 及 ANN。它们可能进一步减少调用或改善长期扩展能力，但都会改变记忆演化、提示词或候选结果，应逐项消融，并检查 QA 质量。

## 7. 本次实现与 smoke 验证的独立复核

本次实现把两组共享的测量 client 接到未修改的上游 robust 构造、解析和演化方法。基线依旧逐条调用 `add_note`；优化版只把 metadata 请求提前放入线程池，随后原序调用相同的 `add_note`。完整 system message、user prompt、temperature、seed、token 上限、thinking 配置、模型名、revision 和服务地址共同构成 SHA-256 请求键。`temperature=0` 和 seed 一致降低随机性，但不能保证推理服务在不同并发调度下逐字输出相同。

`scripts/benchmark.py` 在两个 live 空缓存实验开始前调用本地 vLLM 的 `/reset_prefix_cache`，避免第二组无意继承第一组的 KV 前缀缓存。模型权重已加载，模型下载与服务启动不计入 build/QA，初始化单独报告。基线和优化两组均使用 temperature=0，这与上游默认 0.7 不同，属于两组共同的实验配置变化。

当前 `logical_calls` 包含 metadata 预取及上游随后读取缓存的两次 client 入口，不等于两次模型推理。比较算法步骤应使用 `upstream_calls`，比较实际推理负担应使用 `http_requests` 和 live tokens。并发阶段 `seconds_sum` 是各请求耗时之和，可以大于阶段 wall time，不可相加得端到端耗时。

质量指标是 harness 明确定义的 normalized exact match 和 multiset token F1：先转小写、去标点和冠词，再按重复 token 次数求交集。它与上游 `utils.py` 的 set token F1 不同，也没有执行上游 BERTScore、ROUGE 或 METEOR。因此指标只用于同一 harness 内的对照，应保留该定义与原始预测。

检查 `results/smoke8/baseline.sqlite` 与 `optimized.sqlite`：两组各有 26 个不同请求键，交集为 15；8 个 metadata 请求键全部相同。在共享请求中恰有 1 个响应正文不同，键为 `901bafda49aa0f9fe8b6839e4a36697796cc20076e5eb1d9a64f2e9b78de8d01`。完整 identity 与 HTTP payload 相同，内容为第 3 条 turn 的 LGBTQ support group 分析。两组 keywords/tags 相同，但 context 分别是“shares a powerful experience from attending…”和“shares a positive experience at…, highlighting its impactful nature.”。memory trace 从第 3 条起分叉，与这个差异的时间一致。

这直接证明同一推理请求得到不同的服务端响应，并非该条 metadata 提示词或缓存键不同。vLLM 并发批次及浮点计算差异是合理解释，但当前证据没有单独识别具体底层机制，不能写成已经证明的原因。两组 live 输出分叉后，后续提示词和推理负载也可能不同，单次 live 耗时差异因此不能全归于 Python/数据库优化。固定响应的 baseline→optimized replay 中，逐条 memory trace、上游请求键和 QA 结果一致，提供了隔离推理服务差异后的实现等价证据。

`tests/test_system.py` 使用确定性 HTTP 与 embedding doubles，在 threshold=100 和 threshold=2 下分别验证：每一步记忆相同、连接和邻居更新一致、重建时点与索引文本相同、模型复用、非 metadata 请求仍在主线程原序执行、完整 QA 上下文相同，以及 strict replay 零 HTTP。另验证 metadata 预取缺失响应会抛错且不添加不完整 note。这些测试覆盖控制流，不证明真实模型在所有输入上数值完全一致。

最后，小规模主实验默认 threshold=100，不会触发 consolidation；重建收益必须以低阈值压力实验或索引 microbenchmark 的独立证据支持。空缓存构建、相同响应重放、热响应缓存重跑和索引 microbenchmark 应分别报告，不能混用加速比。
