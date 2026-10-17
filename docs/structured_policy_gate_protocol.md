# Structured fused/joint：固定三项格式门禁

预设时间：2026-10-17T18:07:50Z。任何本协议实时输出之前固定。当前九个 observed 主实验保持原源码与无结构约束请求，继续按原协议执行；本门禁只能在它结束并完成必要的零 HTTP 验证后执行，不替代或修复主实验结果。

## 固定输入，恰好三次请求

1. **已知 fused 格式失败。** 使用 observed 主实验 sample 0 / repeat 1 的原始请求键 `a216cfc4f1cf3f395e7692acbafdbe137a00d170797567ef2f1452721307801b`，从本地 `fused.sqlite` 读取完整原请求。其失败响应是 84-token、正常停止但多一个末尾右括号的 JSON。保留原 system/user 文本，只加入本次结构约束字段；不能把修剪后的旧响应当作新模型结果。候选 ID 从该固定请求读取并保存在控制 manifest。
2. **joint 首条。** sample 0 的第一个 chronological turn，邻居为空，allowed indices 为空。用冻结的 joint 模板构造一次请求，不先运行独立 metadata 分析。
3. **joint 有邻居。** sample 0 的第二个 chronological turn，唯一候选为全局行 0。行 0 的 content/timestamp 来自第一个 turn，metadata 来自 observed 主实验 sample 0 / repeat 1 baseline 缓存的第一个 analysis 响应，使用同一 robust parser。固定这一邻居快照，不能用第 2 项刚生成的 metadata 改变第 3 项输入。

三个控制都来自主实验 sample 0，不使用 held-out sample 3/4。第 2、3 项在调用前保存内容、模板、来源请求键、解析后邻居快照和哈希；若无法精确构造，先记录门禁未执行，不能临时更换更容易通过的样本。

## 共同请求与结构

沿用 `observed-tags-v1` 的 Qwen3-8B、temperature=0、seed=42、max_tokens=2000、RP=1.05、16-tag system 指令和 observe 接受策略。使用 vLLM 的 JSON schema 输出约束，schema、response_format、后端配置、模型/源码身份全部存档并进入请求身份。fused/joint 各自固定通用 schema，至少限定顶层对象、必需字段、字段类型、decision 枚举和邻居更新对象；实际候选 ID、唯一引用、分支组合和首条语义继续由现有 parser 验证。是否支持每项 schema 关键字由固定实现和服务验证，不假定接口存在就代表完全支持。

不把 tags≤16 改成新的硬 schema 上限；该提示仍是观察项。结构化请求只用于 fused/joint 的 JSON 阶段，baseline/exact 保持原文本协议，因此这是新策略配置，不是所有组语义中性的服务修复。

## 一次性接受标准

每项仅发起一次 HTTP，无自动重试、修剪、修复或参数搜索。三项必须全部满足：

- HTTP 成功，正文非空，finish_reason 正常停止且未触及截断；输出不超过共同 2000-token 预算。
- 对完整正文直接 `json.loads` 成功，无 markdown 包裹、额外解释或 trailing data。
- 对应现有 fused/joint parser 校验通过，包括候选全局行号、字段组合与引用唯一性。
- joint 空邻居控制为 `NO_EVOLUTION`，connections/tags/neighbor_updates 均为空；metadata 满足现有 parser 的非空条件。
- tags 超过 16 时记录 actual lengths 和观察计数，不改变本门禁的结构通过判定。

不要求生成特定非空连接或触发特定演化分支；不检查末端 QA 分数，也不据此搜索提示词。记录每项请求/响应、usage、耗时、schema 身份、失败理由；不写入主实验缓存。三项通过只允许继续独立的 held-out structured pilot，不证明任意长链路格式可靠或质量保持。

若任一项失败，保持失败并取消本轮 structured held-out pilot；继续可完成的默认 baseline/exact 长会话验证与报告，不另开一轮 schema/惩罚系数搜索。实现错误若在任何模型调用前被离线测试发现可修正，最终调用前冻结源码与协议。

## 实际门禁结果（不修改上述预设标准）

UTC2026-10-17 18:50:01–18:50:06 执行，源码 checkpoint `21b5b23de2089e6b096a86242a7d51a67cc06d29`。三个固定输入均通过，恰好三次直接 POST，无重试、缓存写入、修剪或 QA 评分。已知 fused 输入、joint 空邻居、joint 固定邻居分别使用170、88、100 output tokens，均为HTTP200/finish_reason=stop；记录wall为2.156、1.133、1.229秒。空邻居严格为NO_EVOLUTION，有邻居引用仅使用allowed row0。

独立复核重新计算全部request/schema/source哈希，确认已知失败请求只增加response_format，joint固定邻居来自原baseline分析而非前一个控制输出；对完整正文执行JSON解析、JSON Schema校验和现有领域parser，全部通过。记录见 [原始结果](../results/diagnostics/structured-policy-v1-fixed-controls/results.json) 与 [独立审计](structured_gate_audit.json)。

该结果允许继续预设pilot，不证明完整长链无失败或QA质量保持。已知fused输入的新输出为170tokens，原非法响应为84tokens，邻居更新也发生变化；schema会改变生成分布，不只是把旧文本修成合法格式。wall可能包含schema编译，但没有强制清空schema缓存，不能宣称首次编译成本已独立测量。
