# fused/joint 输出格式：三个固定输入的检查

方案确定于 2026-10-17T18:07:50Z，早于本检查的实时输出。九次 observed 主实验继续使用原源码和无结构约束请求；本检查在主实验及其零 HTTP 回放结束后执行。

## 三个固定输入

1. **fused 的已知格式失败请求。** 使用主实验 sample 0 / repeat 1 的请求键 `a216cfc4f1cf3f395e7692acbafdbe137a00d170797567ef2f1452721307801b`，从本地 `fused.sqlite` 读取原请求。旧响应为 84 tokens，正常停止，JSON 末尾多一个右括号。保留 system/user 文本，加入结构约束字段；候选 ID 从原请求读取并保存。
2. **joint 首条记忆。** sample 0 的第一条记录，邻居和 allowed indices 均为空。使用固定 joint 模板直接构造请求。
3. **joint 有邻居。** sample 0 的第二条记录，唯一候选为全局行 0。行 0 的 content/timestamp 来自第一条记录，笔记信息来自主实验 sample 0 / repeat 1 baseline 的第一个 analysis 缓存响应，经原 robust 解析器解析。该邻居快照固定，与第 2 项的新输出无关。

三个输入均来自 sample 0，后续独立实验使用 sample 3/4。调用前保存内容、模板、来源请求键、邻居快照和校验值。固定输入无法构造时记录未执行。

## 请求配置与校验

沿用 `observed-tags-v1`：Qwen3-8B、temperature=0、seed=42、max_tokens=2000、RP=1.05、16-tag system 指令和 observe 接受规则。fused/joint 各自使用固定通用 JSON schema，限定顶层对象、必需字段、类型、更新决策枚举和邻居更新对象。程序继续检查候选全局行号、重复引用、动作与字段的关系，以及首条记忆的处理。

schema、response_format、服务配置及模型版本随请求保存，参与缓存区分。结构约束只用于 fused/joint 的 JSON 阶段，baseline/exact 沿用原文本格式。tags≤16 仍作为生成指令，超限时记录实际长度。

每项发起一次 HTTP，直接检查完整返回正文。接受条件如下：

- HTTP 成功、正文非空、正常停止，输出在共同 2000-token 预算内。
- 整段正文可被 `json.loads` 解析，内容为 JSON 本身。
- 现有 fused/joint 解析器校验通过，包括候选行号、字段组合和引用唯一性。
- joint 空邻居结果为 `NO_EVOLUTION`，connections/tags/neighbor_updates 为空，笔记信息满足解析器的非空要求。
- tags 超过 16 时记录实际长度和发生次数，结构判定保持上述规则。

记录请求、响应、usage、耗时、schema 及失败原因，使用独立于主实验的输出。三项全部通过后继续 structured held-out 实验；任一项失败则取消该实验，继续默认长会话验证。模型调用前发现的实现错误可由离线测试修正，调用后按固定配置报告结果。

## 执行结果

UTC2026-10-17 18:50:01–18:50:06 执行，源码提交为 `21b5b23de2089e6b096a86242a7d51a67cc06d29`。三个输入全部通过，各发起一次 POST。fused、joint 空邻居和 joint 固定邻居的 output tokens 分别为 170、88、100，均为 HTTP200 / finish_reason=stop；耗时为 2.156、1.133、1.229 秒。空邻居返回 NO_EVOLUTION，有邻居结果仅引用允许的 row0。

请求、schema 和源码校验值已核对。fused 原请求仅增加 response_format；joint 邻居来自固定的 baseline 分析。完整正文的 JSON、schema 和程序校验均通过。详见 [原始结果](../results/diagnostics/structured-policy-v1-fixed-controls/results.json) 和 [检查记录](structured_gate_audit.json)。

本检查覆盖上述三个输入。fused 的新输出为 170 tokens，旧输出为 84 tokens，邻居更新内容也发生了变化。结构约束会影响生成结果，长期完成率和 QA 质量另行测量。计时可能包含 schema 编译，但首次编译成本未单独测量。
