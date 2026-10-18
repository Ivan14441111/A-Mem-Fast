# 使用指南

[项目首页](README.md)介绍主要功能和结果；[性能分析](docs/deep_analysis.md)解释调用链和优化方法；[实验报告](docs/experiments_v2.md)汇总测量。

## 准备环境

在仓库根目录运行以下命令，创建 Python 3.12 客户端环境：

```bash
scripts/setup_environment.sh
uv pip install --python .venv-client/bin/python -e '.[test]'
export NLTK_DATA="$PWD/.cache/nltk_data"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

脚本使用 `requirements-fast.txt` 安装已测试的依赖，并准备 MiniLM 和 NLTK 资源。包支持的 Python 版本范围为 3.10–3.13，现有测试使用 3.12。

若已有客户端环境，也可执行 `python -m pip install -e '.[test]'`，再自行准备模型与 NLTK 资源。

### 本地模型服务

准备 Qwen3-8B 权重，在独立环境启动 vLLM：

```bash
uv venv --python 3.12 .venv-serve
uv pip install --python .venv-serve/bin/python vllm==0.11.2 transformers==4.57.6
INFERENCE_ENV="$PWD/.venv-serve" MODEL_DIR=/path/to/Qwen3-8B GPU_ID=0 \
  scripts/serve_qwen.sh
```

服务默认监听 `127.0.0.1:21014`，显存用量等参数可在[启动脚本](scripts/serve_qwen.sh)中配置。脚本启用前缀缓存重置接口，供实验开始前清理服务端缓存。

## Python 接口

将路径替换为本机模型和缓存目录。`AMEM_MODEL_REVISION` 用于标识实际使用的权重、分词器和聊天模板；更换它们时使用新的版本名。

```bash
export AMEM_EMBEDDING_MODEL=/path/to/all-MiniLM-L6-v2
export AMEM_BASE_URL=http://127.0.0.1:21014/v1
export AMEM_LLM_MODEL=Qwen3-8B
export AMEM_MODEL_REVISION=my-weights-tokenizer-template-v1
export AMEM_RESPONSE_CACHE=/path/to/responses.sqlite
export AMEM_CACHE_MODE=readwrite
```

```python
import os
from sentence_transformers import SentenceTransformer
from fast_amem.client import MeasuredClient, SQLiteResponseCache
from fast_amem.system import FastAgenticMemorySystem

encoder = SentenceTransformer(os.environ["AMEM_EMBEDDING_MODEL"], device="cpu")
def make_client(cache_path, *, structured_policy_outputs=False):
    return MeasuredClient(
        os.environ["AMEM_BASE_URL"], os.environ["AMEM_LLM_MODEL"],
        cache=SQLiteResponseCache(cache_path),
        cache_mode=os.environ.get("AMEM_CACHE_MODE", "readwrite"),
        revision=os.environ["AMEM_MODEL_REVISION"],
        max_tokens=2000, repetition_penalty=1.05, tag_budget=16,
        tag_budget_policy="observe",
        structured_policy_outputs=structured_policy_outputs,
    )

client = make_client(os.environ["AMEM_RESPONSE_CACHE"])
memory = FastAgenticMemorySystem(encoder, client)
turns = [
    {"content": "Maya started a robotics course in October.",
     "timestamp": "2026-10-01T09:00:00+08:00"},
    {"content": "Maya plans to demonstrate the robot in December.",
     "timestamp": "2026-10-02T09:00:00+08:00"},
]
memory.ingest(iter(turns), analysis_workers=4, prefetch_window=16)
context, row_indices = memory.find_related_memories("Maya's robotics plans", k=2)
print(context)
```

`ingest` 提前处理独立的关键词和摘要提取，并按输入顺序完成检索与记忆更新。默认最多使用 4 个分析线程，预取窗口为 16 条记录。

### 选择记忆策略

| 方案 | 行为 | 配置 |
| --- | --- | --- |
| 默认流程 | 保留上游提示词与更新顺序，并行准备关键词和摘要 | `FastAgenticMemorySystem` |
| 合并演化 | 一次请求给出关联、标签和旧笔记更新 | `evolution_policy="fused"` |
| 联合整理 | 每条记录用一次请求完成关键词、摘要和演化 | `JointAgenticMemorySystem` |
| 回答材料去重 | 同一条记忆在回答材料中保留一次 | `context_policy="deduplicated"` |

`fused`、`joint` 和去重会改变模型输入或任务，适合分别比较耗时与回答质量。比较时为每个方案使用独立缓存。

```python
from pathlib import Path
from fast_amem.joint import JointAgenticMemorySystem

cache_dir = Path(os.environ["AMEM_RESPONSE_CACHE"]).parent
fused = FastAgenticMemorySystem(
    encoder, make_client(cache_dir / "fused.sqlite", structured_policy_outputs=True),
    evolution_policy="fused",
)
joint = JointAgenticMemorySystem(
    encoder, make_client(cache_dir / "joint.sqlite", structured_policy_outputs=True),
)
# Choose fused or joint, then use ingest(iter(turns)) and retrieval as above.
```

`structured_policy_outputs=True` 要求服务支持 `response_format: json_schema`。它约束 fused/joint 响应的字段、类型和动作取值，程序随后检查记忆编号、动作组合、关键词与摘要。标签数量由提示词和客户端规则控制。原流程与问答请求沿用各自的格式。

联合整理按顺序执行，每条记录直接生成完整结果。默认流程提前并行提取关键词与摘要，两者使用不同的调度方式。

### 缓存与回放

缓存键包含完整请求、服务地址和模型版本。在 `readwrite` 模式下，相同请求并发到达可共享一次真实调用。支持以下模式：

| 模式 | 用途 |
| --- | --- |
| `off` | 直接调用模型，适合单次请求或顺序处理 |
| `write` | 每次调用模型并记录响应，用于原版对照 |
| `readwrite` | 读取已有响应，缺失时调用模型并保存，用于并行预取 |
| `replay` | 只读取录制响应，缺失时立即报错，用于重复检查 |

并行预取通过缓存交接分析结果，使用 `readwrite` 或 `replay`。

回放时设置 `AMEM_CACHE_MODE=replay`，使用相同的记录顺序、编码器、服务地址、模型版本、生成参数和策略。回放仍需本地文本编码器。

笔记和向量索引保存在内存中，SQLite 保存模型响应。重新处理相同输入可从响应记录重建记忆。客户端会记录格式完整的模型响应；如果后续记忆字段校验失败，构建停止，录制内容可用于排查。

## 运行实验

模型服务就绪后，设置 MiniLM 路径和模型版本：

```bash
export EMBEDDING_MODEL=/path/to/all-MiniLM-L6-v2
export MODEL_REVISION=your-model-and-template-revision
```

先运行一个 64 条记录的对照：

```bash
.venv-client/bin/python scripts/benchmark.py \
  --embedding-model "$EMBEDDING_MODEL" --model-revision "$MODEL_REVISION" \
  --max-turns 64 --max-questions 12 \
  --max-tokens 2000 --repetition-penalty 1.05 --tag-budget 16 --tag-budget-policy observe \
  --modes baseline,optimized,replay,warm \
  --output results/my-smoke
```

多个会话的重复对照：

```bash
.venv-client/bin/python scripts/benchmark_suite.py \
  --embedding-model "$EMBEDDING_MODEL" --model-revision "$MODEL_REVISION" \
  --samples 0 1 2 --turns 128 --repeats 2 --max-questions 12 \
  --max-tokens 2000 --repetition-penalty 1.05 --tag-budget 16 --tag-budget-policy observe \
  --continue-on-error --output results/my-suite
```

加入 fused 与 joint 比较：

```bash
.venv-client/bin/python scripts/benchmark_suite.py \
  --embedding-model "$EMBEDDING_MODEL" --model-revision "$MODEL_REVISION" \
  --samples 3 4 --turns 64 --repeats 2 --max-questions 24 \
  --max-tokens 2000 --repetition-penalty 1.05 --tag-budget 16 --tag-budget-policy observe \
  --include-fused --include-joint --structured-policy-outputs \
  --continue-on-error --output results/my-policy-pilot
```

每次真实推理实验使用新的输出目录。`--continue-on-error` 保存当前组的错误后继续下一组；源码、配置、缓存或回放检查出错时停止。`--resume` 可跳过源码和配置相符的完整实验组，失败组保留原记录。

以上命令使用统一的生成配置：最多输出 2,000 tokens，重复惩罚 1.05，提示词要求最多 16 个标签。`observe` 记录超出标签数量的完整响应；截断和解析错误会中止当前运行。支持前缀缓存重置的服务会在各次真实运行前清理缓存，其他服务可用 `--prefix-cache-reset off` 并记录这一设置。

问题从类别 1–4 中轮流选取，所需事实均位于选定对话范围。报告包含词项 F1、标准化完全匹配率及逐题答案。

### 查看和绘制结果

```bash
.venv-client/bin/python scripts/summarize_suite.py results/my-suite/suite.json \
  --output results/my-suite/derived.json
uv pip install --python .venv-client/bin/python -e '.[analysis]'
.venv-client/bin/python scripts/plot_suite.py results/my-suite/suite.json \
  --output results/my-suite/overview.svg --png
```

`summary.json` 分别记录构建和问答耗时。模型加载、预热、原版响应批量落盘和报告写入单独处理；`total_seconds` 为构建与问答耗时之和。并行请求的计时可能重叠，阶段累计值适合定位工作量，整次运行耗时用于比较速度。

程序在构建结束时保存录制响应和构建报告，再进入问答。失败时保留已完成部分与错误信息。现有测量对应的源码版本、完成次数和失败情况见[结果目录](results/README.md)。

### 保存响应记录

实验进程退出后，可把 SQLite 缓存归档到私有目录：

```bash
.venv-client/bin/python scripts/archive_replay_cache.py --run-dir results/my-suite
```

归档位于 `.private/artifacts/`，保存数据库备份及其校验值。对已有默认流程结果，可使用原配置及 `--modes replay,warm` 重新检查。Git 默认忽略 SQLite 文件和 `.private/`。

## 检索实现

`FastEmbeddingRetriever` 共享编码器，缓存文本向量，按容量扩展连续存储，并使用归一化向量做精确余弦搜索。定期重建时，变化文本批量编码。

为了沿用原流程的检索结果，实现保留上游的刷新时点、插入和重建时的文本格式，以及按行号记录的关联。编码器输出类型固定，发生变化时会报错，便于发现可能影响排序的数值差异。

## 测试

```bash
.venv-client/bin/python -m pytest -q
.venv-client/bin/python scripts/benchmark_index.py \
  --embedding-model "$EMBEDDING_MODEL" --output results/my-index.json
```

自动测试覆盖请求缓存、并发预取、检索排序、记忆更新、结构校验、回放和失败保存。索引实验使用真实 MiniLM 与 LoCoMo 文本，单独测量索引工作。
