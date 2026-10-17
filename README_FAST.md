# Engineering notes

The [project README](README.md) contains portable setup and benchmark commands. Current measurements and limits live in [docs/experiments_v2.md](docs/experiments_v2.md), with [technical analysis](docs/deep_analysis.md) and [historical measurements](docs/experiments.md); the [source audit](docs/audit.md) explains the original bottlenecks.

## Library example

After setup, point the encoder at downloaded local weights and configure the running vLLM OpenAI-compatible endpoint. Set the revision to identify the actual weights, tokenizer and chat template.

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
client = MeasuredClient(
    os.environ["AMEM_BASE_URL"], os.environ["AMEM_LLM_MODEL"],
    cache=SQLiteResponseCache(os.environ["AMEM_RESPONSE_CACHE"]),
    cache_mode=os.environ.get("AMEM_CACHE_MODE", "readwrite"),
    revision=os.environ["AMEM_MODEL_REVISION"],
    max_tokens=2000, repetition_penalty=1.05, tag_budget=16,
    tag_budget_policy="observe",
)
memory = FastAgenticMemorySystem(encoder, client)
turns = iter([
    {"content": "Maya started a robotics course in October.",
     "timestamp": "2026-10-01T09:00:00+08:00"},
    {"content": "Maya plans to demonstrate the robot in December.",
     "timestamp": "2026-10-02T09:00:00+08:00"},
])
memory.ingest(turns, analysis_workers=4, prefetch_window=16)
context, row_indices = memory.find_related_memories("Maya's robotics plans", k=2)
print(context)
```

For fused evolution, add `evolution_policy="fused"` to the constructor. For joint metadata/evolution, import `JointAgenticMemorySystem` from `fast_amem.joint` and use `JointAgenticMemorySystem(encoder, client)`; its writes are sequential. These optional policies can change quality; use separate response caches when comparing them.

The graph and vector index live in memory; SQLite persists LLM responses. To reconstruct a graph, rerun the same ordered turns with `AMEM_CACHE_MODE=replay` and unchanged encoder, endpoint, revision, decoding and policy. Strict replay makes no live LLM calls and raises on a missing response; the local encoder is still required.

## Default execution path

`FastAgenticMemorySystem` reuses the shared CPU encoder and `FastEmbeddingRetriever`. Independent metadata requests run in a rolling, bounded thread pool. Graph mutation, neighbor retrieval, conditional upstream evolution, and note insertion run chronologically. Prefetch responses enter an exact-request cache and are consumed through the same parser. The successful path follows the robust reference; inference/cache errors propagate in the default strict mode instead of producing a silent benchmark fallback.

The exact retriever uses normalized vectors, growing contiguous storage, an embedding cache and a bounded query cache. Consolidation preserves the upstream threshold and its distinct document format; only changed text is encoded, in batches. It intentionally preserves the upstream refresh schedule after neighbor mutations and positional graph links. Immediately refreshing every changed neighbor, remapping links to stable IDs, approximate nearest-neighbor search and context removal would change behavior. The fast retriever is in memory and does not implement the upstream retriever save/load API. SQLite persists responses; it does not provide transactional graph persistence. Benchmark graph reconstruction uses strict response replay.

`evolution_policy="fused"` uses one structured call for evolution, validates its decision and references, and records mutation/parser counters. It changes the prompting contract and can change memory and QA quality. `context_policy="deduplicated"` removes repeated memory rows before answering; it is another independent opt-in policy. Default values for both policies are `"upstream"`.

## Cache and experiment identities

`MeasuredClient` hashes the complete request, endpoint and explicit model revision. Only complete successful HTTP text responses are persisted in SQLite WAL mode. Semantic/schema validation occurs afterward; a schema-invalid response can be recorded, and policy failures remain visible in counters or raised exceptions. Concurrent identical requests share one live request. `write` records the reference while bypassing reads; `readwrite` enables the optimized pipeline; `replay` raises on every miss and never contacts the server.

The benchmark uses separate configured-reference, optimized, fused and optional joint databases. The measured `observed-tags-v1` profile passes max_tokens=2000, repetition_penalty=1.05, tag_budget=16 and tag_budget_policy="observe" uniformly. A common system instruction requests tag lists of at most 16 items. Otherwise valid complete overshoots are recorded without clipping; live-generation counts are separated from cached response inspections. Truncated or malformed HTTP responses remain errors. The optional default `"error"` policy rejects oversized lists. User prompt templates remain unchanged, and observe uses a separate cache identity. Historical omitted parameters retain their original identities. The unbounded 1000-token pilot, failed RP-only diagnostic and strict tag-budget abort remain separate from the observed-profile primary. Cold-arm prefix caches are reset, and initialization/warmup is reported separately. The reference's write-only recording is buffered in memory and flushed outside timing; optimized cache overhead remains timed. Reported nested stage timings are inclusive and must not be summed into wall time. Per-call latency percentiles also include overlapping requests and are not end-to-end latency.

Three different comparisons serve different purposes:

- Live reference versus fast pipeline measures elapsed time and output/QA differences under separate inference. Greedy decoding and a fixed seed do not guarantee identical responses under different batching.
- Reference responses replayed through the default fast pipeline check every memory snapshot, chronological request key, and complete QA output for equivalence.
- Each optimized policy replayed from its own cold responses checks reproducibility and cache integrity. Fused self replay cannot establish equivalence with upstream prompts.

The suite records source hashes, subset fingerprints, source QA indices, complete per-question predictions, stage token use, errors, parser counters and GPU load. It refuses source drift between cells, and preserves failed cells rather than silently averaging them away. A three-arm suite rotates cold-arm order over repeated pairs. Small samples and shared GPU activity limit timing and quality conclusions.

## Reproduction assets

- `requirements-fast.txt`: exact tested client versions; `pyproject.toml`: installable core package and optional test dependencies.
- `scripts/setup_environment.sh`: fresh CPU-client environment and pinned embedding/NLTK assets.
- `scripts/serve_qwen.sh`: separate local vLLM server; model/environment paths are explicit configuration.
- `scripts/benchmark.py`: one subset, live arms, strict replay and detailed JSON.
- `scripts/benchmark_suite.py`: evidence coverage, repeated multi-conversation protocol, GPU monitor and paired summaries.
- `scripts/benchmark_index.py`: real-text/real-embedding index microbenchmark, excluding LLM calls and QA.

Historical raw manifests record the machine on which they were collected. Their local paths are evidence, not instructions for another machine. Historical results may correspond to earlier source hashes; use the manifest's fingerprint when interpreting them.
