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

Optional fused evolution and joint metadata/evolution change the model task and can change memory and QA quality. Use separate caches when comparing policies. A server supporting `response_format: json_schema` can additionally constrain these policies' output structure:

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

`structured_policy_outputs` defaults to `False`; unconstrained generation remains available when it is disabled. When enabled, it adds a fixed JSON schema only to fused-evolution and joint-note requests. It constrains fields, types and decision values; application parsers still validate retrieved candidate IDs, action consistency and metadata. It provides no semantic or QA-quality guarantee, adds no hard tag-list limit, and leaves upstream evolution and QA requests unchanged. Joint ingestion is sequential and does not prefetch metadata.

The graph and vector index live in memory; SQLite persists LLM responses. To reconstruct a graph, rerun the same ordered turns with `AMEM_CACHE_MODE=replay` and unchanged encoder, endpoint, revision, decoding, policy and structured-output setting. Strict replay makes no live LLM calls and raises on a missing response; the local encoder is still required.

## Default execution path

`FastAgenticMemorySystem` reuses the shared CPU encoder and `FastEmbeddingRetriever`. Independent metadata requests run in a rolling, bounded thread pool. Graph mutation, neighbor retrieval, conditional upstream evolution, and note insertion run chronologically. Prefetch responses enter an exact-request cache and are consumed through the same parser. The successful path follows the robust reference; inference/cache errors propagate in the default strict mode instead of producing a silent benchmark fallback.

The exact retriever uses normalized vectors, growing contiguous storage, an embedding cache and a bounded query cache. Consolidation preserves the upstream threshold and its distinct document format; only changed text is encoded, in batches. It intentionally preserves the upstream refresh schedule after neighbor mutations and positional graph links. Immediately refreshing every changed neighbor, remapping links to stable IDs, approximate nearest-neighbor search and context removal would change behavior. The fast retriever is in memory and does not implement the upstream retriever save/load API. SQLite persists responses; it does not provide transactional graph persistence. Benchmark graph reconstruction uses strict response replay.

`evolution_policy="fused"` uses one JSON-output call for evolution, validates its decision and references, and records mutation/parser counters. It changes the prompting contract and can change memory and QA quality. `context_policy="deduplicated"` removes repeated memory rows before answering; it is another independent opt-in policy. Default values for both policies are `"upstream"`.

## Cache and experiment identities

`MeasuredClient` hashes the complete request, endpoint and explicit model revision. Transport, truncation and configured tag-budget checks precede SQLite WAL cache writes. Fused/joint policy parsing happens afterward, so invalid policy JSON can remain recorded; strict ingestion raises instead of repairing it. Concurrent identical requests share one live request. `write` records the reference while bypassing reads; `readwrite` enables the optimized pipeline; `replay` raises on every miss and never contacts the server.

The schema enters the complete request and therefore the fused/joint cache key. Replay must use the recorded setting; enabling it cannot reuse an unstructured fused/joint response. With the default `False`, historical keys are preserved. Upstream and QA keys remain unchanged even when it is enabled.

The benchmark uses separate configured-reference, optimized, fused and optional joint databases. The measured `observed-tags-v1` profile passes max_tokens=2000, repetition_penalty=1.05, tag_budget=16 and tag_budget_policy="observe" uniformly. A common system instruction requests tag lists of at most 16 items. Otherwise valid complete overshoots are recorded without clipping; live-generation counts are separated from cached response inspections. Truncated or malformed HTTP responses remain errors. The optional default `"error"` policy rejects oversized lists. User prompt templates remain unchanged, and observe uses a separate cache identity. Historical omitted parameters retain their original identities. The unbounded 1000-token pilot, failed RP-only diagnostic and strict tag-budget abort remain separate from the observed-profile primary. Cold-arm prefix caches are reset, and initialization/warmup is reported separately. The reference's write-only recording is buffered in memory and flushed outside timing; optimized cache overhead remains timed. Build/QA also exclude checkpoint/report JSON writes; `total_seconds` is their sum, rather than full process elapsed time. Reported nested stage timings are inclusive and must not be summed into wall time. Per-call latency percentiles also include overlapping requests and are not end-to-end latency.

The full419 run at measured checkpoint `21b5b23` exposed an evidence-loss bug: the baseline reached answer 35 and truncated before its buffered responses were flushed, so its original cache, live trace and build timing were lost. The independent exact arm completed and passed its own replay; reconstruction from that arm's cache does not recover the lost baseline or provide a paired speed result. See the [evidence audit](docs/full419_evidence_audit.json). The final harness now flushes successful baseline responses and writes a build-only checkpoint before QA, then preserves available partial evidence on failure. This does not restore the lost run; archived measurements retain their original source identities.

Three different comparisons serve different purposes:

- Live reference versus fast pipeline measures elapsed time and output/QA differences under separate inference. Greedy decoding and a fixed seed do not guarantee identical responses under different batching.
- Reference responses replayed through the default fast pipeline check every memory snapshot, chronological request key, and complete QA output for equivalence.
- Each optimized policy replayed from its own cold responses checks reproducibility and cache integrity. Fused self replay cannot establish equivalence with upstream prompts.

For completed arms, the suite records source hashes, subset fingerprints, source QA indices, complete per-question predictions, stage token use, errors, parser counters and GPU load. It refuses source drift between cells and retains failed-cell status, diagnostics and the planned denominator. Available partial evidence depends on the recorded harness version. A three-arm suite rotates cold-arm order over repeated pairs. Four-arm counterbalanced runs use four fixed Williams orders with paired reversals. Small samples and shared GPU activity limit timing and quality conclusions.

## Reproduction assets

- `requirements-fast.txt`: exact tested client versions; `pyproject.toml`: installable core package and optional test dependencies.
- `scripts/setup_environment.sh`: fresh CPU-client environment and pinned embedding/NLTK assets.
- `scripts/serve_qwen.sh`: separate local vLLM server; model/environment paths are explicit configuration.
- `scripts/benchmark.py`: one subset, live arms, strict replay and detailed JSON.
- `scripts/benchmark_suite.py`: evidence coverage, repeated multi-conversation protocol, GPU monitor and paired summaries.
- `scripts/benchmark_index.py`: real-text/real-embedding index microbenchmark, excluding LLM calls and QA.

Historical raw manifests record the machine on which they were collected. Their local paths are evidence, not instructions for another machine. Historical results may correspond to earlier source hashes; use the manifest's fingerprint when interpreting them.
