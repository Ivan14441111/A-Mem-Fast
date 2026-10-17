# A-Mem-Fast

A measured performance engineering project built on [A-Mem](https://github.com/WujiangXu/A-mem). It provides a bounded ingestion pipeline, an incremental exact vector index, a persistent response cache, and an opt-in fused memory evolution policy. The benchmark harness separates live inference, response replay, and index microbenchmarks so their speedups can be assessed independently.

The reference implementation is the upstream **robust** memory layer at commit `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`. The upstream source files and prompts remain in the repository. The fast implementation follows the upstream successful execution path with stricter error propagation; fused evolution and context deduplication are explicit policy changes. See [engineering details](README_FAST.md), [source audit](docs/audit.md), and [current experiment report](docs/experiments_v2.md).

```mermaid
flowchart LR
    T[Chronological turns] --> A[Bounded parallel metadata analysis]
    A --> C[Exact request cache]
    C --> S[Sequential note insertion]
    S --> E{Evolution policy}
    E --> U[Upstream conditional calls]
    E --> F[Fused structured call]
    U --> G[Memory graph]
    F --> G
    G --> I[Shared encoder and exact index]
    I --> Q[Retrieval and QA]
```

| Component | Purpose | Semantic boundary |
| --- | --- | --- |
| Rolling metadata prefetch | Overlap independent analysis with ordered graph updates | Default retains upstream prompts and mutation order |
| Exact incremental index | Reuse encoder, grow vector storage, batch changed documents at consolidation | Retains cosine ranking, row indices, consolidation schedule and stale-vector behavior |
| SQLite WAL response cache | Reuse complete deterministic requests and audit replay | Includes model revision, endpoint and decoding settings in identity; HTTP failures and truncated outputs are rejected |
| Fused evolution | Replace conditional decision/strengthen/update requests with one structured request | Changes prompts and potentially memory/QA outcomes; opt-in |
| Joint note construction | Combine metadata and evolution in one ordered request | Changes conditioning and retrieval inputs; experimental opt-in |
| Context deduplication | Remove duplicate graph rows from QA context | Changes QA prompts; separate opt-in ablation |

## Evidence and limits

Real Qwen3-8B / MiniLM / LoCoMo measurements, per-question outputs and negative results are documented in [the current experiment report](docs/experiments_v2.md). [The technical analysis](docs/deep_analysis.md) explains where inference time goes and why request fusion changes graph decisions. Fixed-response replay checks the reference memory trace, chronological request sequence and complete QA outputs; it does not guarantee identical independently inferred live outputs. Earlier 64-turn measurements remain in [the historical report](docs/experiments.md).

The suite supports multiple conversations, repeated paired runs, three rotated arm orders, evidence-complete QA selection, token counts, stage latencies, parser/error counters, and GPU load logs. Repeats of the same questions are not independent new samples. Measurements on shared hardware and this small LoCoMo subset do not establish a general speed or accuracy guarantee. Cache replay and index-only speedups are reported separately from live end-to-end inference. This project does not reproduce the paper's full evaluation.

## Install

Python 3.12 is the tested environment. The package metadata allows Python 3.10–3.13, but those versions have not been benchmarked. A CPU embedding client and a separate GPU inference environment keep benchmark resource use explicit.

```bash
# From a checkout of this repository. Requires uv; refuses an existing target.
scripts/setup_environment.sh

# Optional editable package installation in that client environment.
uv pip install --python .venv-client/bin/python -e '.[test]'
export NLTK_DATA="$PWD/.cache/nltk_data"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

The setup script installs [pinned client dependencies](requirements-fast.txt), CPU PyTorch, NLTK resources, and the pinned `all-MiniLM-L6-v2` snapshot. It prints the embedding model directory; use it as `EMBEDDING_MODEL` below. For an existing environment, `python -m pip install -e '.[test]'` installs the package; download embedding/NLTK assets separately.

A compatible OpenAI-style inference endpoint is required for live benchmarks. The provided local Qwen server setup is:

```bash
uv venv --python 3.12 .venv-serve
uv pip install --python .venv-serve/bin/python vllm==0.11.2 transformers==4.57.6
INFERENCE_ENV="$PWD/.venv-serve" MODEL_DIR=/path/to/Qwen3-8B GPU_ID=0 \
  scripts/serve_qwen.sh
```

Download model weights before starting the server. The script binds to `127.0.0.1:21014` and enables the vLLM development prefix-cache reset endpoint used to isolate cold arms. GPU memory limits and other server options are configurable through environment variables in [the script](scripts/serve_qwen.sh).

## Run

Inspect coverage and the default paired order without loading models or contacting a server:

```bash
python scripts/benchmark_suite.py --plan-only \
  --samples 0 1 2 --turns 128 --repeats 2 --max-questions 12 \
  --max-tokens 2000 --repetition-penalty 1.05 --tag-budget 16 --tag-budget-policy observe \
  --output results/my-plan
```

Run an isolated configured reference / exact pipeline comparison with strict replay:

```bash
export EMBEDDING_MODEL=/path/to/all-MiniLM-L6-v2
# A stable identity for the actual weights, tokenizer, chat template and serving configuration.
export MODEL_REVISION=your-model-and-template-revision

.venv-client/bin/python scripts/benchmark.py \
  --embedding-model "$EMBEDDING_MODEL" --model-revision "$MODEL_REVISION" \
  --max-turns 64 --max-questions 12 \
  --max-tokens 2000 --repetition-penalty 1.05 --tag-budget 16 --tag-budget-policy observe \
  --modes baseline,optimized,replay,warm \
  --output results/my-smoke
```

Run the repeated suite:

```bash
.venv-client/bin/python scripts/benchmark_suite.py \
  --embedding-model "$EMBEDDING_MODEL" --model-revision "$MODEL_REVISION" \
  --samples 0 1 2 --turns 128 --repeats 2 --max-questions 12 \
  --max-tokens 2000 --repetition-penalty 1.05 --tag-budget 16 --tag-budget-policy observe \
  --continue-on-error --output results/my-suite
```

These commands use the explicit `observed-tags-v1` profile: deterministic decoding, repetition penalty 1.05, and a common system instruction requesting tag lists of at most 16 items. Oversized complete lists are recorded without clipping; malformed HTTP responses and truncation remain errors. This contract was introduced after an unbounded strengthen response repeated until truncation; the original failed attempt, fixed diagnostic controls and subsequent strict tag-budget abort are retained. It changes the generator contract equally for configured arms and is separate from the untouched upstream prompts. Omitting the optional flags preserves historical request identities.

The suite attempts each planned cell once. `--continue-on-error` preserves and counts generation/execution failures while attempting remaining independent cells; source, configuration and replay integrity failures stop the suite. Timing/QA comparisons use completed baseline/candidate arm pairs, including pairs in a cell where another optional arm later failed. Completion and replay verification are reported separately, and the full planned denominator remains in `suite.json`.

Use a fresh output directory for each cold run. `--resume` skips completed suite cells only when source and configuration hashes match; partial failed cells are preserved for inspection and require a new suite output. `--context-policy deduplicated` changes QA context for fast arms; the default is `upstream`. If a server does not support prefix-cache reset, `--prefix-cache-reset off` runs with that limitation recorded in the manifest.

Every subset keeps chronological turns and selects QA whose nonempty evidence references are fully contained in those turns. Categories 1–4 are selected in round-robin order; category 5 is excluded. Metrics are normalized exact match and multiset token F1, defined in the harness. `summary.json` separates build and QA times. Model loading/startup/warmup and the reference's write-only recording flush are excluded from live timings and reported separately; real inference and fast-cache overhead are included.

Response-cache SQLite files stay local and are excluded from git; the repository includes reports and source fingerprints. After a run has exited, `python scripts/archive_replay_cache.py --run-dir results/my-suite` creates a consistent private cache archive with SHA256 checks under `.private/artifacts/`.

Existing default-path outputs can be replayed with the same benchmark configuration and `--modes replay,warm`. Replay is strict, never falls back to inference, and needs no live server. A changed model or template requires a new revision identity.

## Test and inspect

```bash
.venv-client/bin/python -m pytest -q
.venv-client/bin/python scripts/benchmark_index.py \
  --embedding-model "$EMBEDDING_MODEL" --output results/my-index.json
```

Saved suites can be summarized and plotted without running inference:

```bash
python scripts/summarize_suite.py results/my-suite/suite.json --output results/my-suite/derived.json
uv venv --python 3.12 .venv-analysis
uv pip install --python .venv-analysis/bin/python 'matplotlib>=3.8,<3.11'
.venv-analysis/bin/python scripts/plot_suite.py results/my-suite/suite.json \
  --output results/my-suite/overview.svg --png
```

The figure includes individual repeat dots and saves the plotted data with a source-suite hash. Unit tests use deterministic HTTP and embedding doubles; they do not download models or contact a live service. They cover graph/request/QA replay equivalence, rolling prefetch, failures, parser validation, exact ranking and ties, index rebuilds, cache isolation and concurrent request coalescing. The index microbenchmark uses real MiniLM embeddings and LoCoMo text, and measures index work only.

## Attribution

The original A-Mem README, paper links and citation are preserved in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md). Original code copyright and MIT terms remain in [LICENSE](LICENSE); [NOTICE](NOTICE) records provenance and the scope of this engineering work. The dataset and original figures are inherited from upstream. LoCoMo data is separately licensed under [CC BY-NC 4.0](data/LICENSE-LOCOMO.txt); see [dataset attribution](data/README.md). No upstream paper results are claimed as results of A-Mem-Fast.
