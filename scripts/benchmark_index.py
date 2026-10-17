#!/usr/bin/env python3
"""Real MiniLM / LoCoMo index microbenchmark, independent of LLM ingestion.

This measures embedding-index construction and retrieval only.  The JSON must
not be interpreted as an end-to-end A-Mem speedup or a memory-quality result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MeasuredEncoder:
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.texts = 0
        self.seconds = 0.0

    def encode(self, documents):
        self.calls += 1
        self.texts += len(documents)
        started = time.perf_counter()
        try:
            return self.model.encode(documents, show_progress_bar=False)
        finally:
            self.seconds += time.perf_counter() - started

    def snapshot(self):
        return {"encode_calls": self.calls, "encode_texts": self.texts, "encode_seconds": self.seconds}


class UpstreamIndex:
    """The upstream SimpleEmbeddingRetriever add/search algorithms verbatim."""

    def __init__(self, model):
        self.model = model
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    def add_documents(self, documents):
        import numpy as np

        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query, k=5):
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return np.argsort(similarities)[-k:][::-1]


def load_documents(path):
    data = json.loads(path.read_text())
    documents, questions = [], []
    for sample in data:
        conversation = sample["conversation"]
        for name, turns in conversation.items():
            if not name.startswith("session_") or not isinstance(turns, list):
                continue
            timestamp = conversation[name + "_date_time"]
            for turn in turns:
                text = turn.get("text", "")
                if "img_url" in turn and "blip_caption" in turn:
                    text = f"[Image: {turn['blip_caption']}]" + (f" {text}" if text else "")
                content = f"Speaker {turn['speaker']}says : {text}"
                # A deterministic stand-in for consolidation's metadata field.
                # No LLM-generated keywords or evolved memories are fabricated.
                documents.append(f"{content} , {timestamp}  ")
        questions.extend(q["question"] for q in sample["qa"] if q.get("question"))
    return documents, list(dict.fromkeys(questions))


def timed(operation):
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def stage(operation, encoder):
    before = encoder.snapshot()
    _, seconds = timed(operation)
    after = encoder.snapshot()
    return {"seconds": seconds, **{name: after[name] - before[name] for name in before}}


def compare_rankings(reference, candidate, questions, k):
    import numpy as np

    rows = []
    for question in questions:
        expected = reference.search(question, k)
        actual = candidate.search(question, k)
        rows.append({"question": question, "equal": bool(np.array_equal(expected, actual)),
                     "baseline_indices": expected.tolist(), "fast_indices": actual.tolist()})
    return {"all_equal": all(row["equal"] for row in rows), "queries": rows}


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (source archive)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", required=True, help="Local SentenceTransformer model directory")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/locomo10.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/index_benchmark.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--query-repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a fresh --output")
    if min(args.sizes + [args.threads, args.batch_size, args.queries, args.query_repeats, args.k]) < 1:
        parser.error("sizes, threads, batch size, query counts, and k must be positive")

    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import sklearn
    import torch
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from fast_amem.retriever import FastEmbeddingRetriever

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    documents, questions = load_documents(args.dataset)
    if max(args.sizes) > len(documents):
        parser.error(f"dataset has only {len(documents)} documents")
    questions = questions[: args.queries]
    shared_model, shared_load_seconds = timed(lambda: SentenceTransformer(args.embedding_model, device="cpu"))
    _, warmup_seconds = timed(lambda: shared_model.encode(documents[:2], show_progress_bar=False))
    result = {
        "scope": "embedding index microbenchmark only; excludes LLM calls, memory evolution, and QA quality",
        "document_format": "LoCoMo turns plus session timestamp in upstream consolidation format; no LLM metadata",
        "embedding_model": args.embedding_model,
        "embedding_device": "cpu",
        "cpu_threads": args.threads,
        "shared_model_initial_load_seconds": shared_load_seconds,
        "shared_model_warmup_seconds": warmup_seconds,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "git_revision": git_revision(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__},
        "query_count": len(questions),
        "query_repeats": args.query_repeats,
        "k": args.k,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        selected = documents[:size]
        baseline_model, reload_seconds = timed(lambda: SentenceTransformer(args.embedding_model, device="cpu"))
        baseline_encoder = MeasuredEncoder(baseline_model)
        baseline = UpstreamIndex(baseline_encoder)
        baseline_build = stage(lambda: [baseline.add_documents([doc]) for doc in selected], baseline_encoder)
        baseline_build["model_reload_seconds"] = reload_seconds
        baseline_build["with_model_reload_seconds"] = baseline_build["seconds"] + reload_seconds
        fast_encoder = MeasuredEncoder(shared_model)
        fast = FastEmbeddingRetriever(fast_encoder, batch_size=args.batch_size, query_cache_size=0)
        fast_build = stage(lambda: fast.rebuild(selected), fast_encoder)
        fast_unchanged = stage(lambda: fast.rebuild(selected), fast_encoder)
        ranking = compare_rankings(baseline, fast, questions, args.k)
        # Both cold-query measurements deliberately encode every query.  Model
        # load and first forward initialization were already reported above.
        repeated = questions * args.query_repeats
        baseline_cold = stage(lambda: [baseline.search(query, args.k) for query in repeated], baseline_encoder)
        fast_cold = stage(lambda: [fast.search(query, args.k) for query in repeated], fast_encoder)
        fast.query_cache_size = len(questions)
        fast.clear_query_cache()
        cache_fill = stage(lambda: [fast.search(query, args.k) for query in questions], fast_encoder)
        fast_warm = stage(lambda: [fast.search(query, args.k) for query in repeated], fast_encoder)
        changes = max(1, math_ceil_fraction(size, 100))
        changed = selected.copy()
        for position in range(changes):
            changed[position] += " benchmark metadata update"
        fast_changed = stage(lambda: fast.rebuild(changed), fast_encoder)
        fast_changed["changed_documents"] = changes
        # Validate the changed index against a fresh batched sklearn reference;
        # this additional encoding is explicitly outside measured construction.
        validation_encoder = MeasuredEncoder(shared_model)
        changed_reference = UpstreamIndex(validation_encoder)
        changed_reference.add_documents(changed)
        changed_ranking = compare_rankings(changed_reference, fast, questions, args.k)
        case = {
            "documents": size,
            "document_sha256": hashlib.sha256(json.dumps(selected, ensure_ascii=False).encode()).hexdigest(),
            "unique_documents": len(set(selected)),
            "baseline_single_document_rebuild": baseline_build,
            "fast_batched_rebuild": fast_build,
            "fast_unchanged_rebuild": fast_unchanged,
            "fast_changed_rebuild": fast_changed,
            "baseline_queries_without_cache": baseline_cold,
            "fast_queries_without_cache": fast_cold,
            "fast_query_cache_fill": cache_fill,
            "fast_queries_with_warm_cache": fast_warm,
            "ranking_equivalence": ranking,
            "changed_ranking_equivalence": changed_ranking,
            "extra_validation_encoding": validation_encoder.snapshot(),
            "fast_metrics_cumulative": dict(fast.metrics),
            "rebuild_speedup_without_model_load": baseline_build["seconds"] / fast_build["seconds"],
            "rebuild_speedup_with_upstream_model_reload": baseline_build["with_model_reload_seconds"] / fast_build["seconds"],
        }
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"documents": size, "baseline_rebuild_seconds": baseline_build["seconds"],
                          "fast_rebuild_seconds": fast_build["seconds"],
                          "unchanged_rebuild_seconds": fast_unchanged["seconds"],
                          "changed_rebuild_seconds": fast_changed["seconds"],
                          "ranking_equal": ranking["all_equal"],
                          "changed_ranking_equal": changed_ranking["all_equal"]}), flush=True)
        if not ranking["all_equal"] or not changed_ranking["all_equal"]:
            raise RuntimeError("Ranking mismatch; inspect the saved benchmark JSON")
        del baseline, baseline_encoder, baseline_model
    print(str(args.output.resolve()), flush=True)


def math_ceil_fraction(numerator, denominator):
    return (numerator + denominator - 1) // denominator


if __name__ == "__main__":
    main()
