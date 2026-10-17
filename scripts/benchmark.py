#!/usr/bin/env python3
"""Small, evidence-complete LoCoMo experiment; upstream files remain untouched."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from types import SimpleNamespace

UPSTREAM_COMMIT = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "memory_layer.py").is_file())
sys.path.insert(0, str(ROOT))


def project_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (source archive)"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class BufferedRecordingCache:
    """Record reference outputs without adding an fsync to the baseline hot path.

    Only the baseline's write-only recorder uses this adapter. Optimized cache
    reads/writes remain synchronous and their overhead stays inside its timing.
    """
    def __init__(self, cache):
        self.cache = cache
        self.rows = {}

    def put(self, key, payload, response):
        if key in self.rows:
            previous = self.rows[key][1]["choices"][0]["message"]["content"]
            current = response["choices"][0]["message"]["content"]
            if previous != current:
                raise RuntimeError("Same baseline request returned different text; exact cache replay is ambiguous")
        self.rows[key] = (payload, response)

    def flush(self):
        for key, (payload, response) in self.rows.items():
            self.cache.put(key, payload, response)
        self.rows.clear()


def select_subset(path, sample_index, max_turns, max_questions):
    data = json.loads(Path(path).read_text())
    sample = data[sample_index]
    turns = []
    conv = sample["conversation"]
    for key, session in conv.items():
        if not key.startswith("session_") or not isinstance(session, list):
            continue
        for t in session:
            text = t.get("text", "")
            if "img_url" in t and "blip_caption" in t:
                text = f"[Image: {t['blip_caption']}]" + (f" {text}" if text else "")
            turns.append({"content": "Speaker " + t["speaker"] + "says : " + text,
                          "timestamp": conv[key + "_date_time"], "dia_id": t["dia_id"]})
    turns = turns[:max_turns]
    ids = {t["dia_id"] for t in turns}
    eligible = [{**q, "source_qa_index": index} for index, q in enumerate(sample["qa"]) if q.get("evidence") and
                all(e in ids for e in q["evidence"]) and q["category"] in (1, 2, 3, 4)]
    buckets = defaultdict(list)
    for q in eligible:
        buckets[q["category"]].append(q)
    questions = []
    while len(questions) < max_questions and any(buckets.values()):
        for c in sorted(buckets):
            if buckets[c] and len(questions) < max_questions:
                questions.append(buckets[c].pop(0))
    if not turns or not questions:
        raise ValueError("Subset contains no turns or evidence-complete QA")
    return {"sample_id": sample["sample_id"], "turns": turns, "questions": questions,
            "eligible_qa": len(eligible), "category_counts": dict(Counter(q["category"] for q in questions))}


def normalize(text):
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(prediction, answer):
    p, a = normalize(prediction).split(), normalize(answer).split()
    common = sum((Counter(p) & Counter(a)).values())
    f1 = 2 * common / (len(p) + len(a)) if p or a else 1.0
    return {"exact_match": float(p == a), "token_f1": f1}


def canonical_memories(system):
    return [{k: v for k, v in vars(m).items() if k not in {"id", "last_accessed"}}
            for m in system.memories.values()]


def answer_question(system, client, qa, k):
    from llm_text_parsers import parse_keywords_response, parse_plain_text_answer
    question = qa["question"]
    query_prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
    keywords = parse_keywords_response(client.get_completion(query_prompt))
    context = system.find_related_memories_raw(keywords, k=k)
    if qa["category"] == 2:
        prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
    else:
        prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
    prediction = parse_plain_text_answer(client.get_completion(prompt))
    return {**qa, "prediction": prediction, "metrics": score(prediction, qa["answer"]),
            "context_digest": digest(context), "prompt_digest": digest(prompt),
            "context_utf8_bytes": len(context.encode("utf-8")),
            "context_rows": context.count("talk start time:"),
            "context_unique_rows": len(set(context.split("talk start time:")[1:]))}


def cache_name_for(mode):
    return {"replay": "baseline", "warm": "optimized", "fused_warm": "fused", "joint_warm": "joint"}.get(mode, mode)


def policy_for(args, mode):
    if mode in {"joint", "joint_warm"}:
        return "joint"
    if mode in {"fused", "fused_warm"}:
        return "fused"
    return args.evolution_policy if mode in {"optimized", "warm"} else "upstream"


def run_mode(args, subset, mode, shared_model):
    try:
        return _run_mode(args, subset, mode, shared_model)
    except Exception as exc:
        failure = {"mode": mode, "exception_type": type(exc).__name__, "error": str(exc),
                   "failed_utc": datetime.now(timezone.utc).isoformat(),
                   "subset_fingerprint": subset["fingerprint"],
                   "evolution_policy": policy_for(args, mode),
                   "note": "Incomplete arm; excluded from timing and quality aggregates. Inspect suite log and local response cache."}
        progress = getattr(exc, "benchmark_progress", None)
        if progress is not None:
            failure["progress"] = progress
        diagnostic = getattr(exc, "llm_diagnostic", None)
        if diagnostic is not None:
            failure["llm_diagnostic"] = diagnostic
        (Path(args.output) / f"{mode}-failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


def _run_mode(args, subset, mode, shared_model):
    import memory_layer
    from memory_layer import SimpleEmbeddingRetriever
    from memory_layer_robust import RobustAgenticMemorySystem
    from fast_amem.client import MeasuredClient, SQLiteResponseCache
    from fast_amem.system import FastAgenticMemorySystem
    output = Path(args.output)
    wall_utc = {"run_start": datetime.now(timezone.utc).isoformat()}
    reset_seconds = None
    if mode in {"baseline", "optimized", "fused", "joint"} and args.prefix_cache_reset == "required":
        import requests
        started = time.perf_counter()
        reset_url = args.base_url.rstrip("/").removesuffix("/v1") + "/reset_prefix_cache"
        requests.post(reset_url, timeout=60).raise_for_status()
        reset_seconds = time.perf_counter() - started
    cache_name = cache_name_for(mode)
    cache_path = output / f"{cache_name}.sqlite"
    if mode in {"baseline", "optimized", "fused", "joint"} and cache_path.exists():
        raise FileExistsError(f"Cold run cache exists: {cache_path}; use a new output directory")
    if mode in {"replay", "warm", "fused_warm", "joint_warm"} and not cache_path.exists():
        raise FileNotFoundError(f"Required recorded run missing: {cache_path}")
    cache = SQLiteResponseCache(cache_path)
    if mode == "baseline":
        cache = BufferedRecordingCache(cache)
    client = MeasuredClient(args.base_url, args.model,
                            cache=cache,
                            cache_mode={"baseline": "write", "optimized": "readwrite",
                                        "replay": "replay", "warm": "replay",
                                        "fused": "readwrite", "fused_warm": "replay",
                                        "joint": "readwrite", "joint_warm": "replay"}[mode],
                            revision=args.model_revision, max_tokens=args.max_tokens, seed=args.seed,
                            repetition_penalty=args.repetition_penalty, tag_budget=args.tag_budget)
    wall_utc["init_start"] = datetime.now(timezone.utc).isoformat()
    init_start = time.perf_counter()
    if mode == "baseline":
        # Initialize identical upstream state without constructing an unused LLM client.
        system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
        system.memories = {}
        system.retriever = SimpleEmbeddingRetriever(args.embedding_model)
        system.llm_controller = SimpleNamespace(llm=client)
        system.evo_cnt = 0
        system.evo_threshold = args.evo_threshold
    elif mode in {"joint", "joint_warm"}:
        from fast_amem.joint import JointAgenticMemorySystem
        system = JointAgenticMemorySystem(shared_model, client, evo_threshold=args.evo_threshold,
                                         context_policy=args.context_policy)
    else:
        system = FastAgenticMemorySystem(
            shared_model, client, evo_threshold=args.evo_threshold,
            evolution_policy=policy_for(args, mode),
            context_policy=args.context_policy if mode in {"optimized", "warm", "fused", "fused_warm", "joint", "joint_warm"} else "upstream")
    init_seconds = time.perf_counter() - init_start
    wall_utc["init_end"] = datetime.now(timezone.utc).isoformat()
    # Every baseline consolidation may create a new model. Track each through
    # the shared factory below, and report deltas for this run only.
    encoder_before = dict(ENCODER_METRICS)
    timings = defaultdict(float)
    counts = Counter()
    # Profile nested work. These inclusive timings must NOT be added together.
    for name in ("process_memory", "find_related_memories", "consolidate_memories"):
        original = getattr(system, name)
        def measured(*a, _fn=original, _name=name, **kw):
            started = time.perf_counter()
            try:
                return _fn(*a, **kw)
            finally:
                timings[_name] += time.perf_counter() - started
                counts[_name] += 1
        setattr(system, name, measured)
    trace = []
    last_completed = [0]
    baseline_event_cursor = [0]
    def on_note(n):
        if mode == "baseline":
            # Upstream catches inference errors internally. Fail this benchmark
            # at the current insertion boundary instead of timing later fallback.
            events = client.events[baseline_event_cursor[0]:]
            baseline_event_cursor[0] = len(client.events)
            errors = [event for event in events if event["error"]]
            if errors:
                error = RuntimeError("Configured robust arm encountered an LLM error; fallback is excluded")
                error.llm_diagnostic = [event.get("llm_diagnostic", {"stage": event["stage"], "error": event["error"]})
                                        for event in errors]
                raise error
        trace.append(digest(canonical_memories(system)))
        if n % 8 == 0 or n == len(subset["turns"]):
            print(f"{mode}: memory {n}/{len(subset['turns'])}, HTTP={client.summary()['http_requests']}", flush=True)
        last_completed[0] = n
    wall_utc["build_start"] = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        if mode == "baseline":
            for n, turn in enumerate(subset["turns"], 1):
                system.add_note(turn["content"], time=turn["timestamp"])
                on_note(n)
        else:
            system.ingest(subset["turns"], analysis_workers=args.workers,
                          prefetch_window=args.prefetch_window, on_note=on_note)
    except Exception as exc:
        index = last_completed[0]
        exc.benchmark_progress = {"phase": "build", "last_completed_turn": index,
                                  "failed_turn_1based": index + 1,
                                  "dia_id": subset["turns"][index]["dia_id"] if index < len(subset["turns"]) else None,
                                  "memory_count_at_failure": len(system.memories)}
        raise
    build_seconds = time.perf_counter() - started
    wall_utc["build_end"] = datetime.now(timezone.utc).isoformat()
    build_summary = client.summary()
    wall_utc["qa_start"] = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    qa_results = []
    for index, question in enumerate(subset["questions"]):
        try:
            qa_results.append(answer_question(system, client, question, args.retrieve_k))
        except Exception as exc:
            exc.benchmark_progress = {"phase": "qa", "question_index_0based": index,
                                      "source_qa_index": question["source_qa_index"]}
            raise
    qa_seconds = time.perf_counter() - started
    wall_utc["qa_end"] = datetime.now(timezone.utc).isoformat()
    recording_flush_seconds = 0.0
    if mode == "baseline":
        started = time.perf_counter()
        cache.flush()
        recording_flush_seconds = time.perf_counter() - started
    report = {"mode": mode, "wall_utc": wall_utc, "init_seconds": init_seconds, "build_seconds": build_seconds,
              "subset_fingerprint": subset["fingerprint"],
              "evolution_policy": policy_for(args, mode),
              "context_policy": args.context_policy if mode in {"optimized", "warm", "fused", "fused_warm", "joint", "joint_warm"} else "upstream",
              "context_metrics": getattr(system, "context_metrics", {}),
              "repetition_penalty": args.repetition_penalty,
              "tag_budget": args.tag_budget,
              "prefix_cache_reset_seconds": reset_seconds,
              "recording_flush_seconds_excluded": recording_flush_seconds,
              "qa_seconds": qa_seconds, "total_seconds": build_seconds + qa_seconds,
              "build_llm": build_summary, "all_llm": client.summary(),
              "inclusive_seconds": dict(timings), "operation_counts": dict(counts),
              "retriever_metrics": getattr(system.retriever, "metrics", {}),
              "encoder_metrics": {k: ENCODER_METRICS[k] - encoder_before.get(k, 0)
                                  for k in ENCODER_METRICS},
              "evolution_metrics": getattr(system, "evolution_metrics", {}),
              "evolution_count": system.evo_cnt, "memory_count": len(system.memories),
              "memory_trace": trace, "memory_digest": digest(canonical_memories(system)),
              "upstream_request_keys": [e["key"] for e in client.events if not e["prefetch"]],
              "metrics": {k: sum(q["metrics"][k] for q in qa_results) / len(qa_results)
                          for k in ("exact_match", "token_f1")},
              "qa": qa_results, "llm_events": client.events}
    (output / f"{mode}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if report["all_llm"]["errors"]:
        error = RuntimeError(f"{mode}: errors detected; see {output / (mode + '.json')}")
        error.llm_diagnostic = [event.get("llm_diagnostic", {"stage": event["stage"], "error": event["error"]})
                                for event in client.events if event["error"]]
        raise error
    print(f"{mode}: build={build_seconds:.3f}s QA={qa_seconds:.3f}s F1={report['metrics']['token_f1']:.3f}", flush=True)
    return report


ENCODER_METRICS = Counter()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(ROOT / "data/locomo10.json"))
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=64)
    p.add_argument("--max-questions", type=int, default=12)
    p.add_argument("--retrieve-k", type=int, default=10)
    p.add_argument("--evo-threshold", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--context-policy", choices=("upstream", "deduplicated"), default="upstream")
    p.add_argument("--evolution-policy", choices=("upstream", "fused"), default="upstream")
    p.add_argument("--prefix-cache-reset", choices=("required", "off"), default="required",
                   help="required uses vLLM dev reset endpoint; off records that cache was uncontrolled")
    p.add_argument("--skip-server-warmup", action="store_true",
                   help="Skip live warmup (automatically skipped for replay-only invocations)")
    p.add_argument("--prefetch-window", type=int, default=16)
    p.add_argument("--base-url", default="http://127.0.0.1:21014/v1")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--model-revision", required=True)
    p.add_argument("--embedding-model", required=True)
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--tag-budget", type=int, help="Optional common system-message tag-list bound; configured protocol bounded-tags-v1")
    p.add_argument("--repetition-penalty", type=float, help="Optional deterministic vLLM decoding parameter; common to all live/replay arms")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--modes", default="baseline,optimized,replay,warm")
    args = p.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if min(args.max_turns, args.max_questions, args.retrieve_k, args.evo_threshold,
           args.workers, args.prefetch_window, args.max_tokens) < 1:
        p.error("Subset size, retrieval k and evolution threshold must be positive")
    if args.tag_budget is not None and args.tag_budget <= 0:
        p.error("tag-budget must be positive")
    if args.repetition_penalty is not None and (not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0):
        p.error("repetition-penalty must be finite and positive")
    modes = [mode.strip() for mode in args.modes.split(",")]
    if not modes or len(modes) != len(set(modes)):
        p.error("Modes must be nonempty and unique")
    if args.sample_index < 0:
        p.error("sample-index must be nonnegative")
    if set(modes) - {"baseline", "optimized", "replay", "warm", "fused", "fused_warm", "joint", "joint_warm"}:
        p.error("Unknown mode")
    output = Path(args.output)
    # Refuse destructive reuse before writing any manifest or subset artifact.
    available = {name for name in ("baseline", "optimized", "fused", "joint")
                 if (output / f"{name}.sqlite").exists()}
    for mode in modes:
        cache_name = cache_name_for(mode)
        exists = (output / f"{cache_name}.sqlite").exists()
        if mode in {"baseline", "optimized", "fused", "joint"} and exists:
            raise FileExistsError(f"Cold run cache exists in {output}; choose a new output directory")
        if mode in {"replay", "warm", "fused_warm", "joint_warm"} and cache_name not in available:
            raise FileNotFoundError(f"Missing prior recorded {cache_name} cache in {output}; order cold modes first")
        if mode in {"baseline", "optimized", "fused", "joint"}:
            available.add(cache_name)
    output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    import memory_layer
    # CPU inference avoids disturbing unrelated GPU workloads. Resolve the default
    # consolidation alias locally, while preserving actual baseline model reloads.
    def encoder_factory(name):
        started = time.perf_counter()
        encoder = SentenceTransformer(args.embedding_model if name == "all-MiniLM-L6-v2" else name,
                                      device="cpu")
        ENCODER_METRICS["model_loads"] += 1
        ENCODER_METRICS["model_load_seconds"] += time.perf_counter() - started
        original_encode = encoder.encode
        def encode(texts, *a, **kw):
            ENCODER_METRICS["encode_calls"] += 1
            ENCODER_METRICS["encode_texts"] += len(texts)
            began = time.perf_counter()
            try:
                return original_encode(texts, *a, **kw)
            finally:
                ENCODER_METRICS["encode_seconds"] += time.perf_counter() - began
        encoder.encode = encode
        return encoder
    memory_layer.SentenceTransformer = encoder_factory
    subset = select_subset(args.dataset, args.sample_index, args.max_turns, args.max_questions)
    subset["fingerprint"] = digest(subset)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        keys = ("model", "model_revision", "base_url", "embedding_model", "max_tokens",
                "seed", "retrieve_k", "evo_threshold", "evolution_policy", "context_policy", "prefix_cache_reset", "repetition_penalty", "tag_budget")
        if (old["subset_fingerprint"] != subset["fingerprint"] or
                any(old["args"].get(k) != getattr(args, k) for k in keys)):
            raise ValueError("Existing output belongs to a different subset or model configuration")
    (output / "subset.json").write_text(json.dumps(subset, indent=2, ensure_ascii=False))
    versions = {}
    from importlib.metadata import PackageNotFoundError, version
    for package in ("torch", "transformers", "sentence-transformers", "numpy", "scikit-learn", "vllm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            if package != "vllm":
                raise
            versions[package] = "external server; see server environment manifest"
    manifest = {"args": vars(args), "python": platform.python_version(), "versions": versions,
                "upstream_commit": UPSTREAM_COMMIT,
                "project_commit": project_revision(),
                "upstream_repository": "https://github.com/WujiangXu/A-mem",
                "subset_fingerprint": subset["fingerprint"],
                "timing_note": "Model download/server startup/model loading/warmup excluded; real live inference and fast-cache overhead included. Init reported separately. Nested timing is inclusive.",
                "decoding": {"temperature": 0, "seed": args.seed, "enable_thinking": False,
                             "max_tokens": args.max_tokens, "repetition_penalty": args.repetition_penalty,
                             "tag_budget": args.tag_budget,
                             "prompt_protocol": "bounded-tags-v1" if args.tag_budget is not None else "upstream-system",
                             "note": "Common configured decoding in all arms; differs from upstream default temperature."},
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [ROOT / "scripts/benchmark.py", ROOT / "memory_layer.py",
                                               ROOT / "memory_layer_robust.py", ROOT / "llm_text_parsers.py",
                                               *sorted((ROOT / "fast_amem").glob("*.py"))]},
                "baseline_recording": "Responses buffered in RAM; SQLite flush excluded. Optimized persistent cache overhead included."}
    if manifest_path.exists():
        manifest_path = output / f"invocation-{int(time.time())}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    # Warm both the local embedding kernels and server before timing either arm.
    started = time.perf_counter()
    shared_model = encoder_factory(args.embedding_model)
    shared_model.encode([t["content"] for t in subset["turns"][:4]])
    from fast_amem.client import MeasuredClient
    if not args.skip_server_warmup and any(mode in {"baseline", "optimized", "fused", "joint"} for mode in modes):
        MeasuredClient(args.base_url, args.model, max_tokens=args.max_tokens, seed=args.seed,
                       repetition_penalty=args.repetition_penalty, tag_budget=args.tag_budget).get_completion(
                           "Reply with exactly one word: READY")
    manifest["shared_encoder_and_warmup_seconds"] = time.perf_counter() - started
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    reports = {}
    for mode in modes:
        reports[mode] = run_mode(args, subset, mode, shared_model)
    for mode in ("baseline", "optimized", "fused", "joint"):
        if mode not in reports and (output / f"{mode}.json").exists():
            reports[mode] = json.loads((output / f"{mode}.json").read_text())
            if reports[mode].get("subset_fingerprint") != subset["fingerprint"]:
                raise ValueError("Stored report has a different subset fingerprint")
    comparisons = {}
    for base, other in (("baseline", "optimized"), ("baseline", "fused"), ("baseline", "replay"),
                        ("optimized", "warm"), ("fused", "fused_warm"),
                        ("baseline", "joint"), ("joint", "joint_warm")):
        if base not in reports or other not in reports:
            continue
        a, b = reports[base], reports[other]
        comparisons[f"{base}_vs_{other}"] = {
            "build_speedup": a["build_seconds"] / b["build_seconds"],
            "total_speedup": a["total_seconds"] / b["total_seconds"],
            "memory_trace_equal": a["memory_trace"] == b["memory_trace"],
            "upstream_requests_equal": a["upstream_request_keys"] == b["upstream_request_keys"],
            "qa_equal": a["qa"] == b["qa"],
            "f1_delta": b["metrics"]["token_f1"] - a["metrics"]["token_f1"],
            "same_evolution_policy": a.get("evolution_policy", "upstream") == b.get("evolution_policy", "upstream"),
            "same_context_policy": a.get("context_policy", "upstream") == b.get("context_policy", "upstream")}
    summary = {"runs": {m: {k: r[k] for k in ("build_seconds", "qa_seconds", "metrics", "build_llm", "all_llm")}
                        for m, r in reports.items()}, "comparisons": comparisons}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(comparisons, indent=2), flush=True)
    for comparison in ("baseline_vs_replay", "optimized_vs_warm", "fused_vs_fused_warm", "joint_vs_joint_warm"):
        if comparison in comparisons and not all(comparisons[comparison][k]
                                                for k in ("memory_trace_equal", "upstream_requests_equal", "qa_equal")):
            raise AssertionError(f"Semantic replay mismatch: {comparison}")


if __name__ == "__main__":
    main()
