#!/usr/bin/env python3
"""Isolate QA context deduplication on one recorded graph and fixed keywords.

The graph and keyword responses are reconstructed strictly from a completed
benchmark cache. Only the two answer arms call inference. This isolates the
prompt change from evolution, retrieval candidates, and keyword generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark import answer_question, canonical_memories, digest


class ReadOnlySnapshotCache:
    """Load recorded responses once; worker threads never write the source run."""

    def __init__(self, path):
        uri = "file:" + quote(str(Path(path).resolve()), safe="/") + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            self.rows = {key: json.loads(response)
                         for key, response in connection.execute("SELECT key, response FROM responses")}

    def get(self, key):
        return self.rows.get(key)


class FrozenQueryClient:
    """Reuse exact keyword responses while measuring live answer requests only."""

    def __init__(self, query_responses, answer_client):
        self.query_responses = query_responses
        self.answer_client = answer_client

    def get_completion(self, prompt, **kwargs):
        from fast_amem.client import stage_for
        if stage_for(prompt) == "query_keywords":
            if prompt not in self.query_responses:
                raise RuntimeError("Question was not recorded in the fixed keyword snapshot")
            return self.query_responses[prompt]
        if stage_for(prompt) != "answer":
            raise RuntimeError("Context ablation permits only fixed keywords and answer inference")
        return self.answer_client.get_completion(prompt, **kwargs)


def query_prompt(question):
    # This is exactly the query prompt in benchmark.answer_question. A proxy
    # rejects any drift instead of silently regenerating different keywords.
    return f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""


def reconstruct_graph(run_dir, graph_mode, model):
    from fast_amem.client import MeasuredClient
    from fast_amem.system import FastAgenticMemorySystem

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    subset = json.loads((run_dir / "subset.json").read_text())
    report = json.loads((run_dir / f"{graph_mode}.json").read_text())
    if digest({key: value for key, value in subset.items() if key != "fingerprint"}) != subset["fingerprint"]:
        raise ValueError("Recorded subset fingerprint is invalid")
    if report["subset_fingerprint"] != subset["fingerprint"]:
        raise ValueError("Recorded report has a different subset fingerprint")
    config = manifest["args"]
    cache = ReadOnlySnapshotCache(run_dir / f"{graph_mode}.sqlite")
    client = MeasuredClient(config["base_url"], config["model"], cache=cache, cache_mode="replay",
                            revision=config["model_revision"], seed=config["seed"],
                            max_tokens=config["max_tokens"],
                            repetition_penalty=config.get("repetition_penalty"),
                            tag_budget=config.get("tag_budget"),
                            tag_budget_policy=config.get("tag_budget_policy", "error"))
    system = FastAgenticMemorySystem(model, client, evo_threshold=config["evo_threshold"],
                                     evolution_policy=report.get("evolution_policy", "upstream"))
    trace = []
    system.ingest(subset["turns"], analysis_workers=config["workers"],
                  prefetch_window=config["prefetch_window"],
                  on_note=lambda _: trace.append(digest(canonical_memories(system))))
    if trace != report["memory_trace"] or digest(canonical_memories(system)) != report["memory_digest"]:
        raise AssertionError("Fixed graph does not exactly replay the recorded memory trace")
    query_responses = {query_prompt(qa["question"]): client.get_completion(query_prompt(qa["question"]))
                       for qa in subset["questions"]}
    if client.summary()["http_requests"] != 0:
        raise AssertionError("Graph reconstruction and keyword lookup must make zero HTTP requests")
    return system, query_responses, config, subset, report, client.summary()


def context_plan(system, query_responses, questions, k):
    from llm_text_parsers import parse_keywords_response
    results = []
    for qa in questions:
        keywords = parse_keywords_response(query_responses[query_prompt(qa["question"])])
        row = {"source_qa_index": qa.get("source_qa_index"), "question": qa["question"],
               "category": qa["category"]}
        for policy in ("upstream", "deduplicated"):
            system.context_policy = policy
            context = system.find_related_memories_raw(keywords, k)
            row[policy] = {"context_digest": digest(context),
                           "context_utf8_bytes": len(context.encode("utf-8")),
                           "context_rows": context.count("talk start time:")}
        results.append(row)
    return results


def evaluate_arm(system, query_responses, config, subset, policy, output,
                 base_url, prefix_cache_reset):
    from fast_amem.client import MeasuredClient, SQLiteResponseCache

    if prefix_cache_reset == "required":
        import requests
        reset_url = base_url.rstrip("/").removesuffix("/v1") + "/reset_prefix_cache"
        requests.post(reset_url, timeout=60).raise_for_status()
    system.context_policy = policy
    system.context_metrics = {"queries": 0, "rows_before": 0, "rows_after": 0}
    answer_client = MeasuredClient(base_url, config["model"], cache=SQLiteResponseCache(output / f"{policy}.sqlite"),
                                   cache_mode="write", revision=config["model_revision"],
                                   max_tokens=config["max_tokens"], seed=config["seed"],
                                   repetition_penalty=config.get("repetition_penalty"),
                                   tag_budget=config.get("tag_budget"),
                                   tag_budget_policy=config.get("tag_budget_policy", "error"))
    proxy = FrozenQueryClient(query_responses, answer_client)
    before = digest(canonical_memories(system))
    started = time.perf_counter()
    qa_results = [answer_question(system, proxy, qa, config["retrieve_k"]) for qa in subset["questions"]]
    elapsed = time.perf_counter() - started
    if digest(canonical_memories(system)) != before:
        raise AssertionError("QA context ablation mutated the fixed graph")
    summary = answer_client.summary()
    if summary["errors"] or summary["http_requests"] != len(qa_results):
        raise AssertionError("Every question must have exactly one successful cold answer request")
    result = {"context_policy": policy, "qa_seconds": elapsed, "answer_llm": summary,
              "memory_digest": before, "context_metrics": dict(system.context_metrics),
              "metrics": {name: sum(qa["metrics"][name] for qa in qa_results) / len(qa_results)
                          for name in ("exact_match", "token_f1")},
              "context_utf8_bytes": sum(qa["context_utf8_bytes"] for qa in qa_results),
              "context_rows": sum(qa["context_rows"] for qa in qa_results),
              "qa": qa_results, "llm_events": answer_client.events}
    (output / f"{policy}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"{policy}: answer_seconds={elapsed:.3f}, prompt_tokens={summary['prompt_tokens_live']}, "
          f"F1={result['metrics']['token_f1']:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed benchmark.py output with manifest, subset, and response cache")
    parser.add_argument("--graph-mode", choices=("baseline", "optimized", "fused"), default="baseline")
    parser.add_argument("--output", required=True, help="Fresh directory for isolated context-ablation artifacts")
    parser.add_argument("--order", choices=("upstream,deduplicated", "deduplicated,upstream"),
                        default="upstream,deduplicated")
    parser.add_argument("--base-url", help="Live answer endpoint; defaults to the recorded endpoint")
    parser.add_argument("--embedding-model", help="Optional relocated copy of the same embedding weights")
    parser.add_argument("--prefix-cache-reset", choices=("required", "off"), default="required")
    parser.add_argument("--plan-only", action="store_true", help="Strictly replay and report context sizes without live inference")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty or absent")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    manifest = json.loads((Path(args.run_dir) / "manifest.json").read_text())
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model = SentenceTransformer(args.embedding_model or manifest["args"]["embedding_model"], device="cpu")
    system, queries, config, subset, report, replay_summary = reconstruct_graph(args.run_dir, args.graph_mode, model)
    plan = context_plan(system, queries, subset["questions"], config["retrieve_k"])
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"args": vars(args), "fixed_memory_digest": report["memory_digest"],
                  "subset_fingerprint": subset["fingerprint"], "fixed_keywords_digest": digest(queries),
                  "graph_replay_llm": replay_summary,
                  "source_run_manifest": manifest,
                  "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in [Path(__file__), ROOT / "scripts/benchmark.py",
                                                 *sorted((ROOT / "fast_amem").glob("*.py"))]},
                  "timing_note": "Graph replay, keyword lookup, encoder load, and one READY warmup excluded. "
                                 "Each timed arm includes one cold answer request per question and SQLite recording. "
                                 "QA context changes intentionally; this is not an exact-behavior optimization.",
                  "context_plan": plan}
    (output / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    if args.plan_only:
        print(json.dumps({"plan_only": True, "questions": len(plan), "graph_replay_http": 0,
                          **{policy + "_context_rows": sum(row[policy]["context_rows"] for row in plan)
                             for policy in ("upstream", "deduplicated")}}, indent=2))
        return
    base_url = args.base_url or config["base_url"]
    from fast_amem.client import MeasuredClient
    MeasuredClient(base_url, config["model"], repetition_penalty=config.get("repetition_penalty"),
                   tag_budget=config.get("tag_budget"),
                   tag_budget_policy=config.get("tag_budget_policy", "error")).get_completion(
        "Reply with exactly one word: READY")
    results = {policy: evaluate_arm(system, queries, config, subset, policy, output,
                                    base_url, args.prefix_cache_reset)
               for policy in args.order.split(",")}
    original, dedup = results["upstream"], results["deduplicated"]
    def reduction(before, after):
        return 1 - after / before if before else 0.0
    comparison = {"answer_speedup": original["qa_seconds"] / dedup["qa_seconds"],
                  "prompt_token_reduction": reduction(original["answer_llm"]["prompt_tokens_live"],
                                                      dedup["answer_llm"]["prompt_tokens_live"]),
                  "context_byte_reduction": reduction(original["context_utf8_bytes"], dedup["context_utf8_bytes"]),
                  "context_row_reduction": reduction(original["context_rows"], dedup["context_rows"]),
                  "f1_delta": dedup["metrics"]["token_f1"] - original["metrics"]["token_f1"],
                  "exact_match_delta": dedup["metrics"]["exact_match"] - original["metrics"]["exact_match"],
                  "predictions_equal": [qa["prediction"] for qa in original["qa"]] ==
                                       [qa["prediction"] for qa in dedup["qa"]]}
    summary = {"comparison": comparison,
               "runs": {policy: {key: value for key, value in result.items() if key not in {"qa", "llm_events"}}
                        for policy, result in results.items()}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
