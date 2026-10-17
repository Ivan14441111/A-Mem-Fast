#!/usr/bin/env python3
"""Derive transparent per-cell and per-conversation tables from saved suite JSON.

This script does no inference. It includes failed cells explicitly, calculates
QA-stage tokens as all-stage minus build-stage tokens, and does not treat
repeated questions as new independent evaluation questions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def average(rows, key):
    return statistics.mean(row[key] for row in rows)


def tokens(summary, field):
    return sum(stage.get(field, 0) for stage in summary["by_stage"].values())


def latency_summary(events):
    stages = defaultdict(list)
    for event in events:
        if event.get("http_requests", 0):
            stages[event["stage"]].append(event)
    result = {}
    for stage, selected in stages.items():
        values = sorted(event["http_seconds"] for event in selected)
        def percentile(q):
            position = (len(values) - 1) * q
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            return values[lower] + (values[upper] - values[lower]) * (position - lower)
        result[stage] = {"live_calls": len(selected),
                         "http_attempts": sum(event["http_requests"] for event in selected),
                         "call_http_seconds_p50": percentile(0.5),
                         "call_http_seconds_p95": percentile(0.95),
                         "call_http_seconds_max": values[-1]}
    return result


def summarize(report, suite_dir=None):
    rows = []
    groups = defaultdict(list)
    failures = []
    for cell in report["cells"]:
        if cell["status"] != "complete":
            failures.append({key: cell.get(key) for key in ("name", "status", "returncode")})
            continue
        runs = cell["summary"]["runs"]
        baseline = runs["baseline"]
        for arm in ("baseline", "optimized", "fused"):
            if arm not in runs:
                continue
            run = runs[arm]
            row = {"cell": cell["name"], "sample_index": cell["sample_index"],
                   "sample_id": cell["sample_id"], "turns": cell["actual_turns"],
                   "question_count": cell["question_count"], "repeat": cell["repeat"],
                   "arm": arm, "experiment_policy": cell.get("evolution_policy", "upstream"), "order": cell["modes"].split(",")[:3],
                   "build_seconds": run["build_seconds"], "qa_seconds": run["qa_seconds"],
                   "total_seconds": run["build_seconds"] + run["qa_seconds"],
                   "http_requests_build": run["build_llm"]["http_requests"],
                   "http_requests_all": run["all_llm"]["http_requests"],
                   "token_f1": run["metrics"]["token_f1"], "exact_match": run["metrics"]["exact_match"],
                   "build_speedup": baseline["build_seconds"] / run["build_seconds"],
                   "f1_delta": run["metrics"]["token_f1"] - baseline["metrics"]["token_f1"]}
            if suite_dir is not None:
                raw = json.loads((suite_dir / cell["name"] / f"{arm}.json").read_text())
                row["live_stage_latency"] = latency_summary(raw.get("llm_events", []))
                row["llm_errors"] = raw.get("all_llm", {}).get("errors", 0)
                row["evolution_count"] = raw["evolution_count"]
                row["consolidation_count"] = raw["operation_counts"].get("consolidate_memories", 0)
                row["memory_count"] = raw["memory_count"]
                evolution = raw.get("evolution_metrics", {})
                row["evolution_decisions"] = evolution.get("fused_decisions", {})
                for metric in ("fused_attempts", "fused_parse_failures", "fused_invalid_reference_failures",
                               "fused_connections", "fused_neighbor_updates"):
                    row[metric] = evolution.get(metric, 0)
                row["mean_context_rows"] = statistics.mean(q.get("context_rows", 0) for q in raw["qa"])
                row["mean_context_unique_rows"] = statistics.mean(q.get("context_unique_rows", 0) for q in raw["qa"])
                row["mean_context_utf8_bytes"] = statistics.mean(q.get("context_utf8_bytes", 0) for q in raw["qa"])
            for field in ("prompt_tokens_live", "completion_tokens_live"):
                row["build_" + field] = tokens(run["build_llm"], field)
                row["qa_" + field] = tokens(run["all_llm"], field) - row["build_" + field]
            rows.append(row)
            groups[(row["sample_index"], row["turns"], row["experiment_policy"], row["arm"])].append(row)
    conversations = []
    numeric = ["build_seconds", "qa_seconds", "total_seconds", "http_requests_build", "http_requests_all",
               "token_f1", "exact_match", "build_speedup", "f1_delta", "build_prompt_tokens_live",
               "build_completion_tokens_live", "qa_prompt_tokens_live", "qa_completion_tokens_live"]
    if suite_dir is not None:
        numeric += ["llm_errors", "evolution_count", "consolidation_count", "memory_count", "fused_attempts",
                    "fused_parse_failures", "fused_invalid_reference_failures", "fused_connections",
                    "fused_neighbor_updates", "mean_context_rows", "mean_context_unique_rows", "mean_context_utf8_bytes"]
    for (sample, turns, policy, arm), selected in sorted(groups.items()):
        conversations.append({"sample_index": sample, "turns": turns, "experiment_policy": policy, "arm": arm,
                              "repeats": len(selected), "question_count_per_repeat": selected[0]["question_count"],
                              "mean": {key: average(selected, key) for key in numeric},
                              "build_seconds_min": min(row["build_seconds"] for row in selected),
                              "build_seconds_max": max(row["build_seconds"] for row in selected),
                              "evolution_decisions_per_repeat_mean": {
                                  decision: statistics.mean(row.get("evolution_decisions", {}).get(decision, 0)
                                                            for row in selected)
                                  for decision in sorted({decision for row in selected
                                                         for decision in row.get("evolution_decisions", {})})}})
    return {"aggregation_unit": "Per-conversation rows average repeated runs of the same selected questions. Raw cell rows are paired within repeat. Repeated questions are not independent QA samples.",
            "cells": rows, "conversations": conversations, "noncomplete_cells": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, help="Optional derived JSON output; default prints JSON")
    args = parser.parse_args()
    result = summarize(json.loads(args.suite.read_text()), args.suite.parent)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
