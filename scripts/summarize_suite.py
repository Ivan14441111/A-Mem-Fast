#!/usr/bin/env python3
"""Derive transparent per-cell and per-conversation tables from saved suite JSON.

This script does no inference. It includes failed cells explicitly, calculates
QA-stage tokens as all-stage minus build-stage tokens, and does not treat
repeated questions as new independent evaluation questions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def average(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.mean(values) if values else None


def saved_complete_runs(cell, suite_dir=None):
    """Return finished cold arms, including those preceding another arm's failure.

    A cold-arm JSON is written only after its build and all selected QA finish.
    Its presence does not imply that later replay verification ran or passed.
    """
    runs = dict(cell.get("summary", {}).get("runs", {}))
    if suite_dir is not None:
        for arm in ("baseline", "optimized", "fused", "joint"):
            path = suite_dir / cell["name"] / f"{arm}.json"
            if path.exists():
                raw = json.loads(path.read_text())
                if all(key in raw for key in ("build_seconds", "qa_seconds", "metrics", "build_llm", "all_llm")):
                    runs[arm] = raw
    return runs


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


def tag_observations_by_stage(events):
    stages = defaultdict(list)
    for event in events:
        if "tag_budget_observation" in event and not event.get("cache_hit") and event.get("http_requests", 0):
            stages[event["stage"]].append(event["tag_budget_observation"])
    return {stage: {"live_responses_inspected": len(observations),
                    "live_lists_inspected": sum(row["list_count"] for row in observations),
                    "overshoot_responses": sum(row["over_budget_list_count"] > 0 for row in observations),
                    "overshoot_lists": sum(row["over_budget_list_count"] for row in observations),
                    "max_actual_tags": max(row["max_actual_tags"] for row in observations)}
            for stage, observations in stages.items()}


def summarize(report, suite_dir=None):
    rows = []
    groups = defaultdict(list)
    failures = []
    for cell in report["cells"]:
        if cell["status"] != "complete":
            failures.append({key: cell.get(key) for key in ("name", "status", "returncode", "failure_category",
                                                           "failure_artifacts", "failure_reason")})
        runs = saved_complete_runs(cell, suite_dir)
        if not runs:
            continue
        baseline = runs.get("baseline")
        live_order = [arm for arm in cell["modes"].split(",")
                      if arm in {"baseline", "optimized", "fused", "joint"}]
        for arm in ("baseline", "optimized", "fused", "joint"):
            if arm not in runs:
                continue
            run = runs[arm]
            row = {"cell": cell["name"], "sample_index": cell["sample_index"],
                   "sample_id": cell["sample_id"], "turns": cell["actual_turns"],
                   "question_count": cell["question_count"], "repeat": cell["repeat"],
                   "arm": arm, "experiment_policy": cell.get("evolution_policy", "upstream"), "order": live_order,
                   "whole_cell_status": cell["status"], "baseline_pair_complete": baseline is not None,
                   "build_seconds": run["build_seconds"], "qa_seconds": run["qa_seconds"],
                   "total_seconds": run["build_seconds"] + run["qa_seconds"],
                   "http_requests_build": run["build_llm"]["http_requests"],
                   "http_requests_all": run["all_llm"]["http_requests"],
                   "token_f1": run["metrics"]["token_f1"], "exact_match": run["metrics"]["exact_match"],
                   "build_speedup": baseline["build_seconds"] / run["build_seconds"] if baseline else None,
                   "total_speedup": (baseline["build_seconds"] + baseline["qa_seconds"]) /
                                    (run["build_seconds"] + run["qa_seconds"]) if baseline else None,
                   "f1_delta": run["metrics"]["token_f1"] - baseline["metrics"]["token_f1"] if baseline else None}
            if suite_dir is not None:
                raw = json.loads((suite_dir / cell["name"] / f"{arm}.json").read_text())
                row["live_stage_latency"] = latency_summary(raw.get("llm_events", []))
                row["live_stage_tag_observations"] = tag_observations_by_stage(raw.get("llm_events", []))
                row["llm_errors"] = raw.get("all_llm", {}).get("errors", 0)
                row["evolution_count"] = raw["evolution_count"]
                row["consolidation_count"] = raw["operation_counts"].get("consolidate_memories", 0)
                row["memory_count"] = raw["memory_count"]
                evolution = raw.get("evolution_metrics", {})
                row["evolution_decisions"] = evolution.get("joint_decisions" if arm == "joint" else "fused_decisions", {})
                for policy in ("fused", "joint"):
                    for metric in ("attempts", "parse_failures", "invalid_reference_failures", "connections", "neighbor_updates"):
                        key = policy + "_" + metric
                        row[key] = evolution.get(key, 0)
                for phase in ("build", "all"):
                    observed = raw.get(phase + "_llm", {}).get("tag_budget_overshoots", {})
                    row[phase + "_tag_budget_observations"] = observed
                    for metric in ("live_response_inspections", "live_tag_lists_inspected",
                                   "live_overshoot_responses", "over_budget_lists_live", "max_actual_tags"):
                        row[phase + "_tags_" + metric] = observed.get(metric, 0)
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
               "token_f1", "exact_match", "build_speedup", "total_speedup", "f1_delta", "build_prompt_tokens_live",
               "build_completion_tokens_live", "qa_prompt_tokens_live", "qa_completion_tokens_live"]
    if suite_dir is not None:
        numeric += ["llm_errors", "evolution_count", "consolidation_count", "memory_count", "fused_attempts",
                    "fused_parse_failures", "fused_invalid_reference_failures", "fused_connections",
                    "fused_neighbor_updates", "mean_context_rows", "mean_context_unique_rows", "mean_context_utf8_bytes"]
        numeric += ["joint_" + metric for metric in ("attempts", "parse_failures", "invalid_reference_failures",
                                                    "connections", "neighbor_updates")]
        numeric += [phase + "_tags_" + metric for phase in ("build", "all")
                    for metric in ("live_response_inspections", "live_tag_lists_inspected",
                                   "live_overshoot_responses", "over_budget_lists_live", "max_actual_tags")]
    for (sample, turns, policy, arm), selected in sorted(groups.items()):
        conversations.append({"sample_index": sample, "turns": turns, "experiment_policy": policy, "arm": arm,
                              "repeats": len(selected), "question_count_per_repeat": selected[0]["question_count"],
                              "baseline_paired_repeats": sum(row["baseline_pair_complete"] for row in selected),
                              "mean": {key: average(selected, key) for key in numeric},
                              "build_seconds_min": min(row["build_seconds"] for row in selected),
                              "build_seconds_max": max(row["build_seconds"] for row in selected),
                              "tag_observations_across_repeats": {
                                  phase: {**{metric: sum(row.get(phase + "_tags_" + metric, 0) for row in selected)
                                             for metric in ("live_response_inspections", "live_tag_lists_inspected",
                                                            "live_overshoot_responses", "over_budget_lists_live")},
                                          "max_actual_tags": max(row.get(phase + "_tags_max_actual_tags", 0)
                                                                 for row in selected)}
                                  for phase in ("build", "all")},
                              "evolution_decisions_per_repeat_mean": {
                                  decision: statistics.mean(row.get("evolution_decisions", {}).get(decision, 0)
                                                            for row in selected)
                                  for decision in sorted({decision for row in selected
                                                         for decision in row.get("evolution_decisions", {})})}})
    statuses = {status: sum(cell["status"] == status for cell in report["cells"])
                for status in sorted({cell["status"] for cell in report["cells"]})}
    arm_outcomes = {}
    arm_failures = []
    for arm in ("baseline", "optimized", "fused", "joint"):
        planned = [cell for cell in report["cells"] if arm in cell.get("modes", "").split(",")]
        if not planned:
            continue
        complete = [row for row in rows if row["arm"] == arm]
        failed = [cell for cell in planned if suite_dir is not None and
                  (suite_dir / cell["name"] / f"{arm}-failure.json").exists()]
        for cell in failed:
            detail = json.loads((suite_dir / cell["name"] / f"{arm}-failure.json").read_text())
            arm_failures.append({"cell": cell["name"], "arm": arm,
                                 **{key: detail.get(key) for key in ("exception_type", "error", "failed_utc", "progress")}})
        arm_outcomes[arm] = {"planned": len(planned), "attempted_with_artifact": len(complete) + len(failed),
                             "completed": len(complete), "failed": len(failed),
                             "not_completed_or_failed": len(planned) - len(complete) - len(failed),
                             "completed_baseline_pairs": sum(row["baseline_pair_complete"] for row in complete)}
    return {"aggregation_unit": "Per-conversation rows use arithmetic means over completed cold arms, including those preceding another arm's failure. Paired ratios require both baseline and candidate to finish. Replay success is separate. For equal-conversation geometric speedup claims use analyze_suite_views.py. Repeated questions are not independent QA samples.",
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "planned_cells": len(report["cells"]), "cell_status_counts": statuses,
            "cold_arm_outcomes": arm_outcomes,
            "cold_arm_failures": arm_failures,
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
