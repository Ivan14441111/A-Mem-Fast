#!/usr/bin/env python3
"""Compute all-planned, allocation-stable and one-confirmation sensitivity views.

Averages log speedups within each conversation, then gives conversations equal
weight. QA averages repeats within conversation before averaging conversations.
The confirmation is retained separately and is never used to silently overwrite
a planned observation. This script performs no inference or retry selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def describe(cells):
    arms = sorted({arm for entry in cells for arm in entry["cell"].get("summary", {}).get("runs", {})
                   if arm in {"optimized", "fused", "joint"}})
    metrics = {}
    for arm in arms:
        grouped = defaultdict(list)
        for entry in cells:
            cell = entry["cell"]
            if cell["status"] != "complete" or arm not in cell["summary"]["runs"]:
                continue
            runs = cell["summary"]["runs"]
            baseline, candidate = runs["baseline"], runs[arm]
            grouped[(cell["sample_index"], cell["actual_turns"])].append({
                "build_log_speedup": math.log(baseline["build_seconds"] / candidate["build_seconds"]),
                "total_log_speedup": math.log((baseline["build_seconds"] + baseline["qa_seconds"]) /
                                             (candidate["build_seconds"] + candidate["qa_seconds"])),
                "baseline_f1": baseline["metrics"]["token_f1"], "candidate_f1": candidate["metrics"]["token_f1"],
                "f1_delta": candidate["metrics"]["token_f1"] - baseline["metrics"]["token_f1"]})
        conversations = [{"sample_index": sample, "turns": turns, "repeats": len(rows),
                          **{key: statistics.mean(row[key] for row in rows) for key in rows[0]}}
                         for (sample, turns), rows in sorted(grouped.items())]
        if conversations:
            metrics[arm] = {"paired_runs": sum(row["repeats"] for row in conversations),
                            "independent_conversations": len({row["sample_index"] for row in conversations}),
                            "build_speedup_equal_conversation_geomean": math.exp(statistics.mean(row["build_log_speedup"] for row in conversations)),
                            "total_speedup_equal_conversation_geomean": math.exp(statistics.mean(row["total_log_speedup"] for row in conversations)),
                            **{key + "_equal_conversation_mean": statistics.mean(row[key] for row in conversations)
                               for key in ("baseline_f1", "candidate_f1", "f1_delta")},
                            "conversations": conversations}
    unique_qa = {(entry["cell"]["sample_index"], index) for entry in cells
                 if entry["cell"]["status"] == "complete" for index in entry["cell"]["source_qa_indices"]}
    return {"cell_refs": [{"origin": entry["origin"], "name": entry["cell"]["name"]} for entry in cells],
            "unique_selected_qa": len(unique_qa), "metrics": metrics}


def entries(suite, audit, origin):
    classifications = {row["name"]: row["classification"] for row in audit["cells"]}
    return [{"origin": origin, "cell": cell,
             "classification": classifications.get(cell["name"], "unavailable")}
            for cell in suite["cells"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--confirmation-suite", type=Path)
    parser.add_argument("--confirmation-audit", type=Path)
    parser.add_argument("--replace-cell", help="The predetermined first-cell observation, retained in all-planned")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.confirmation_suite) != bool(args.confirmation_audit):
        parser.error("Confirmation suite and audit must be supplied together")
    if args.confirmation_suite and not args.replace_cell:
        parser.error("Confirmation requires an explicit predetermined --replace-cell")
    primary = read(args.suite)
    if len({cell["actual_turns"] for cell in primary["cells"]}) > 1:
        parser.error("Analyze one turn-count suite at a time so conversation weights remain explicit")
    primary_entries = entries(primary, read(args.audit), "primary")
    stable = [entry for entry in primary_entries if entry["cell"]["status"] == "complete"
              and entry["classification"] == "allocation-stable"]
    retained = list(primary_entries)
    views = {"planned_all": describe(primary_entries), "planned_allocation_stable": describe(stable),
             "supplemental_balanced_stable": {"available": False, "reason": "Confirmation not yet supplied"}}
    if args.confirmation_suite:
        confirmation = read(args.confirmation_suite)
        confirmations = entries(confirmation, read(args.confirmation_audit), "confirmation")
        retained += confirmations
        target = next((entry for entry in primary_entries if entry["cell"]["name"] == args.replace_cell), None)
        others = [entry for entry in primary_entries if entry["cell"]["name"] != args.replace_cell]
        reason = None
        if target is None:
            reason = "Predetermined replacement cell is absent"
        elif len(confirmations) != 1:
            reason = "Exactly one predetermined confirmation cell is required"
        elif any(entry["cell"]["status"] != "complete" or entry["classification"] != "allocation-stable" for entry in others + confirmations):
            reason = "Other planned cells or confirmation are incomplete or fail allocation criterion"
        elif target["cell"]["subset_fingerprint"] != confirmations[0]["cell"]["subset_fingerprint"]:
            reason = "Confirmation is a different evidence subset"
        elif primary["source_sha256"] != confirmation["source_sha256"]:
            reason = "Confirmation source hashes differ from primary suite"
        else:
            keys = ("model", "model_revision", "base_url", "embedding_model", "max_tokens", "seed",
                    "retrieve_k", "evo_threshold", "workers", "prefetch_window", "context_policy", "prefix_cache_reset", "repetition_penalty", "tag_budget")
            if any(primary["args"].get(key) != confirmation["args"].get(key) for key in keys):
                reason = "Confirmation model or inference configuration differs"
        if reason is None:
            views["supplemental_balanced_stable"] = {"available": True, **describe(others + confirmations)}
        else:
            views["supplemental_balanced_stable"] = {"available": False, "reason": reason}
    result = {"aggregation": "Mean log paired speedup within conversation, then equal-conversation mean and exponentiation. F1 averages repeats within conversation then conversations equally. Repeated QA and confirmation do not add independent conversations or questions.",
              "sensitivity_limit": "Allocation-stable only classifies observed GPU memory allocation range, not all compute-load changes. Primary planned data is always retained.",
              "inputs": {"primary_suite": str(args.suite), "primary_suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                         "primary_audit": str(args.audit)},
              "views": views,
              "retained_cells": [{"origin": entry["origin"], "name": entry["cell"]["name"],
                                  "status": entry["cell"]["status"], "classification": entry["classification"]}
                                 for entry in retained]}
    if args.confirmation_suite:
        result["inputs"]["confirmation_suite"] = str(args.confirmation_suite)
        result["inputs"]["confirmation_suite_sha256"] = hashlib.sha256(args.confirmation_suite.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
