#!/usr/bin/env python3
"""Apply the prerecorded conservative 64-to-128-turn scheduling rule.

Scheduling uses timings and question counts, never QA quality or relative policy speed.
Full recorded QA equality is checked only to establish replay integrity.
Missing arms use conservative global maxima, never early-failure elapsed time.
This is a scheduling estimate, not a performance bound or experiment result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PAIRS = (("baseline", "replay"), ("optimized", "warm"),
         ("fused", "fused_warm"), ("joint", "joint_warm"))


def elapsed(run):
    return run["init_seconds"] + run["build_seconds"] + run["qa_seconds"]


def project(suite, directory, target, now, deadline):
    reasons = []
    if not all(cell["status"] in {"complete", "failed"} for cell in suite["cells"]):
        reasons.append("All planned64 attempts must finish first")
    if any(cell.get("failure_category") == "configuration_or_integrity" for cell in suite["cells"]):
        reasons.append("Unresolved source/configuration/replay integrity failure")
    if suite["source_sha256"] != target["source_sha256"]:
        reasons.append("Source differs between the fixed64 and128 plans")
    config_keys = ("samples", "repeats", "max_questions", "base_url", "model", "model_revision", "embedding_model",
                   "max_tokens", "seed", "repetition_penalty", "tag_budget", "tag_budget_policy",
                   "structured_policy_outputs", "retrieve_k", "evo_threshold", "workers", "prefetch_window", "context_policy", "prefix_cache_reset")
    if any(suite["args"].get(key) != target["args"].get(key) for key in config_keys):
        reasons.append("Generator/policy settings differ between the fixed plans")
    if {cell["actual_turns"] for cell in suite["cells"]} != {64} or {cell["actual_turns"] for cell in target["cells"]} != {128}:
        raise ValueError("This preregistered projection is specifically64→128 turns")
    cold, replays = {}, {}
    for cell in suite["cells"]:
        for arm, replay in PAIRS:
            for mode, records in ((arm, cold), (replay, replays)):
                path = directory / cell["name"] / f"{mode}.json"
                if not path.exists():
                    continue
                run = json.loads(path.read_text())
                if run["all_llm"]["errors"]:
                    reasons.append(f"Stored complete report contains client errors: {cell['name']}/{mode}")
                if records is replays and run["all_llm"]["http_requests"]:
                    reasons.append(f"Replay unexpectedly made HTTP requests: {cell['name']}/{mode}")
                records[(cell["sample_index"], cell["repeat"], mode)] = run
    for (sample, repeat, replay), recorded in replays.items():
        arm = next(arm for arm, mode in PAIRS if mode == replay)
        original = cold.get((sample, repeat, arm))
        if original is None:
            reasons.append(f"Stored replay lacks its completed cold source: {sample}/{repeat}/{replay}")
            continue
        fields = ("subset_fingerprint", "memory_trace", "memory_digest", "upstream_request_keys", "qa")
        if any(field not in original or field not in recorded or original[field] != recorded[field] for field in fields):
            reasons.append(f"Stored replay differs from its cold source: {sample}/{repeat}/{replay}")
    recovery_path = directory / "replay_recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        if not recovery.get("finished_utc") or any(cell.get("status") != "passed" for cell in recovery.get("cells", [])):
            reasons.append("Recorded replay recovery is incomplete or failed")
        if any(not check.get("passed", False) for cell in recovery.get("cells", []) for check in cell.get("checks", [])):
            reasons.append("Recorded replay recovery contains a failed semantic check")
    if not cold or not replays:
        return {"available": False, "can_start": False,
                "reasons": reasons + ["No usable complete cold/replay timing maxima"]}
    maximum_build = max(run["build_seconds"] for run in cold.values())
    maximum_qa_per_question = max(run["qa_seconds"] / len(run["qa"]) for run in cold.values())
    maximum_init = max(run["init_seconds"] for run in cold.values())
    maximum_replay = max(elapsed(run) for run in replays.values())
    cells = []
    for cell in suite["cells"]:
        corresponding = next(item for item in target["cells"] if (item["sample_index"], item["repeat"]) == (cell["sample_index"], cell["repeat"]))
        if cell["modes"] != corresponding["modes"]:
            reasons.append("Corresponding arm/replay order differs between fixed plans")
        planned_arms = [arm for arm, _ in PAIRS if arm in cell["modes"].split(",")]
        cold_estimates, replay_estimates = [], []
        original_reported = 0.0
        for arm, replay in PAIRS:
            if arm not in planned_arms:
                continue
            key = (cell["sample_index"], cell["repeat"], arm)
            run = cold.get(key)
            build = run["build_seconds"] if run else maximum_build
            qa_per_question = run["qa_seconds"] / len(run["qa"]) if run else maximum_qa_per_question
            initialization = run["init_seconds"] if run else maximum_init
            cold_estimates.append({"arm": arm, "used_complete_timing": run is not None,
                                   "build64_seconds": build, "qa_seconds_per_question": qa_per_question,
                                   "init_seconds": initialization,
                                   "projected128_seconds": 2 * build + qa_per_question * corresponding["question_count"] + initialization})
            if run:
                original_reported += elapsed(run)
            replay_run = replays.get((cell["sample_index"], cell["repeat"], replay))
            replay_elapsed = elapsed(replay_run) if replay_run else maximum_replay
            replay_estimates.append({"mode": replay, "used_complete_timing": replay_run is not None,
                                     "elapsed64_seconds": replay_elapsed, "projected128_seconds": 2 * replay_elapsed})
            # A failed cold cell's recovered replays occurred after its original
            # process. Subtract only replays present in the original suite summary.
            if replay_run and replay in cell.get("summary", {}).get("runs", {}):
                original_reported += elapsed(replay_run)
        residual = max(0.0, cell.get("process_seconds", 0) - original_reported)
        cells.append({"source_cell": cell["name"], "target_cell": corresponding["name"],
                      "cold_arms": cold_estimates, "replays": replay_estimates,
                      "residual_process_seconds": residual,
                      "residual_note": "Conservatively retained; can include unreported failed-arm elapsed, not a speed estimate.",
                      "projected_seconds_before_margin": residual + sum(item["projected128_seconds"] for item in cold_estimates + replay_estimates)})
    projected = 1.20 * sum(cell["projected_seconds_before_margin"] for cell in cells)
    remaining = (deadline - now).total_seconds()
    if remaining < 120 * 60:
        reasons.append("Less than120 minutes remain until the fixed deadline")
    if projected > remaining - 40 * 60:
        reasons.append("Projection exceeds remaining time minus40-minute finalization reserve")
    return {"available": True, "can_start": not reasons, "reasons": reasons,
            "observed_utc": now.isoformat(), "deadline_utc": deadline.isoformat(),
            "remaining_seconds": remaining, "finalization_reserve_seconds": 2400,
            "margin_multiplier": 1.20, "projected_seconds": projected,
            "maxima": {"build64_seconds": maximum_build, "qa_seconds_per_question": maximum_qa_per_question,
                       "cold_init_seconds": maximum_init, "replay_elapsed64_seconds": maximum_replay},
            "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite64", type=Path)
    parser.add_argument("--target-suite", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True, help="ISO8601 fixed deadline, e.g.2026-10-17T22:42:00+00:00")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the recorded decision; choose a fresh output path")
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        parser.error("Deadline must include a timezone")
    result = project(json.loads(args.suite64.read_text()), args.suite64.parent,
                     json.loads(args.target_suite.read_text()), datetime.now(timezone.utc), deadline)
    result["preregistration"] = "docs/joint128_extension_protocol.md"
    result["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (args.suite64, args.target_suite)}
    result["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result.get(key) for key in ("available", "can_start", "projected_seconds", "reasons")}, indent=2))


if __name__ == "__main__":
    main()
