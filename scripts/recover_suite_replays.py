#!/usr/bin/env python3
"""Verify finished cold arms when a later arm prevented the suite's replay phase.

Run after all live cells have stopped and before changing measured sources.
Uses only strict replay modes; a cache miss is an error, never live inference.
The original suite status and failed arm remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("baseline", "replay"), ("optimized", "warm"),
         ("fused", "fused_warm"), ("joint", "joint_warm"))


def verify_pair(directory, cold, replay):
    original = json.loads((directory / f"{cold}.json").read_text())
    recorded = json.loads((directory / f"{replay}.json").read_text())
    checks = {"memory_trace_equal": original["memory_trace"] == recorded["memory_trace"],
              "request_sequence_equal": original["upstream_request_keys"] == recorded["upstream_request_keys"],
              "qa_equal": original["qa"] == recorded["qa"],
              "zero_http": recorded["all_llm"]["http_requests"] == 0,
              "zero_client_errors": recorded["all_llm"]["errors"] == 0}
    return {"cold_arm": cold, "replay_arm": replay, "passed": all(checks.values()), **checks,
            "replay_report_sha256": hashlib.sha256((directory / f"{replay}.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if any(cell["status"] in {"running", "pending"} for cell in suite["cells"]):
        parser.error("Finish all planned live attempts before recovery")
    if args.output.exists():
        parser.error("Choose a fresh recovery report path")

    def check_sources():
        current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in suite["source_sha256"]}
        if current != suite["source_sha256"]:
            raise RuntimeError("Measured source changed; use the archived measured version for replay")

    check_sources()
    result = {"source_suite": str(args.suite), "source_suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "note": "Post-suite strict zero-HTTP verification of completed cold arms; no failed generation is retried.",
              "plan_only": args.plan_only, "cells": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for cell in suite["cells"]:
        directory = args.suite.parent / cell["name"]
        available = [(cold, replay) for cold, replay in PAIRS if (directory / f"{cold}.json").exists()]
        missing = [replay for cold, replay in available if not (directory / f"{replay}.json").exists()]
        item = {"name": cell["name"], "original_cell_status": cell["status"],
                "completed_cold_arms": [cold for cold, _ in available], "missing_replay_modes": missing}
        result["cells"].append(item)
        if missing:
            command = list(cell["command"])
            command[command.index("--modes") + 1] = ",".join(missing)
            if "--skip-server-warmup" not in command:
                command.append("--skip-server-warmup")
            item["command"] = command
            save()
            if not args.plan_only:
                check_sources()
                log = args.output.parent / f"{cell['name']}-replay-recovery.txt"
                environment = dict(os.environ)
                environment.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
                environment.setdefault("HF_HUB_OFFLINE", "1")
                environment.setdefault("TRANSFORMERS_OFFLINE", "1")
                with log.open("w") as handle:
                    process = subprocess.run(command, cwd=ROOT, env=environment,
                                             stdout=handle, stderr=subprocess.STDOUT)
                item["returncode"] = process.returncode
                check_sources()
                if process.returncode:
                    item["status"] = "failed"
                    save()
                    raise SystemExit(process.returncode)
        if not args.plan_only:
            item["checks"] = [verify_pair(directory, cold, replay) for cold, replay in available]
            item["status"] = "passed" if all(row["passed"] for row in item["checks"]) else "failed"
            save()
            if item["status"] == "failed":
                raise RuntimeError(f"Strict replay verification failed: {cell['name']}")
        print(f"{cell['name']}: {len(available)} completed cold arms; {len(missing)} recovery modes", flush=True)
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(args.output)


if __name__ == "__main__":
    main()
