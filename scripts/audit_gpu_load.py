#!/usr/bin/env python3
"""Classify allocation transitions in saved suite GPU logs, preserving all pairs.

Allocation-stable does not mean compute-load stable. Legacy arm phase times are
approximate, inferred from original report mtimes and durations; this audit
persists those estimates before a git checkout can replace filesystem mtimes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from benchmark_suite import aggregate


def gpu_rows(path, gpu_index, local_timezone):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for record in reader:
            record = {str(key).strip(): str(value).strip() for key, value in record.items()}
            if record.get("index") != str(gpu_index):
                continue
            try:
                timestamp = datetime.strptime(record["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                timestamp = timestamp.replace(tzinfo=ZoneInfo(local_timezone)).astimezone(timezone.utc)
                rows.append({"utc": timestamp, "memory_mib": float(record["memory.used [MiB]"].split()[0]),
                             "utilization_percent": float(record["utilization.gpu [%]"].split()[0])})
            except (ValueError, KeyError):
                continue
    return sorted(rows, key=lambda row: row["utc"])


def allocation_summary(rows):
    if not rows:
        return {"samples": 0}
    memory = [row["memory_mib"] for row in rows]
    return {"samples": len(rows), "start_utc": rows[0]["utc"].isoformat(),
            "end_utc": rows[-1]["utc"].isoformat(), "memory_min_mib": min(memory),
            "memory_max_mib": max(memory), "allocation_range_mib": max(memory) - min(memory),
            "max_sample_gap_seconds": max(((b["utc"] - a["utc"]).total_seconds()
                                           for a, b in zip(rows, rows[1:])), default=0),
            "utilization_percent_mean": statistics.mean(row["utilization_percent"] for row in rows)}


def phase_window(raw, path):
    if "wall_utc" in raw:
        return {"method": "reported UTC phase timestamps",
                "build_start": raw["wall_utc"]["build_start"], "build_end": raw["wall_utc"]["build_end"]}
    written = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    qa_end = written - timedelta(seconds=raw.get("recording_flush_seconds_excluded", 0))
    build_end = qa_end - timedelta(seconds=raw["qa_seconds"])
    build_start = build_end - timedelta(seconds=raw["build_seconds"])
    return {"method": "approximate: original JSON mtime minus recording flush, QA and build durations; unmeasured serialization/scheduling gaps possible",
            "original_report_mtime_utc": written.isoformat(),
            "build_start": build_start.isoformat(), "build_end": build_end.isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--timezone", required=True, help="Timezone of nvidia-smi CSV timestamps")
    parser.add_argument("--transition-mib", type=float, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.suite.read_text())
    directory = args.suite.parent
    cells = []
    stable = []
    for cell in report["cells"]:
        csv_path = directory / f"{cell['name']}-gpu.csv"
        rows = gpu_rows(csv_path, args.gpu_index, args.timezone)
        allocation = allocation_summary(rows)
        classification = ("unavailable" if not rows else "resource-transition-confounded"
                          if allocation["allocation_range_mib"] > args.transition_mib else "allocation-stable")
        result = {"name": cell["name"], "status": cell["status"], "classification": classification,
                  "whole_cell": allocation, "arms": {}}
        if csv_path.exists():
            result["gpu_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        for arm in ("baseline", "optimized", "fused", "joint"):
            raw_path = directory / cell["name"] / f"{arm}.json"
            if not raw_path.exists():
                continue
            raw = json.loads(raw_path.read_text())
            window = phase_window(raw, raw_path)
            start = datetime.fromisoformat(window["build_start"])
            end = datetime.fromisoformat(window["build_end"])
            result["arms"][arm] = {"report_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                                    "window": window,
                                    "build_gpu_samples": allocation_summary([row for row in rows if start <= row["utc"] <= end])}
        cells.append(result)
        if cell["status"] == "complete" and classification == "allocation-stable":
            stable.append(cell)
    result = {"gpu_index": args.gpu_index, "gpu_timestamp_timezone": args.timezone,
              "allocation_transition_threshold_mib": args.transition_mib,
              "interpretation": "All completed pairs remain the primary aggregate. Sensitivity uses cells whose whole-cell observed target GPU allocation range is <= threshold. This does not establish constant utilization or remove all confounding. Missing/sparse logs limit classification; inspect sample counts/gaps. Legacy arm timing estimates depend on original filesystem mtimes, preserved here with source hashes.",
              "cells": cells, "all_completed_aggregate": aggregate(report["cells"]),
              "allocation_stable_cell_names": [cell["name"] for cell in stable],
              "allocation_stable_aggregate": aggregate(stable)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
