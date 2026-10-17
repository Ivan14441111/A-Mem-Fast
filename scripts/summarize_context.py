#!/usr/bin/env python3
"""Summarize fixed-graph context cases without model inference.

Timing uses geometric paired answer-time ratios across completed conversations;
quality gives each conversation equal weight. Resource reductions use pooled
token/byte totals. Planned and unavailable cases remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, execution = read(args.protocol), read(args.execution)
    statuses = {case["sample_index"]: case for case in execution["cases"]}
    rows, cases, unique_qa = [], [], set()
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.protocol, args.execution)}
    for case in protocol["cases"]:
        sample = case["sample_index"]
        status = statuses.get(sample, {"status": "not-run"})["status"]
        cases.append({"sample_index": sample, "source_run": case["source_run"], "status": status})
        if status != "complete":
            continue
        command = case["command"]
        directory = Path(command[command.index("--output") + 1])
        summary_path, manifest_path = directory / "summary.json", directory / "manifest.json"
        summary, manifest = read(summary_path), read(manifest_path)
        if manifest["graph_replay_llm"]["http_requests"] != 0:
            raise ValueError("Graph replay unexpectedly made HTTP requests")
        for path in (summary_path, manifest_path):
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for question in manifest["context_plan"]:
            unique_qa.add((sample, question["source_qa_index"]))
        for policy, run in summary["runs"].items():
            if run["memory_digest"] != manifest["fixed_memory_digest"]:
                raise ValueError("Context arms did not use the fixed graph")
            rows.append({"sample_index": sample, "context_policy": policy,
                         "qa_seconds": run["qa_seconds"], "metrics": run["metrics"],
                         "http_requests": run["answer_llm"]["http_requests"],
                         "prompt_tokens": run["answer_llm"]["prompt_tokens_live"],
                         "completion_tokens": run["answer_llm"]["completion_tokens_live"],
                         "context_rows": run["context_rows"], "context_utf8_bytes": run["context_utf8_bytes"]})
    aggregates = {}
    completed = [case["sample_index"] for case in cases if case["status"] == "complete"]
    if completed:
        for policy in ("upstream", "deduplicated"):
            selected = [row for row in rows if row["context_policy"] == policy]
            aggregates[policy] = {"conversations": len(selected),
                                  "qa_seconds_sum": sum(row["qa_seconds"] for row in selected),
                                  **{key + "_sum": sum(row[key] for row in selected)
                                     for key in ("http_requests", "prompt_tokens", "completion_tokens", "context_rows", "context_utf8_bytes")},
                                  **{key + "_equal_conversation_mean": statistics.mean(row["metrics"][key] for row in selected)
                                     for key in ("token_f1", "exact_match")}}
        original, dedup = aggregates["upstream"], aggregates["deduplicated"]
        paired = []
        for sample in completed:
            by_policy = {row["context_policy"]: row for row in rows if row["sample_index"] == sample}
            paired.append(by_policy["upstream"]["qa_seconds"] / by_policy["deduplicated"]["qa_seconds"])
        aggregates["comparison"] = {"answer_speedup_equal_conversation_geomean": math.exp(statistics.mean(math.log(value) for value in paired)),
                                     "f1_delta_equal_conversation_mean": dedup["token_f1_equal_conversation_mean"] - original["token_f1_equal_conversation_mean"],
                                     **{key + "_pooled_reduction": 1 - dedup[key + "_sum"] / original[key + "_sum"]
                                        for key in ("prompt_tokens", "context_rows", "context_utf8_bytes")}}
    result = {"aggregation": __doc__, "planned_cases": len(cases), "completed_cases": len(completed),
              "unique_selected_qa_in_completed_cases": len(unique_qa), "cases": cases, "rows": rows,
              "aggregates": aggregates, "input_sha256": inputs,
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
