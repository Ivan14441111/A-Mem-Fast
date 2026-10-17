#!/usr/bin/env python3
"""Run the prerecorded three-control structured policy gate, exactly one POST each."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--server-config-audit", type=Path, help="Optional saved backend configuration audit")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    from fast_amem.client import MeasuredClient
    from fast_amem.structured import policy_response_format
    from fast_amem.joint import JOINT_NOTE_PROMPT, parse_joint_note
    from fast_amem.evolution import parse_fused_evolution
    from llm_text_parsers import parse_analyze_content
    from scripts.benchmark import project_revision
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    subset = json.loads((args.run_dir / "subset.json").read_text())
    config = manifest["args"]
    key = "a216cfc4f1cf3f395e7692acbafdbe137a00d170797567ef2f1452721307801b"
    with sqlite3.connect((args.run_dir / "fused.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as connection:
        payload = connection.execute("SELECT payload FROM responses WHERE key=?", (key,)).fetchone()
    if payload is None:
        raise RuntimeError("Prerecorded failed fused request is unavailable")
    old = json.loads(payload[0])
    first, second = subset["turns"][:2]
    with sqlite3.connect((args.run_dir / "baseline.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as connection:
        matches = []
        for analysis_key, identity, response in connection.execute("SELECT key,payload,response FROM responses"):
            prompt = json.loads(identity)["request"]["messages"][-1]["content"]
            if prompt.startswith("Analyze the following content") and prompt.endswith(first["content"]):
                matches.append((analysis_key, json.loads(response)))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one frozen first-turn baseline analysis response")
    analysis_key, body = matches[0]
    metadata = parse_analyze_content(body["choices"][0]["message"]["content"], first["content"])
    neighbors = ("memory index:0\t talk start time:" + first["timestamp"] +
                 "\t memory content: " + first["content"] +
                 "\t memory context: " + metadata["context"] +
                 "\t memory keywords: " + str(metadata["keywords"]) +
                 "\t memory tags: " + str(metadata["tags"]) + "\n")
    controls = [
        {"name": "known_fused_failure", "stage": "evolution_fused", "source_key": key,
         "allowed_indices": list(dict.fromkeys(int(value) for value in re.findall(r"memory index:(\d+)", old["request"]["messages"][-1]["content"]))),
         "prompt": old["request"]["messages"][-1]["content"]},
        {"name": "joint_empty", "stage": "joint_note", "allowed_indices": [],
         "prompt": JOINT_NOTE_PROMPT.format(content=first["content"], neighbors="", allowed_indices=[])},
        {"name": "joint_one_frozen_neighbor", "stage": "joint_note", "allowed_indices": [0],
         "source_analysis_key": analysis_key, "fixed_neighbor_metadata": metadata,
         "fixed_neighbor_text": neighbors,
         "prompt": JOINT_NOTE_PROMPT.format(content=second["content"], neighbors=neighbors, allowed_indices=[0])},
    ]
    for control in controls:
        request = copy.deepcopy(old["request"])
        request["messages"][-1]["content"] = control.pop("prompt")
        request["response_format"] = policy_response_format(control["stage"])
        control["request"] = request
        control["request_sha256"] = sha(request)
        control["cache_identity_sha256"] = sha({**old, "request": request})
        control["schema_sha256"] = sha(request["response_format"])
    args.output.mkdir(parents=True)
    protocol = {"defined_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(args.run_dir),
                "source_manifest_sha256": hashlib.sha256((args.run_dir / "manifest.json").read_bytes()).hexdigest(),
                "preregistration": "docs/structured_policy_gate_protocol.md", "configuration": config,
                "project_commit": project_revision(), "structured_policy_outputs": True,
                "structured_backend_configuration": json.loads(args.server_config_audit.read_text()) if args.server_config_audit else {"requested_backend": "not externally verified", "selected_runtime_backend": "not asserted"},
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__).resolve(), *sorted((ROOT / "fast_amem").glob("*.py"))]},
                "http_policy": "Exactly one POST attempt per fixed control, no retries, no cache writes, no trimming or repair; no QA scoring. Counts describe client invocations, not guaranteed server receipt on connection failure.",
                "timing_scope": "Any schema/backend compilation is included in the control wall time; schema caches are not forcibly cleared. The model is already loaded. A successful gate prewarms both fixed policy schemas for any later pilot.",
                "controls": controls}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+"\n")
    if args.plan_only:
        print("Prepared exactly three fixed controls; zero HTTP.")
        return
    import requests
    results = []
    for control in controls:
        started = time.perf_counter()
        record = {"name": control["name"], "started_utc": datetime.now(timezone.utc).isoformat(),
                  "request_sha256": control["request_sha256"], "http_requests": 1, "passed": False}
        try:
            response = requests.post(old["base_url"].rstrip("/")+"/chat/completions", json=control["request"], timeout=180)
            record["http_status"] = response.status_code
            record["response"] = response.json()
            response.raise_for_status()
            content = MeasuredClient._validate_response(record["response"])
            if record["response"]["choices"][0].get("finish_reason") != "stop":
                raise ValueError("Gate requires an explicit normal stop")
            if record["response"].get("usage", {}).get("completion_tokens", 0) > 2000:
                raise ValueError("Gate exceeded the common completion-token budget")
            parsed = json.loads(content)
            if control["stage"] == "joint_note":
                parsed = parse_joint_note(content, control["allowed_indices"])
            else:
                parsed = parse_fused_evolution(content, control["allowed_indices"])
            tag_lists = []
            def visit(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key == "tags" and isinstance(child, list): tag_lists.append(len(child))
                        visit(child)
                elif isinstance(value, list):
                    for child in value: visit(child)
            visit(parsed)
            record.update({"passed": True, "parsed": parsed, "tag_list_counts": tag_lists,
                           "over_budget_lists": sum(count > 16 for count in tag_lists)})
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_seconds"] = time.perf_counter() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        results.append(record)
        report = {"passed": len(results)==3 and all(row["passed"] for row in results),
                  "attempted": len(results), "http_requests": sum(row["http_requests"] for row in results),
                  "controls": results}
        (args.output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        print(f"{control['name']}: passed={record['passed']}, wall={record['wall_seconds']:.3f}s", flush=True)
    if not all(row["passed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
