#!/usr/bin/env python3
"""Run isolated, paired LoCoMo experiments with explicit coverage and arm order.

The suite runs one process at a time against a shared server. Timing is paired
within a conversation/repeat, never across unrelated cells. Replay equivalence
and live-model quality are reported separately; fused evolution changes prompts.
Use --plan-only to inspect evidence-complete QA coverage without loading models.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

from benchmark import ROOT, digest, project_revision, select_subset


def distribution(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def aggregate(cells):
    groups = {}
    expanded = []
    for cell in cells:
        expanded.append(cell)
        for alternate in ("fused", "joint"):
            if cell.get("status") == "complete" and f"baseline_vs_{alternate}" in cell["summary"]["comparisons"]:
                alternate_summary = {**cell["summary"],
                    "runs": {**cell["summary"]["runs"], "optimized": cell["summary"]["runs"][alternate]},
                    "comparisons": {**cell["summary"]["comparisons"],
                        "baseline_vs_optimized": cell["summary"]["comparisons"][f"baseline_vs_{alternate}"],
                        "optimized_vs_warm": cell["summary"]["comparisons"][f"{alternate}_vs_{alternate}_warm"]}}
                expanded.append({**cell, "evolution_policy": alternate, "summary": alternate_summary})
    for cell in expanded:
        if cell.get("status") != "complete":
            continue
        key = f"turns={cell['max_turns']}/policy={cell['evolution_policy']}"
        group = groups.setdefault(key, [])
        summary = cell["summary"]
        comparison = summary["comparisons"]["baseline_vs_optimized"]
        group.append((cell, summary, comparison))
    result = {}
    for key, rows in groups.items():
        speedups = [comparison["total_speedup"] for _, _, comparison in rows]
        result[key] = {
            "completed_pairs": len(rows),
            "conversation_indices": sorted({cell["sample_index"] for cell, _, _ in rows}),
            "arm_orders": dict(Counter(cell["modes"] for cell, _, _ in rows)),
            "build_speedup": distribution([c["build_speedup"] for _, _, c in rows]),
            "total_speedup": distribution(speedups),
            "total_speedup_geomean": math.exp(statistics.mean(math.log(x) for x in speedups)),
            "f1_delta": distribution([c["f1_delta"] for _, _, c in rows]),
            "exact_live_memory_trace_pairs": sum(c["memory_trace_equal"] for _, _, c in rows),
            "exact_live_qa_pairs": sum(c["qa_equal"] for _, _, c in rows),
            "replay_validated_pairs": sum(all(
                all(summary["comparisons"].get(pair, {}).get(field, False)
                    for field in ("memory_trace_equal", "upstream_requests_equal", "qa_equal"))
                for pair in ("baseline_vs_replay", "optimized_vs_warm"))
                for _, summary, _ in rows),
            "baseline_f1": distribution([s["runs"]["baseline"]["metrics"]["token_f1"] for _, s, _ in rows]),
            "optimized_f1": distribution([s["runs"]["optimized"]["metrics"]["token_f1"] for _, s, _ in rows]),
            "baseline_http_requests": distribution([s["runs"]["baseline"]["all_llm"]["http_requests"] for _, s, _ in rows]),
            "optimized_http_requests": distribution([s["runs"]["optimized"]["all_llm"]["http_requests"] for _, s, _ in rows]),
            "tokens_live": {arm: {stage: {token: distribution([
                sum(values.get(token, 0) for values in summary["runs"][arm][stage]["by_stage"].values())
                for _, summary, _ in rows]) for token in ("prompt_tokens_live", "completion_tokens_live")}
                for stage in ("build_llm", "all_llm")} for arm in ("baseline", "optimized")},
        }
    return result


def build_plan(args):
    rng = random.Random(args.order_seed)
    cells = []
    skipped = []
    pair_index = 0
    for turns in args.turns:
        for sample_index in args.samples:
            try:
                subset = select_subset(args.dataset, sample_index, turns, args.max_questions)
            except (ValueError, IndexError) as exc:
                skipped.append({"sample_index": sample_index, "max_turns": turns, "reason": str(exc)})
                continue
            for policy in args.policies:
                for repeat in range(args.repeats):
                    # AB/BA by repeat, with alternating starting order by cell.
                    reverse = bool((pair_index + repeat) % 2)
                    if args.order == "random":
                        reverse = bool(rng.randrange(2))
                    elif args.order == "baseline-first":
                        reverse = False
                    modes = "optimized,baseline,replay,warm" if reverse else "baseline,optimized,replay,warm"
                    if args.include_fused:
                        arms = ["baseline", "optimized", "fused"]
                        if args.order == "random":
                            rng.shuffle(arms)
                        elif args.order == "counterbalanced":
                            rotation = (pair_index + repeat) % 3
                            arms = arms[rotation:] + arms[:rotation]
                        modes = ",".join(arms + ["replay", "warm", "fused_warm"])
                    if getattr(args, "include_joint", False):
                        arms = ["baseline", "optimized", "joint", "fused"]
                        if args.order == "random":
                            rng.shuffle(arms)
                        elif args.order == "counterbalanced":
                            # Four-arm Williams design; paired rows are reversals.
                            orders = [
                                ["baseline", "optimized", "joint", "fused"],
                                ["optimized", "fused", "baseline", "joint"],
                                ["fused", "joint", "optimized", "baseline"],
                                ["joint", "baseline", "fused", "optimized"],
                            ]
                            arms = orders[(pair_index + 2 * repeat) % 4]
                        modes = ",".join(arms + ["replay", "warm", "fused_warm", "joint_warm"])
                    name = f"turns{turns}-sample{sample_index}-{policy}-r{repeat + 1}"
                    cells.append({"name": name, "sample_index": sample_index, "sample_id": subset["sample_id"],
                                  "max_turns": turns, "actual_turns": len(subset["turns"]),
                                  "question_count": len(subset["questions"]),
                                  "eligible_qa": subset["eligible_qa"], "category_counts": subset["category_counts"],
                                  "source_qa_indices": [q["source_qa_index"] for q in subset["questions"]],
                                  "subset_fingerprint": digest(subset), "evolution_policy": policy,
                                  "repeat": repeat + 1, "modes": modes, "status": "pending"})
                pair_index += 1
    return {"schema_version": 1, "args": vars(args), "cells": cells, "skipped_subsets": skipped,
            "interpretation": "Each speedup is a within-cell live baseline/optimized ratio. Repeats reuse questions and are not independent conversations. Fused changes prompts; live equality is diagnostic, not guaranteed. Replay compares each policy against its own responses. Shared GPU load can affect timings; small suites do not establish a population confidence interval."}


def command_for(args, cell):
    command = [sys.executable, str(ROOT / "scripts/benchmark.py"),
               "--dataset", str(args.dataset), "--sample-index", str(cell["sample_index"]),
               "--max-turns", str(cell["max_turns"]), "--max-questions", str(args.max_questions),
               "--evolution-policy", cell["evolution_policy"], "--modes", cell["modes"],
               "--output", str(Path(args.output) / cell["name"])]
    for flag in ("base_url", "model", "model_revision", "embedding_model", "max_tokens", "seed",
                 "retrieve_k", "evo_threshold", "workers", "prefetch_window", "prefix_cache_reset", "context_policy", "repetition_penalty", "tag_budget"):
        value = getattr(args, flag)
        if value is not None:
            command.extend(["--" + flag.replace("_", "-"), str(value)])
    return command


def write_report(path, report):
    report["aggregate"] = aggregate(report["cells"])
    report["status_counts"] = dict(Counter(cell["status"] for cell in report["cells"]))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "data/locomo10.json"))
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--turns", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--policies", nargs="+", choices=("upstream", "fused"), default=["upstream"])
    parser.add_argument("--context-policy", choices=("upstream", "deduplicated"), default="upstream")
    parser.add_argument("--include-joint", action="store_true", help="Four live arms including experimental joint note construction, with Williams arm orders and each arm replay")
    parser.add_argument("--include-fused", action="store_true", help="Run three cold arms per cell with rotated order and each arm replay")
    parser.add_argument("--order", choices=("counterbalanced", "random", "baseline-first"), default="counterbalanced")
    parser.add_argument("--order-seed", type=int, default=20261018)
    parser.add_argument("--max-questions", type=int, default=24)
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--evo-threshold", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-window", type=int, default=16)
    parser.add_argument("--base-url", default="http://127.0.0.1:21014/v1")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--tag-budget", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-cache-reset", choices=("required", "off"), default="required")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip validated complete cells; failed partial directories require a fresh suite")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    args.include_fused = args.include_fused or args.include_joint
    if args.tag_budget is not None and args.tag_budget <= 0:
        parser.error("tag-budget must be positive")
    if args.repetition_penalty is not None and (not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0):
        parser.error("repetition-penalty must be finite and positive")
    if args.include_fused and args.policies != ["upstream"]:
        parser.error("--include-fused requires --policies upstream (the default)")
    if min(args.turns + [args.repeats, args.max_questions, args.workers, args.prefetch_window,
                         args.max_tokens, args.retrieve_k, args.evo_threshold]) < 1 or min(args.samples) < 0:
        parser.error("Sizes/repeats/workers must be positive and samples nonnegative")
    if not args.plan_only and (not args.model_revision or not args.embedding_model):
        parser.error("Live runs require --model-revision and --embedding-model")
    for name in ("samples", "turns", "policies"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"Duplicate {name} would reuse output directories")
    report = build_plan(args)
    output = Path(args.output)
    path = output / "suite.json"
    identity_args = {key: value for key, value in vars(args).items()
                     if key not in {"resume", "keep_going", "plan_only"}}
    report["configuration_fingerprint"] = digest(identity_args)
    report["project_commit"] = project_revision()
    report["source_sha256"] = {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
                               for file in [ROOT / "scripts/benchmark_suite.py", ROOT / "scripts/benchmark.py",
                                            ROOT / "memory_layer.py", ROOT / "memory_layer_robust.py",
                                            ROOT / "llm_text_parsers.py", *sorted((ROOT / "fast_amem").glob("*.py"))]}
    if path.exists():
        old = json.loads(path.read_text())
        if not args.resume:
            parser.error("Suite output exists; use --resume or a fresh output")
        if old["configuration_fingerprint"] != report["configuration_fingerprint"]:
            parser.error("Existing suite configuration differs")
        if old["source_sha256"] != report["source_sha256"]:
            parser.error("Source files changed; use a new output directory")
        old_cells = {cell["name"]: cell for cell in old["cells"]}
        for index, cell in enumerate(report["cells"]):
            previous = old_cells.get(cell["name"], {})
            if previous.get("status") == "complete":
                report["cells"][index] = previous
    output.mkdir(parents=True, exist_ok=True)
    write_report(path, report)
    print(json.dumps({"cells": len(report["cells"]), "skipped_subsets": report["skipped_subsets"],
                      "plan": str(path), "status": report["status_counts"]}, indent=2), flush=True)
    if args.plan_only:
        return
    for cell in report["cells"]:
        if cell["status"] == "complete":
            continue
        current_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in report["source_sha256"]}
        if current_hashes != report["source_sha256"]:
            raise RuntimeError("Benchmark sources changed during suite; preserve this output and start a new suite")
        cell_output = output / cell["name"]
        if cell_output.exists():
            raise FileExistsError(f"Partial cell exists: {cell_output}; preserve it and choose a new suite output")
        print(f"Running {cell['name']}: {cell['modes']}", flush=True)
        command = command_for(args, cell)
        cell["command"] = command
        cell["status"] = "running"
        write_report(path, report)
        started = time.perf_counter()
        with (output / f"{cell['name']}.log").open("w") as log, (output / f"{cell['name']}-gpu.csv").open("w") as gpu_log:
            monitor = None
            if shutil.which("nvidia-smi"):
                monitor = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw",
                    "--format=csv", "--loop=2"], stdout=gpu_log, stderr=subprocess.STDOUT)
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            finally:
                if monitor is not None:
                    monitor.terminate()
                    try:
                        monitor.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        monitor.kill()
                        monitor.wait()
        post_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in report["source_sha256"]}
        if post_hashes != report["source_sha256"]:
            cell["status"] = "failed"
            cell["failure_reason"] = "Source files changed during cell; outputs preserved but excluded from aggregate"
            write_report(path, report)
            raise RuntimeError(cell["failure_reason"])
        cell["process_seconds"] = time.perf_counter() - started
        cell["returncode"] = process.returncode
        cell["status"] = "complete" if process.returncode == 0 else "failed"
        if process.returncode == 0:
            cell["summary"] = json.loads((cell_output / "summary.json").read_text())
        else:
            lines = (output / f"{cell['name']}.log").read_text(errors="replace").splitlines()
            cell["failure_log_tail"] = lines[-30:]
        write_report(path, report)
        print(f"{cell['name']}: {cell['status']} ({cell['process_seconds']:.1f}s)", flush=True)
        if process.returncode and not args.keep_going:
            raise SystemExit(process.returncode)
    if report["skipped_subsets"]:
        print("Some requested subsets had no evidence-complete QA; see suite.json.", flush=True)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    if any(cell["status"] == "failed" for cell in report["cells"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
