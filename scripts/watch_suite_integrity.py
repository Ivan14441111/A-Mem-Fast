#!/usr/bin/env python3
"""Supervise one owned benchmark runner without changing measured code.

On source drift, ambiguous baseline responses, or explicit replay/identity
errors, terminate only the verified runner and its descendants. Never manages
the inference server or other GPU processes. Normal policy failures continue
according to the frozen suite protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
SQLITE_ERRORS = frozenset({"DatabaseError", "OperationalError", "IntegrityError", "ProgrammingError",
                          "InternalError", "InterfaceError", "DataError", "NotSupportedError"})
PATTERNS = (
    "Same baseline request returned different text; exact cache replay is ambiguous",
    "Replay cache miss:", "Semantic replay mismatch:",
    "Existing output belongs to a different subset or model configuration",
    "Stored report has a different subset fingerprint",
    "Benchmark sources changed during suite",
    "Source files changed during cell",
)


def command(pid):
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return b""


def error_fields(value):
    """Read only typed error fields, never generated content previews/bodies."""
    if isinstance(value, list):
        return [message for item in value for message in error_fields(item)]
    if not isinstance(value, dict):
        return []
    errors = [value["error"]] if isinstance(value.get("error"), str) else []
    if "llm_diagnostic" in value:
        errors.extend(error_fields(value["llm_diagnostic"]))
    return errors


def owned_descendants(pid):
    parents = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            # The comm field can contain spaces, so split after its final ')'.
            tail = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(directory.name)] = int(tail[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
            continue
    found = {pid}
    while True:
        extended = found | {child for child, parent in parents.items() if parent in found}
        if extended == found:
            return sorted(found - {pid})
        found = extended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh supervision record")
    initial = command(args.pid)
    if (Path("/proc") / str(args.pid)).stat().st_uid != os.getuid():
        parser.error("Runner must belong to the current user")
    argv = [part.decode(errors="replace") for part in initial.split(b"\0") if part]
    if not any(Path(arg).name in {"benchmark.py", "benchmark_suite.py"} for arg in argv) or "--output" not in argv:
        parser.error("PID is not an identifiable benchmark runner")
    try:
        cwd = (Path("/proc") / str(args.pid) / "cwd").resolve()
        run_output = (cwd / argv[argv.index("--output") + 1]).resolve()
        if not run_output.is_relative_to(args.suite.parent.resolve()):
            raise ValueError("Runner output does not belong to this suite")
    except (IndexError, OSError, ValueError) as exc:
        parser.error(str(exc))
    expected = json.loads(args.suite.read_text())["source_sha256"]
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "runner_pid": args.pid,
              "runner_argv_sha256": hashlib.sha256(initial).hexdigest(),
              "source_suite": str(args.suite), "poll_seconds": 1,
              "supervisor_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "patterns": PATTERNS, "expected_source_sha256": expected,
              "fatal_sqlite_exception_types": sorted(SQLITE_ERRORS),
              "inspection_scope": "Failure JSON top-level error and nested llm_diagnostic[*].error fields only; generated content_prefix/content_suffix/body are never searched. Every replay-mode failure is fatal.",
              "termination_scope": "Only verified benchmark runner PID and its same-user descendants; server and other GPU jobs are outside this tree.",
              "global_stop_required": False, "status": "watching"}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    save()
    while command(args.pid) == initial:
        trigger = None
        current = {}
        for name in expected:
            try:
                current[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            except OSError as exc:
                current[name] = "read_error:" + type(exc).__name__
        if current != expected:
            trigger = {"kind": "source_drift", "current_source_sha256": current}
        else:
            for path in sorted(args.suite.parent.glob("*/*-failure.json")):
                try:
                    failure = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue  # The writer can be between open and close.
                if failure.get("mode") in {"replay", "warm", "fused_warm", "joint_warm"}:
                    trigger = {"kind": "replay_failure", "artifact": str(path)}
                    break
                messages = error_fields(failure)
                if failure.get("exception_type") in SQLITE_ERRORS or any(
                        message.startswith(error + ":") for message in messages for error in SQLITE_ERRORS):
                    trigger = {"kind": "sqlite_persistence_failure", "artifact": str(path)}
                    break
                match = next((pattern for message in messages for pattern in PATTERNS if pattern in message), None)
                if match:
                    trigger = {"kind": "explicit_integrity_error", "pattern": match, "artifact": str(path)}
                    break
        if trigger:
            descendants = owned_descendants(args.pid)
            child_commands = {pid: command(pid) for pid in descendants}
            # Verify root identity again before any signal, guarding PID reuse.
            if command(args.pid) != initial:
                break
            record.update({"status": "integrity_stop", "global_stop_required": True,
                           "trigger": trigger, "signaled_pids": []})
            save()
            for pid in [args.pid, *reversed(descendants)]:
                try:
                    if pid != args.pid and (not child_commands[pid] or command(pid) != child_commands[pid]):
                        continue
                    os.kill(pid, signal.SIGTERM)
                    record["signaled_pids"].append(pid)
                except ProcessLookupError:
                    pass
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()
            print(json.dumps(trigger), flush=True)
            raise SystemExit(2)
        time.sleep(1)
    record.update({"status": "runner_exited", "finished_utc": datetime.now(timezone.utc).isoformat()})
    save()
    print(args.output)


if __name__ == "__main__":
    main()
