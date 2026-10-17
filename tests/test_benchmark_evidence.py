"""Evidence durability regressions using real application code and SQLite."""
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark
import benchmark_suite as suite
import fast_amem.client as client

@pytest.fixture
def offline_run(tmp_path, monkeypatch):
    import memory_layer
    import fast_amem.client
    import requests

    class Encoder:
        def encode(self, texts):
            return np.asarray([[1., 2., 3.] for text in texts], dtype=np.float32)
    monkeypatch.setattr(memory_layer, "SentenceTransformer", lambda *a, **kw: Encoder())
    monkeypatch.setattr(fast_amem.client, "MeasuredClient", client.MeasuredClient)
    state = {"analyses": 0, "answers": 0, "failure": "qa", "advance": lambda value: None}
    def post(session, url, *, json: dict, timeout):
        stage = client.stage_for(json["messages"][-1]["content"])
        state["advance"](1)
        finish = "stop"
        if stage == "analysis":
            state["analyses"] += 1
            body = {"keywords": ["test", "memory", "topic"], "context": "A fixture.", "tags": ["test", "memory", "topic"]}
            if state["failure"] == "build" and state["analyses"] == 2: finish = "length"
        elif stage == "evolution_decision": body = {"decision": "NO_EVOLUTION", "reason": "Offline fixture"}
        elif stage == "query_keywords":
            assert (tmp_path / "baseline-build-checkpoint.json").exists()
            with sqlite3.connect(tmp_path / "baseline.sqlite") as connection:
                assert connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0] >= 3
            body = {"keywords": "test memory"}
        elif stage == "answer":
            state["answers"] += 1
            body = {"answer": "test"}
            if state["failure"] == "qa" and state["answers"] == 2: finish = "length"
        else: raise AssertionError(stage)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": __import__("json").dumps(body)}, "finish_reason": finish}],
                          "usage": {"prompt_tokens": 12, "completion_tokens": 2000 if finish == "length" else 6}})
    monkeypatch.setattr(requests.Session, "post", post)
    args = SimpleNamespace(output=tmp_path, prefix_cache_reset="off", base_url="http://mock.invalid/v1",
                           model="model", model_revision="weights-template", embedding_model="mock",
                           max_tokens=2000, seed=42, repetition_penalty=None, tag_budget=None,
                           tag_budget_policy="error", structured_policy_outputs=False, evo_threshold=100,
                           evolution_policy="upstream", context_policy="upstream", workers=1, prefetch_window=1, retrieve_k=2)
    subset = {"fingerprint": "subset", "turns": [{"content": "First memory", "timestamp": "2024-01-01", "dia_id": "D1:1"},
                                                 {"content": "Second memory", "timestamp": "2024-01-02", "dia_id": "D1:2"}],
              "questions": [{"question": f"Question {i}?", "category": 1, "answer": "test", "source_qa_index": i} for i in range(2)]}
    return args, subset, state


def test_qa_failure_preserves_full_build_and_successful_partial_qa(tmp_path, offline_run):
    args, subset, state = offline_run
    with pytest.raises(ValueError, match="truncated"):
        benchmark.run_mode(args, subset, "baseline", None)
    build = json.loads((tmp_path / "baseline-build-checkpoint.json").read_text())
    partial = json.loads((tmp_path / "baseline-partial.json").read_text())
    failure = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert build["build_complete"] and not build["qa_complete"]
    assert build["last_completed_turn"] == len(build["memory_trace"]) == build["memory_count"] == 2
    assert build["recording_pending_responses"] == 0 and build["qa"] == []
    assert build["source_identity"]["source_sha256"] and build["source_identity"]["configuration"]["model_revision"] == "weights-template"
    assert partial["build_seconds"] == build["build_seconds"] >= 0
    assert partial["memory_trace"] == build["memory_trace"] and partial["memory_digest"] == build["memory_digest"]
    assert partial["phase"] == "qa" and len(partial["qa"]) == 1 and partial["qa_seconds_partial"] >= 0
    assert failure["progress"]["question_index_0based"] == 1
    assert failure["llm_diagnostic"]["finish_reason"] == "length"
    with sqlite3.connect(tmp_path / "baseline.sqlite") as connection:
        keys = {row[0] for row in connection.execute("SELECT key FROM responses")}
    failed_key = next(event["key"] for event in partial["llm_events"] if event["error"])
    assert set(build["upstream_request_keys"]) <= keys and len(keys) == 6
    assert failed_key not in keys
    assert not (tmp_path / "baseline.json").exists()


def test_build_failure_labels_current_state_separately_from_completed_prefix(tmp_path, offline_run):
    args, subset, state = offline_run
    state["failure"] = "build"
    with pytest.raises(RuntimeError, match="Configured robust arm encountered an LLM error"):
        benchmark.run_mode(args, subset, "baseline", None)
    partial = json.loads((tmp_path / "baseline-partial.json").read_text())
    assert partial["last_completed_turn"] == len(partial["memory_trace"]) == 1
    assert partial["memory_count"] == 2  # upstream committed a fallback note before its error was detected
    assert partial["memory_state_status"] == "current_state_at_failure"
    assert not partial["build_complete"] and partial["build_seconds"] is None
    assert partial["build_seconds_partial"] >= 0
    assert not (tmp_path / "baseline-build-checkpoint.json").exists()


def test_flush_failure_does_not_write_successful_build_checkpoint(tmp_path, offline_run, monkeypatch):
    args, subset, state = offline_run
    def fail_flush(cache): raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(benchmark.BufferedRecordingCache, "flush", fail_flush)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        benchmark.run_mode(args, subset, "baseline", None)
    assert not (tmp_path / "baseline-build-checkpoint.json").exists()
    failure = json.loads((tmp_path / "baseline-failure.json").read_text())
    partial = json.loads((tmp_path / "baseline-partial.json").read_text())
    assert partial["recording_pending_responses"] == 3
    assert failure["evidence_errors"][0]["operation"] == "flush_response_cache"
    assert suite.is_integrity_failure([failure])


def test_secondary_flush_error_preserves_primary_truncation_and_diagnostic(tmp_path, offline_run, monkeypatch):
    args, subset, state = offline_run
    original = benchmark.BufferedRecordingCache.flush
    flushed = []
    def fail_second(cache):
        flushed.append(True)
        if len(flushed) > 1: raise sqlite3.OperationalError("disk I/O error")
        return original(cache)
    monkeypatch.setattr(benchmark.BufferedRecordingCache, "flush", fail_second)
    with pytest.raises(ValueError, match="truncated"):
        benchmark.run_mode(args, subset, "baseline", None)
    failure = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert failure["exception_type"] == "ValueError" and failure["llm_diagnostic"]["finish_reason"] == "length"
    assert failure["evidence_errors"][0]["exception_type"] == "OperationalError"
    assert suite.is_integrity_failure([failure])
    assert (tmp_path / "baseline-build-checkpoint.json").exists()


def test_secondary_partial_write_error_is_fatal_without_masking_primary(tmp_path, offline_run, monkeypatch):
    args, subset, state = offline_run
    original = benchmark.write_json_atomic
    def fail_partial(path, value):
        if path.name.endswith("-partial.json"): raise PermissionError("checkpoint denied")
        return original(path, value)
    monkeypatch.setattr(benchmark, "write_json_atomic", fail_partial)
    with pytest.raises(ValueError, match="truncated"):
        benchmark.run_mode(args, subset, "baseline", None)
    failure = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert failure["evidence_errors"][0]["exception_type"] == "PermissionError"
    assert suite.is_integrity_failure([failure])
    assert (tmp_path / "baseline-build-checkpoint.json").exists()


def test_build_checkpoint_replace_failure_keeps_cache_and_is_fatal(tmp_path, offline_run, monkeypatch):
    args, subset, state = offline_run
    original = benchmark.os.replace
    def fail_build(source, destination):
        if str(destination).endswith("-build-checkpoint.json"): raise OSError("checkpoint replace failed")
        return original(source, destination)
    monkeypatch.setattr(benchmark.os, "replace", fail_build)
    with pytest.raises(OSError, match="checkpoint replace failed"):
        benchmark.run_mode(args, subset, "baseline", None)
    assert not (tmp_path / "baseline-build-checkpoint.json").exists()
    assert not (tmp_path / ".baseline-build-checkpoint.json.tmp").exists()
    with sqlite3.connect(tmp_path / "baseline.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 3
    failure = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert failure["operation"] == "write_build_checkpoint"
    assert suite.is_integrity_failure([failure])


def test_failure_artifact_write_error_does_not_mask_original_exception(tmp_path, offline_run, monkeypatch):
    args, subset, state = offline_run
    original = benchmark.write_json_atomic
    def fail_artifact(path, value):
        if path.name.endswith("-failure.json"): raise PermissionError("failure artifact denied")
        return original(path, value)
    monkeypatch.setattr(benchmark, "write_json_atomic", fail_artifact)
    with pytest.raises(ValueError, match="truncated") as caught:
        benchmark.run_mode(args, subset, "baseline", None)
    assert caught.value.llm_diagnostic["finish_reason"] == "length"
    assert caught.value.evidence_errors[-1]["operation"] == "write_failure_artifact"
    assert (tmp_path / "baseline-build-checkpoint.json").exists()
    assert (tmp_path / "baseline-partial.json").exists()
    assert not (tmp_path / "baseline-failure.json").exists()
    assert suite.is_integrity_failure([])


def test_arbitrary_secondary_generation_message_is_not_storage_provenance():
    detail = {"mode": "fused", "exception_type": "ValueError", "error": "generation failed",
              "evidence_errors": [{"operation": "model_generation", "exception_type": "OSError",
                                    "exception_module": "builtins", "error": "database is locked"}]}
    assert not suite.is_integrity_failure([detail])


@pytest.mark.parametrize("operation", ["replace", "fsync"])
def test_atomic_write_failure_retains_previous_file_and_cleans_temp(tmp_path, monkeypatch, operation):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"previous":"complete"}')
    def fail(*args): raise OSError("simulated storage failure")
    monkeypatch.setattr(benchmark.os, operation, fail)
    with pytest.raises(OSError): benchmark.write_json_atomic(path, {"replacement": "incomplete"})
    assert json.loads(path.read_text()) == {"previous": "complete"}
    assert not (tmp_path / ".checkpoint.json.tmp").exists()


def test_flush_keeps_duplicate_response_consistency_across_boundaries(tmp_path):
    cache = benchmark.BufferedRecordingCache(client.SQLiteResponseCache(tmp_path / "responses.sqlite"))
    def response(text): return {"choices": [{"message": {"content": text}}]}
    cache.put("key", "payload", response("first"))
    cache.flush()
    assert cache.rows == {}
    with pytest.raises(RuntimeError, match="ambiguous"):
        cache.put("key", "payload", response("different"))


def test_successful_report_excludes_flush_and_checkpoint_time(tmp_path, offline_run, monkeypatch):
    args, subset, state = offline_run
    state["failure"] = None
    clock = [0.]
    state["advance"] = lambda value: clock.__setitem__(0, clock[0] + value)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    original_flush, original_write = benchmark.BufferedRecordingCache.flush, benchmark.write_json_atomic
    def slow_flush(cache):
        state["advance"](100)
        return original_flush(cache)
    def slow_write(path, value):
        state["advance"](200)
        return original_write(path, value)
    monkeypatch.setattr(benchmark.BufferedRecordingCache, "flush", slow_flush)
    monkeypatch.setattr(benchmark, "write_json_atomic", slow_write)
    report = benchmark.run_mode(args, subset, "baseline", None)
    assert report["build_seconds"] == 3 and report["qa_seconds"] == 4
    assert report["total_seconds"] == 7 and report["recording_flush_seconds_excluded"] == 200
    assert report["memory_count"] == 2 and len(report["qa"]) == 2
    assert json.loads((tmp_path / "baseline.json").read_text())["metrics"] == report["metrics"]
