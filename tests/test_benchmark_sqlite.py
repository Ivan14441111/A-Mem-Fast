"""Error provenance regressions with real SQLite errors and offline clients."""
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import benchmark_suite as suite
import fast_amem.client as public_client

client = public_client

def real_sqlite_error(kind):
    connection = sqlite3.connect(":memory:")
    try:
        if kind == "ProgrammingError": connection.close()
        connection.execute("SELECT missing FROM nonexistent_table")
    except sqlite3.Error as exc:
        return exc
    finally:
        connection.close()
    raise AssertionError("Expected a real SQLite exception")


@pytest.mark.parametrize("kind", ["OperationalError", "ProgrammingError"])
def test_real_sqlite_failure_artifact_keeps_origin_and_aborts(tmp_path, monkeypatch, kind):
    exception = real_sqlite_error(kind)
    def fail(*args): raise exception
    monkeypatch.setattr(benchmark, "_run_mode", fail)
    with pytest.raises(sqlite3.Error):
        benchmark.run_mode(SimpleNamespace(output=tmp_path), {"fingerprint": "subset"}, "baseline", None)
    detail = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert detail["exception_module"] == "sqlite3" and detail["exception_type"] == kind
    assert suite.is_integrity_failure([detail])


def test_real_sqlite_cache_read_error_records_client_provenance(tmp_path):
    cache = client.SQLiteResponseCache(tmp_path / "broken.sqlite")
    cache.connection().execute("DROP TABLE responses")
    controller = client.MeasuredClient("http://mock.invalid/v1", "model", cache=cache,
                                       cache_mode="readwrite", revision="weights-template")
    controller._session = lambda: (_ for _ in ()).throw(AssertionError("No HTTP is permitted"))
    with pytest.raises(sqlite3.OperationalError): controller.get_completion("cached request")
    event = controller.events[-1]
    assert event["error_module"] == "sqlite3" and event["error_type"] == "OperationalError"
    assert event["http_requests"] == 0
    assert suite.is_integrity_failure([{"mode": "baseline", "exception_type": "RuntimeError",
        "exception_module": "builtins", "error": "Configured robust arm encountered an LLM error; fallback is excluded",
        "llm_diagnostic": [event]}])


def test_actual_baseline_wrapper_preserves_sqlite_origin(tmp_path, monkeypatch):
    import memory_layer
    import numpy as np
    import requests

    class Encoder:
        def encode(self, texts): return np.asarray([[1., 2., 3.] for text in texts], dtype=np.float32)
    monkeypatch.setattr(memory_layer, "SentenceTransformer", lambda *a, **kw: Encoder())
    monkeypatch.setattr(public_client, "MeasuredClient", client.MeasuredClient)
    failure = real_sqlite_error("ProgrammingError")
    class FailingRecorder(benchmark.BufferedRecordingCache):
        def put(self, *args): raise failure
    monkeypatch.setattr(benchmark, "BufferedRecordingCache", FailingRecorder)
    def post(session, url, **kwargs):
        body = {"keywords": ["test", "memory", "topic"], "context": "A fixture.", "tags": ["test", "memory", "topic"]}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": json.dumps(body)}, "finish_reason": "stop"}]})
    monkeypatch.setattr(requests.Session, "post", post)
    args = SimpleNamespace(output=tmp_path, prefix_cache_reset="off", base_url="http://mock.invalid/v1",
                           model="model", model_revision="weights-template", embedding_model="mock",
                           max_tokens=2000, seed=42, repetition_penalty=None, tag_budget=None,
                           tag_budget_policy="error", structured_policy_outputs=False, evo_threshold=100,
                           evolution_policy="upstream", context_policy="upstream", workers=1, prefetch_window=1)
    subset = {"fingerprint": "subset", "turns": [{"content": "One memory", "timestamp": "2024-01-01", "dia_id": "D1:1"}], "questions": []}
    with pytest.raises(RuntimeError, match="Configured robust arm encountered an LLM error"):
        benchmark.run_mode(args, subset, "baseline", None)
    detail = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert detail["exception_type"] == "RuntimeError" and detail["exception_module"] == "builtins"
    diagnostic = detail["llm_diagnostic"][0]
    assert diagnostic["error_type"] == "ProgrammingError" and diagnostic["error_module"] == "sqlite3"
    assert suite.is_integrity_failure([detail])


@pytest.mark.parametrize("detail", [
    {"mode": "optimized", "exception_type": "OperationalError", "error": "database is locked"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "wrapper",
     "llm_diagnostic": [{"error": "ProgrammingError: Cannot operate on a closed database."}]},
    {"mode": "optimized", "exception_type": "Error", "exception_module": "sqlite3", "error": "base SQLite error"},
    {"mode": "optimized", "exception_type": "OperationalError", "exception_module": "_sqlite3", "error": "database is locked"},
])
def test_specific_legacy_and_explicit_sqlite_records_are_fatal(detail):
    assert suite.is_integrity_failure([detail])


@pytest.mark.parametrize("detail", [
    {"mode": "optimized", "exception_type": "OperationalError", "exception_module": "model_runtime", "error": "model failure"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "wrapper",
     "llm_diagnostic": [{"error_type": "OperationalError", "error_module": "model_runtime", "error": "OperationalError: model failure"}]},
    {"mode": "optimized", "exception_type": "Error", "error": "generic unknown error"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "wrapper", "llm_diagnostic": [{"error": "Error: generic unknown error"}]},
    {"mode": "optimized", "exception_type": "JSONDecodeError", "exception_module": "json.decoder", "error": "Expecting value"},
    {"mode": "fused", "exception_type": "ValueError", "exception_module": "builtins", "error": "fused evolution must return a JSON object"},
    {"mode": "baseline", "exception_type": "HTTPError", "exception_module": "requests.exceptions", "error": "server error"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "generation failed",
     "llm_diagnostic": [{"content_prefix": "OperationalError: database is locked", "content_suffix": "sqlite3 ProgrammingError"}]},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "wrapper",
     "llm_diagnostic": [{"error": "ValueError: model mentioned OperationalError: database is locked"}]},
])
def test_generation_errors_non_sqlite_origin_and_generated_text_remain_nonfatal(detail):
    assert not suite.is_integrity_failure([detail])
