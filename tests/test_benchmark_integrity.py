"""Exact-cache integrity classification with offline application regressions."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import benchmark_suite as draft

def response(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def test_real_buffered_cache_exception_artifact_aborts_suite(tmp_path, monkeypatch):
    cache = benchmark.BufferedRecordingCache(None)
    cache.put("same-key", "same-request", response("first response"))
    def fail(*args):
        cache.put("same-key", "same-request", response("different response"))
    monkeypatch.setattr(benchmark, "_run_mode", fail)
    with pytest.raises(RuntimeError, match="exact cache replay is ambiguous"):
        benchmark.run_mode(SimpleNamespace(output=tmp_path), {"fingerprint": "subset"}, "baseline", None)
    detail = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert draft.is_integrity_failure([detail])


def test_actual_baseline_wrapper_preserves_ambiguity_as_fatal(tmp_path, monkeypatch):
    import numpy as np
    import requests
    import memory_layer

    class Encoder:
        def encode(self, texts):
            return np.asarray([[1., 2., 3.] for text in texts], dtype=np.float32)

    monkeypatch.setattr(memory_layer, "SentenceTransformer", lambda *a, **kw: Encoder())
    analyses = []
    def post(session, url, *, json, timeout):
        prompt = json["messages"][-1]["content"]
        if prompt.startswith("Analyze the following content"):
            analyses.append(prompt)
            body = {"keywords": ["robotics", "course", "test"], "tags": ["topic", "test", "memory"],
                    "context": "Different response " + str(len(analyses))}
        else:
            assert prompt.startswith("You are an AI memory evolution agent")
            body = {"decision": "NO_EVOLUTION", "reason": "Offline fixture"}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: response(__import__("json").dumps(body)))
    monkeypatch.setattr(requests.Session, "post", post)
    args = SimpleNamespace(output=tmp_path, prefix_cache_reset="off", base_url="http://mock.invalid/v1",
                           model="model", model_revision="weights-template", embedding_model="mock",
                           max_tokens=2000, seed=42, repetition_penalty=None, tag_budget=None,
                           tag_budget_policy="error", structured_policy_outputs=False, evo_threshold=100,
                           evolution_policy="upstream", context_policy="upstream", workers=1, prefetch_window=1)
    subset = {"fingerprint": "subset", "turns": [{"content": "Same robotics course", "timestamp": "2024-01-01", "dia_id": "D1:1"},
                                                  {"content": "Same robotics course", "timestamp": "2024-01-02", "dia_id": "D1:2"}], "questions": []}
    with pytest.raises(RuntimeError, match="Configured robust arm encountered an LLM error"):
        benchmark.run_mode(args, subset, "baseline", None)
    detail = json.loads((tmp_path / "baseline-failure.json").read_text())
    assert len(analyses) == 2 and analyses[0] == analyses[1]
    assert detail["progress"]["failed_turn_1based"] == 2
    assert "ambiguous" not in detail["error"]
    assert "exact cache replay is ambiguous" in detail["llm_diagnostic"][0]["error"]
    assert draft.is_integrity_failure([detail])


@pytest.mark.parametrize("detail", [
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "Configured robust arm encountered an LLM error; fallback is excluded",
     "llm_diagnostic": [{"stage": "analysis", "error": "RuntimeError: Replay cache miss: analysis key"}]},
    {"mode": "optimized", "exception_type": "RuntimeError", "error": "Replay cache miss: analysis key"},
    {"mode": "warm", "exception_type": "ValueError", "error": "LLM returned no text"},
    {"mode": "baseline", "exception_type": "TypeError", "error": "unexpected keyword argument 'tag_budget_policy'"},
])
def test_existing_replay_and_configuration_failures_still_abort(detail):
    assert draft.is_integrity_failure([detail])


@pytest.mark.parametrize("detail", [
    {"mode": "fused", "exception_type": "ValueError", "error": "fused evolution must return a JSON object"},
    {"mode": "optimized", "exception_type": "ValueError", "error": "LLM reached max_tokens; completion is truncated"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "Configured robust arm encountered an LLM error; fallback is excluded",
     "llm_diagnostic": [{"stage": "analysis", "error": "ValueError: LLM returned no text"}]},
    {"mode": "joint", "exception_type": "RuntimeError", "error": "HTTP server failed after retries"},
    {"mode": "optimized", "exception_type": "JSONDecodeError", "error": "Expecting value: line 1 column 1"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "ambiguous model decision"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "Model description mentions Replay cache miss without an actual replay lookup"},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "generation failed",
     "llm_diagnostic": {"content_prefix": "Same baseline request returned different text; exact cache replay is ambiguous",
                        "content_suffix": "Replay cache miss"}},
    {"mode": "baseline", "exception_type": "RuntimeError", "error": "generation failed", "llm_diagnostic": None},
])
def test_generation_failures_and_model_text_do_not_become_global_integrity_failures(detail):
    assert not draft.is_integrity_failure([detail])


@pytest.mark.parametrize("ambiguous", [True, False])
def test_continue_on_error_stops_only_for_integrity_failure(tmp_path, monkeypatch, ambiguous):
    cells = [{"name": f"case-{i}", "status": "pending", "sample_index": i, "max_turns": 2,
              "evolution_policy": "upstream", "modes": "baseline"} for i in range(2)]
    monkeypatch.setattr(draft, "build_plan", lambda args: {"cells": cells, "skipped_subsets": []})
    monkeypatch.setattr(draft, "project_revision", lambda: "offline-fixture")
    monkeypatch.setattr(draft.shutil, "which", lambda name: None)
    attempts = []
    def failed_process(command, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        attempts.append(destination.name)
        original_error = ("RuntimeError: Same baseline request returned different text; exact cache replay is ambiguous"
                          if ambiguous else "ValueError: LLM reached max_tokens; completion is truncated")
        detail = {"mode": "baseline", "exception_type": "RuntimeError",
                  "error": "Configured robust arm encountered an LLM error; fallback is excluded",
                  "llm_diagnostic": [{"stage": "analysis", "error": original_error}]}
        (destination / "baseline-failure.json").write_text(json.dumps(detail))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(draft.subprocess, "run", failed_process)
    monkeypatch.setattr(sys, "argv", ["benchmark_suite.py", "--output", str(tmp_path / "suite"),
                                     "--model-revision", "mock-weights", "--embedding-model", "mock-encoder",
                                     "--continue-on-error"])
    if ambiguous:
        with pytest.raises(RuntimeError, match="configuration/replay integrity failed"):
            draft.main()
        assert attempts == ["case-0"]
    else:
        with pytest.raises(SystemExit) as caught:
            draft.main()
        assert caught.value.code == 1
        assert attempts == ["case-0", "case-1"]
    saved = json.loads((tmp_path / "suite/suite.json").read_text())
    assert saved["cells"][0]["failure_category"] == ("configuration_or_integrity" if ambiguous else "generation_or_execution")
    assert saved["cells"][1]["status"] == ("pending" if ambiguous else "failed")
