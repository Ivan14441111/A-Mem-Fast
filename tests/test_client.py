from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from fast_amem.client import MeasuredClient, SQLiteResponseCache


def response(content="ok", finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}}


class Session:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = self.result
        class Response:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return result
        return Response()


def client(tmp_path, **kwargs):
    return MeasuredClient("http://local/v1", "qwen", revision="weights-v1+template-v1",
                          cache=SQLiteResponseCache(tmp_path / "responses.sqlite"),
                          cache_mode="readwrite", **kwargs)


def test_persistent_cache_exact_request_and_revision(tmp_path):
    first = client(tmp_path)
    session = Session(response())
    first._local.session = session
    assert first.get_completion("hello") == "ok"
    assert first.get_completion("hello") == "ok"
    assert len(session.calls) == 1
    second = client(tmp_path)
    second.cache_mode = "replay"
    assert second.get_completion("hello") == "ok"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        second.get_completion("changed prompt")
    second.revision = "different weights"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        second.get_completion("hello")
    assert first.summary()["cache_hits"] == 1
    assert first.summary()["by_stage"]["other"]["prompt_tokens_live"] == 12


def test_cache_separates_decoding_budget(tmp_path):
    first = client(tmp_path)
    first._local.session = Session(response())
    first.get_completion("hello")
    second = client(tmp_path, max_tokens=2000)
    second.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        second.get_completion("hello")


def test_truncated_outputs_never_cached(tmp_path):
    c = client(tmp_path)
    c._local.session = Session(response("unfinished", "length"))
    with pytest.raises(ValueError, match="max_tokens"):
        c.get_completion("hello")
    c.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        c.get_completion("hello")
    assert c.summary()["errors"] == 2


def test_wal_concurrent_threads_and_reopen(tmp_path):
    cache = SQLiteResponseCache(tmp_path / "parallel.sqlite")
    def work(i):
        cache.put(str(i), json.dumps({"i": i}), response(str(i)))
        return cache.get(str(i))
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(work, range(40)))
    reopened = SQLiteResponseCache(tmp_path / "parallel.sqlite")
    assert all(r == reopened.get(str(i)) == response(str(i)) for i, r in enumerate(actual))


def test_identical_concurrent_requests_share_one_http_call(tmp_path):
    c = client(tmp_path)
    session = Session(response())
    c._session = lambda: session
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda _: c.get_completion("same input"), range(24)))
    assert actual == ["ok"] * 24
    assert len(session.calls) == 1
    assert c.summary()["cache_hits"] == 23
    assert not c._inflight


def test_refuse_unversioned_or_stochastic_cache(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        client(tmp_path, temperature=0.7)
    with pytest.raises(ValueError, match="revision"):
        MeasuredClient("http://local/v1", "qwen", cache_mode="readwrite",
                       cache=SQLiteResponseCache(tmp_path / "x.sqlite"))


@pytest.mark.parametrize("result", [
    None, {}, {"choices": []}, response(None), response("  "),
    response("blocked", "content_filter"), response("call", "tool_calls"),
    {**response(), "usage": []}, {**response(), "usage": {"prompt_tokens": "12"}},
])
def test_malformed_responses_are_not_persisted(tmp_path, result):
    c = client(tmp_path)
    c._local.session = Session(result)
    with pytest.raises(ValueError, match="LLM"):
        c.get_completion("hello")
    c.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        c.get_completion("hello")


def test_cached_response_is_validated_before_replay(tmp_path):
    c = client(tmp_path)
    c._local.session = Session(response())
    c.get_completion("hello")
    key = c.events[-1]["key"]
    c.cache.put(key, "replaced for regression test", response("truncated", "length"))
    c.cache_mode = "replay"
    with pytest.raises(ValueError, match="max_tokens"):
        c.get_completion("hello")


def test_missing_usage_is_accepted(tmp_path):
    c = client(tmp_path)
    c._local.session = Session({**response(), "usage": None})
    assert c.get_completion("hello") == "ok"
    assert c.summary()["by_stage"]["other"]["prompt_tokens_live"] == 0


def test_http_error_without_response_preserves_original_error(tmp_path):
    import requests

    error = requests.HTTPError("no attached response")

    class BrokenSession:
        def post(self, *args, **kwargs):
            raise error

    c = client(tmp_path)
    c._local.session = BrokenSession()
    with pytest.raises(requests.HTTPError) as raised:
        c.get_completion("hello")
    assert raised.value is error
    assert c.summary()["http_requests"] == 1
    assert not c._inflight


def test_summary_reports_live_token_totals_and_logical_latency_quantiles(tmp_path):
    c = client(tmp_path)
    c._local.session = Session(response())
    for prompt in ["Analyze the following content one", "Analyze the following content two", "other"]:
        c.get_completion(prompt)
    c.get_completion("other")  # Cache reads must not duplicate live token totals.
    for event, duration in zip(c.events, [1.0, 2.0, 3.0, 4.0]):
        event["seconds"] = duration
    summary = c.summary()
    assert summary["prompt_tokens_live"] == 36
    assert summary["completion_tokens_live"] == 9
    assert summary["latency_seconds_p50"] == 2.5
    assert summary["latency_seconds_p95"] == pytest.approx(3.85)
    assert summary["latency_seconds_max"] == 4.0
    assert summary["by_stage"]["analysis"]["latency_seconds_p50"] == 1.5
    assert summary["by_stage"]["other"]["latency_seconds_p95"] == pytest.approx(3.95)


def test_truncated_output_retains_bounded_diagnostic_and_wasted_token_usage(tmp_path):
    import hashlib

    content = "begin " + "repeated output " * 500 + " end"
    result = response(content, "length")
    result["usage"] = {"prompt_tokens": 123, "completion_tokens": 2000, "total_tokens": 2123}
    c = client(tmp_path)
    c._local.session = Session(result)
    with pytest.raises(ValueError, match="max_tokens") as caught:
        c.get_completion("Analyze the following content: private prompt body")
    diagnostic = caught.value.llm_diagnostic
    assert diagnostic == c.events[0]["llm_diagnostic"]
    assert diagnostic["stage"] == "analysis"
    assert diagnostic["request_key"] == c.events[0]["key"]
    assert diagnostic["finish_reason"] == "length"
    assert diagnostic["usage"] == result["usage"]
    assert diagnostic["content_chars"] == len(content)
    assert diagnostic["content_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert diagnostic["content_prefix"] == content[:512]
    assert diagnostic["content_suffix"] == content[-512:]
    assert "private prompt body" not in json.dumps(diagnostic)
    summary = c.summary()
    assert summary["completion_tokens_live"] == 2000
    assert summary["prompt_tokens_live"] == 123
    assert summary["by_stage"]["analysis"]["completion_tokens_live"] == 2000
    c.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        c.get_completion("Analyze the following content: private prompt body")
    assert c.summary()["completion_tokens_live"] == 2000


def test_successful_response_events_do_not_store_content_previews(tmp_path):
    c = client(tmp_path)
    c._local.session = Session(response("successful completion"))
    assert c.get_completion("hello") == "successful completion"
    assert "llm_diagnostic" not in c.events[0]


def test_repetition_penalty_omission_preserves_historical_request_identity(tmp_path):
    import hashlib
    from fast_amem.client import SYSTEM_MESSAGE

    c = client(tmp_path)
    session = Session(response())
    c._local.session = session
    c.get_completion("hello")
    historical = {"revision": c.revision, "base_url": c.base_url, "request": {
        "model": c.model,
        "messages": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": "hello"}],
        "temperature": c.temperature, "seed": c.seed, "max_tokens": c.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }}
    identity = json.dumps(historical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert c.events[-1]["key"] == hashlib.sha256(identity.encode()).hexdigest()
    assert "repetition_penalty" not in session.calls[0][1]["json"]
    explicit_none = client(tmp_path, repetition_penalty=None)
    explicit_none.cache_mode = "replay"
    assert explicit_none.get_completion("hello") == "ok"
    assert explicit_none.summary()["http_requests"] == 0


def test_repetition_penalty_is_explicit_and_isolates_request_cache(tmp_path):
    original = client(tmp_path)
    original._local.session = Session(response("original"))
    original.get_completion("hello")
    changed = client(tmp_path, repetition_penalty=1.05)
    changed.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        changed.get_completion("hello")
    changed.cache_mode = "readwrite"
    session = Session(response("penalized"))
    changed._local.session = session
    assert changed.get_completion("hello") == "penalized"
    assert session.calls[0][1]["json"]["repetition_penalty"] == 1.05
    assert changed.summary()["repetition_penalty"] == 1.05
    assert original.events[0]["key"] != changed.events[-1]["key"]
    same = client(tmp_path, repetition_penalty=1.05)
    same.cache_mode = "replay"
    assert same.get_completion("hello") == "penalized"
    different = client(tmp_path, repetition_penalty=1.1)
    different.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        different.get_completion("hello")


@pytest.mark.parametrize("penalty", [0, -0.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_repetition_penalty_is_rejected(tmp_path, penalty):
    with pytest.raises(ValueError, match="finite and positive"):
        client(tmp_path, repetition_penalty=penalty)
