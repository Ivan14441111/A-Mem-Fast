"""End-to-end semantic checks using deterministic HTTP and embedding doubles."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json as json_module
import re
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import memory_layer
from memory_layer import SimpleEmbeddingRetriever
from memory_layer_robust import RobustAgenticMemorySystem
from fast_amem.client import MeasuredClient, SQLiteResponseCache, stage_for
from fast_amem.system import FastAgenticMemorySystem
from scripts.benchmark import answer_question, canonical_memories, digest


class DeterministicEncoder:
    def encode(self, texts):
        # This is a test double, not an embedding model used for experiment data.
        return np.asarray([
            np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8)[:19].astype(float) - 127
            for text in texts
        ])

    def get_config_dict(self):
        return {"model_name": "deterministic-test-encoder"}


class DeterministicService:
    """One shared fake server; request scheduling cannot change its responses."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def post(self, url, *, json, timeout):
        payload = deepcopy(json)
        prompt = payload["messages"][1]["content"]
        stage = stage_for(prompt)
        neighbor_ids = [int(i) for i in re.findall(r"memory index:(\d+)", prompt)]
        marker = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        if stage == "analysis":
            result = {"keywords": ["topic", marker, "concept"],
                      "context": "Analysis of " + marker,
                      "tags": ["test", "metadata", marker]}
        elif stage == "evolution_decision":
            result = {"decision": "STRENGTHEN_AND_UPDATE", "reason": "exercise both mutations"}
        elif stage == "strengthen":
            result = {"connections": [neighbor_ids[0]], "tags": ["linked", marker]}
        elif stage == "update_neighbors":
            result = {"new_context_neighborhood": [f"Updated {i} by {marker}" for i in neighbor_ids],
                      "new_tags_neighborhood": [["updated", str(i), marker] for i in neighbor_ids]}
        elif stage == "query_keywords":
            result = {"keywords": "test topic"}
        elif stage == "answer":
            result = {"answer": "deterministic answer"}
        else:
            raise AssertionError("Unexpected test prompt: " + prompt[:100])
        with self._lock:
            self.calls.append({"payload": payload, "stage": stage, "thread": threading.get_ident()})
        content = json_module.dumps(result)

        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        return Response()


def make_client(path, mode, service, **kwargs):
    client = MeasuredClient("http://fake-test-server/v1", "test-qwen",
                            revision="test-weights-and-template-v1", cache=SQLiteResponseCache(path),
                            cache_mode=mode, **kwargs)
    client._session = lambda: service
    return client


def upstream_system(client, threshold):
    system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
    system.memories = {}
    system.retriever = SimpleEmbeddingRetriever("deterministic-test-encoder")
    system.llm_controller = SimpleNamespace(llm=client)
    system.evo_cnt = 0
    system.evo_threshold = threshold
    return system


@pytest.mark.parametrize("threshold", [2, 100])
def test_prefetch_preserves_sequential_evolution_and_rebuild(tmp_path, monkeypatch, threshold):
    model_loads = []

    def encoder_factory(name):
        model_loads.append(name)
        return DeterministicEncoder()

    monkeypatch.setattr(memory_layer, "SentenceTransformer", encoder_factory)
    turns = [{"content": f"Speaker Asays : Topic {i} contains distinct information.",
              "timestamp": "10:00 am on 1 January, 2024"} for i in range(9)]
    original = DeterministicService()
    client_a = make_client(tmp_path / "baseline.sqlite", "write", original)
    baseline = upstream_system(client_a, threshold)
    trace_a = []
    for turn in turns:
        baseline.add_note(turn["content"], time=turn["timestamp"])
        trace_a.append(digest(canonical_memories(baseline)))

    concurrent = DeterministicService()
    client_b = make_client(tmp_path / "optimized.sqlite", "readwrite", concurrent)
    shared_encoder = DeterministicEncoder()
    optimized = FastAgenticMemorySystem(shared_encoder, client_b, evo_threshold=threshold)
    trace_b = []
    optimized.ingest(turns, analysis_workers=3, prefetch_window=4,
                     on_note=lambda _: trace_b.append(digest(canonical_memories(optimized))))

    assert trace_a == trace_b
    assert canonical_memories(baseline) == canonical_memories(optimized)
    assert baseline.evo_cnt == optimized.evo_cnt == len(turns) - 1
    assert len(optimized.consolidation_seconds) == (len(turns) - 1) // threshold
    assert optimized.retriever.model is shared_encoder
    assert len(model_loads) == 1 + (len(turns) - 1) // threshold
    assert baseline.retriever.corpus == optimized.retriever.corpus
    assert any(c["stage"] == "update_neighbors" for c in original.calls)
    assert any(c["stage"] == "strengthen" for c in original.calls)

    # Only metadata may run in the executor; all state-dependent requests retain
    # their exact chronological order, including their full neighbor prompts.
    main_thread = threading.get_ident()
    nonanalysis_a = [c["payload"] for c in original.calls if c["stage"] != "analysis"]
    nonanalysis_b = [c["payload"] for c in concurrent.calls if c["stage"] != "analysis"]
    assert nonanalysis_a == nonanalysis_b
    assert all(c["thread"] == main_thread for c in concurrent.calls if c["stage"] != "analysis")
    assert all(c["thread"] != main_thread for c in concurrent.calls if c["stage"] == "analysis")
    assert client_a.summary()["http_requests"] == client_b.summary()["http_requests"]
    assert client_b.summary()["prefetch_calls"] == len(turns)
    assert client_b.summary()["cache_hits"] == len(turns)
    assert client_a.summary()["errors"] == client_b.summary()["errors"] == 0

    qa = {"question": "What is the test answer?", "answer": "deterministic answer", "category": 4}
    assert answer_question(baseline, client_a, qa, 5) == answer_question(optimized, client_b, qa, 5)

    # Strict replay must recover every intermediate state and QA prompt while
    # making zero requests, including across the consolidation boundary.
    replay_service = DeterministicService()
    replay_client = make_client(tmp_path / "baseline.sqlite", "replay", replay_service)
    replay = FastAgenticMemorySystem(DeterministicEncoder(), replay_client, evo_threshold=threshold)
    replay_trace = []
    replay.ingest(turns, analysis_workers=3, prefetch_window=4,
                  on_note=lambda _: replay_trace.append(digest(canonical_memories(replay))))
    assert replay_trace == trace_a
    assert answer_question(replay, replay_client, qa, 5) == answer_question(baseline, client_a, qa, 5)
    assert replay_client.summary()["http_requests"] == 0
    assert replay_client.summary()["errors"] == 0
    assert not replay_service.calls


def test_analysis_prefetch_failure_propagates_and_does_not_add_partial_note(tmp_path):
    service = DeterministicService()
    client = make_client(tmp_path / "missing.sqlite", "replay", service)
    system = FastAgenticMemorySystem(DeterministicEncoder(), client)
    turns = [{"content": "Unrecorded input", "timestamp": "2024-01-01"}]
    with pytest.raises(RuntimeError, match="Replay cache miss: analysis"):
        system.ingest(turns)
    assert not system.memories
    assert client.summary()["errors"] == 1
    assert not service.calls


def test_rolling_prefetch_refills_before_graph_work_and_supports_iterators():
    future_analysis_started = threading.Event()

    class Client:
        cache_mode = "readwrite"

        def get_completion(self, prompt, **kwargs):
            if prompt.endswith("content-2"):
                future_analysis_started.set()
            return "prefetched"

    system = FastAgenticMemorySystem(DeterministicEncoder(), Client())
    added = []

    def add_note(content, **kwargs):
        if not added:
            # A fixed-window implementation would not schedule content-2 until
            # after both initial notes, leaving this event unset.
            assert future_analysis_started.wait(timeout=5)
        added.append(content)

    system.add_note = add_note
    turns = ({"content": f"content-{i}", "timestamp": "2024-01-01"} for i in range(6))
    callbacks = []
    system.ingest(turns, analysis_workers=2, prefetch_window=2, on_note=callbacks.append)
    assert added == [f"content-{i}" for i in range(6)]
    assert callbacks == list(range(1, 7))


def test_strict_add_note_does_not_hide_metadata_replay_miss(tmp_path):
    client = make_client(tmp_path / "strict.sqlite", "replay", DeterministicService())
    system = FastAgenticMemorySystem(DeterministicEncoder(), client)
    with pytest.raises(RuntimeError, match="Replay cache miss: analysis"):
        system.add_note("Unrecorded input", time="2024-01-01")
    assert not system.memories
    assert system.retriever.corpus == []


def test_strict_add_note_does_not_hide_evolution_replay_miss(tmp_path):
    client = make_client(tmp_path / "strict.sqlite", "readwrite", DeterministicService())
    system = FastAgenticMemorySystem(DeterministicEncoder(), client)
    system.add_note("First note", time="2024-01-01")
    client.get_completion(__import__("llm_text_parsers").ANALYZE_CONTENT_PROMPT.format(content="Second note"))
    before = deepcopy(canonical_memories(system))
    corpus = list(system.retriever.corpus)
    client.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss: evolution_decision"):
        system.add_note("Second note", time="2024-01-02")
    assert canonical_memories(system) == before
    assert system.retriever.corpus == corpus
    assert system.evo_cnt == 0


class FusedService(DeterministicService):
    def __init__(self, decision="STRENGTHEN_AND_UPDATE", invalid=False):
        super().__init__()
        self.decision = decision
        self.invalid = invalid

    def post(self, url, *, json, timeout):
        prompt = json["messages"][1]["content"]
        if stage_for(prompt) != "evolution_fused":
            return super().post(url, json=json, timeout=timeout)
        indices = [int(i) for i in re.findall(r"memory index:(\d+)", prompt)]
        result = {"decision": self.decision, "connections": [], "tags": [], "neighbor_updates": []}
        if self.decision in {"STRENGTHEN", "STRENGTHEN_AND_UPDATE"}:
            result["connections"] = [indices[-1], indices[-1]]
            result["tags"] = ["fused-new"]
        if self.decision in {"UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"}:
            result["neighbor_updates"] = [{"memory_index": indices[-1], "context": "fused context", "tags": []}]
        if self.invalid:
            result["neighbor_updates"].append({"memory_index": 999, "context": "bad", "tags": []})
        with self._lock:
            self.calls.append({"payload": deepcopy(json), "stage": "evolution_fused",
                               "thread": threading.get_ident()})
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": json_module.dumps(result)},
                                                          "finish_reason": "stop"}]})


@pytest.mark.parametrize("decision", ["NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"])
def test_fused_policy_one_call_per_evolution_and_strict_replay(tmp_path, decision):
    service = FusedService(decision)
    path = tmp_path / "fused.sqlite"
    client = make_client(path, "readwrite", service)
    system = FastAgenticMemorySystem(DeterministicEncoder(), client, evo_threshold=2,
                                     evolution_policy="fused")
    turns = [{"content": f"Topic {i}", "timestamp": "2024-01-01"} for i in range(7)]
    system.ingest(turns, analysis_workers=3, prefetch_window=3)
    summary = client.summary()
    assert summary["by_stage"]["analysis"]["http_requests"] == 7
    assert summary["by_stage"]["evolution_fused"]["http_requests"] == 6
    assert summary["http_requests"] == 13
    assert system.evo_cnt == (0 if decision == "NO_EVOLUTION" else 6)
    assert len(system.consolidation_seconds) == (0 if decision == "NO_EVOLUTION" else 3)
    memories = list(system.memories.values())
    if decision in {"STRENGTHEN", "STRENGTHEN_AND_UPDATE"}:
        assert all(len(note.links) == 1 for note in memories[1:])
        assert all(0 <= row < i for i, note in enumerate(memories) for row in note.links)
    else:
        assert all(not note.links for note in memories)
    if decision in {"UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"}:
        assert any(note.context == "fused context" for note in memories)
    replay_client = make_client(path, "replay", DeterministicService())
    replay = FastAgenticMemorySystem(DeterministicEncoder(), replay_client, evo_threshold=2,
                                     evolution_policy="fused")
    replay.ingest(turns, analysis_workers=2, prefetch_window=4)
    assert canonical_memories(system) == canonical_memories(replay)
    assert system.retriever.corpus == replay.retriever.corpus
    assert replay_client.summary()["http_requests"] == 0


def test_invalid_fused_response_does_not_apply_any_mutations(tmp_path):
    client = make_client(tmp_path / "fused.sqlite", "readwrite", FusedService(invalid=True))
    system = FastAgenticMemorySystem(DeterministicEncoder(), client, evolution_policy="fused")
    system.add_note("First topic", time="2024-01-01")
    before = deepcopy(canonical_memories(system))
    corpus = list(system.retriever.corpus)
    with pytest.raises(ValueError, match="unique retrieved global index"):
        system.add_note("Second topic", time="2024-01-02")
    assert canonical_memories(system) == before
    assert system.retriever.corpus == corpus
    assert system.evo_cnt == 0


def test_encoding_failure_rolls_back_neighbor_updates(tmp_path):
    class FailingEncoder(DeterministicEncoder):
        def encode(self, texts):
            if any(text.startswith("content:Second") for text in texts):
                raise ValueError("test encoder failure")
            return super().encode(texts)

    client = make_client(tmp_path / "fused.sqlite", "readwrite", FusedService())
    system = FastAgenticMemorySystem(FailingEncoder(), client, evolution_policy="fused")
    system.add_note("First topic", time="2024-01-01")
    before = deepcopy(canonical_memories(system))
    with pytest.raises(ValueError, match="test encoder failure"):
        system.add_note("Second topic", time="2024-01-02")
    assert canonical_memories(system) == before
    assert len(system.retriever.corpus) == 1


@pytest.mark.parametrize("kwargs", [{"evo_threshold": 0}, {"evo_threshold": 1.5},
                                   {"evolution_policy": "unknown"}])
def test_invalid_system_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        FastAgenticMemorySystem(DeterministicEncoder(), SimpleNamespace(), **kwargs)


def test_consolidation_failure_is_visible_after_the_note_commits(tmp_path):
    # Refresh follows the upstream commit boundary: this exception means the
    # note exists and callers must not blindly retry the same turn.
    client = make_client(tmp_path / "boundary.sqlite", "readwrite", FusedService())
    system = FastAgenticMemorySystem(DeterministicEncoder(), client, evolution_policy="fused", evo_threshold=1)
    system.add_note("First topic", time="2024-01-01")

    def fail_refresh():
        raise ValueError("consolidation failed")

    system.consolidate_memories = fail_refresh
    with pytest.raises(ValueError, match="consolidation failed"):
        system.add_note("Second topic", time="2024-01-02")
    assert len(system.memories) == len(system.retriever.corpus) == 2
    assert system.evo_cnt == 1


@pytest.mark.parametrize("repetition_penalty,tag_budget", [(None, None), (1.05, 16)])
def test_context_ablation_reconstructs_recorded_graph_and_keywords_without_http(tmp_path, repetition_penalty, tag_budget):
    from scripts.benchmark_context import reconstruct_graph, query_prompt

    service = DeterministicService()
    client = make_client(tmp_path / "optimized.sqlite", "readwrite", service,
                         repetition_penalty=repetition_penalty, tag_budget=tag_budget)
    system = FastAgenticMemorySystem(DeterministicEncoder(), client, evo_threshold=2)
    turns = [{"content": f"Topic {i}", "timestamp": "2024-01-01"} for i in range(5)]
    questions = [{"question": "A test question?", "answer": "deterministic answer", "category": 4}]
    subset = {"turns": turns, "questions": questions}
    subset["fingerprint"] = digest(subset)
    trace = []
    system.ingest(turns, on_note=lambda _: trace.append(digest(canonical_memories(system))))
    answer_question(system, client, questions[0], 5)
    (tmp_path / "manifest.json").write_text(json_module.dumps({"args": {
        "base_url": client.base_url, "model": client.model, "model_revision": client.revision,
        "seed": client.seed, "max_tokens": client.max_tokens, "evo_threshold": 2,
        "workers": 2, "prefetch_window": 2, "retrieve_k": 5,
        "repetition_penalty": repetition_penalty, "tag_budget": tag_budget}}))
    (tmp_path / "subset.json").write_text(json_module.dumps(subset))
    (tmp_path / "optimized.json").write_text(json_module.dumps({
        "subset_fingerprint": subset["fingerprint"], "memory_trace": trace,
        "memory_digest": digest(canonical_memories(system)), "evolution_policy": "upstream"}))
    calls_before = len(service.calls)
    replay, query_responses, _, _, _, summary = reconstruct_graph(tmp_path, "optimized", DeterministicEncoder())
    assert canonical_memories(replay) == canonical_memories(system)
    assert len(service.calls) == calls_before
    assert summary["http_requests"] == 0
    assert json_module.loads(query_responses[query_prompt(questions[0]["question"])] ) == {"keywords": "test topic"}


def test_none_turn_does_not_silently_truncate_later_turns():
    class Client:
        cache_mode = "readwrite"

        def get_completion(self, *args, **kwargs):
            return "prefetched"

    system = FastAgenticMemorySystem(None, Client())
    added = []
    system.add_note = lambda content, **kwargs: added.append(content)
    turns = [{"content": "first", "timestamp": "date"}, None,
             {"content": "last", "timestamp": "date"}]
    with pytest.raises(TypeError):
        system.ingest(iter(turns), analysis_workers=1, prefetch_window=1)
    assert added == []
