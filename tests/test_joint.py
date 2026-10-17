"""Validate the experimental one-request metadata/evolution policy."""

from copy import deepcopy
import hashlib
import json
import re
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fast_amem.client import MeasuredClient, SQLiteResponseCache
from scripts.benchmark import canonical_memories, digest

import fast_amem.joint as joint


class Encoder:
    def encode(self, texts):
        return np.asarray([np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8)[:13].astype(float) - 127
                           for text in texts])


class Service:
    def __init__(self, decision="STRENGTHEN_AND_UPDATE", invalid=False):
        self.decision = decision
        self.invalid = invalid
        self.calls = []

    def post(self, url, *, json: dict, timeout):
        prompt = json["messages"][1]["content"]
        assert prompt.startswith("Create and evolve one memory note in one operation.")
        indices = [int(row) for row in re.findall(r"memory index:(\d+)", prompt)]
        decision = self.decision if indices else "NO_EVOLUTION"
        evolution = {"decision": decision, "connections": [], "tags": [], "neighbor_updates": []}
        if decision in {"STRENGTHEN", "STRENGTHEN_AND_UPDATE"}:
            evolution.update(connections=[indices[-1]], tags=["revised"])
        if decision in {"UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"}:
            evolution["neighbor_updates"] = [{"memory_index": indices[-1], "context": "joint update", "tags": []}]
        if self.invalid and indices:
            evolution["neighbor_updates"].append({"memory_index": 999, "context": "invented", "tags": []})
        content = prompt.split("NEW CONTENT:\n", 1)[1].split("\n\nRETRIEVED NEIGHBORS", 1)[0]
        marker = hashlib.sha256(content.encode()).hexdigest()[:8]
        result = {"metadata": {"keywords": ["topic", marker, "concept"],
                               "context": "Summary " + marker, "tags": ["test", "topic", marker]},
                  "evolution": evolution}
        self.calls.append({"payload": deepcopy(json), "thread": threading.get_ident()})
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": __import__("json").dumps(result)},
                                                          "finish_reason": "stop"}],
                                             "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def client(path, mode, service):
    controller = MeasuredClient("http://fake/v1", "model", cache=SQLiteResponseCache(path),
                                cache_mode=mode, revision="test-joint-v1")
    controller._session = lambda: service
    return controller


@pytest.mark.parametrize("decision", ["NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"])
def test_joint_one_request_per_note_and_own_cache_replay(tmp_path, decision):
    service = Service(decision)
    controller = client(tmp_path / "joint.sqlite", "readwrite", service)
    system = joint.JointAgenticMemorySystem(Encoder(), controller, evo_threshold=2)
    turns = [{"content": f"Content {i}", "timestamp": "2024-01-01"} for i in range(7)]
    trace = []
    system.ingest(iter(turns), on_note=lambda _: trace.append(digest(canonical_memories(system))))
    assert len(service.calls) == controller.summary()["http_requests"] == len(turns)
    assert controller.summary()["by_stage"]["joint_note"]["http_requests"] == len(turns)
    assert controller.summary()["prefetch_calls"] == 0
    assert all(call["thread"] == threading.get_ident() for call in service.calls)
    assert system.evo_cnt == (0 if decision == "NO_EVOLUTION" else 6)
    assert system.evolution_metrics["joint_decisions"]["NO_EVOLUTION"] == (7 if decision == "NO_EVOLUTION" else 1)
    assert controller.summary()["prompt_tokens_live"] == 70
    replay_client = client(tmp_path / "joint.sqlite", "replay", None)
    replay = joint.JointAgenticMemorySystem(Encoder(), replay_client, evo_threshold=2)
    replay_trace = []
    replay.ingest(turns, on_note=lambda _: replay_trace.append(digest(canonical_memories(replay))))
    assert trace == replay_trace
    assert canonical_memories(system) == canonical_memories(replay)
    assert system.retriever.corpus == replay.retriever.corpus
    assert replay_client.summary()["http_requests"] == 0


def result():
    return {"metadata": {"keywords": ["one"], "context": "summary", "tags": ["topic"]},
            "evolution": {"decision": "NO_EVOLUTION", "connections": [], "tags": [], "neighbor_updates": []}}


@pytest.mark.parametrize("metadata", [
    None, {}, {"keywords": [], "context": "summary", "tags": ["topic"]},
    {"keywords": [""], "context": "summary", "tags": ["topic"]},
    {"keywords": [1], "context": "summary", "tags": ["topic"]},
    {"keywords": ["one"], "context": " ", "tags": ["topic"]},
    {"keywords": ["one"], "context": ["summary"], "tags": ["topic"]},
    {"keywords": ["one"], "context": "summary", "tags": "topic"},
    {"keywords": ["one"], "context": "summary", "tags": []},
])
def test_joint_rejects_metadata_without_heuristic_repair(metadata):
    with pytest.raises(ValueError, match="joint metadata"):
        joint.parse_joint_note(json.dumps({**result(), "metadata": metadata}), [3])


def test_joint_first_note_cannot_evolve_without_neighbors():
    payload = result()
    payload["evolution"]["decision"] = "STRENGTHEN"
    with pytest.raises(ValueError, match="without neighbors"):
        joint.parse_joint_note(json.dumps(payload), [])


def test_joint_validates_all_fields_before_applying_any_changes(tmp_path):
    controller = client(tmp_path / "joint.sqlite", "readwrite", Service(invalid=True))
    system = joint.JointAgenticMemorySystem(Encoder(), controller)
    system.add_note("First", time="2024-01-01")
    before = deepcopy(canonical_memories(system))
    with pytest.raises(ValueError, match="global index"):
        system.add_note("Second", time="2024-01-02")
    assert canonical_memories(system) == before
    assert len(system.retriever.corpus) == 1
    assert system.evo_cnt == 0
    assert system.evolution_metrics["joint_parse_failures"] == 1


def test_joint_missing_response_cannot_fall_back_to_heuristics(tmp_path):
    controller = client(tmp_path / "missing.sqlite", "replay", None)
    system = joint.JointAgenticMemorySystem(Encoder(), controller)
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        system.add_note("First", time="2024-01-01")
    assert not system.memories


def test_joint_encoder_failure_restores_neighbor_fields(tmp_path):
    class FailingEncoder(Encoder):
        def encode(self, texts):
            if any(text.startswith("content:Second") for text in texts):
                raise ValueError("encoder failed")
            return super().encode(texts)

    controller = client(tmp_path / "joint.sqlite", "readwrite", Service())
    system = joint.JointAgenticMemorySystem(FailingEncoder(), controller)
    system.add_note("First", time="2024-01-01")
    before = deepcopy(canonical_memories(system))
    with pytest.raises(ValueError, match="encoder failed"):
        system.add_note("Second", time="2024-01-02")
    assert canonical_memories(system) == before
    assert system.evo_cnt == 0


def test_joint_rejects_metadata_overrides_before_any_request(tmp_path):
    service = Service()
    system = joint.JointAgenticMemorySystem(Encoder(), client(tmp_path / "joint.sqlite", "readwrite", service))
    with pytest.raises(ValueError, match="metadata overrides"):
        system.add_note("content", tags=["manual"])
    assert not service.calls
