"""Opt-in schema request guidance preserves historical defaults and domain checks."""

from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

from jsonschema import Draft202012Validator, ValidationError
import pytest

from fast_amem.client import MeasuredClient, SQLiteResponseCache
from fast_amem.evolution import FUSED_EVOLUTION_PROMPT, parse_fused_evolution
from fast_amem.joint import JOINT_NOTE_PROMPT, parse_joint_note
from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT, EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT, UPDATE_NEIGHBORS_PROMPT,
)

import fast_amem.structured as structured


FUSED = FUSED_EVOLUTION_PROMPT.format(content="new", context="summary", keywords=["topic"], tags=["topic"], neighbors="")
JOINT = JOINT_NOTE_PROMPT.format(content="new", neighbors="", allowed_indices=[])
UPSTREAM = [
    ANALYZE_CONTENT_PROMPT.format(content="new"),
    EVOLUTION_DECISION_PROMPT.format(content="new", context="summary", keywords=["topic"], nearest_neighbors_memories=""),
    STRENGTHEN_DETAILS_PROMPT.format(content="new", keywords=["topic"], nearest_neighbors_memories=""),
    UPDATE_NEIGHBORS_PROMPT.format(content="new", context="summary", nearest_neighbors_memories="", max_neighbor_idx=1, neighbor_count=2),
    "Given the following question, generate several keywords separated by commas.\nQuestion: test?",
    "Based on the context: quoted context, answer the question.",
]


def evolution():
    return {"decision": "NO_EVOLUTION", "connections": [], "tags": [], "neighbor_updates": []}


def joint_note():
    return {"metadata": {"keywords": ["topic"], "context": "Summary.", "tags": ["topic"]},
            "evolution": evolution()}


class Session:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append(deepcopy(json))
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}]})


def make_client(tmp_path, cls=MeasuredClient, **kwargs):
    return cls("http://mock.invalid/v1", "model", cache=SQLiteResponseCache(tmp_path / "cache.sqlite"),
               cache_mode="readwrite", revision="weights-template-v1", **kwargs)


@pytest.mark.parametrize("stage,expected_name,valid", [
    ("evolution_fused", "amem_fused_evolution_v1", evolution()),
    ("joint_note", "amem_joint_note_v1", joint_note()),
])
def test_fixed_generic_schema_is_standard_and_accepts_domain_examples(stage, expected_name, valid):
    response_format = structured.policy_response_format(stage)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == expected_name
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(valid)
    assert "maxItems" not in json.dumps(schema)

    def objects(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                yield value
            for child in value.values():
                yield from objects(child)
        elif isinstance(value, list):
            for child in value:
                yield from objects(child)

    for obj in objects(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_schema_enforces_keys_types_and_decision_enum():
    validator = Draft202012Validator(structured.policy_response_format("evolution_fused")["json_schema"]["schema"])
    for invalid in [{**evolution(), "decision": "invented"},
                    {**evolution(), "connections": ["0"]},
                    {**evolution(), "extra": "not allowed"},
                    {key: value for key, value in evolution().items() if key != "tags"},
                    {**evolution(), "neighbor_updates": [{"memory_index": 0, "tags": []}]}]:
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def test_schema_does_not_replace_candidate_branch_or_metadata_domain_validation():
    schema = structured.policy_response_format("evolution_fused")["json_schema"]["schema"]
    assert schema["properties"]["connections"]["items"] == {"type": "integer"}
    invalid_candidate = {**evolution(), "decision": "STRENGTHEN", "connections": [999]}
    Draft202012Validator(schema).validate(invalid_candidate)
    with pytest.raises(ValueError, match="retrieved global"):
        parse_fused_evolution(json.dumps(invalid_candidate), [0])
    contradictory = {**evolution(), "tags": ["unexpected mutation"]}
    Draft202012Validator(schema).validate(contradictory)
    with pytest.raises(ValueError, match="cannot contain mutations"):
        parse_fused_evolution(json.dumps(contradictory), [0])
    empty_metadata = joint_note()
    empty_metadata["metadata"]["tags"] = []
    joint_schema = structured.policy_response_format("joint_note")["json_schema"]["schema"]
    Draft202012Validator(joint_schema).validate(empty_metadata)
    with pytest.raises(ValueError, match="nonempty"):
        parse_joint_note(json.dumps(empty_metadata), [])


def test_response_format_returns_independent_values():
    first = structured.policy_response_format("evolution_fused")
    first["json_schema"]["schema"]["properties"]["decision"]["enum"].clear()
    assert len(structured.policy_response_format("evolution_fused")["json_schema"]["schema"]["properties"]["decision"]["enum"]) == 4


# Captured from frozen pre-structured commit 2019a73d4ce7135acff4fc2cf51978ccee087124.
# These constants catch accidental request/cache identity changes independently
# of the current implementation constructing both sides of the comparison.
HISTORICAL_DEFAULT_KEYS = {
    "error-evolution_fused": "900c170dcac6622f6c581919b10c12ef41d31b2d551c3004a2b9e06dcde62867",
    "observe-evolution_fused": "70d68eaff138b7dbcc7ba84725f8a0194d4277717732ba5ad8daacef3c4a680f",
    "error-joint_note": "e437cc148cb3a813b9cd9d70e368034476c323f40654ba198abf3279617dce31",
    "observe-joint_note": "b5ce9f4ef81e8a1f721992b7d7809c9edbea7f26b114a7129496a8ef351b55dd"
}


@pytest.mark.parametrize("stage,prompt,content", [("evolution_fused", FUSED, json.dumps(evolution())),
                                                ("joint_note", JOINT, json.dumps(joint_note()))])
@pytest.mark.parametrize("tag_policy", ["error", "observe"])
def test_disabled_option_preserves_historical_request_identity(tmp_path, stage, prompt, content, tag_policy):
    current = make_client(tmp_path, tag_budget=16, tag_budget_policy=tag_policy)
    current._local.session = Session(content)
    assert current.get_completion(prompt) == content
    assert current.events[-1]["key"] == HISTORICAL_DEFAULT_KEYS[tag_policy + "-" + stage]
    assert "response_format" not in current._local.session.calls[0]
    current.cache_mode = "replay"
    assert current.get_completion(prompt) == content
    assert current.summary()["cache_hits"] == 1
    assert current.summary()["structured_policy_outputs"] is False
    assert inspect.signature(MeasuredClient).parameters["structured_policy_outputs"].default is False


@pytest.mark.parametrize("prompt", UPSTREAM)
def test_enabling_option_leaves_upstream_and_qa_requests_unchanged(tmp_path, prompt):
    original = make_client(tmp_path, cls=MeasuredClient)
    original._local.session = Session("plain completion")
    original.get_completion(prompt)
    enabled = make_client(tmp_path, structured_policy_outputs=True)
    enabled._local.session = Session("plain completion")
    enabled.cache_mode = "write"
    enabled.get_completion(prompt)
    assert enabled._local.session.calls[0] == original._local.session.calls[0]
    assert enabled.events[-1]["key"] == original.events[-1]["key"]
    assert "response_format" not in enabled._local.session.calls[0]
    assert enabled.events[-1]["structured_output_applied"] is False


@pytest.mark.parametrize("prompt,content,stage", [(FUSED, json.dumps(evolution()), "evolution_fused"),
                                                (JOINT, json.dumps(joint_note()), "joint_note")])
def test_enabled_policy_request_includes_schema_and_owns_cache_identity(tmp_path, prompt, content, stage):
    original = make_client(tmp_path)
    original._local.session = Session(content)
    original.get_completion(prompt)
    enabled = make_client(tmp_path, structured_policy_outputs=True)
    enabled.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        enabled.get_completion(prompt)
    enabled.cache_mode = "readwrite"
    enabled._local.session = Session(content)
    assert enabled.get_completion(prompt) == content
    assert enabled._local.session.calls[0]["response_format"] == structured.policy_response_format(stage)
    assert enabled.events[-1]["key"] != original.events[-1]["key"]
    assert enabled.events[-1]["structured_output_applied"] is True
    assert enabled.summary()["structured_policy_outputs"] is True
    replay = make_client(tmp_path, structured_policy_outputs=True)
    replay.cache_mode = "replay"
    assert replay.get_completion(prompt) == content
    assert replay.summary()["http_requests"] == 0


def test_structured_option_does_not_clip_observed_tags_or_repair_invalid_json(tmp_path):
    oversized = {**evolution(), "decision": "STRENGTHEN", "tags": ["tag"] * 18}
    schema = structured.policy_response_format("evolution_fused")["json_schema"]["schema"]
    Draft202012Validator(schema).validate(oversized)
    client = make_client(tmp_path, structured_policy_outputs=True, tag_budget=16, tag_budget_policy="observe")
    client._local.session = Session(json.dumps(oversized))
    assert len(json.loads(client.get_completion(FUSED))["tags"]) == 18
    assert client.summary()["tag_budget_overshoots"]["live_overshoot_responses"] == 1
    malformed = json.dumps(evolution()) + "}"
    client._local.session = Session(malformed)
    assert client.get_completion(FUSED + "\nDifferent diagnostic request") == malformed
    with pytest.raises(ValueError, match="JSON object"):
        parse_fused_evolution(malformed, [])


def test_flag_requires_a_boolean(tmp_path):
    with pytest.raises(TypeError, match="boolean"):
        make_client(tmp_path, structured_policy_outputs="true")
