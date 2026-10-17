"""The common bounded-tag profile changes the system request and rejects excess."""

import json

import pytest

from fast_amem.client import SYSTEM_MESSAGE, bounded_tags_instruction
from fast_amem.evolution import FUSED_EVOLUTION_PROMPT, parse_fused_evolution
from fast_amem.joint import JOINT_NOTE_PROMPT
from llm_text_parsers import ANALYZE_CONTENT_PROMPT, STRENGTHEN_DETAILS_PROMPT, UPDATE_NEIGHBORS_PROMPT
from test_client import Session, client, response


ANALYSIS = ANALYZE_CONTENT_PROMPT.format(content="New content")
STRENGTHEN = STRENGTHEN_DETAILS_PROMPT.format(content="New content", keywords=["topic"], nearest_neighbors_memories="")
UPDATE = UPDATE_NEIGHBORS_PROMPT.format(content="New content", context="Summary", nearest_neighbors_memories="",
                                       max_neighbor_idx=1, neighbor_count=2)
FUSED = FUSED_EVOLUTION_PROMPT.format(content="New content", context="Summary", keywords=["topic"],
                                     tags=["topic"], neighbors="")
JOINT = JOINT_NOTE_PROMPT.format(content="New content", neighbors="", allowed_indices=[0])


def fused(tags, nested=False):
    return {"decision": "STRENGTHEN_AND_UPDATE", "connections": [], "tags": [] if nested else tags,
            "neighbor_updates": [{"memory_index": 0, "context": "", "tags": tags}] if nested else []}


CASES = [
    ("analysis", ANALYSIS, lambda tags: "KEYWORDS: topic\nCONTEXT: Summary\nTAGS: " + ", ".join(tags)),
    ("analysis", ANALYSIS, lambda tags: json.dumps({"keywords": ["topic"], "context": "Summary", "tags": tags})),
    ("strengthen", STRENGTHEN, lambda tags: "CONNECTIONS: 0\nTAGS: " + ", ".join(tags)),
    ("strengthen", STRENGTHEN, lambda tags: json.dumps({"suggested_connections": [0], "tags_to_update": tags})),
    ("update_neighbors", UPDATE, lambda tags: json.dumps({"new_context_neighborhood": ["", ""],
                                                           "new_tags_neighborhood": [["first"], tags]})),
    ("update_neighbors", UPDATE, lambda tags: "NEIGHBOR 0:\nCONTEXT: Original\nTAGS: first\n\nNEIGHBOR 1:\nCONTEXT: Other\nTAGS: " + ", ".join(tags)),
    ("evolution_fused", FUSED, lambda tags: json.dumps(fused(tags))),
    ("evolution_fused", FUSED, lambda tags: json.dumps(fused(tags, nested=True))),
    ("joint_note", JOINT, lambda tags: json.dumps({"metadata": {"keywords": ["topic"], "context": "Summary", "tags": tags},
                                                  "evolution": fused([])})),
    ("joint_note", JOINT, lambda tags: json.dumps({"metadata": {"keywords": ["topic"], "context": "Summary", "tags": ["topic"]},
                                                  "evolution": fused(tags, nested=True)})),
]


@pytest.mark.parametrize("stage,prompt,output", CASES)
def test_sixteen_tags_are_accepted_exactly_and_cached(tmp_path, stage, prompt, output):
    content = output([f"tag-{i}" for i in range(16)])
    c = client(tmp_path, tag_budget=16)
    session = Session(response(content))
    c._local.session = session
    assert c.get_completion(prompt) == content
    assert c.get_completion(prompt) == content
    assert len(session.calls) == 1
    assert c.summary()["tag_budget"] == 16
    assert c.summary()["generation_profile"] == "bounded-tags-v1"
    assert c.events[0]["stage"] == stage


@pytest.mark.parametrize("stage,prompt,output", CASES)
def test_seventeen_tags_fail_before_caching_and_retain_usage(tmp_path, stage, prompt, output):
    content = output([f"tag-{i}" for i in range(17)])
    c = client(tmp_path, tag_budget=16)
    c._local.session = Session(response(content))
    with pytest.raises(ValueError, match="tag_budget=16: 17 items") as caught:
        c.get_completion(prompt)
    assert caught.value.llm_diagnostic["stage"] == stage
    assert c.summary()["completion_tokens_live"] == 3
    c.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        c.get_completion(prompt)


def test_tag_budget_changes_only_system_instruction_and_cache_identity(tmp_path):
    original = client(tmp_path)
    original._local.session = Session(response("KEYWORDS: topic\nCONTEXT: Summary\nTAGS: topic"))
    original.get_completion(ANALYSIS)
    explicit_none = client(tmp_path, tag_budget=None)
    explicit_none.cache_mode = "replay"
    assert explicit_none.get_completion(ANALYSIS).endswith("TAGS: topic")
    bounded = client(tmp_path, tag_budget=16)
    bounded.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        bounded.get_completion(ANALYSIS)
    bounded.cache_mode = "readwrite"
    session = Session(response("KEYWORDS: topic\nCONTEXT: Summary\nTAGS: topic"))
    bounded._local.session = session
    bounded.get_completion(ANALYSIS)
    messages = session.calls[0][1]["json"]["messages"]
    assert messages[0]["content"] == SYSTEM_MESSAGE + "\n\n" + (
        "Return no more than 16 items in each TAGS/tags list, including lists for updated neighbors. "
        "Follow the requested format and stop after completing its required fields.")
    assert messages[1]["content"] == ANALYSIS
    assert bounded.events[-1]["key"] != original.events[0]["key"]
    different = client(tmp_path, tag_budget=15)
    different.cache_mode = "replay"
    with pytest.raises(RuntimeError, match="Replay cache miss"):
        different.get_completion(ANALYSIS)
    assert bounded_tags_instruction(16) in messages[0]["content"]


def test_over_budget_cached_response_is_rejected_on_read(tmp_path):
    c = client(tmp_path, tag_budget=16)
    c._local.session = Session(response("CONNECTIONS: 0\nTAGS: first"))
    c.get_completion(STRENGTHEN)
    key = c.events[-1]["key"]
    c.cache.put(key, "externally replaced regression fixture", response("CONNECTIONS: 0\nTAGS: " + ", ".join(["repeat"] * 17)))
    c.cache_mode = "replay"
    with pytest.raises(ValueError, match="17 items"):
        c.get_completion(STRENGTHEN)
    assert c.events[-1]["cache_hit"]


@pytest.mark.parametrize("prompt", ["Based on the context: a question", "Given the following question, generate several keywords separated by commas."])
def test_qa_text_is_not_interpreted_as_metadata_tags(tmp_path, prompt):
    content = "TAGS: " + ", ".join(["quoted answer text"] * 17)
    c = client(tmp_path, tag_budget=16)
    c._local.session = Session(response(content))
    assert c.get_completion(prompt) == content


def test_malformed_fused_json_is_returned_unchanged_for_the_policy_parser(tmp_path):
    content = '{"tags": ["unfinished"'
    c = client(tmp_path, tag_budget=16)
    c._local.session = Session(response(content))
    assert c.get_completion(FUSED) == content
    with pytest.raises(ValueError, match="JSON object"):
        parse_fused_evolution(content, [0])


def test_parser_type_failure_also_retains_bounded_diagnostic(tmp_path):
    c = client(tmp_path, tag_budget=16)
    c._local.session = Session(response(json.dumps({"keywords": 1, "context": "Summary", "tags": []})))
    with pytest.raises(TypeError) as caught:
        c.get_completion(ANALYSIS)
    assert caught.value.llm_diagnostic["stage"] == "analysis"
    assert c.events[-1]["llm_diagnostic"] == caught.value.llm_diagnostic


@pytest.mark.parametrize("budget", [0, -1, 1.5, "16"])
def test_invalid_tag_budget_is_rejected(tmp_path, budget):
    with pytest.raises((ValueError, TypeError)):
        client(tmp_path, tag_budget=budget)
