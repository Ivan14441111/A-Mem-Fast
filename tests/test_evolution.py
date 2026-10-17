"""Validate the intentionally different, one-request evolution policy."""

import json

import pytest

from fast_amem.evolution import parse_fused_evolution


def operation(decision="NO_EVOLUTION", **kwargs):
    return {"decision": decision, "connections": [], "tags": [], "neighbor_updates": [], **kwargs}


def test_fused_parser_accepts_sparse_global_indices_and_preserves_order():
    result = operation("STRENGTHEN_AND_UPDATE", connections=[12, 4, 12], tags=["topic"],
                       neighbor_updates=[{"memory_index": 4, "context": "new context", "tags": []}])
    actual = parse_fused_evolution("```json\n" + json.dumps(result) + "\n```", [12, 4, 9])
    assert actual == {**result, "connections": [12, 4]}


@pytest.mark.parametrize("decision", ["NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"])
def test_every_fused_decision_can_choose_no_concrete_changes(decision):
    assert parse_fused_evolution(json.dumps(operation(decision)), [4]) == operation(decision)


@pytest.mark.parametrize("result", [
    [], {}, operation("invented"), operation(None), {**operation(), "reason": "unused"},
    operation("STRENGTHEN", connections=[0]),  # 0 is a local position, not retrieved global row.
    operation("STRENGTHEN", connections=[True]),
    operation("STRENGTHEN", connections=[4.0]),
    operation("STRENGTHEN", connections="4"),
    operation("STRENGTHEN", tags="tag"),
    operation("STRENGTHEN", tags=[1]),
    operation("UPDATE_NEIGHBOR", neighbor_updates={}),
    operation("UPDATE_NEIGHBOR", neighbor_updates=[{"memory_index": 4, "tags": []}]),
    operation("UPDATE_NEIGHBOR", neighbor_updates=[{"memory_index": 0, "context": "", "tags": []}]),
    operation("UPDATE_NEIGHBOR", neighbor_updates=[{"memory_index": True, "context": "", "tags": []}]),
    operation("UPDATE_NEIGHBOR", neighbor_updates=[{"memory_index": 4, "context": None, "tags": []}]),
    operation("UPDATE_NEIGHBOR", neighbor_updates=[{"memory_index": 4, "context": "", "tags": [None]}]),
    operation("UPDATE_NEIGHBOR", neighbor_updates=[{"memory_index": 4, "context": "", "tags": []}] * 2),
    operation("NO_EVOLUTION", connections=[4]),
    operation("NO_EVOLUTION", tags=["tag"]),
    operation("STRENGTHEN", neighbor_updates=[{"memory_index": 4, "context": "new", "tags": []}]),
    operation("UPDATE_NEIGHBOR", connections=[4]),
    operation("UPDATE_NEIGHBOR", tags=["new"]),
])
def test_fused_parser_rejects_invalid_or_contradictory_mutations(result):
    with pytest.raises(ValueError, match="fused|EVOLUTION|STRENGTHEN|UPDATE_NEIGHBOR"):
        parse_fused_evolution(json.dumps(result), [4, 12])


def test_fused_parser_rejects_freeform_or_truncated_text():
    for text in ["I think no changes are needed.", '{"decision":']:
        with pytest.raises(ValueError, match="JSON"):
            parse_fused_evolution(text, [4])
