"""Fixed generic JSON schemas for explicitly opt-in policy output guidance.

These schemas constrain syntax and field types. Candidate IDs, conditional
actions, nonempty metadata, and other domain rules remain application checks.
"""

from __future__ import annotations


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _strings():
    return {"type": "array", "items": {"type": "string"}}


def _evolution_schema():
    update = _object({"memory_index": {"type": "integer"},
                      "context": {"type": "string"}, "tags": _strings()})
    return _object({
        "decision": {"type": "string", "enum": [
            "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"]},
        "connections": {"type": "array", "items": {"type": "integer"}},
        "tags": _strings(),
        "neighbor_updates": {"type": "array", "items": update},
    })


def policy_response_format(stage):
    """Return a fresh standard response_format only for fused or joint calls.

    There is no dynamic candidate enum or array-size cap. In particular, the
    observed tag instruction is not converted into a hard schema maximum.
    """
    if stage == "evolution_fused":
        schema = _evolution_schema()
        name = "amem_fused_evolution_v1"
    elif stage == "joint_note":
        metadata = _object({"keywords": _strings(), "context": {"type": "string"}, "tags": _strings()})
        schema = _object({"metadata": metadata, "evolution": _evolution_schema()})
        name = "amem_joint_note_v1"
    else:
        return None
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
