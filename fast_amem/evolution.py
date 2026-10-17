"""An explicit, experimental one-request evolution policy.

This policy changes the inference task. It is deliberately separate from the
upstream three-request path and must be evaluated as an algorithm ablation.
"""

from __future__ import annotations

import json

from llm_text_parsers import strip_markdown_fences


FUSED_EVOLUTION_PROMPT = """Evolve this new memory with its retrieved neighbors in one operation.
Choose exactly one decision:
- NO_EVOLUTION: this memory stands alone.
- STRENGTHEN: connect the new memory to relevant neighbors and optionally replace its tags.
- UPDATE_NEIGHBOR: revise relevant neighbors' context or tags using the new information.
- STRENGTHEN_AND_UPDATE: do both.

New memory:
- Content: {content}
- Context: {context}
- Keywords: {keywords}
- Tags: {tags}

Retrieved neighbors (each 'memory index' is a GLOBAL corpus row):
{neighbors}

Return only one JSON object, with exactly these fields:
{{"decision":"NO_EVOLUTION","connections":[],"tags":[],"neighbor_updates":[]}}

Rules:
- connections contains integer GLOBAL memory index values from the retrieved neighbors, never local positions in this list.
- tags is a list of strings replacing the NEW memory's tags. Empty means keep its current tags.
- neighbor_updates contains only changed neighbors. Each entry has exactly these fields:
  {{"memory_index":0,"context":"updated context","tags":["updated tag"]}}
- Each memory_index must be a GLOBAL index appearing in the retrieved neighbors, used at most once.
- In an update, an empty context or empty tags means preserve that field.
- NO_EVOLUTION requires all three lists empty.
- STRENGTHEN requires neighbor_updates empty.
- UPDATE_NEIGHBOR requires connections and tags empty.
- Do not add explanations or repeat unchanged neighbors.
"""

DECISIONS = frozenset({"NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"})


def _strings(value, field):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"fused evolution {field} must be a list of strings")
    return value


def parse_fused_evolution(response: str, neighbor_indices) -> dict:
    """Validate the entire response before any memory is mutated.

    Reject invented rows, duplicate updates, contradictory actions, and malformed
    fields. Connections are deduplicated in their first-occurrence order.
    """
    try:
        result = json.loads(strip_markdown_fences(response))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("fused evolution must return a JSON object") from exc
    fields = {"decision", "connections", "tags", "neighbor_updates"}
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError("fused evolution must contain exactly decision, connections, tags, neighbor_updates")
    decision = result["decision"]
    if not isinstance(decision, str) or decision not in DECISIONS:
        raise ValueError("fused evolution has an invalid decision")
    allowed = {int(row) for row in neighbor_indices}
    connections = result["connections"]
    if (not isinstance(connections, list) or
            any(type(row) is not int or row not in allowed for row in connections)):
        raise ValueError("fused evolution connections must reference retrieved global indices")
    result["connections"] = list(dict.fromkeys(connections))
    _strings(result["tags"], "tags")
    updates = result["neighbor_updates"]
    if not isinstance(updates, list):
        raise ValueError("fused evolution neighbor_updates must be a list")
    seen = set()
    for update in updates:
        if not isinstance(update, dict) or set(update) != {"memory_index", "context", "tags"}:
            raise ValueError("fused evolution update requires memory_index, context, tags")
        row = update["memory_index"]
        if type(row) is not int or row not in allowed or row in seen:
            raise ValueError("fused evolution update must reference a unique retrieved global index")
        seen.add(row)
        if not isinstance(update["context"], str):
            raise ValueError("fused evolution update context must be a string")
        _strings(update["tags"], "update tags")
    if decision == "NO_EVOLUTION" and (connections or result["tags"] or updates):
        raise ValueError("NO_EVOLUTION cannot contain mutations")
    if decision == "STRENGTHEN" and updates:
        raise ValueError("STRENGTHEN cannot update neighbors")
    if decision == "UPDATE_NEIGHBOR" and (connections or result["tags"]):
        raise ValueError("UPDATE_NEIGHBOR cannot strengthen the new memory")
    return result
