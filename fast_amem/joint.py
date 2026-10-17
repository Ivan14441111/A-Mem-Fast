"""Experimental policy: one ordered LLM request for metadata and evolution.

This changes the model's conditioning and output task. It is an algorithm
ablation, not an exact replacement for upstream or metadata-prefetched fused
evolution. No speculative metadata requests are made by this class.
"""

from __future__ import annotations

import json
from operator import index

from llm_text_parsers import strip_markdown_fences
from memory_layer_robust import RobustMemoryNote
from fast_amem.evolution import parse_fused_evolution
from fast_amem.system import FastAgenticMemorySystem


JOINT_NOTE_PROMPT = """Create and evolve one memory note in one operation.

First derive metadata about the NEW CONTENT ONLY:
- keywords: at least three important keywords, most important first; exclude speaker names and time references.
- context: one sentence summarizing the new content's main topic, key points, and purpose.
- tags: at least three broad themes or categories.
Use the retrieved neighbors only for evolution, not as new facts in the metadata.

Then choose one evolution decision:
- NO_EVOLUTION: this memory stands alone.
- STRENGTHEN: link the new note to relevant neighbors and optionally replace its tags.
- UPDATE_NEIGHBOR: revise relevant neighbors' context or tags using the new information.
- STRENGTHEN_AND_UPDATE: do both.

NEW CONTENT:
{content}

RETRIEVED NEIGHBORS (memory index values are GLOBAL corpus rows):
{neighbors}

Allowed global row indices: {allowed_indices}

Return ONLY one JSON object with exactly this structure:
{{"metadata":{{"keywords":["keyword1","keyword2","keyword3"],"context":"One sentence.","tags":["theme1","theme2","theme3"]}},"evolution":{{"decision":"NO_EVOLUTION","connections":[],"tags":[],"neighbor_updates":[]}}}}

Evolution rules:
- connections is a list of integer GLOBAL row indices from the allowed indices; never use local positions.
- evolution.tags replaces the NEW note's tags; an empty list preserves metadata.tags.
- neighbor_updates includes only changed retrieved neighbors. Each entry has exactly memory_index (an allowed integer GLOBAL row), context (string), and tags (list of strings).
- Use each memory_index at most once. An empty context or empty tags preserves the original field.
- NO_EVOLUTION requires connections, evolution.tags, and neighbor_updates all empty.
- STRENGTHEN requires neighbor_updates empty.
- UPDATE_NEIGHBOR requires connections and evolution.tags empty.
- With no retrieved neighbors, choose NO_EVOLUTION and leave all evolution lists empty.
- Do not include a reason, explanations, or unchanged neighbor copies.
"""


def parse_joint_note(response, neighbor_indices):
    """Validate both metadata and all evolution mutations before constructing a note."""
    try:
        result = json.loads(strip_markdown_fences(response))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("joint note must return a JSON object") from exc
    if not isinstance(result, dict) or set(result) != {"metadata", "evolution"}:
        raise ValueError("joint note requires exactly metadata and evolution")
    metadata = result["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {"keywords", "context", "tags"}:
        raise ValueError("joint metadata requires exactly keywords, context, tags")
    for field in ("keywords", "tags"):
        values = metadata[field]
        if (not isinstance(values, list) or not values or
                any(not isinstance(value, str) or not value.strip() for value in values)):
            raise ValueError(f"joint metadata {field} must be a nonempty list of nonempty strings")
    if not isinstance(metadata["context"], str) or not metadata["context"].strip():
        raise ValueError("joint metadata context must be a nonempty string")
    indices = tuple(int(row) for row in neighbor_indices)
    evolution = parse_fused_evolution(json.dumps(result["evolution"]), indices)
    if not indices and evolution["decision"] != "NO_EVOLUTION":
        raise ValueError("joint note without neighbors requires NO_EVOLUTION")
    return {"metadata": metadata, "evolution": evolution}


class JointAgenticMemorySystem(FastAgenticMemorySystem):
    """Experimental sequential write path with exactly one request per new note."""

    def __init__(self, model, client, *, evo_threshold=100, context_policy="upstream"):
        super().__init__(model, client, evo_threshold=evo_threshold,
                         evolution_policy="fused", strict_failures=True,
                         context_policy=context_policy)
        self.evolution_policy = "joint"
        self.evolution_metrics.update({"joint_attempts": 0, "joint_parse_failures": 0,
                                       "joint_invalid_reference_failures": 0,
                                       "joint_decisions": {}, "joint_connections": 0,
                                       "joint_neighbor_updates": 0})

    def add_note(self, content, time=None, **kwargs):
        """Commit a validated note, then perform any scheduled consolidation.

        A consolidation failure occurs after insertion has committed, matching
        upstream. The note remains present; repair refresh rather than retrying
        the same turn.
        """
        if set(kwargs) & {"keywords", "context", "tags", "category", "llm_controller", "timestamp"}:
            raise ValueError("joint policy derives metadata in its one request; metadata overrides are unsupported")
        if kwargs.get("id") in self.memories:
            raise ValueError("memory id already exists")
        neighbors, indices = self.find_related_memories(content, k=5)
        self.evolution_metrics["joint_attempts"] += 1
        response = self.llm_controller.llm.get_completion(JOINT_NOTE_PROMPT.format(
            content=content, neighbors=neighbors, allowed_indices=[int(row) for row in indices]))
        try:
            result = parse_joint_note(response, indices)
        except ValueError as exc:
            self.evolution_metrics["joint_parse_failures"] += 1
            if "global ind" in str(exc):
                self.evolution_metrics["joint_invalid_reference_failures"] += 1
            raise
        note = RobustMemoryNote(content=content, timestamp=time, category="Uncategorized",
                                **result["metadata"], **kwargs)
        evolution = result["evolution"]
        note.links.extend(evolution["connections"])
        if evolution["tags"]:
            note.tags = evolution["tags"]
        journal = {}
        self._mutation_journal = journal
        try:
            for update in evolution["neighbor_updates"]:
                self._update_neighbor(update["memory_index"], update["context"], update["tags"])
            self.retriever.add_documents([
                "content:" + note.content + " context:" + note.context +
                " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)])
        except Exception:
            for row, (context, tags) in journal.items():
                memory = list(self.memories.values())[row]
                memory.context, memory.tags = context, tags
            raise
        finally:
            self._mutation_journal = None
        self.memories[note.id] = note
        decision_counts = self.evolution_metrics["joint_decisions"]
        decision_counts[evolution["decision"]] = decision_counts.get(evolution["decision"], 0) + 1
        self.evolution_metrics["joint_connections"] += len(evolution["connections"])
        self.evolution_metrics["joint_neighbor_updates"] += len(evolution["neighbor_updates"])
        if evolution["decision"] != "NO_EVOLUTION":
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def process_memory(self, note):
        raise RuntimeError("Joint policy evolves the note inside add_note; call add_note or ingest")

    def ingest(self, turns, *, analysis_workers=4, prefetch_window=16, on_note=None):
        # Keep the harness interface, but never run content-only analysis for this
        # state-dependent policy. Both knobs remain validated for consistent CLI.
        if index(analysis_workers) < 1 or index(prefetch_window) < 1:
            raise ValueError("Worker count and prefetch window must be positive")
        for count, turn in enumerate(turns, 1):
            self.add_note(turn["content"], time=turn["timestamp"])
            if on_note:
                on_note(count)
