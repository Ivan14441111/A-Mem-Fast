"""Keep upstream evolution sequential; parallelize only independent note analysis."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import logging
from operator import index
from types import SimpleNamespace
import time

from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT, EVOLUTION_DECISION_PROMPT, FOCUSED_KEYWORDS_PROMPT,
    STRENGTHEN_DETAILS_PROMPT, UPDATE_NEIGHBORS_PROMPT, _parse_list_items,
    parse_analyze_content, parse_evolution_decision, parse_strengthen_details,
    parse_update_neighbors, validate_analysis_result,
)
from memory_layer_robust import RobustAgenticMemorySystem, RobustMemoryNote
from fast_amem.evolution import FUSED_EVOLUTION_PROMPT, parse_fused_evolution
from fast_amem.retriever import FastEmbeddingRetriever

logger = logging.getLogger(__name__)


class _StrictMemoryNote(RobustMemoryNote):
    """Use upstream parsing while allowing inference failures to reach callers."""

    @staticmethod
    def analyze_content(content, llm_controller):
        response = llm_controller.llm.get_completion(ANALYZE_CONTENT_PROMPT.format(content=content))
        analysis = parse_analyze_content(response, content)
        if not analysis["keywords"]:
            response = llm_controller.llm.get_completion(
                FOCUSED_KEYWORDS_PROMPT.format(content=content), temperature=0.3)
            analysis["keywords"] = _parse_list_items(response)
        return validate_analysis_result(analysis, content)


class FastAgenticMemorySystem(RobustAgenticMemorySystem):
    def __init__(self, model, client, *, evo_threshold=100,
                 evolution_policy="upstream", strict_failures=True,
                 context_policy="upstream"):
        evo_threshold = index(evo_threshold)
        if evo_threshold <= 0:
            raise ValueError("evo_threshold must be positive")
        if evolution_policy not in {"upstream", "fused"}:
            raise ValueError("evolution_policy must be 'upstream' or 'fused'")
        if context_policy not in {"upstream", "deduplicated"}:
            raise ValueError("context_policy must be 'upstream' or 'deduplicated'")
        # Same upstream state, with shared encoder and the measured controller.
        self.memories = {}
        self.retriever = FastEmbeddingRetriever(model)
        self.llm_controller = SimpleNamespace(llm=client)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.evolution_policy = evolution_policy
        self.strict_failures = strict_failures
        self.context_policy = context_policy
        self.evolution_metrics = {
            "fused_attempts": 0, "fused_parse_failures": 0,
            "fused_invalid_reference_failures": 0,
            "fused_decisions": {}, "fused_connections": 0, "fused_neighbor_updates": 0,
        }
        self.context_metrics = {"queries": 0, "rows_before": 0, "rows_after": 0}
        self.consolidation_seconds = []

    def add_note(self, content, time=None, **kwargs):
        """Retain upstream construction while making failed inference visible.

        The default uses upstream prompts and parsers unchanged on successful
        requests. ``strict_failures=False`` restores its heuristic metadata
        fallback and graceful evolution fallback for interactive use.

        Like upstream, note insertion commits before periodic consolidation.
        If consolidation raises, the note remains stored; repair the refresh
        instead of blindly retrying the same turn.
        """
        note_class = _StrictMemoryNote if self.strict_failures else RobustMemoryNote
        note = note_class(content=content, llm_controller=self.llm_controller,
                          timestamp=time, **kwargs)
        if note.id in self.memories:
            raise ValueError("memory id already exists")
        # Evolution changes only the new note and retrieved neighbors. Journal
        # those few fields so an encoding failure cannot leave neighbor changes
        # from a note that was never added.
        journal = {}
        self._mutation_journal = journal
        try:
            evo_label, note = self.process_memory(note)
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
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def _update_neighbor(self, row, context, tags):
        memory = list(self.memories.values())[row]
        journal = getattr(self, "_mutation_journal", None)
        if journal is not None and row not in journal:
            journal[row] = (memory.context, memory.tags)
        if context:
            memory.context = context
        if tags:
            memory.tags = tags

    def process_memory(self, note):
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        if len(indices) == 0:
            return False, note
        try:
            if self.evolution_policy == "fused":
                return self._evolve_fused(note, neighbor_memory, indices)
            return self._evolve_upstream(note, neighbor_memory, indices)
        except Exception:
            if self.strict_failures:
                raise
            logger.exception("Evolution failed for note %s; storing without evolution", note.id)
            return False, note

    def _evolve_upstream(self, note, neighbor_memory, indices):
        # This is the successful path of RobustAgenticMemorySystem.process_memory.
        # Keep its prompt order, parsers, and local-to-global update mapping.
        client = self.llm_controller.llm
        response = client.get_completion(EVOLUTION_DECISION_PROMPT.format(
            context=note.context, content=note.content, keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory))
        decision = parse_evolution_decision(response)["decision"]
        if decision == "NO_EVOLUTION":
            return False, note
        if decision in {"STRENGTHEN", "STRENGTHEN_AND_UPDATE"}:
            response = client.get_completion(STRENGTHEN_DETAILS_PROMPT.format(
                content=note.content, keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory))
            strengthen = parse_strengthen_details(response)
            note.links.extend(strengthen["connections"])
            if strengthen["tags"]:
                note.tags = strengthen["tags"]
        if decision in {"UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"}:
            response = client.get_completion(UPDATE_NEIGHBORS_PROMPT.format(
                content=note.content, context=note.context,
                nearest_neighbors_memories=neighbor_memory,
                max_neighbor_idx=len(indices) - 1, neighbor_count=len(indices)))
            updates = parse_update_neighbors(response, len(indices))
            for local, update in enumerate(updates[:len(indices)]):
                row = int(indices[local])
                if row < len(self.memories):
                    self._update_neighbor(row, update["context"], update["tags"])
        return True, note

    def _evolve_fused(self, note, neighbor_memory, indices):
        self.evolution_metrics["fused_attempts"] += 1
        response = self.llm_controller.llm.get_completion(FUSED_EVOLUTION_PROMPT.format(
            content=note.content, context=note.context, keywords=note.keywords,
            tags=note.tags, neighbors=neighbor_memory))
        try:
            result = parse_fused_evolution(response, indices)
        except ValueError as exc:
            self.evolution_metrics["fused_parse_failures"] += 1
            if "global ind" in str(exc):
                self.evolution_metrics["fused_invalid_reference_failures"] += 1
            raise
        decisions = self.evolution_metrics["fused_decisions"]
        decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
        self.evolution_metrics["fused_connections"] += len(result["connections"])
        self.evolution_metrics["fused_neighbor_updates"] += len(result["neighbor_updates"])
        # All fields and row references have been validated before any mutation.
        note.links.extend(result["connections"])
        if result["tags"]:
            note.tags = result["tags"]
        for update in result["neighbor_updates"]:
            self._update_neighbor(update["memory_index"], update["context"], update["tags"])
        return result["decision"] != "NO_EVOLUTION", note

    def find_related_memories_raw(self, query, k=5):
        """Optionally emit each QA memory row once, in first-occurrence order.

        Deduplication changes the answer prompt and is a separate ablation. The
        upstream policy delegates to the original formatter and expansion loop.
        The deduplicated policy retains that loop's k+1-neighbor expansion cap.
        """
        if self.context_policy == "upstream":
            return super().find_related_memories_raw(query, k)
        if not self.memories:
            return ""
        indices = self.retriever.search(query, k)
        memories = list(self.memories.values())
        occurrences = []
        for row in indices:
            occurrences.append(int(row))
            for position, neighbor in enumerate(memories[row].links):
                # Access before deduplication to retain upstream bounds errors.
                memories[neighbor]
                occurrences.append(neighbor if neighbor >= 0 else len(memories) + neighbor)
                if position >= k:
                    break
        rows = list(dict.fromkeys(occurrences))
        self.context_metrics["queries"] += 1
        self.context_metrics["rows_before"] += len(occurrences)
        self.context_metrics["rows_after"] += len(rows)
        return "".join(
            "talk start time:" + memories[row].timestamp +
            "memory content: " + memories[row].content +
            "memory context: " + memories[row].context +
            "memory keywords: " + str(memories[row].keywords) +
            "memory tags: " + str(memories[row].tags) + "\n"
            for row in rows)

    def consolidate_memories(self):
        started = time.perf_counter()
        # Preserve upstream refresh timing AND its distinct consolidation format.
        docs = [m.content + " , " + f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
                for m in self.memories.values()]
        self.retriever.rebuild(docs)
        self.consolidation_seconds.append(time.perf_counter() - started)

    def ingest(self, turns, *, analysis_workers=4, prefetch_window=16, on_note=None):
        """A bounded pipeline. Background tasks see content only, never graph state.

        Their exact responses enter the request cache; add_note still invokes the
        upstream constructor/parser and process_memory in chronological order.
        Failed futures propagate instead of silently reporting a successful run.
        """
        client = self.llm_controller.llm
        if client.cache_mode not in {"readwrite", "replay"}:
            raise ValueError("Analysis prefetch requires a response cache")
        analysis_workers, prefetch_window = index(analysis_workers), index(prefetch_window)
        if analysis_workers < 1 or prefetch_window < 1:
            raise ValueError("Worker count and prefetch window must be positive")
        iterator = iter(turns)
        exhausted = object()
        pool = ThreadPoolExecutor(max_workers=analysis_workers)
        pending = deque()

        def submit(turn):
            return turn, pool.submit(client.get_completion,
                                     ANALYZE_CONTENT_PROMPT.format(content=turn["content"]),
                                     _prefetch=True)

        try:
            pending.extend(submit(turn) for turn in islice(iterator, prefetch_window))
            count = 0
            while pending:
                turn, future = pending.popleft()
                future.result()
                # Refill before graph evolution to keep analysis workers busy
                # across former window boundaries. At most window analysis futures remain pending.
                next_turn = next(iterator, exhausted)
                if next_turn is not exhausted:
                    pending.append(submit(next_turn))
                self.add_note(turn["content"], time=turn["timestamp"])
                count += 1
                if on_note:
                    on_note(count)
        finally:
            # Cancel queued speculative analysis after an error; running requests
            # complete before returning, avoiding writes after the caller exits.
            pool.shutdown(wait=True, cancel_futures=True)
