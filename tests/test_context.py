"""Context deduplication is an explicit QA-only prompt ablation."""

from types import SimpleNamespace
import json

import pytest

from fast_amem.client import SQLiteResponseCache
from fast_amem.system import FastAgenticMemorySystem
from scripts.benchmark_context import FrozenQueryClient, ReadOnlySnapshotCache, context_plan, query_prompt


class Retriever:
    def __init__(self, indices):
        self.indices = indices
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.indices


def memory(row, links):
    return SimpleNamespace(timestamp=f"date-{row}", content=f"text-{row}", context=f"context-{row}",
                           keywords=[f"keyword-{row}"], tags=[f"tag-{row}"], links=links)


def system(policy="upstream"):
    instance = FastAgenticMemorySystem(None, None, context_policy=policy)
    instance.memories = {"a": memory(0, [1]), "b": memory(1, []), "c": memory(2, [0, 1, 0])}
    instance.retriever = Retriever([2, 0])
    return instance


def test_dedup_keeps_first_occurrence_exact_format_and_graph():
    instance = system()
    original = instance.find_related_memories_raw("query", 5)
    assert original.count("talk start time:") == 6
    instance.context_policy = "deduplicated"
    actual = instance.find_related_memories_raw("query", 5)
    # First occurrences in upstream output are main row2, linked row0, row1.
    assert actual == "".join(line + "\n" for line in original.splitlines()[:3])
    assert instance.context_metrics == {"queries": 1, "rows_before": 6, "rows_after": 3}
    assert instance.memories["c"].links == [0, 1, 0]
    assert instance.retriever.calls == [("query", 5), ("query", 5)]


def test_dedup_retains_original_k_plus_one_expansion_cap():
    instance = system("deduplicated")
    instance.retriever.indices = [2]
    actual = instance.find_related_memories_raw("query", 1)
    assert actual.count("talk start time:") == 3  # main plus two neighbors.
    assert instance.context_metrics["rows_before"] == 3


def test_negative_links_alias_the_same_row_for_deduplication():
    instance = system("deduplicated")
    instance.memories["c"].links = [-1, 2]
    instance.retriever.indices = [2]
    assert instance.find_related_memories_raw("query", 5).count("talk start time:") == 1


def test_dedup_preserves_bounds_errors_from_invalid_links():
    instance = system("deduplicated")
    instance.memories["c"].links = [999]
    with pytest.raises(IndexError):
        instance.find_related_memories_raw("query")


def test_read_only_response_snapshot_never_changes_the_source(tmp_path):
    path = tmp_path / "source.sqlite"
    source = SQLiteResponseCache(path)
    source.put("key", "{}", {"response": "original"})
    snapshot = ReadOnlySnapshotCache(path)
    source.put("key", "{}", {"response": "later"})
    assert snapshot.get("key") == {"response": "original"}
    assert snapshot.get("missing") is None


def test_fixed_keyword_proxy_cannot_silently_regenerate_keywords():
    calls = []
    client = SimpleNamespace(get_completion=lambda prompt, **kwargs: calls.append(prompt) or "answer")
    recorded = query_prompt("A question?")
    proxy = FrozenQueryClient({recorded: "fixed keywords"}, client)
    assert proxy.get_completion(recorded) == "fixed keywords"
    assert calls == []
    assert proxy.get_completion("Based on the context: text") == "answer"
    assert calls == ["Based on the context: text"]
    with pytest.raises(RuntimeError, match="not recorded"):
        proxy.get_completion(query_prompt("Different question?"))
    with pytest.raises(RuntimeError, match="permits only"):
        proxy.get_completion("An unexpected evolution request")


def test_offline_context_plan_uses_one_graph_and_fixed_keywords():
    instance = system()
    qa = {"question": "Question?", "category": 4, "source_qa_index": 7}
    plan = context_plan(instance, {query_prompt(qa["question"]): json.dumps({"keywords": "shared query"})},
                        [qa], 5)
    assert len(plan) == 1
    assert plan[0]["upstream"]["context_rows"] == 6
    assert plan[0]["deduplicated"]["context_rows"] == 3
    assert instance.retriever.calls == [("shared query", 5), ("shared query", 5)]
    assert plan[0]["deduplicated"]["context_utf8_bytes"] < plan[0]["upstream"]["context_utf8_bytes"]


def test_empty_context_dedup_returns_no_text():
    instance = FastAgenticMemorySystem(None, None, context_policy="deduplicated")
    assert instance.find_related_memories_raw("query") == ""


def test_invalid_context_policy_is_rejected():
    with pytest.raises(ValueError, match="context_policy"):
        FastAgenticMemorySystem(None, None, context_policy="unknown")
