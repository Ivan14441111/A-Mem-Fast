"""Protocol checks that run without model loading or a live inference server."""
from pathlib import Path
from types import SimpleNamespace
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from benchmark import cache_name_for, select_subset
from benchmark_suite import aggregate, build_plan


def args(**changes):
    values = dict(dataset=str(SCRIPTS.parent / "data/locomo10.json"), samples=[0, 1, 2],
                  turns=[128], repeats=3, policies=["upstream"], order="counterbalanced",
                  order_seed=42, max_questions=12, include_fused=True)
    return SimpleNamespace(**(values | changes))


def test_three_arm_order_is_balanced_within_every_conversation():
    plan = build_plan(args())
    assert not plan["skipped_subsets"]
    assert len(plan["cells"]) == 9
    for sample in [0, 1, 2]:
        cells = [cell for cell in plan["cells"] if cell["sample_index"] == sample]
        for position in range(3):
            assert {cell["modes"].split(",")[position] for cell in cells} == {"baseline", "optimized", "fused"}
        assert all(cell["question_count"] == 12 for cell in cells)
        assert all(len(set(cell["source_qa_indices"])) == 12 for cell in cells)


def test_subset_is_evidence_complete():
    subset = select_subset(SCRIPTS.parent / "data/locomo10.json", 0, 128, 12)
    turn_ids = {turn["dia_id"] for turn in subset["turns"]}
    assert all(set(question["evidence"]) <= turn_ids for question in subset["questions"])


def test_self_replay_caches_are_separate():
    assert [cache_name_for(mode) for mode in ["baseline", "replay", "optimized", "warm", "fused", "fused_warm"]] == [
        "baseline", "baseline", "optimized", "optimized", "fused", "fused"]


def test_incomplete_cells_are_excluded_from_statistics():
    assert aggregate([{"status": "failed"}, {"status": "pending"}]) == {}


def test_derived_report_splits_qa_tokens_and_preserves_evolution_counts(tmp_path):
    import json
    from summarize_suite import summarize

    def run(build_tokens, all_tokens):
        def summary(tokens):
            return {"http_requests": 2, "by_stage": {"stage": {
                "prompt_tokens_live": tokens, "completion_tokens_live": tokens // 2}}}
        return {"build_seconds": 10, "qa_seconds": 2,
                "build_llm": summary(build_tokens), "all_llm": summary(all_tokens),
                "metrics": {"token_f1": 0.5, "exact_match": 0}}

    cell = {"name": "case", "status": "complete", "sample_index": 0, "sample_id": "sample",
            "actual_turns": 128, "question_count": 12, "repeat": 1,
            "modes": "baseline,optimized,fused,replay,warm,fused_warm",
            "summary": {"runs": {"baseline": run(100, 160), "fused": run(60, 100)}}}
    directory = tmp_path / "case"
    directory.mkdir()
    for arm, count in [("baseline", 127), ("fused", 10)]:
        raw = {"evolution_count": count, "memory_count": 128,
               "operation_counts": {"consolidate_memories": int(count >= 100)},
               "evolution_metrics": {"fused_decisions": {"NO_EVOLUTION": 117}} if arm == "fused" else {},
               "qa": [{"context_rows": 10, "context_unique_rows": 6, "context_utf8_bytes": 256}]}
        (directory / f"{arm}.json").write_text(json.dumps(raw))
    result = summarize({"cells": [cell, {"name": "failed", "status": "failed", "returncode": 1}]}, tmp_path)
    baseline, fused = result["cells"]
    assert baseline["build_prompt_tokens_live"] == 100
    assert baseline["qa_prompt_tokens_live"] == 60
    assert baseline["consolidation_count"] == 1
    assert fused["consolidation_count"] == 0
    assert fused["evolution_decisions"]["NO_EVOLUTION"] == 117
    assert len(result["noncomplete_cells"]) == 1


def test_sensitivity_aggregation_gives_conversations_equal_weight():
    import math
    from analyze_suite_views import describe

    def cell(sample, repeat, speedup, f1):
        baseline = {"build_seconds": 4, "qa_seconds": 0, "metrics": {"token_f1": 0.5}}
        candidate = {"build_seconds": 4 / speedup, "qa_seconds": 0, "metrics": {"token_f1": f1}}
        return {"origin": "primary", "cell": {"name": f"s{sample}r{repeat}", "status": "complete",
                "sample_index": sample, "actual_turns": 128, "source_qa_indices": [0, 1],
                "summary": {"runs": {"baseline": baseline, "optimized": candidate}}}}

    data = [cell(0, 1, 4, 0.4), *[cell(1, repeat, 1, 0.6) for repeat in [1, 2, 3]]]
    result = describe(data)
    metric = result["metrics"]["optimized"]
    assert math.isclose(metric["build_speedup_equal_conversation_geomean"], 2.0)
    assert math.isclose(metric["candidate_f1_equal_conversation_mean"], 0.5)
    assert metric["paired_runs"] == 4
    assert metric["independent_conversations"] == 2
    assert result["unique_selected_qa"] == 4


def test_failed_arm_artifact_preserves_bounded_llm_diagnostic_and_progress(tmp_path, monkeypatch):
    import json
    import benchmark
    import pytest

    error = ValueError("LLM reached max_tokens; increase budget")
    error.llm_diagnostic = {"stage": "strengthen", "usage": {"completion_tokens": 2000},
                            "content_prefix": "bounded preview"}
    error.benchmark_progress = {"phase": "build", "failed_turn_1based": 116, "dia_id": "D7:8"}
    def fail(*args):
        raise error
    monkeypatch.setattr(benchmark, "_run_mode", fail)
    with pytest.raises(ValueError, match="max_tokens"):
        benchmark.run_mode(SimpleNamespace(output=tmp_path, evolution_policy="upstream"),
                           {"fingerprint": "subset"}, "optimized", None)
    saved = json.loads((tmp_path / "optimized-failure.json").read_text())
    assert saved["llm_diagnostic"]["usage"]["completion_tokens"] == 2000
    assert saved["progress"]["failed_turn_1based"] == 116


def test_continue_on_error_cannot_hide_replay_or_configuration_failures():
    from benchmark_suite import is_integrity_failure

    assert not is_integrity_failure([{"mode": "optimized", "exception_type": "ValueError",
                                      "error": "LLM reached max_tokens; increase budget"}])
    assert is_integrity_failure([])
    assert is_integrity_failure([{"mode": "warm", "exception_type": "RuntimeError",
                                  "error": "Replay cache miss"}])
    assert is_integrity_failure([{"mode": "baseline", "exception_type": "TypeError",
                                  "error": "unexpected keyword argument 'tag_budget_policy'"}])


def test_completion_report_keeps_failed_and_pending_planned_denominator(tmp_path):
    import json
    from benchmark_suite import write_report

    report = {"cells": [{"status": "failed"}, {"status": "failed"}, {"status": "pending"}]}
    path = tmp_path / "suite.json"
    write_report(path, report)
    saved = json.loads(path.read_text())
    assert saved["completion"]["planned_cells"] == 3
    assert saved["completion"]["failed_cells"] == 2
    assert saved["completion"]["attempted_cells"] == 2
    assert saved["completion"]["all_attempted"] is False
    assert saved["aggregate"] == {}
