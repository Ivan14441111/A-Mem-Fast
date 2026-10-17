from pathlib import Path

from scripts.benchmark import select_subset, score


def test_subset_uses_only_evidence_inside_prefix():
    path = Path(__file__).resolve().parents[1] / "data/locomo10.json"
    subset = select_subset(path, 0, 32, 12)
    ids = {t["dia_id"] for t in subset["turns"]}
    assert len(subset["turns"]) == 32
    assert len(subset["questions"]) == 12
    assert {q["category"] for q in subset["questions"]} == {1, 2, 3, 4}
    assert all(q["evidence"] and all(e in ids for e in q["evidence"])
               for q in subset["questions"])


def test_scoring_is_explicit_normalized_em_and_token_f1():
    assert score("The red, dog!", "red dog") == {"exact_match": 1.0, "token_f1": 1.0}
    assert score("dog", "red dog")["token_f1"] == 2 / 3
