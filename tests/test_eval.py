"""Eval-harness tests — deterministic scorers and the CI gate, no network."""
from eval.scorers import (
    FakeJudge,
    citation_validity,
    keyword_correctness,
)
from eval.run_eval import check_thresholds
from app.rag.vector_store import Hit


def test_citation_validity_accepts_in_range():
    assert citation_validity("Give fluids [1] then antibiotics [2].", 4) is True


def test_citation_validity_rejects_fabricated():
    assert citation_validity("As shown [9].", 4) is False  # only 4 retrieved


def test_citation_validity_rejects_no_citation():
    assert citation_validity("Give fluids then antibiotics.", 4) is False


def test_keyword_correctness_fraction():
    assert keyword_correctness("antibiotic within one hour", ["antibiotic", "hour"]) == 1.0
    assert keyword_correctness("antibiotic only", ["antibiotic", "hour"]) == 0.5
    assert keyword_correctness("anything", []) == 1.0


def test_fake_judge_is_deterministic_and_offline():
    j = FakeJudge()
    ctx = [Hit("give antibiotics within one hour", 0.9, "S", "u", "s::0")]
    assert j.correctness("q", "antibiotic hour", ["antibiotic", "hour"]) == 1.0
    g = j.groundedness("give antibiotics within one hour", ctx)
    assert 0.0 <= g <= 1.0
    assert j.groundedness("", ctx) == 0.0


def test_check_thresholds_passes_when_met():
    report = {"hit@k": 0.92, "keyword_correctness": 0.8, "citation_validity": 0.9}
    ok, failures = check_thresholds(report)
    # thresholds.yaml floors: 0.85 / 0.55 / 0.85 -> all met
    assert ok is True
    assert failures == []


def test_check_thresholds_fails_on_regression():
    report = {"hit@k": 0.50, "keyword_correctness": 0.8, "citation_validity": 0.9}
    ok, failures = check_thresholds(report)
    assert ok is False
    assert any("hit@k" in f for f in failures)


def test_check_thresholds_skips_unmeasured_metrics():
    # fake-provider run: judge/citation are None -> skipped, not failed
    report = {"hit@k": 0.92, "keyword_correctness": 0.8, "citation_validity": None}
    ok, _ = check_thresholds(report)
    assert ok is True
