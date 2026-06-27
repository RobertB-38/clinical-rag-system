"""Evaluate the clinical RAG system and (optionally) gate CI on regression.

    python -m eval.run_eval            # report only
    python -m eval.run_eval --check    # also exit 1 if below eval/thresholds.yaml

Metrics
-------
Deterministic (free, reproducible, CI-gating):
  hit@k               retrieval surfaced the expected guideline
  keyword_correctness expected key terms appear in the answer
  citation_validity   answer [n] cites only retrieved passages

LLM-judge (needs a real model; reported, not gated):
  judge_correctness   frontier model rates clinical correctness 0-1
  judge_groundedness  frontier model rates claim support 0-1

Providers come from the environment. With the free local stack
(RAG_LLM_PROVIDER=fake) you still get hit@k and the deterministic scorers; set
RAG_LLM_PROVIDER=claude + RAG_ANTHROPIC_API_KEY for generation and the judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from app.config import get_settings
from app.rag.pipeline import RagPipeline
from eval.scorers import citation_validity, get_judge, keyword_correctness

REPORT_PATH = Path("eval/last_report.json")
THRESHOLDS_PATH = Path("eval/thresholds.yaml")


def load_eval(path: str = "eval/qa.yaml") -> list[dict]:
    return (yaml.safe_load(Path(path).read_text()) or {}).get("questions", [])


def source_id_from_url(url: str, sources: list[dict]) -> str:
    for s in sources:
        if s.get("url") == url:
            return s["id"]
    return url


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def run() -> dict:
    settings = get_settings()
    pipeline = RagPipeline(settings)
    judge = get_judge(settings)
    real_llm = settings.llm_provider != "fake"

    manifest = (yaml.safe_load(Path(settings.sources_file).read_text()) or {}).get("sources", [])
    items = load_eval()

    hits = 0
    kw_scores: list[float] = []
    cite_scores: list[float] = []
    judge_corr: list[float] = []
    judge_grnd: list[float] = []

    for item in items:
        result = pipeline.answer(item["question"])
        retrieved_ids = {source_id_from_url(h.source_url, manifest) for h in result.contexts}
        hit = item["expected_source_id"] in retrieved_ids
        hits += int(hit)

        keywords = item.get("expected_keywords", [])
        kw_scores.append(keyword_correctness(result.answer, keywords))

        if real_llm and not result.refused:
            cite_scores.append(1.0 if citation_validity(result.answer, len(result.contexts)) else 0.0)
            judge_corr.append(judge.correctness(item["question"], result.answer, keywords))
            judge_grnd.append(judge.groundedness(result.answer, result.contexts))

        print(f"  {'HIT ' if hit else 'MISS'}  {item['question']}")

    n = len(items) or 1
    report = {
        "n_questions": len(items),
        "provider": settings.llm_provider,
        "hit@k": round(hits / n, 3),
        "keyword_correctness": _mean(kw_scores),
        "citation_validity": _mean(cite_scores),
        "judge_correctness": _mean(judge_corr),
        "judge_groundedness": _mean(judge_grnd),
    }

    print("\n=== Evaluation ===")
    for k, v in report.items():
        print(f"  {k:22}: {v}")
    if not real_llm:
        print("  (note: provider=fake — judge & citation metrics need a real LLM)")

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


def check_thresholds(report: dict) -> tuple[bool, list[str]]:
    """Compare a report to eval/thresholds.yaml. Returns (ok, failures)."""
    if not THRESHOLDS_PATH.exists():
        return True, []
    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text()) or {}
    failures = []
    for metric, floor in thresholds.items():
        value = report.get(metric)
        if value is None:
            continue  # metric not measured in this run (e.g. fake provider)
        if value < floor:
            failures.append(f"{metric}={value} < required {floor}")
    return len(failures) == 0, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 on threshold regression")
    args = parser.parse_args()

    report = run()
    if args.check:
        ok, failures = check_thresholds(report)
        if not ok:
            print("\nFAIL — eval regression:")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)
        print("\nPASS — all measured metrics meet thresholds.")
