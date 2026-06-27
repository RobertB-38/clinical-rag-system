"""Cost-aware model router.

Easy questions go to a cheap model, hard ones to a frontier model. The
classifier is a deliberately simple, explainable heuristic — length plus a set
of clinical-reasoning markers (dosing, interactions, differentials, management
steps). It is NOT machine learning; the point is measurable cost control you
can defend, and the actual saving is reported by the cost dashboard.

`RouterGenerator` implements the same Generator interface as a single model, so
the pipeline does not know or care that routing is happening.
"""
from __future__ import annotations

from typing import Iterator

from app.config import Settings
from app.observability import record_router
from app.rag.vector_store import Hit

# Markers that suggest genuine clinical reasoning rather than a lookup.
COMPLEX_MARKERS = (
    "differential", "compare", "contraindicat", "interaction", "dose", "dosage",
    "titrat", "manage", "management", "why", "stepwise", "step-wise", "pathophysiolog",
    "escalat", "adjust", "renal", "hepatic", "pregnan", "monitor", "first-line",
)
LONG_QUESTION_WORDS = 25


def classify(question: str) -> str:
    """Return the tier name: 'frontier' for hard queries, else 'cheap'."""
    q = question.lower()
    if len(q.split()) > LONG_QUESTION_WORDS:
        return "frontier"
    if any(marker in q for marker in COMPLEX_MARKERS):
        return "frontier"
    return "cheap"


class RouterGenerator:
    """Picks a generator per query and records which tier was used."""

    def __init__(self, settings: Settings, cheap, frontier) -> None:
        self._settings = settings
        self._cheap = cheap
        self._frontier = frontier
        self._cheap_model = settings.router_cheap_model
        self._frontier_model = settings.router_frontier_model

    def _pick(self, question: str):
        tier = classify(question)
        if tier == "frontier":
            record_router("frontier", self._frontier_model)
            return self._frontier
        record_router("cheap", self._cheap_model)
        return self._cheap

    def answer(self, question: str, contexts: list[Hit]) -> str:
        return self._pick(question).answer(question, contexts)

    def stream(self, question: str, contexts: list[Hit]) -> Iterator[str]:
        return self._pick(question).stream(question, contexts)
