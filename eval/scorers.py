"""Scorers for the clinical RAG eval.

Two tiers, on purpose:

* Deterministic scorers (no LLM, run offline and in CI):
    - hit@k                 retrieval found the expected guideline
    - keyword_correctness   expected key terms appear in the answer
    - citation_validity     answer [n] cites only retrieved passages
  These gate CI because they are reproducible and free.

* LLM-judge scorers (need a real model; reported, not gated by default):
    - judge_correctness     a frontier model rates clinical correctness 0-1
    - judge_groundedness    a frontier model rates whether each claim is
                            supported by the retrieved passages 0-1
  These are richer but noisy, so they inform rather than fail the build.

The Judge sits behind an interface with a FakeJudge, so the harness and its
tests run with no key and no network.
"""
from __future__ import annotations

import re
from typing import Protocol

from app.config import Settings
from app.rag.vector_store import Hit

CITATION_RE = re.compile(r"\[(\d+)\]")


# --- deterministic scorers ---------------------------------------------------

def citation_validity(answer: str, n_contexts: int) -> bool:
    """True if the answer cites at least one passage and every [n] is in range.

    A fabricated citation (e.g. [9] when only 8 passages were retrieved) fails.
    """
    cited = {int(n) for n in CITATION_RE.findall(answer)}
    return bool(cited) and all(1 <= n <= n_contexts for n in cited)


def keyword_correctness(answer: str, keywords: list[str]) -> float:
    """Fraction of expected key terms present in the answer (0-1)."""
    if not keywords:
        return 1.0
    low = answer.lower()
    found = sum(1 for k in keywords if k.lower() in low)
    return round(found / len(keywords), 3)


# --- LLM-judge ---------------------------------------------------------------

class Judge(Protocol):
    def correctness(self, question: str, answer: str, keywords: list[str]) -> float: ...
    def groundedness(self, answer: str, contexts: list[Hit]) -> float: ...


class FakeJudge:
    """Offline judge. Deterministic, no network — used in tests and CI.

    It proxies correctness with keyword recall and groundedness with lexical
    overlap, so the harness produces *some* signal offline. These are NOT a
    substitute for the real judge; real numbers need ClaudeJudge + a key.
    """

    def correctness(self, question: str, answer: str, keywords: list[str]) -> float:
        return keyword_correctness(answer, keywords)

    def groundedness(self, answer: str, contexts: list[Hit]) -> float:
        if not answer.strip():
            return 0.0
        ctx = " ".join(h.text.lower() for h in contexts)
        words = [w for w in re.findall(r"[a-z]{4,}", answer.lower())]
        if not words:
            return 0.0
        supported = sum(1 for w in words if w in ctx)
        return round(supported / len(words), 3)


_JUDGE_SYSTEM = (
    "You are a strict evaluator of a clinical question-answering system. "
    "You return ONLY a single number between 0 and 1 with two decimals, no words."
)


class ClaudeJudge:
    """Real LLM judge. Uses a (preferably frontier) Claude model."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import

        if not settings.anthropic_api_key:
            raise RuntimeError("RAG_ANTHROPIC_API_KEY is required for the LLM judge.")
        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.eval_judge_model

    def _ask(self, prompt: str) -> float:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=8,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        m = re.search(r"[01](?:\.\d+)?", text)
        return float(m.group()) if m else 0.0

    def correctness(self, question: str, answer: str, keywords: list[str]) -> float:
        prompt = (
            f"Question: {question}\n\nAnswer to grade:\n{answer}\n\n"
            f"Expected key concepts: {', '.join(keywords) or '(none given)'}\n\n"
            "Rate how clinically correct and complete the answer is, 0 to 1."
        )
        return self._ask(prompt)

    def groundedness(self, answer: str, contexts: list[Hit]) -> float:
        passages = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(contexts, 1))
        prompt = (
            f"Passages:\n{passages}\n\nAnswer:\n{answer}\n\n"
            "Rate, 0 to 1, the fraction of the answer's claims that are directly "
            "supported by the passages above. Unsupported claims lower the score."
        )
        return self._ask(prompt)


def get_judge(settings: Settings) -> Judge:
    """Real judge when a key + claude provider are configured, else the fake."""
    if settings.llm_provider == "claude" and settings.anthropic_api_key:
        return ClaudeJudge(settings)
    return FakeJudge()
