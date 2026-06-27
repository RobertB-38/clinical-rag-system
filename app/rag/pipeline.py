"""End-to-end RAG: retrieve -> relevance gate -> grounded generation.

The relevance gate is the clinical-safety control: if the best retrieved
passage scores below the configured threshold, we refuse rather than letting
the model answer from weak or irrelevant context.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from app.config import Settings, get_settings
from app.observability import GENERATE_LATENCY, RETRIEVE_LATENCY, TOP_SCORE, span
from app.rag.generator import REFUSAL, Generator, get_generator
from app.rag.retriever import Retriever
from app.rag.vector_store import Hit


@dataclass
class RagResult:
    answer: str
    contexts: list[Hit]
    refused: bool


@dataclass
class RagStream:
    """Streaming counterpart of RagResult.

    `contexts` and `refused` are known up front (retrieval + relevance gate run
    synchronously and are fast), so the API can send sources immediately and
    then stream answer tokens from `tokens`.
    """
    contexts: list[Hit]
    refused: bool
    tokens: Iterator[str]


class RagPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(self.settings)
        self.generator = generator or get_generator(self.settings.llm_provider, self.settings)

    def answer(self, question: str, top_k: int | None = None) -> RagResult:
        with span("rag.answer"):
            contexts, best = self._retrieve(question, top_k)

            if not contexts or best < self.settings.score_threshold:
                return RagResult(answer=REFUSAL, contexts=contexts, refused=True)

            t = time.perf_counter()
            with span("rag.generate", model=self.settings.llm_provider):
                answer = self.generator.answer(question, contexts)
            GENERATE_LATENCY.observe(time.perf_counter() - t)

            # Output-side guardrail: verify the generated answer is actually
            # supported by the retrieved passages. Skipped for the fake provider
            # (which doesn't cite). This catches the "retrieved good context but
            # answered from elsewhere" failure the score gate alone cannot.
            if self.settings.enforce_groundedness and self.settings.llm_provider != "fake":
                from app.guardrails import check_groundedness

                grounded, _overlap = check_groundedness(
                    answer, contexts, self.settings.groundedness_min_overlap
                )
                if not grounded:
                    return RagResult(answer=REFUSAL, contexts=contexts, refused=True)

            return RagResult(answer=answer, contexts=contexts, refused=False)

    def _retrieve(self, question: str, top_k: int | None) -> tuple[list[Hit], float]:
        """Retrieve + record retrieval latency and best-score metrics/span."""
        t = time.perf_counter()
        with span("rag.retrieve", top_k=top_k or self.settings.top_k):
            contexts = self.retriever.search(question, top_k)
        RETRIEVE_LATENCY.observe(time.perf_counter() - t)
        best = max((h.score for h in contexts), default=0.0)
        TOP_SCORE.observe(best)
        return contexts, best

    def stream(self, question: str, top_k: int | None = None) -> RagStream:
        """Same retrieve -> gate -> generate flow, but the answer is streamed.

        The relevance gate is applied before any token is generated, so a
        refusal never streams a partial (and potentially ungrounded) answer.
        """
        contexts, best = self._retrieve(question, top_k)

        if not contexts or best < self.settings.score_threshold:
            def refusal_tokens() -> Iterator[str]:
                yield REFUSAL

            return RagStream(contexts=contexts, refused=True, tokens=refusal_tokens())

        tokens = self.generator.stream(question, contexts)
        return RagStream(contexts=contexts, refused=False, tokens=tokens)
