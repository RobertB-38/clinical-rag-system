"""Grounded answer generation behind one interface.

`ClaudeGenerator` calls Anthropic Claude with a strict, grounded prompt:
answer only from the supplied passages, cite source titles, and refuse when
the context does not contain the answer. `FakeGenerator` returns a
deterministic answer for tests with no API call.

Both share `build_prompt` so tests can assert the grounding instructions and
the retrieved contexts are actually sent to the model.
"""
from __future__ import annotations

from typing import Iterator, Protocol

from app.config import Settings, get_settings
from app.rag.vector_store import Hit

REFUSAL = (
    "I don't have guidance on that in the indexed clinical guidelines, "
    "so I can't answer it reliably."
)

SYSTEM_PROMPT = (
    "You are a clinical guideline assistant. Answer the user's question using "
    "ONLY the numbered guideline passages provided. After each statement, cite the "
    "passage(s) you used by their bracketed number, for example [1] or [2][3]. "
    "Every claim must carry a citation to one of the provided passages. If the "
    "passages do not contain enough information to answer, reply exactly with: "
    f"'{REFUSAL}' Do not use any outside medical knowledge. This is an information "
    "retrieval tool, not medical advice."
)


def build_prompt(question: str, contexts: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(contexts, 1):
        blocks.append(f"[{i}] Source: {h.source_title}\n{h.text}")
    joined = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return (
        f"Guideline passages:\n\n{joined}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above. Cite each statement with the "
        "bracketed passage number(s) it comes from, e.g. [1]."
    )


class Generator(Protocol):
    def answer(self, question: str, contexts: list[Hit]) -> str: ...

    def stream(self, question: str, contexts: list[Hit]) -> Iterator[str]:
        """Yield the answer incrementally as text chunks."""
        ...


class FakeGenerator:
    """Deterministic generator for tests — no network, no key."""

    def answer(self, question: str, contexts: list[Hit]) -> str:
        if not contexts:
            return REFUSAL
        return f"[fake answer] Based on {contexts[0].source_title}."

    def stream(self, question: str, contexts: list[Hit]) -> Iterator[str]:
        # Emit the same deterministic answer word-by-word so streaming tests
        # exercise the chunking path without a network call.
        for word in self.answer(question, contexts).split(" "):
            yield word + " "


class ClaudeGenerator:
    def __init__(self, settings: Settings, model: str | None = None) -> None:
        from anthropic import Anthropic  # lazy import

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "RAG_ANTHROPIC_API_KEY is not set. Set it, or run with "
                "RAG_LLM_PROVIDER=fake to test retrieval only."
            )
        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.claude_model  # router passes cheap/frontier models
        self._max_tokens = settings.max_answer_tokens

    def answer(self, question: str, contexts: list[Hit]) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        )
        _record_usage(self._model, resp.usage.input_tokens, resp.usage.output_tokens)
        return "".join(block.text for block in resp.content if block.type == "text").strip()

    def stream(self, question: str, contexts: list[Hit]) -> Iterator[str]:
        # Anthropic's streaming context manager yields text deltas as they arrive.
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
            _record_usage(self._model, final.usage.input_tokens, final.usage.output_tokens)


class OpenAIGenerator:
    """Cheap-tier generator (e.g. gpt-4o-mini) used by the router."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Needed for the cheap router tier; "
                "set it, or use RAG_LLM_PROVIDER=claude / fake."
            )
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._max_tokens = settings.max_answer_tokens

    def answer(self, question: str, contexts: list[Hit]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, contexts)},
            ],
        )
        if resp.usage:
            _record_usage(self._model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return (resp.choices[0].message.content or "").strip()

    def stream(self, question: str, contexts: list[Hit]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, contexts)},
            ],
        )
        for chunk in stream:
            if chunk.usage:
                _record_usage(self._model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def _record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    from app.cost import Usage, record

    record(Usage(model=model, input_tokens=input_tokens, output_tokens=output_tokens))


def get_generator(provider: str, settings: Settings | None = None) -> Generator:
    settings = settings or get_settings()
    if provider == "fake":
        return FakeGenerator()
    if provider == "claude":
        return ClaudeGenerator(settings)
    if provider == "openai":
        return OpenAIGenerator(settings)
    if provider == "router":
        from app.rag.router import RouterGenerator

        # All-Claude by default: cheap=Haiku, frontier=Sonnet (one key, real cost delta).
        cheap = ClaudeGenerator(settings, model=settings.router_cheap_model)
        frontier = ClaudeGenerator(settings, model=settings.router_frontier_model)
        return RouterGenerator(settings, cheap=cheap, frontier=frontier)
    raise ValueError(f"Unknown llm_provider: {provider!r}")
