"""Per-request token + cost accounting.

Prices are approximate published list prices in USD per 1M tokens and are kept
in one editable table — swap in your actual contracted rates. `compute_cost`
turns a token count into dollars; generators call `record` after each call so
cost shows up on the dashboard per model.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass

# Per-request usage accumulator so the API can return cost-per-answer (not just
# aggregate it into Prometheus). Set at the start of a request; generators add
# to it as they record usage.
_acc: contextvars.ContextVar[dict | None] = contextvars.ContextVar("rag_usage_acc", default=None)


def begin_request() -> None:
    _acc.set({"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "model": None})


def current_usage() -> dict:
    return _acc.get() or {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "model": None}

# (input_per_million, output_per_million) USD. Approximate — edit to match your rates.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-8": (15.00, 75.00),
    "fake": (0.0, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    return PRICES.get(model, (0.0, 0.0))


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = price_for(model)
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for when an API usage object is
    unavailable (e.g. the fake generator)."""
    return max(1, len(text) // 4)


@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        return compute_cost(self.model, self.input_tokens, self.output_tokens)


def record(usage: Usage) -> None:
    """Record usage to Prometheus (tokens + cost) and the per-request accumulator."""
    from app.observability import record_generation

    record_generation(usage.model, usage.input_tokens, usage.output_tokens, usage.cost_usd)

    acc = _acc.get()
    if acc is not None:
        acc["cost_usd"] = round(acc["cost_usd"] + usage.cost_usd, 6)
        acc["input_tokens"] += usage.input_tokens
        acc["output_tokens"] += usage.output_tokens
        acc["model"] = usage.model
