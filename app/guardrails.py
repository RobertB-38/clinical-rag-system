"""Guardrails: input filters + an output groundedness check.

Layers (defence in depth, never "bulletproof"):

Input
  redact_pii        strip emails, phone/PPS-like numbers, long digit runs so a
                    user who pastes patient identifiers does not have them sent
                    to the model or logged.
  detect_injection  flag common prompt-injection phrasings ("ignore previous
                    instructions", "reveal your system prompt", ...). Heuristic,
                    not a guarantee — pairs with the grounded-only system prompt.

Output
  check_groundedness  the NEW output-side control. The existing refusal was a
                    retrieval-score gate; this verifies the *generated answer*
                    is actually supported by the retrieved passages (citations
                    present and lexical overlap), and refuses if not. This is
                    the strongest clinical-safety property: it catches a model
                    that retrieved good context but then answered from elsewhere.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque

from app.rag.vector_store import Hit

# --- input: PII redaction ----------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\b(?:\+?\d[\d\s-]{7,}\d)\b")
_PPS = re.compile(r"\b\d{7}[A-Za-z]{1,2}\b")           # Irish PPS number shape
_LONGNUM = re.compile(r"\b\d{6,}\b")                   # MRN / long identifiers
_DOB = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, kinds_found)."""
    found: list[str] = []

    def sub(pattern: re.Pattern, label: str, repl: str, s: str) -> str:
        nonlocal found
        if pattern.search(s):
            found.append(label)
            return pattern.sub(repl, s)
        return s

    out = text
    out = sub(_EMAIL, "email", "[REDACTED_EMAIL]", out)
    out = sub(_PPS, "pps", "[REDACTED_ID]", out)
    out = sub(_DOB, "dob", "[REDACTED_DATE]", out)
    out = sub(_PHONE, "phone", "[REDACTED_PHONE]", out)
    out = sub(_LONGNUM, "long_number", "[REDACTED_ID]", out)
    return out, found


# --- input: prompt-injection detection ---------------------------------------
_INJECTION_PATTERNS = (
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous|above)",
    r"reveal|show|print|repeat .{0,20}(system )?(prompt|instructions)",
    r"you are now",
    r"act as (?:a|an|the)",
    r"developer mode|jailbreak|do anything now|\bDAN\b",
    r"override .{0,20}(rules|safety|guardrails)",
)
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> tuple[bool, str]:
    for rx in _INJECTION_RE:
        if rx.search(text):
            return True, f"matched injection pattern: {rx.pattern!r}"
    return False, ""


# --- output: groundedness check ----------------------------------------------
_WORD = re.compile(r"[a-z]{4,}")
_CITATION = re.compile(r"\[(\d+)\]")


def check_groundedness(answer: str, contexts: list[Hit], min_overlap: float = 0.4) -> tuple[bool, float]:
    """Cheap, deterministic output check.

    Grounded if the answer carries at least one valid in-range citation AND a
    meaningful fraction of its content words appear in the retrieved passages.
    Returns (is_grounded, overlap_score). Designed to run on every request with
    no extra LLM call; an LLM-judge groundedness score also exists in the eval
    harness for offline depth.
    """
    if not answer.strip():
        return False, 0.0

    cited = {int(n) for n in _CITATION.findall(answer)}
    n_ctx = len(contexts)
    citations_ok = bool(cited) and all(1 <= n <= n_ctx for n in cited)

    ctx_blob = " ".join(h.text.lower() for h in contexts)
    words = _WORD.findall(answer.lower())
    overlap = (sum(1 for w in words if w in ctx_blob) / len(words)) if words else 0.0

    return (citations_ok and overlap >= min_overlap), round(overlap, 3)


# --- rate limiting (dependency-light, per-client fixed window) ----------------
def parse_rate(spec: str) -> tuple[int, int]:
    """Parse '30/minute' -> (30, 60). Supports second|minute|hour."""
    units = {"second": 1, "minute": 60, "hour": 3600}
    count, _, unit = spec.partition("/")
    return int(count), units.get(unit.strip().lower().rstrip("s"), 60)


class RateLimiter:
    """In-process fixed-window limiter keyed by client. Adequate for a single
    replica or a sticky-session demo; a shared store (Redis) would be the
    multi-replica production form, noted in the SLO doc."""

    def __init__(self, spec: str = "30/minute") -> None:
        self.max_requests, self.window = parse_rate(spec)
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        dq = self._hits[key]
        while dq and dq[0] <= now - self.window:
            dq.popleft()
        if len(dq) >= self.max_requests:
            return False
        dq.append(now)
        return True


def screen_input(text: str) -> tuple[str, str | None]:
    """Apply input guardrails. Returns (clean_text, refusal_message_or_None).

    Injection -> refuse outright. PII -> redact and continue (so a careless
    paste of identifiers is scrubbed rather than rejected)."""
    injected, _ = detect_injection(text)
    if injected:
        return text, (
            "This request looks like a prompt-injection attempt and was blocked. "
            "Ask a clinical question grounded in the guidelines."
        )
    clean, _found = redact_pii(text)
    return clean, None
