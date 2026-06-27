"""/v1/query — retrieve guideline passages and answer, grounded and cited.

The pipeline is built lazily on first use and cached on the router, so model
loading happens once and tests can inject a fake pipeline via set_pipeline().
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.cost import begin_request, current_usage
from app.guardrails import RateLimiter, check_groundedness, screen_input
from app.models import QueryRequest, QueryResponse, RetrievedContext
from app.observability import ERRORS, REQUESTS
from app.rag.pipeline import RagPipeline

router = APIRouter(prefix="/v1", tags=["query"])

_pipeline: RagPipeline | None = None
_limiter = RateLimiter(get_settings().rate_limit)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "anonymous"


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


def set_pipeline(pipeline: RagPipeline) -> None:
    """Inject a pipeline (used by tests)."""
    global _pipeline
    _pipeline = pipeline


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, http_request: Request):
    if not _limiter.allow(_client_key(http_request)):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})

    clean_question, refusal = screen_input(request.question)
    if refusal:
        REQUESTS.labels(endpoint="/v1/query", refused="true").inc()
        return QueryResponse(
            question=request.question, answer=refusal, sources=[], contexts=[], refused=True
        )

    begin_request()
    t0 = time.perf_counter()
    try:
        result = get_pipeline().answer(clean_question, request.top_k)
    except Exception as exc:  # config/runtime issue (e.g. missing API key, no index)
        ERRORS.labels(endpoint="/v1/query").inc()
        return QueryResponse(
            question=request.question,
            answer=f"The service is not fully configured: {exc}",
            sources=[],
            contexts=[],
            refused=True,
        )
    REQUESTS.labels(endpoint="/v1/query", refused=str(result.refused).lower()).inc()
    usage = current_usage()
    sources = list(dict.fromkeys(h.source_url for h in result.contexts if h.source_url))
    return QueryResponse(
        question=request.question,
        answer=result.answer,
        sources=sources,
        contexts=[
            RetrievedContext(
                text=h.text, score=h.score, source_title=h.source_title, source_url=h.source_url
            )
            for h in result.contexts
        ],
        refused=result.refused,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        cost_usd=usage["cost_usd"],
        model=usage["model"],
    )


def _sse(payload: dict) -> str:
    """Format a dict as one Server-Sent Events message."""
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest, http_request: Request) -> StreamingResponse:
    """Streaming variant of /v1/query as Server-Sent Events.

    Event sequence (each a `data:` line of JSON with a `type` field):
      1. `meta`  — retrieved sources/contexts and the refusal flag, sent before
                   any token so the UI can render the sources panel immediately.
      2. `token` — incremental answer text (zero or more).
      3. `done`  — the full answer plus server-side latency in ms.
      4. `error` — emitted instead if the pipeline is misconfigured.
    """
    t0 = time.perf_counter()

    if not _limiter.allow(_client_key(http_request)):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})

    clean_question, refusal = screen_input(request.question)

    # async generator (not sync) so it runs in the request's task context — the
    # per-request cost contextvar persists across yields here, where a threadpool
    # sync generator would lose it (each chunk would get a fresh context copy).
    async def event_stream():
        begin_request()
        if refusal:
            REQUESTS.labels(endpoint="/v1/query/stream", refused="true").inc()
            yield _sse({"type": "meta", "question": request.question, "refused": True, "sources": [], "contexts": []})
            yield _sse({"type": "token", "text": refusal})
            yield _sse({"type": "done", "answer": refusal, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "grounded": False, "groundedness": 0.0})
            return

        try:
            rs = get_pipeline().stream(clean_question, request.top_k)
        except Exception as exc:  # missing key / no index, etc.
            ERRORS.labels(endpoint="/v1/query/stream").inc()
            yield _sse({"type": "error", "message": f"The service is not fully configured: {exc}"})
            return

        REQUESTS.labels(endpoint="/v1/query/stream", refused=str(rs.refused).lower()).inc()
        sources = list(dict.fromkeys(h.source_url for h in rs.contexts if h.source_url))
        yield _sse(
            {
                "type": "meta",
                "question": request.question,
                "refused": rs.refused,
                "sources": sources,
                "contexts": [
                    {
                        "text": h.text,
                        "score": h.score,
                        "source_title": h.source_title,
                        "source_url": h.source_url,
                    }
                    for h in rs.contexts
                ],
            }
        )

        parts: list[str] = []
        for chunk in rs.tokens:
            parts.append(chunk)
            yield _sse({"type": "token", "text": chunk})

        answer = "".join(parts)
        grounded, overlap = (False, 0.0) if rs.refused else check_groundedness(answer, rs.contexts)
        usage = current_usage()
        yield _sse(
            {
                "type": "done",
                "answer": answer,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "grounded": grounded,
                "groundedness": overlap,
                "cost_usd": usage["cost_usd"],
                "model": usage["model"],
            }
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
