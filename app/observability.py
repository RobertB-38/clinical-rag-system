"""Observability: Prometheus metrics + OpenTelemetry tracing.

Why this module exists
----------------------
You cannot operate or debug what you cannot see. Every request and pipeline
step emits a metric (Prometheus) and a trace span (OpenTelemetry) so latency,
cost, errors and retrieval quality are visible on a dashboard.

Design choice: both Prometheus and OpenTelemetry are imported defensively. If
the libraries are not installed (e.g. a minimal test environment) the metrics
and spans degrade to no-ops, so the core service and its unit tests never
depend on the observability stack being present.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

# --- Prometheus (optional) ---------------------------------------------------
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _PROM = True
except Exception:  # pragma: no cover - exercised only when lib absent
    _PROM = False
    CONTENT_TYPE_LATEST = "text/plain"


class _NoopMetric:
    def labels(self, *a, **k):
        return self

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass


if _PROM:
    REQUESTS = Counter("rag_requests_total", "RAG queries", ["endpoint", "refused"])
    ERRORS = Counter("rag_errors_total", "Errors during a query", ["endpoint"])
    LATENCY = Histogram(
        "rag_request_latency_seconds", "End-to-end request latency", ["endpoint"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
    )
    RETRIEVE_LATENCY = Histogram(
        "rag_retrieve_latency_seconds", "Retrieval step latency",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
    )
    GENERATE_LATENCY = Histogram(
        "rag_generate_latency_seconds", "Generation step latency",
        buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
    )
    TOP_SCORE = Histogram(
        "rag_top_retrieval_score", "Best retrieval similarity per query",
        buckets=(0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    TOKENS = Counter("rag_tokens_total", "LLM tokens", ["model", "kind"])  # kind=input|output
    COST = Counter("rag_cost_usd_total", "Estimated LLM spend (USD)", ["model"])
    ROUTER = Counter("rag_router_selections_total", "Router model picks", ["tier", "model"])
else:  # pragma: no cover
    REQUESTS = ERRORS = LATENCY = RETRIEVE_LATENCY = GENERATE_LATENCY = _NoopMetric()
    TOP_SCORE = TOKENS = COST = ROUTER = _NoopMetric()


def metrics_payload() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    if _PROM:
        return generate_latest(), CONTENT_TYPE_LATEST
    return b"# prometheus_client not installed\n", "text/plain"


def record_generation(model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    """Record per-request token usage and cost (called by the cost accountant)."""
    TOKENS.labels(model=model, kind="input").inc(input_tokens)
    TOKENS.labels(model=model, kind="output").inc(output_tokens)
    COST.labels(model=model).inc(cost_usd)


def record_router(tier: str, model: str) -> None:
    ROUTER.labels(tier=tier, model=model).inc()


# --- OpenTelemetry (optional) ------------------------------------------------
_tracer = None


def setup_tracing(service_name: str = "clinical-rag") -> None:
    """Initialise an OTel tracer provider.

    Exports via OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set (e.g. a collector
    or Tempo); otherwise falls back to a console exporter so spans are still
    visible locally. Safe no-op if opentelemetry is not installed.
    """
    global _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception:  # pragma: no cover
        _tracer = None
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    exporter = None
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()
        except Exception:  # pragma: no cover
            exporter = None
    if exporter is None and os.getenv("OTEL_CONSOLE", "0") == "1":
        exporter = ConsoleSpanExporter()
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


@contextmanager
def span(name: str, **attributes):
    """Start a span if tracing is active, else a no-op context."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            try:
                s.set_attribute(k, v)
            except Exception:  # pragma: no cover
                pass
        yield s
