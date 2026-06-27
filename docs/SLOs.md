# Service Level Objectives — Clinical RAG

These are the targets the service is operated against. They are measured from
the Prometheus metrics the app exports and enforced in two places: the k6 load
test thresholds (`load/k6-load-test.js`) and the Grafana dashboard.

| SLO | Target | Metric / source | Notes |
|---|---|---|---|
| Availability | ≥ 99% of `/v1/query` non-5xx over 30 days | `rag_errors_total` vs `rag_requests_total` | Single-replica demo; HPA improves this under load. |
| Latency (non-streaming) | p95 < 8 s, p99 < 12 s | `rag_request_latency_seconds` histogram | Dominated by generation; frontier model is the tail. |
| Time-to-first-token (streaming) | p95 < 2 s | retrieval + gate run before first token | Retrieval is ~10–50 ms; the rest is model first-token. |
| Retrieval quality | p10 top-similarity ≥ 0.30 | `rag_top_retrieval_score` histogram | Below the 0.25 gate → refusal, by design. |
| Refusal rate | < 25% on in-corpus traffic | `rag_requests_total{refused="true"}` | A spike means corpus gaps or retrieval drift. |
| Cost per answer | < $0.01 median | `rag_cost_usd_total` / answered requests | The router keeps the median on the cheap model. |
| Answer correctness | judge_correctness ≥ 0.70 | eval harness (offline) | Gated in CI when keys are present. |
| Groundedness | every served answer passes the output check | `app/guardrails.check_groundedness` | Ungrounded answers are refused, not shown. |

## Error budget
At 99% availability the monthly budget is ~7.2 hours. Burn is reviewed on the
dashboard's error-rate panel; sustained burn pauses risky changes (chunking,
embeddings, prompts, model) until the eval harness confirms no regression.

## Scaling
HPA scales 1→5 replicas at 60% CPU. The k6 ramp (30 VUs) is sized to cross that
threshold so autoscaling can be demonstrated with `kubectl get hpa -w`.

## Known limitations (honest)
- Single-replica rate limiter is in-process; multi-replica production needs a
  shared store (Redis). Documented in `app/guardrails.RateLimiter`.
- Local k3s/minikube proves the manifests + autoscaling, not production uptime;
  the public demo URL remains the Hugging Face Space.
