# Clinical RAG — Portfolio & interview prep

Everything here is backed by code that runs and numbers you measured (June 2026).
Don't inflate them in interviews — the honesty *is* the signal.

## One-line positioning
> A production RAG service over Ireland's NCEC clinical guidelines with the full
> LLMOps layer — evaluation in CI, observability, cost-aware model routing,
> safety guardrails, Kubernetes, and a streaming React UI.

## CV bullets (use 2–3)
- Built and deployed a **production RAG service** (FastAPI, ChromaDB, BGE embeddings) over Irish NCEC clinical guidelines with **token-streaming answers**, inline citations, and a relevance-gated refusal.
- Built an **LLM evaluation harness** (answer correctness, groundedness, citation validity) **gated in CI**; measured **hit@k 1.00, citation-validity 1.00, judge-groundedness 0.95** on a 13-question clinical set.
- Cut inference cost ~**2.5×** with a query-complexity **model router** (Claude Haiku vs Sonnet) plus per-request token/cost accounting ($0.005 vs $0.013 per answer).
- Added an **output-side groundedness guardrail** that refuses answers unsupported by retrieved text, plus PII redaction, prompt-injection blocking, and rate limiting.
- Instrumented with **OpenTelemetry + Prometheus + Grafana** (p50/p95/p99 latency, cost, error/refusal rates, retrieval quality) and packaged for **Kubernetes** with HPA autoscaling and a k6 load test.

## LinkedIn / portfolio paragraph
> I upgraded a retrieve-then-generate RAG demo into a production-grade service.
> The interesting part isn't the RAG — it's the layer around it: a scored
> evaluation harness wired into CI so a regression fails the build, full
> observability (OpenTelemetry/Prometheus/Grafana), a cost-aware model router
> that measurably cut per-answer cost, safety guardrails including an output
> groundedness check that refuses unsupported answers, Kubernetes manifests with
> autoscaling, and a streaming React UI. Grounded only in public NCEC guidelines,
> never patient data.

## Interview talking points (and the honest version)
1. **Why this project?** "The market doesn't have a RAG shortage, it has a
   RAG-in-production shortage. Anyone can wrap an LLM; I wanted to show eval,
   observability, cost control, and guardrails around it."
2. **The router** — be honest it's a heuristic (length + clinical-reasoning
   markers), not ML. "It's a deliberately simple, explainable classifier; the
   point is *measured* cost control — 2.5× cheaper on simple queries — not
   sophistication." Owning that it's manufactured-but-justified reads as senior.
3. **The output groundedness check** — your strongest safety story. "The original
   refusal was a retrieval-score gate. I added an output-side check that verifies
   the generated answer is actually supported by retrieved text. Example: asked
   for an insulin dose in DKA, the model correctly said the corpus doesn't
   contain it rather than inventing a number."
4. **Eval design** — "Deterministic metrics (hit@k, keyword, citation validity)
   gate CI because they're reproducible; LLM-judge correctness/groundedness are
   reported but not gated because they're noisy. judge_correctness came out 0.82,
   not 1.0 — the judge docks incomplete answers, which is the point."
5. **Observability** — "I can answer 'why was that request slow or expensive'
   from the Grafana dashboard: per-step latency spans, tokens, cost, retrieval
   score distribution."

## Honest limitations to volunteer (don't hide them)
- **Kubernetes is local (k3s/minikube)** — proves the manifests + autoscaling
  skill, not production uptime. The public demo stays the Hugging Face Space.
- **Rate limiter is per-replica (in-process)** — multi-replica production needs a
  shared store (Redis).
- **LLM-judge scores are noisy** — treated as a signal, not ground truth.
- **No clinical validation** — demonstrates engineering, not medical correctness.

## If asked "what would you do next?"
Hybrid retrieval / a reranker for the harder queries; a Redis-backed limiter;
ship traces to a real collector (Tempo); and a managed cluster with a live URL
if the deployment story needed to be production-real.
