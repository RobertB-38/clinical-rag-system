# Clinical RAG — Day 2–5 Build Plan

Status after Day 1: FastAPI skeleton with `/health` and a stubbed `/v1/query`. No retrieval, no documents, no LLM. This plan turns the stub into a working, demoable RAG pipeline grounded in real clinical guidelines.

## Architecture decisions (made for you, with reasoning)

**Vector store: Chroma (local, persistent).** Zero infra, runs in-process, persists to disk, ships in a Docker volume. pgvector/Qdrant are better for production scale but add a service to run and explain. For an MSc portfolio piece, local-first wins on reproducibility and demo-ability. Swap later behind a `Retriever` interface if you want to show that maturity.

**Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local) as default, with an OpenAI embedding option behind a flag.** MiniLM is free, fast on CPU, 384-dim, and means your demo runs with no API key. Keep the door open to a hosted embedding model so you can show a quality comparison in your writeup.

**LLM: provider-agnostic via a thin `Generator` interface.** Default to an OpenAI/Anthropic chat call behind an env var; allow a local Ollama option. Never hardcode one vendor — examiners notice abstraction.

**Corpus: a small, fixed, license-clean set of open guidelines.** Do NOT scrape PubMed broadly (licensing + noise). Pick ~5–15 open NICE/NHS/WHO PDF guidelines, commit a manifest (URL + retrieval date + license note), and store processed text — not necessarily the raw PDFs if license is unclear.

## Decision you still own

The one thing I can't decide for you: **the corpus list**. Pick the clinical topics now (e.g. sepsis, hypertension, type-2 diabetes, asthma, sepsis-in-children) so retrieval has a coherent domain. Vague "all of NHS" makes evaluation impossible.

---

## Day 2 — Ingestion pipeline
Goal: documents → clean chunks → embedded → persisted in Chroma.

- Add deps: `chromadb`, `sentence-transformers`, `pypdf`, `langchain-text-splitters` (splitter only; you don't need full LangChain).
- `data/sources.yaml`: manifest of each document (title, url, retrieved_at, license).
- `app/ingest/loader.py`: load PDF/HTML → raw text per document.
- `app/ingest/chunker.py`: recursive splitter, ~500–800 tokens, ~80 overlap. Attach metadata: `source_title`, `source_url`, `chunk_id`, `section` if available.
- `app/ingest/embed_store.py`: embed chunks, upsert into a persistent Chroma collection at `./data/chroma`.
- CLI: `python -m app.ingest.run` rebuilds the index. Print chunk count and timing.
- Test: ingest 1 sample doc, assert collection count > 0.

## Day 3 — Retrieval wired into the API
Goal: `/v1/query` returns real retrieved chunks (still no generation).

- `app/rag/retriever.py`: `Retriever.search(question, top_k)` → list of `{text, source_title, source_url, score}`. This is the interface that lets you swap vector stores later.
- Update `app/routers/query.py` to call the retriever; populate `sources` with deduped `source_url`s.
- Extend `QueryResponse` with a `contexts` field (the retrieved chunks + scores) so retrieval is inspectable in the demo.
- Tests: query a known topic, assert the top source matches the expected guideline.

## Day 4 — Generation + grounding
Goal: grounded answers with citations; refuses when context is weak.

- `app/rag/generator.py`: `Generator.answer(question, contexts)` behind an interface. Prompt must instruct: answer ONLY from provided context, cite source titles inline, and say "I don't have guidance on that" when context is insufficient. This anti-hallucination behaviour is the single most important thing for a *clinical* RAG and the thing to highlight in your writeup.
- Add a relevance floor: if best score < threshold, skip generation and return the refusal. (Cheap, and a great talking point.)
- Config via env: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `TOP_K`, `SCORE_THRESHOLD`. Read with pydantic-settings.
- Tests: mock the LLM; assert the prompt contains the contexts and that a low-score query returns the refusal path.

## Day 5 — Evaluation + Docker + docs
Goal: prove it works, make it runnable, make it presentable.

- `eval/qa.yaml`: 15–25 hand-written question/expected-source pairs over your corpus.
- `eval/run_eval.py`: compute retrieval **hit@k** (is the right guideline retrieved?) and a simple faithfulness check (does the answer only cite retrieved sources?). Output a table. These two numbers are your headline results.
- Update `Dockerfile` + add `docker-compose.yml` mounting `./data` so the index persists; document a one-command run.
- README: architecture diagram, design decisions (copy from above), eval results, limitations, and an explicit **"not a medical device"** disclaimer. The disclaimer matters for a clinical project and signals maturity.

---

## Guardrails / things that will bite you
- **Token-vs-char chunking**: sentence-transformers caps at 256 tokens; oversized chunks get silently truncated, wrecking retrieval. Verify chunk token length, don't assume.
- **Index reproducibility**: commit `sources.yaml` and the ingest script, gitignore `data/chroma`. Anyone should rebuild the index from scratch.
- **Don't commit raw PDFs** unless the licence clearly permits redistribution. Commit the manifest + processed text instead.
- **Keep the `Retriever`/`Generator` interfaces thin** — they're what let you say "swappable, testable" in an interview.

## What "done" looks like for the portfolio
`docker compose up`, POST a clinical question, get a grounded answer citing a real NICE/NHS guideline, plus a README showing hit@k and a faithfulness score. That's a credible, defensible project — far more than a wrapper around an LLM API.
