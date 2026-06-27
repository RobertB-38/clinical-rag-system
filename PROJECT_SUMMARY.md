# Clinical RAG System — Project Summary

**Live demo:** https://robert-b38-clinical-rag.hf.space
**Code:** https://github.com/RobertB-38/clinical-rag-system

A complete, deployed retrieval-augmented generation (RAG) system that answers clinical
questions using Ireland's NCEC National Clinical Guidelines — with inline citations,
full source traceability, and an honest refusal when the guidelines don't cover a
question. Built end to end: data pipeline, retrieval, generation, evaluation, a custom
web frontend, and a public deployment.

---

## 1. What it is and why

Large language models answer fluently but can fabricate facts — unacceptable in a
clinical context. RAG fixes this by retrieving authoritative source passages first and
constraining the model to answer only from them. This project applies that pattern to a
real, regulated domain (Irish national clinical guidelines) and ships it as a working
product, not a notebook.

It was built to demonstrate three capabilities end to end:

1. **GenAI / LLM / RAG engineering** — building a retrieval+generation pipeline by hand.
2. **Production / deployed systems** — a tested, containerised service running at a public URL.
3. **Clinical-domain credibility** — grounding, refusal, evaluation, and correct handling of a regulated, copyrighted corpus.

---

## 2. Approach

The project followed a deliberate methodology rather than ad-hoc building:

- **Corpus first, and bounded.** Chose a single authoritative guideline body (NCEC) so
  retrieval never has to reconcile conflicting advice from different authorities. The
  corpus is catalogued in a manifest (`data/sources.yaml`) for reproducibility; raw PDFs
  are indexed locally but not redistributed (licence-aware).
- **Build by hand, not by framework.** Deliberately avoided LangChain/LlamaIndex so every
  step — chunking, embedding, retrieval, prompting — is explicit and explainable.
- **Interfaces + fakes from the start.** Embeddings, vector store, and the LLM each sit
  behind a small interface with a lightweight "fake" implementation, so the whole pipeline
  is unit-testable with no API key and no model download.
- **Measure, then improve.** Built an evaluation harness early. When the first embedding
  model (MiniLM) produced a perfect-looking retrieval score but answers still refused in
  practice, the metrics exposed it — and drove a switch to a stronger model (BGE-small).
- **Ship it.** Treated a live public URL as a requirement, not an afterthought.

---

## 3. Architecture and pipeline

**Build-time (ingestion):** guideline PDFs → text extraction → ~350-token overlapping
chunks with source metadata → local embedding (BGE-small) → persisted in a Chroma vector
index. Re-runnable and idempotent.

**Request-time (per question):**
1. Embed the query (BGE query-prefix convention).
2. Vector search in Chroma for the top-8 most similar passages.
3. Relevance gate — if the best match is too weak, refuse rather than answer.
4. Grounded generation — passages + question go to Claude with strict instructions:
   answer only from these passages, cite them by number, refuse if insufficient.
5. Return the answer, the citations, and the retrieved passages with relevance scores.

The FastAPI service exposes `POST /v1/query` and `GET /health`, and serves the custom
frontend at `/`.

---

## 4. Tech stack

| Layer | Choice |
| --- | --- |
| Language / runtime | Python 3.11 |
| API | FastAPI, Uvicorn, Pydantic v2, pydantic-settings |
| Embeddings | sentence-transformers — **BGE-small-en-v1.5** (MiniLM selectable), CPU, free |
| Vector store | ChromaDB (persistent), with an in-memory NumPy store for tests |
| Generation | Anthropic Claude (Haiku 4.5) via the official SDK |
| Ingestion | pypdf, langchain-text-splitters (splitter only), PyYAML |
| Frontend | Custom HTML/CSS/JS, Three.js (WebGL liquid-glass UI), light/dark toggle |
| Testing | pytest, Starlette TestClient |
| Evaluation | custom hit@k + citation-validity harness |
| Packaging / deploy | Docker, docker-compose, Hugging Face Docker Space, Git LFS |
| Tooling | Git/GitHub, Hugging Face Hub, conda/venv |

Deliberately **not** used: LangChain/LlamaIndex (hand-rolled for explainability),
Kubernetes (single-container service), fine-tuning (RAG is the right tool for grounding).

---

## 5. Skills demonstrated

- **RAG engineering** — chunking strategy, embedding choice, vector search, grounded
  prompting, query/passage asymmetric embeddings.
- **Backend/software engineering** — typed API design, dependency-injection via
  interfaces, environment-driven config, a green test suite, graceful error handling.
- **ML evaluation** — defining and computing retrieval and faithfulness metrics, and
  acting on them (the MiniLM→BGE decision).
- **MLOps / deployment** — containerisation, port/secret config, large-binary handling
  with Git LFS, shipping a public service.
- **Frontend** — a bespoke WebGL/glassmorphism UI wired to the API.
- **Judgement** — scoping the corpus, licence compliance, responsible-AI framing, and
  knowing what *not* to build.

---

## 6. Key decisions (and the reasoning)

- **Single guideline body (NCEC)** → coherent retrieval, no conflicting guidance.
- **Hand-built pipeline over a framework** → every step is explainable in an interview.
- **BGE-small over MiniLM** → measurably better retrieval; fixed real refusals that a
  document-level hit@k had hidden. Knowing *why* the model was changed is the point.
- **Relevance gate + grounded-only prompt** → the system says "I don't know" instead of
  hallucinating — the single most important property for a clinical tool.
- **Provider interfaces with fakes** → tests run free and offline; production swaps in
  real models via env vars.
- **Prebuilt index shipped via Git LFS** → instant boot on free hardware, within the
  corpus's non-commercial licence.

---

## 7. Evaluation

Scored against a hand-written question set (`eval/qa.yaml`) over the indexed guidelines:

| Metric | Score | Meaning |
| --- | --- | --- |
| hit@k | **0.92** (12/13) | retrieval surfaced the correct guideline |
| citation_validity | **0.92** (12/13) | answers cite only passages actually retrieved |

Honest reading: hit@k *dropped* from a perfect 1.00 (MiniLM) to 0.92 (BGE) while the
system got *better* — because MiniLM's "hits" included cases that found the right
document but still refused. This is exactly why more than one metric matters.

---

## 8. Responsible AI and licensing

- Grounded-only answers with citations; refusal on weak evidence; full source
  traceability (every answer shows its passages and scores).
- Prominent "educational demo — not medical advice" notice; users told not to enter
  patient data; nothing is stored.
- NCEC guideline content reproduced for non-commercial educational use with attribution;
  not affiliated with or endorsed by the Department of Health / HSE.

---

## 9. Deployment

Containerised with Docker (listens on HF's port 7860) and deployed as a Hugging Face
**Docker Space**. The prebuilt Chroma index ships with the image via Git LFS; the
Anthropic API key is injected as a Space secret. The custom frontend and the API are
served from the same container.

---

## 10. Known limitations and next steps

- **One eval miss** — "stop-smoking pharmacotherapy" misses at the document level because
  the COPD guideline also covers smoking cessation and outranks the dedicated guideline.
  Documented, not hidden. A hybrid keyword+vector search or reranker would likely fix it.
- **No clinical validation** — outputs are not reviewed by clinicians; this demonstrates
  engineering, not medical correctness.
- **Free-tier sleep** — the Space sleeps when idle and takes ~1–2 min to wake.
- **Next:** expand the corpus to the full NCEC suite, add hybrid retrieval / reranking,
  add streaming responses, and add per-answer feedback capture for evaluation.
