# Clinical RAG System — Project Description

**Live demo:** https://robert-b38-clinical-rag.hf.space
**Code:** https://github.com/RobertB-38/clinical-rag-system

---

## One-line summary
A deployed retrieval-augmented generation (RAG) system that answers clinical questions using Ireland's NCEC National Clinical Guidelines, returning answers grounded strictly in retrieved guideline text with inline citations, source traceability, and an explicit refusal when the evidence is insufficient.

## Problem / motivation
General LLMs answer medical questions from memory and can hallucinate — unacceptable in a clinical setting. This project demonstrates the safer pattern: retrieve the authoritative guideline passages first, then constrain the model to answer only from them, with citations and a refusal path when coverage is weak. It was built to demonstrate three competencies end to end: building with LLMs (RAG), shipping a deployed production service, and working responsibly in a regulated clinical domain.

## What was built
- An ingestion pipeline that loads guideline PDFs, splits them into ~350-token chunks, embeds them locally, and stores them in a persistent vector index.
- A retrieval + generation pipeline: embed the question, fetch the top-8 most relevant passages from the vector store, apply a relevance gate, and have Claude generate a grounded, cited answer (or refuse).
- A typed FastAPI service exposing `/v1/query` and `/health`, serving a custom single-page frontend.
- A custom Three.js "liquid-glass" web UI (animated DNA-free glassmorphism, light/dark toggle) that calls the API and renders answers, citations, and retrieved source cards.
- An evaluation harness measuring retrieval and citation quality.
- Containerisation and live deployment.

## Approach (how the project was run)
- **Single-authority corpus.** Chose one guideline body (Ireland's NCEC) rather than scraping multiple sources, so retrieval never has to reconcile conflicting advice — a deliberate scoping decision documented in the repo.
- **Licence-aware data handling.** Read the NCEC intellectual-property terms; index guideline text for non-commercial/educational use with attribution, and keep raw PDFs out of the public repo.
- **Interface-driven, testable design.** Every external dependency (embeddings, vector store, LLM) sits behind an interface with a lightweight "fake" implementation, so the automated test suite runs with no API key and no model download.
- **Evaluation-driven iteration.** Measured retrieval before trusting it. An initial embedding model produced a perfect-looking hit@k while answers still refused in practice (right document, wrong passage); diagnosing this led to a deliberate upgrade to a stronger embedding model, which fixed the real refusals.
- **Responsible-AI by construction.** Grounded-only answers, refusal on weak evidence, traceable sources, and clear non-device disclaimers built into the system rather than bolted on.

## Architecture & pipeline
```
PDFs ─► load ─► chunk (~350 tokens) ─► embed (BGE-small) ─► Chroma index   [build-time]

question ─► embed query ─► vector search (top-8) ─► relevance gate ─►
            grounded generation (Claude, cited) ─► answer + sources        [per request]
```
The frontend (Three.js + glassmorphism) is served by FastAPI and calls `/v1/query`; answers render with `[n]` citations and a card per retrieved guideline passage with its relevance score.

## Tech stack
- **Language / runtime:** Python 3.11
- **API:** FastAPI, Uvicorn, Pydantic v2, pydantic-settings (environment-driven config), CORS
- **Embeddings:** sentence-transformers — BAAI/bge-small-en-v1.5 with query/passage prompt prefixes (MiniLM selectable); runs free on CPU
- **Vector store:** ChromaDB (persistent), with an in-memory NumPy cosine store for tests
- **LLM / generation:** Anthropic Claude (Haiku) via the official SDK
- **Document processing:** pypdf, recursive token-aware chunking
- **Frontend:** custom HTML/CSS/JavaScript, Three.js (WebGL), glassmorphism, light/dark theming
- **Testing:** pytest, Starlette TestClient
- **Evaluation:** custom harness (hit@k, citation-validity) over a YAML question set
- **Packaging / deploy:** Docker, docker-compose, Git LFS, Hugging Face Docker Spaces
- **Tooling:** Git/GitHub, environment secrets management

## Skills demonstrated
- Retrieval-Augmented Generation (RAG) design and implementation from first principles (no orchestration framework hiding the mechanics)
- Embedding models, semantic search, vector databases, chunking strategy
- LLM prompt design for grounding, citation, and refusal behaviour
- Backend / API engineering (FastAPI, typed schemas, config management, CORS)
- Frontend development (custom interactive UI, WebGL/Three.js)
- Software engineering practice: dependency-inversion interfaces, fakes/mocks, unit + integration tests
- ML/IR evaluation: defining metrics (hit@k, citation-validity), diagnosing retrieval failures, model selection
- MLOps / deployment: containerisation, Git LFS for model/index artifacts, cloud deployment, secrets management
- Responsible AI for a regulated domain: grounding, refusal, traceability, licensing, disclaimers
- Data sourcing and licence compliance

## Evaluation & results
Hand-written question set scored on:
- **hit@k = 0.92** — did retrieval surface the correct guideline for each question?
- **citation-validity = 0.92** — do the answer's citations point only to passages actually retrieved (no fabricated references)?

Headline insight: upgrading the embedding model dropped document-level hit@k from a misleading 1.00 to an honest 0.92 while making the *system* better — because the earlier model's "hits" included cases that still refused. Documented as evidence of metric-aware iteration rather than chasing a single number.

## Responsible AI
Answers are generated only from retrieved guideline passages; the system refuses when evidence is insufficient; every answer returns its source passages and relevance scores for traceability; a prominent "not a medical device / educational only" disclaimer is shown; the corpus is a single authoritative body; and NCEC content is used under non-commercial/educational terms with attribution.

## Deployment
Containerised with Docker (FastAPI serving both the UI and the API) and deployed as a Docker Space on Hugging Face (free CPU tier). The prebuilt vector index is shipped with the image via Git LFS so the app boots without re-ingesting; the Anthropic API key is injected as a Space secret. The result is a public, live, evaluated demo.

## Outcomes
A finished, deployed, evaluated clinical RAG system with a custom frontend, automated tests, documented design decisions, and honest limitations — published as both source code (GitHub) and a live application (Hugging Face Spaces).
