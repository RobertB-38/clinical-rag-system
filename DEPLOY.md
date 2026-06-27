# Deployment

Two supported targets. The Hugging Face Space is the one to put on your CV — it gives a public URL anyone can click.

## Option A — Hugging Face Docker Space with the custom front-end (recommended)

The showcase is `web/index.html` (Three.js DNA + liquid glass) served by FastAPI, which
also exposes `/v1/query`. This needs a **Docker** Space (not Gradio), because we serve a
custom page + API. Build the Chroma index locally and commit it so the Space boots instantly.

**1. Build the index locally** (once), so `data/chroma/` exists:
```bash
python -m app.ingest.run
```

**2. Create the Space** → huggingface.co/new-space → SDK **Docker** → **Blank**, hardware **CPU basic (free)**.

**3. Set the Space metadata.** Edit the Space's `README.md` frontmatter:
```yaml
---
title: Clinical RAG System
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```
`app_port: 7860` matches the port the container listens on (the Dockerfile honours `$PORT`,
which HF sets to 7860).

**4. Add the API key as a Secret** (Space → Settings → Variables and secrets):
`RAG_ANTHROPIC_API_KEY = sk-ant-...`. Never commit it.

**5. Push code + the prebuilt index to the Space.** `data/chroma/` is gitignored, so
force-add it for the Space only:
```bash
git clone https://huggingface.co/spaces/<you>/clinical-rag hf-space && cd hf-space
rsync -a --exclude='.git' --exclude='.venv' "/path/to/Clinical RAG Build/" .
git add -A
git add -f data/chroma            # override .gitignore for the Space
git commit -m "Deploy clinical RAG (custom UI + prebuilt NCEC index)"
git push
```

**6. Wait for the Docker build**, then open the Space URL — the liquid-glass UI loads and
`/v1/query` answers live.

Notes:
- MiniLM (~90 MB) downloads on first boot to embed the *query*; CPU basic handles it.
- The committed index reproduces NCEC guideline text — keep the Space non-commercial with the attribution already in the UI/README (see `data/sources.yaml`).
- Do **not** force-add `.env` or your API key.
- `app_ui.py` (Gradio) remains in the repo as a lightweight fallback; the Docker Space runs the FastAPI app, not Gradio.

## Option B — Docker / any container host (Render, Railway, Fly.io)

```bash
docker build -t clinical-rag .
docker run -p 8000:8000 -e RAG_ANTHROPIC_API_KEY=sk-ant-... clinical-rag
```

Or with compose (persists the index volume):
```bash
RAG_ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

This serves the FastAPI API (`/health`, `/v1/query`). On Render/Railway, point the service at the Dockerfile and set `RAG_ANTHROPIC_API_KEY` as an environment variable.

## Pre-flight checklist
- [ ] `pytest` green
- [ ] `python -m app.ingest.run` builds the index without errors
- [ ] `python -m eval.run_eval` prints hit@k
- [ ] `RAG_ANTHROPIC_API_KEY` set as a secret (never committed)
- [ ] Real guideline corpus in `data/sources.yaml`
