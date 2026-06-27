# Clinical RAG — React front-end

Streaming chat over the FastAPI service, with a panel showing the retrieved
NCEC sources (and similarity scores), per-answer latency, and cost.

## Run (dev)
```bash
# 1) start the API (from repo root)
uvicorn app.main:app --port 8000

# 2) start the UI
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /v1 -> :8000
```

## How it works
- Calls `POST /v1/query/stream` and parses the Server-Sent Events:
  `meta` (sources first) → `token` (streamed answer) → `done` (latency, cost,
  groundedness).
- Renders the answer as it streams; the side panel fills from the `meta` and
  `done` events. A "refused" badge shows when the question is outside the corpus;
  a "grounded ✓" badge reflects the output groundedness check.

## Build
```bash
npm run build        # static bundle in dist/, serve behind the API or any host
```
