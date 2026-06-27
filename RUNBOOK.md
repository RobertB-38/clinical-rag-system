# RUNBOOK — verify the v0.2 production upgrade

Run these in order on your machine. Each step has the command, **what good looks
like**, and what to paste me if it doesn't. Everything before this was built and
unit-tested offline; this turns "wired" into "measured".

> One-time cleanup first: remove the stray empty git folder from the failed init,
> then make your real commit at the end.
> ```bash
> cd "Clinical RAG Build" && rm -rf .git
> ```

---

## Prerequisites
- Python 3.11, `pip`
- Docker Desktop (for Steps 3 and the image)
- Node 18+ and npm (for Step 4)
- (Optional, Step 5) minikube or k3s, kubectl, k6
- Your keys: an Anthropic key and an OpenAI key

## Step 0 — keys and the index (2 min)
```bash
cd "Clinical RAG Build"
cp .env.example .env
# edit .env:  RAG_ANTHROPIC_API_KEY=sk-ant-...   and   OPENAI_API_KEY=sk-...
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# the service needs a Chroma index. If data/chroma/ is missing, build it:
ls data/chroma 2>/dev/null || python -m app.ingest.run
```
**Good:** `data/chroma/` exists. **If retrieval refuses everything later,** the
index is empty — re-run `python -m app.ingest.run` and confirm the PDFs are in
`data/raw/`.

## Step 1 — run the API and ask it something real (5 min)
```bash
# sanity: offline tests still green
python -m pytest -q

# start the API (uses your .env)
RAG_LLM_PROVIDER=router uvicorn app.main:app --port 8000
```
In another terminal:
```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the recommended timeframe for giving antibiotics in adult sepsis?"}' | python -m json.tool
```
**Good:** JSON with a cited `answer`, `contexts` with scores, and **non-zero
`latency_ms` and `cost_usd`**, plus `"model"` showing which tier the router
picked. Try a complex question ("dose and titration…") and a simple one ("what
is sepsis?") and confirm `model` differs — that's the router working.
**Paste me:** the two JSON responses if the model doesn't switch or cost is 0.

## Step 2 — real eval numbers (5–10 min, costs a little)
```bash
RAG_LLM_PROVIDER=claude python -m eval.run_eval            # report
RAG_LLM_PROVIDER=claude python -m eval.run_eval --check    # CI gate
```
**Good:** a table with real `hit@k`, `keyword_correctness`, `citation_validity`,
`judge_correctness`, `judge_groundedness`, and `eval/last_report.json` written.
For a stronger judge, set `RAG_EVAL_JUDGE_MODEL` to a frontier model in `.env`.
**Then:** paste me the table — I'll put the real numbers into the README where I
left blanks (don't hand-edit them; I'll keep the honest-reading notes consistent).

## Step 3 — observability stack + screenshots (10 min)
```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up --build
```
Drive a little traffic (rerun the Step 1 curl a few times), then open:
- Grafana → http://localhost:3000 (admin/admin) → dashboard **"Clinical RAG — Production"**
- Prometheus → http://localhost:9090 (try query `rag_request_latency_seconds_bucket`)

**Good:** latency, request-rate, cost, and retrieval-quality panels show data.
**Screenshot the dashboard** with a few panels populated → save to `assets/`.
**If panels are empty:** Prometheus may not be scraping — paste the output of
http://localhost:9090/targets (the `clinical-rag` target should be UP).

## Step 4 — React UI (5 min)
```bash
# keep the API running (Step 1), then:
cd frontend
npm install
npm run dev          # http://localhost:5173
```
**Good:** ask a question, the answer **streams token-by-token**, the side panel
fills with sources + scores + latency + cost, and the grounded/refused badge
shows. **Screenshot it mid-stream** → `assets/`.
**If it can't reach the API:** confirm the API is on :8000 (the Vite proxy points
there). Paste me any red console errors.

## Step 5 — Kubernetes autoscaling (optional, 20 min)
```bash
minikube start && minikube addons enable metrics-server
docker build -t clinical-rag:latest . && minikube image load clinical-rag:latest
kubectl create secret generic clinical-rag-secrets \
  --from-literal=RAG_ANTHROPIC_API_KEY=sk-ant-... --from-literal=OPENAI_API_KEY=sk-...
make k8s-deploy
kubectl rollout status deploy/clinical-rag
kubectl port-forward svc/clinical-rag 8080:80 &
kubectl get hpa -w &
k6 run -e BASE_URL=http://localhost:8080 load/k6-load-test.js
```
**Good:** during the 30-VU plateau the HPA REPLICAS climb above 1, then scale back
down. **Screenshot `kubectl get hpa`** before/after.
**If REPLICAS never move:** metrics-server isn't ready — `kubectl top pods` should
return CPU; paste me that.

## Step 6 — lock it in
1. Paste me the Step 2 eval table and the screenshot filenames → I update the README.
2. Then: `git init && git add -A && git commit -m "v0.2: production LLMOps layer"`
   and push to your GitHub repo.
3. Update your CV/LinkedIn from the **Resume bullets** section of the README.

---

### Troubleshooting quick table
| Symptom | Likely cause | Fix |
|---|---|---|
| Every answer refuses | empty/missing Chroma index | `python -m app.ingest.run` |
| `cost_usd` is 0 with real key | provider is `fake` or `claude`-only on a simple query | check `RAG_LLM_PROVIDER=router`, key set |
| Grafana panels empty | Prometheus not scraping | check :9090/targets, the `api:8000` target |
| 429 errors under load | rate limiter (30/min) | expected; raise `RAG_RATE_LIMIT` or ignore |
| React blank/CORS | API not on :8000 | start API first; Vite proxies /v1 → :8000 |
