import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.observability import LATENCY, metrics_payload, setup_tracing
from app.routers import query

app = FastAPI(
    title="Clinical RAG System",
    description="RAG over Ireland's NCEC National Clinical Guidelines — grounded, cited answers.",
    version="0.2.0",
)

setup_tracing("clinical-rag")

# Allow the front-end (same-origin in prod, or a separate host) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


@app.middleware("http")
async def record_latency(request: Request, call_next):
    """Record per-endpoint request latency for Prometheus."""
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path not in ("/metrics", "/health") and not path.startswith("/static"):
        LATENCY.labels(endpoint=path).observe(time.perf_counter() - start)
    return response


@app.get("/metrics", include_in_schema=False)
async def metrics():
    body, content_type = metrics_payload()
    return Response(content=body, media_type=content_type)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/", include_in_schema=False)
async def index():
    if WEB_INDEX.exists():
        return FileResponse(str(WEB_INDEX))
    return JSONResponse({"message": "Clinical RAG API. POST /v1/query. UI not bundled."})
