"""Pipeline tests using the lightweight fake/in-memory implementations:
no model download, no API key, no network. This is the payoff of the
provider-abstraction layer."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingest.chunker import chunk_document
from app.ingest.loader import load_document
from app.main import app
from app.rag.embeddings import FakeEmbedder
from app.rag.generator import REFUSAL, FakeGenerator, build_prompt
from app.rag.pipeline import RagPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import MemoryVectorStore
from app.routers.query import set_pipeline

SAMPLE = Path("tests/fixtures/sample_sepsis.txt")


def build_test_pipeline() -> RagPipeline:
    """Ingest the sample doc into an in-memory store and wire a fake pipeline."""
    settings = Settings(
        embedding_provider="fake",
        vector_store="memory",
        llm_provider="fake",
        score_threshold=0.05,
    )
    embedder = FakeEmbedder()
    store = MemoryVectorStore()

    doc = load_document(SAMPLE, "Sample Sepsis", "https://example.org/sample-sepsis", "sample")
    chunks = chunk_document(doc, settings.chunk_tokens, settings.chunk_overlap)
    vectors = embedder.encode([c.text for c in chunks])
    store.add(
        ids=[c.chunk_id for c in chunks],
        vectors=vectors,
        texts=[c.text for c in chunks],
        metadatas=[
            {"source_title": c.source_title, "source_url": c.source_url, "chunk_id": c.chunk_id}
            for c in chunks
        ],
    )
    retriever = Retriever(settings, embedder=embedder, store=store)
    return RagPipeline(settings=settings, retriever=retriever, generator=FakeGenerator())


@pytest.fixture
def client():
    set_pipeline(build_test_pipeline())
    return TestClient(app)


def test_ingestion_produces_chunks():
    doc = load_document(SAMPLE, "t", "u", "sample")
    chunks = chunk_document(doc, 256, 40)
    assert len(chunks) > 0
    assert all(c.chunk_id.startswith("sample::") for c in chunks)


def test_query_returns_grounded_structure(client):
    r = client.post("/v1/query", json={"question": "What is the Sepsis Six bundle?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert len(body["contexts"]) > 0
    assert "not a medical device" in body["disclaimer"].lower()


def test_relevant_question_retrieves_sepsis_context(client):
    r = client.post("/v1/query", json={"question": "antibiotics for sepsis within one hour"})
    body = r.json()
    joined = " ".join(c["text"].lower() for c in body["contexts"])
    assert "antibiotic" in joined


def test_relevance_gate_refuses_below_threshold():
    """With an impossibly high threshold, even a good retrieval is refused —
    proves the safety gate, not just retrieval, controls answering."""
    settings = Settings(
        embedding_provider="fake", vector_store="memory", llm_provider="fake",
        score_threshold=1.01,  # cosine similarity can never exceed 1.0
    )
    embedder = FakeEmbedder()
    store = MemoryVectorStore()
    doc = load_document(SAMPLE, "Sample Sepsis", "u", "sample")
    chunks = chunk_document(doc, settings.chunk_tokens, settings.chunk_overlap)
    store.add(
        ids=[c.chunk_id for c in chunks],
        vectors=embedder.encode([c.text for c in chunks]),
        texts=[c.text for c in chunks],
        metadatas=[{"source_title": "Sample Sepsis", "source_url": "u", "chunk_id": c.chunk_id}
                   for c in chunks],
    )
    pipe = RagPipeline(
        settings=settings,
        retriever=Retriever(settings, embedder=embedder, store=store),
        generator=FakeGenerator(),
    )
    result = pipe.answer("What antibiotics for sepsis?")
    assert result.refused is True
    assert result.answer == REFUSAL
    assert len(result.contexts) > 0  # contexts still returned for transparency


def test_prompt_includes_contexts_and_grounding_rule():
    from app.rag.vector_store import Hit

    hit = Hit("Give antibiotics within one hour.", 0.9, "Sample", "u", "sample::0")
    prompt = build_prompt("antibiotic timing?", [hit])
    assert "within one hour" in prompt
    assert "antibiotic timing?" in prompt


def _read_sse(response):
    """Collect SSE `data:` lines from a streaming test response into dicts."""
    events = []
    for line in response.iter_lines():
        if not line:
            continue
        text = line if isinstance(line, str) else line.decode()
        if text.startswith("data: "):
            events.append(json.loads(text[len("data: "):]))
    return events


def test_stream_endpoint_emits_meta_tokens_done(client):
    """The SSE stream must lead with sources (meta), stream tokens, then close
    with the full answer and a server-side latency number."""
    with client.stream(
        "POST", "/v1/query/stream", json={"question": "What is the Sepsis Six bundle?"}
    ) as r:
        assert r.status_code == 200
        events = _read_sse(r)

    types = [e["type"] for e in events]
    assert types[0] == "meta"          # sources arrive before any token
    assert "token" in types            # answer was streamed
    assert types[-1] == "done"         # stream closed cleanly

    meta = events[0]
    assert meta["refused"] is False
    assert len(meta["contexts"]) > 0

    done = events[-1]
    assert "fake answer" in done["answer"].lower()
    assert isinstance(done["latency_ms"], (int, float))


def test_stream_refuses_below_threshold():
    """Streaming must also honour the relevance gate: a refusal streams the
    refusal text, never a partial ungrounded answer, but still returns contexts."""
    settings = Settings(
        embedding_provider="fake", vector_store="memory", llm_provider="fake",
        score_threshold=1.01,
    )
    embedder = FakeEmbedder()
    store = MemoryVectorStore()
    doc = load_document(SAMPLE, "Sample Sepsis", "u", "sample")
    chunks = chunk_document(doc, settings.chunk_tokens, settings.chunk_overlap)
    store.add(
        ids=[c.chunk_id for c in chunks],
        vectors=embedder.encode([c.text for c in chunks]),
        texts=[c.text for c in chunks],
        metadatas=[{"source_title": "Sample Sepsis", "source_url": "u", "chunk_id": c.chunk_id}
                   for c in chunks],
    )
    pipe = RagPipeline(
        settings=settings,
        retriever=Retriever(settings, embedder=embedder, store=store),
        generator=FakeGenerator(),
    )
    rs = pipe.stream("What antibiotics for sepsis?")
    assert rs.refused is True
    assert "".join(rs.tokens) == REFUSAL
    assert len(rs.contexts) > 0
