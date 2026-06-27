"""Build the vector index from the guideline corpus.

    python -m app.ingest.run

Reads data/sources.yaml, loads each document from data/raw/, chunks it,
embeds the chunks locally, and writes them to the configured vector store.
Re-runnable: rebuilds the index from scratch each time.
"""
from __future__ import annotations

import time
from pathlib import Path

import yaml

from app.config import get_settings
from app.ingest.chunker import chunk_document
from app.ingest.loader import load_document
from app.rag.embeddings import get_embedder
from app.rag.vector_store import get_vector_store


def load_sources(sources_file: str) -> list[dict]:
    data = yaml.safe_load(Path(sources_file).read_text()) or {}
    return data.get("sources", [])


def build_index(settings=None) -> int:
    settings = settings or get_settings()
    sources = load_sources(settings.sources_file)
    embedder = get_embedder(settings.embedding_provider, settings.embedding_model)
    store = get_vector_store(settings.vector_store, settings.chroma_path, settings.chroma_collection)
    store.reset()  # rebuild from scratch each run -> re-running is always safe

    raw_dir = Path(settings.raw_dir)
    total_chunks = 0
    t0 = time.time()

    for src in sources:
        path = raw_dir / src["file"]
        if not path.exists():
            print(f"  ! missing file, skipping: {path}")
            continue
        doc = load_document(path, src["title"], src.get("url", ""), src["id"])
        chunks = chunk_document(doc, settings.chunk_tokens, settings.chunk_overlap)
        if not chunks:
            print(f"  ! no text extracted: {src['title']}")
            continue
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
        total_chunks += len(chunks)
        print(f"  + {src['title']}: {len(chunks)} chunks")

    dt = time.time() - t0
    print(f"\nIndexed {total_chunks} chunks from {len(sources)} source(s) in {dt:.1f}s")
    print(f"Vector store now holds {store.count()} chunks.")
    return total_chunks


if __name__ == "__main__":
    build_index()
