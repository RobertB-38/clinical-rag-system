"""Retrieval: question -> ranked guideline passages.

This is the seam that lets the vector store be swapped without touching the
API. It embeds the query with the same model used at ingest time and returns
the top-k hits. Filtering by score threshold is the pipeline's job, so the API
can still show what *was* retrieved even on a refusal.

Components can be injected (used by tests to share an in-memory store between
ingestion and retrieval); otherwise they are built from settings.
"""
from __future__ import annotations

from app.config import Settings, get_settings
from app.rag.embeddings import Embedder, get_embedder
from app.rag.vector_store import Hit, VectorStore, get_vector_store


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder or get_embedder(
            self.settings.embedding_provider, self.settings.embedding_model
        )
        self._store = store or get_vector_store(
            self.settings.vector_store, self.settings.chroma_path, self.settings.chroma_collection
        )

    def search(self, question: str, top_k: int | None = None) -> list[Hit]:
        k = top_k or self.settings.top_k
        query_vec = self._embedder.encode_query(question)
        return self._store.search(query_vec, k)
