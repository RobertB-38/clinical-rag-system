"""Vector stores behind one interface.

`ChromaVectorStore` is the real, persistent store (survives restarts, ships
in a Docker volume). `MemoryVectorStore` is a pure-NumPy cosine store used in
tests — no disk, no chromadb dependency, identical search semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Hit:
    text: str
    score: float
    source_title: str
    source_url: str
    chunk_id: str


class VectorStore(Protocol):
    def reset(self) -> None: ...

    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str],
            metadatas: list[dict]) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Hit]: ...

    def count(self) -> int: ...


class MemoryVectorStore:
    """In-memory cosine-similarity store. Vectors are assumed L2-normalised,
    so a dot product is the cosine similarity in [-1, 1]."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._texts: list[str] = []
        self._metas: list[dict] = []

    def reset(self) -> None:
        self._vectors = None
        self._texts = []
        self._metas = []

    def add(self, ids, vectors, texts, metadatas) -> None:
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._texts.extend(texts)
        self._metas.extend(metadatas)

    def search(self, query_vector, top_k) -> list[Hit]:
        if self._vectors is None or len(self._texts) == 0:
            return []
        sims = self._vectors @ query_vector.reshape(-1)
        k = min(top_k, len(self._texts))
        idx = np.argsort(-sims)[:k]
        return [
            Hit(
                text=self._texts[i],
                score=float(sims[i]),
                source_title=self._metas[i].get("source_title", ""),
                source_url=self._metas[i].get("source_url", ""),
                chunk_id=self._metas[i].get("chunk_id", ""),
            )
            for i in idx
        ]

    def count(self) -> int:
        return len(self._texts)


class ChromaVectorStore:
    """Persistent store backed by chromadb. We pass our own embeddings, so
    Chroma does no embedding itself — it's pure vector storage + ANN search."""

    def __init__(self, path: str, collection: str) -> None:
        import chromadb  # lazy import
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._name = collection
        # cosine space to match our normalised vectors
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        """Drop and recreate the collection so a rebuild starts clean."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, vectors, texts, metadatas) -> None:
        self._col.add(
            ids=ids,
            embeddings=[v.tolist() for v in vectors],
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query_vector, top_k) -> list[Hit]:
        res = self._col.query(
            query_embeddings=[query_vector.reshape(-1).tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            # chroma cosine distance = 1 - cosine_similarity
            hits.append(
                Hit(
                    text=text,
                    score=float(1.0 - dist),
                    source_title=meta.get("source_title", ""),
                    source_url=meta.get("source_url", ""),
                    chunk_id=meta.get("chunk_id", ""),
                )
            )
        return hits

    def count(self) -> int:
        return self._col.count()


def get_vector_store(kind: str, path: str, collection: str) -> VectorStore:
    if kind == "memory":
        return MemoryVectorStore()
    if kind == "chroma":
        return ChromaVectorStore(path, collection)
    raise ValueError(f"Unknown vector_store: {kind!r}")
