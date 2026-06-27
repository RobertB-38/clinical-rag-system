"""Embedding models behind a single interface.

`STEmbedder` wraps any sentence-transformers model. We default to
**BAAI/bge-small-en-v1.5**, which retrieves noticeably better than MiniLM on
this corpus (it surfaces the spirometry/FEV1 section for "how is COPD
diagnosed?", where MiniLM returned epidemiology). BGE expects an instruction
prefix on the *query* only; passages are embedded plain. This query/passage
asymmetry is handled here via `encode_query` vs `encode`.

`FakeEmbedder` is a deterministic hash vector for tests/CI — no model
download, no network. All vectors are L2-normalised, so a dot product is the
cosine similarity.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:        # passages -> (n, dim)
        ...

    def encode_query(self, text: str) -> np.ndarray:          # single query -> (dim,)
        ...


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype(np.float32)


class FakeEmbedder:
    """Deterministic, dependency-free embeddings for tests."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for t in text.lower().split():
                h = int(hashlib.md5(t.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalise(out)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class STEmbedder:
    """sentence-transformers model with optional query/passage prompt prefixes."""

    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = "") -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._qp = query_prefix
        self._pp = passage_prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        prepared = [self._pp + t for t in texts] if self._pp else texts
        vecs = self._model.encode(
            prepared, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        v = self._model.encode(
            [self._qp + text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return v[0].astype(np.float32)


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "fake":
        return FakeEmbedder()
    if provider == "minilm":
        return STEmbedder(model_name)  # symmetric, no prefixes
    if provider == "bge":
        return STEmbedder("BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_PREFIX)
    raise ValueError(f"Unknown embedding_provider: {provider!r}")
