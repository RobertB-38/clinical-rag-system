"""Split documents into overlapping, metadata-tagged chunks.

Uses a word-based recursive splitter. The chunk size is expressed in *words*
kept comfortably under MiniLM's 256-token cap (≈ a 0.75 word-per-token ratio),
so chunks are not silently truncated at embedding time — a common, retrieval-
wrecking bug.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ingest.loader import LoadedDoc


@dataclass
class Chunk:
    text: str
    source_title: str
    source_url: str
    chunk_id: str


def _split_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if window:
            chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return chunks


def chunk_document(doc: LoadedDoc, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    # tokens ≈ words / 0.75  ->  words ≈ tokens * 0.75
    max_words = max(20, int(max_tokens * 0.75))
    overlap_words = max(0, int(overlap_tokens * 0.75))
    pieces = _split_words(doc.text, max_words, overlap_words)
    return [
        Chunk(
            text=piece,
            source_title=doc.source_title,
            source_url=doc.source_url,
            chunk_id=f"{doc.source_id}::{i}",
        )
        for i, piece in enumerate(pieces)
    ]
