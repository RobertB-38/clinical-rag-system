"""Load source documents (PDF or plain text) into raw text + metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDoc:
    text: str
    source_title: str
    source_url: str
    source_id: str


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path, source_title: str, source_url: str, source_id: str) -> LoadedDoc:
    if path.suffix.lower() == ".pdf":
        text = _read_pdf(path)
    else:
        text = _read_text(path)
    return LoadedDoc(
        text=text.strip(),
        source_title=source_title,
        source_url=source_url,
        source_id=source_id,
    )
