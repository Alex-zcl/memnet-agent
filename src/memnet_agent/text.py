from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def read_text_input(value: str | Path | Iterable[str]) -> list[str]:
    """Normalize text, a text-file path, or an iterable of texts."""

    if isinstance(value, Path):
        return [value.read_text(encoding="utf-8", errors="replace")]
    if isinstance(value, str):
        possible_path = Path(value)
        if "\n" not in value and len(value) < 4096 and possible_path.is_file():
            return [possible_path.read_text(encoding="utf-8", errors="replace")]
        return [value]
    return [str(item) for item in value]


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_sentences: int = 1,
    min_chars: int = 1,
) -> list[str]:
    """Split long text into sentence-aware overlapping chunks."""

    normalized = " ".join(text.replace("\r", "\n").split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized] if len(normalized) >= min_chars else []

    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(normalized) if part.strip()]
    if len(sentences) == 1:
        chunks = [normalized[i : i + max_chars] for i in range(0, len(normalized), max_chars)]
        return [chunk for chunk in chunks if len(chunk) >= min_chars]

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if len(chunk) >= min_chars]
