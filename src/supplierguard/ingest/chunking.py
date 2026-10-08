"""Paragraph-aware chunking with word-aligned overlap."""

from __future__ import annotations

import re
from typing import List

_SPLIT = re.compile(r"\n|(?<=[.!?])\s+")


def _split_long(paragraph: str, max_chars: int) -> List[str]:
    """Break an oversized paragraph on line and sentence boundaries, then on words."""
    pieces = [p.strip() for p in _SPLIT.split(paragraph) if p.strip()]
    out: List[str] = []
    cur = ""
    for piece in pieces:
        while len(piece) > max_chars:  # a single sentence longer than the limit
            cut = piece.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if not cur:
            cur = piece
        elif len(cur) + 1 + len(piece) <= max_chars:
            cur += " " + piece
        else:
            out.append(cur)
            cur = piece
    if cur:
        out.append(cur)
    return out


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text.strip() if overlap > 0 else ""
    start = len(text) - overlap
    if not text[start - 1].isspace():  # do not start mid-word
        m = re.search(r"\s", text[start:])
        start = start + m.end() if m else len(text)
    return text[start:].strip()


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")
    units: List[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        units.extend([para] if len(para) <= max_chars else _split_long(para, max_chars))

    chunks: List[str] = []
    cur = ""
    for unit in units:
        if not cur:
            cur = unit
        elif len(cur) + 2 + len(unit) <= max_chars:
            cur += "\n\n" + unit
        else:
            chunks.append(cur)
            tail = _overlap_tail(cur, overlap)
            # Overlap is dropped when it would push the next chunk over the size limit.
            cur = f"{tail}\n\n{unit}" if tail and len(tail) + 2 + len(unit) <= max_chars else unit
    if cur:
        chunks.append(cur)
    return chunks
