"""Chunking strategies.

Each strategy takes a list of Documents and returns a list of Chunks. They are
registered by name so the active strategy can be selected from config:

    chunker = get_chunker(config.chunking)
    chunks = chunker(documents)

Adding a new strategy means writing a function and adding it to the registry.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import nltk

from rag_culinary.config import ChunkingConfig
from rag_culinary.corpus import Document


# Lazy-init: only download punkt when sentence chunking is actually used
_punkt_ready = False


def _ensure_punkt() -> None:
    global _punkt_ready
    if _punkt_ready:
        return
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _punkt_ready = True


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str            # original filename
    strategy: str
    text: str
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        # tolerate extra legacy keys (e.g. 'sentence_count')
        return cls(
            chunk_id=d["chunk_id"],
            source=d["source"],
            strategy=d["strategy"],
            text=d["text"],
            token_count=d.get("token_count", len(d["text"].split())),
        )


# ── Strategy implementations ──────────────────────────────────────────────────

def fixed_size(documents: list[Document], size: int) -> list[Chunk]:
    """Non-overlapping fixed-size token windows."""
    chunks: list[Chunk] = []
    for doc in documents:
        tokens = doc.text.split()
        for i in range(0, len(tokens), size):
            window = tokens[i : i + size]
            if window:
                chunks.append(Chunk(
                    chunk_id=f"fixed{size}_{doc.filename}_{i}",
                    source=doc.filename,
                    strategy=f"fixed_{size}_tokens",
                    text=" ".join(window),
                    token_count=len(window),
                ))
    return chunks


def sentence_based(documents: list[Document], max_sentences: int = 5) -> list[Chunk]:
    """Group NLTK sentences into fixed-count chunks."""
    _ensure_punkt()
    chunks: list[Chunk] = []
    for doc in documents:
        sents = nltk.sent_tokenize(doc.text)
        for i in range(0, len(sents), max_sentences):
            group = sents[i : i + max_sentences]
            text = " ".join(group).strip()
            if text:
                chunks.append(Chunk(
                    chunk_id=f"sent_{doc.filename}_{i}",
                    source=doc.filename,
                    strategy="sentence_based",
                    text=text,
                    token_count=len(text.split()),
                ))
    return chunks


def overlapping(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    """Sliding window: chunks of `size` tokens, advancing by `size - overlap`."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be < size ({size})")
    step = size - overlap
    chunks: list[Chunk] = []
    for doc in documents:
        tokens = doc.text.split()
        # Use a do-while pattern so a doc shorter than `size` still yields one chunk
        i = 0
        produced = False
        while i < len(tokens) or not produced:
            window = tokens[i : i + size]
            if not window:
                break
            chunks.append(Chunk(
                chunk_id=f"overlap_{doc.filename}_{i}",
                source=doc.filename,
                strategy=f"overlap_{size}_{overlap}",
                text=" ".join(window),
                token_count=len(window),
            ))
            produced = True
            if i + size >= len(tokens):
                break
            i += step
    return chunks


# ── Registry ──────────────────────────────────────────────────────────────────

ChunkerFn = Callable[[list[Document]], list[Chunk]]


def get_chunker(cfg: ChunkingConfig) -> ChunkerFn:
    """Return a callable that applies the configured chunking strategy."""
    s = cfg.strategy
    if s == "fixed_200":
        return lambda docs: fixed_size(docs, 200)
    if s == "fixed_400":
        return lambda docs: fixed_size(docs, 400)
    if s == "sentence":
        return lambda docs: sentence_based(docs, cfg.max_sentences)
    if s == "overlapping":
        return lambda docs: overlapping(docs, cfg.size, cfg.overlap)
    raise ValueError(
        f"Unknown chunking strategy: {s!r}. "
        f"Valid options: fixed_200, fixed_400, sentence, overlapping"
    )
