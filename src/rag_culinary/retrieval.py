"""Retrievers: Dense, BM25, Hybrid.

Three concrete retrievers behind a common protocol. Each returns the same
RetrievalResult shape, so the rest of the pipeline doesn't need to know which
strategy is active.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rag_culinary.chunking import Chunk
from rag_culinary.config import RetrievalConfig

if TYPE_CHECKING:
    # Only imported for type hints — keeps retrieval module loadable without
    # sentence-transformers (e.g. for unit tests with a fake embedder).
    from rag_culinary.embedding import Embedder


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    chunk: Chunk


# ── Tokenisation used by BM25 ────────────────────────────────────────────────

def bm25_tokenise(text: str) -> list[str]:
    """Lowercase word-level tokenisation; matches what experiments used."""
    return re.findall(r"\b\w+\b", text.lower())


# ── Score normalisation for hybrid combination ───────────────────────────────

def min_max_norm(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# ── Retriever protocol ───────────────────────────────────────────────────────

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]: ...


# ── Dense (FAISS) ────────────────────────────────────────────────────────────

class DenseRetriever:
    """Cosine similarity via FAISS inner-product on L2-normalised embeddings."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must align 1:1")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        q = self.embedder.encode_query(query).reshape(1, -1)
        scores, idxs = self.index.search(q, k)
        return [
            RetrievalResult(rank=r + 1, score=float(scores[0][r]), chunk=self.chunks[int(idxs[0][r])])
            for r in range(len(idxs[0]))
        ]


# ── Sparse (BM25) ────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 keyword retrieval over tokenised chunk text."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokenised = [bm25_tokenise(c.text) for c in chunks]
        self.index = BM25Okapi(self.tokenised)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        scores = self.index.get_scores(bm25_tokenise(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(rank=r + 1, score=float(scores[i]), chunk=self.chunks[i])
            for r, i in enumerate(top)
        ]


# ── Hybrid (alpha-weighted) ──────────────────────────────────────────────────

class HybridRetriever:
    """Hybrid dense + BM25:  alpha * norm(dense) + (1-alpha) * norm(bm25).

    Both signals are min-max normalised per query so they live in the same
    [0, 1] range before weighting.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        alpha: float = 0.6,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.alpha = alpha
        self.bm25 = BM25Okapi([bm25_tokenise(c.text) for c in chunks])

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        q_emb = self.embedder.encode_query(query)
        dense = np.asarray(self.embeddings @ q_emb)
        sparse = np.asarray(self.bm25.get_scores(bm25_tokenise(query)))

        hybrid = self.alpha * min_max_norm(dense) + (1 - self.alpha) * min_max_norm(sparse)
        top = np.argsort(hybrid)[::-1][:k]
        return [
            RetrievalResult(rank=r + 1, score=float(hybrid[i]), chunk=self.chunks[i])
            for r, i in enumerate(top)
        ]


# ── Factory ──────────────────────────────────────────────────────────────────

def build_retriever(
    cfg: RetrievalConfig,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    embedder: Embedder,
) -> Retriever:
    """Build a retriever from config + already-loaded artifacts."""
    s = cfg.strategy
    if s == "dense":
        return DenseRetriever(chunks, embeddings, embedder)
    if s == "bm25":
        return BM25Retriever(chunks)
    if s == "hybrid":
        return HybridRetriever(chunks, embeddings, embedder, cfg.hybrid_alpha)
    raise ValueError(f"Unknown retrieval strategy: {s!r}. Valid: dense, bm25, hybrid")
