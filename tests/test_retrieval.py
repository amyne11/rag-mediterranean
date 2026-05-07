"""Unit tests for retrieval — uses fake embedders to avoid model downloads."""
from __future__ import annotations

import numpy as np
import pytest

from rag_culinary.chunking import Chunk
from rag_culinary.retrieval import (
    BM25Retriever, DenseRetriever, HybridRetriever,
    bm25_tokenise, min_max_norm,
)


# ── A trivial fake Embedder so DenseRetriever tests don't download a model ──

class FakeEmbedder:
    """Embeds queries into a known direction so we can predict the top result."""

    def __init__(self, query_to_vec: dict[str, np.ndarray]):
        self.query_to_vec = query_to_vec

    def encode_query(self, query: str) -> np.ndarray:
        return self.query_to_vec[query]


def make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            source=f"doc{i}.txt",
            strategy="test",
            text=f"chunk {i} text about ingredient_{i}",
            token_count=5,
        )
        for i in range(n)
    ]


# ── tokenisation + normalisation ─────────────────────────────────────────────

def test_bm25_tokenise_lowercases_and_keeps_words():
    assert bm25_tokenise("Hello, World!") == ["hello", "world"]
    assert bm25_tokenise("foo123 bar") == ["foo123", "bar"]


def test_min_max_norm_handles_constant_array():
    out = min_max_norm(np.array([3.0, 3.0, 3.0]))
    assert np.array_equal(out, np.zeros(3))


def test_min_max_norm_scales_to_unit_range():
    out = min_max_norm(np.array([0.0, 5.0, 10.0]))
    assert out[0] == 0.0
    assert out[2] == 1.0
    assert 0.0 < out[1] < 1.0


# ── DenseRetriever ───────────────────────────────────────────────────────────

def test_dense_retriever_returns_top_k_in_rank_order():
    chunks = make_chunks(5)
    # Embeddings are one-hot vectors so we can predict cosine sims exactly
    embeddings = np.eye(5, dtype="float32")
    # Query vector closest to chunk 2
    embedder = FakeEmbedder({"q": embeddings[2]})

    retriever = DenseRetriever(chunks, embeddings, embedder)
    results = retriever.retrieve("q", k=3)

    assert len(results) == 3
    assert results[0].chunk.chunk_id == "c2"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0)


def test_dense_retriever_rejects_size_mismatch():
    with pytest.raises(ValueError):
        DenseRetriever(make_chunks(3), np.eye(5, dtype="float32"), FakeEmbedder({}))


# ── BM25Retriever ────────────────────────────────────────────────────────────

def test_bm25_retriever_finds_keyword_match():
    chunks = make_chunks(5)
    retriever = BM25Retriever(chunks)
    # 'ingredient_3' appears verbatim in chunk 3 only
    results = retriever.retrieve("ingredient_3", k=3)
    assert results[0].chunk.chunk_id == "c3"


# ── HybridRetriever ──────────────────────────────────────────────────────────

def test_hybrid_retriever_alpha_bounds():
    chunks = make_chunks(3)
    embeddings = np.eye(3, dtype="float32")
    embedder = FakeEmbedder({"q": embeddings[0]})

    # Valid bounds
    HybridRetriever(chunks, embeddings, embedder, alpha=0.0)
    HybridRetriever(chunks, embeddings, embedder, alpha=1.0)
    HybridRetriever(chunks, embeddings, embedder, alpha=0.6)

    # Invalid
    with pytest.raises(ValueError):
        HybridRetriever(chunks, embeddings, embedder, alpha=1.5)
    with pytest.raises(ValueError):
        HybridRetriever(chunks, embeddings, embedder, alpha=-0.1)


def test_hybrid_retriever_alpha_one_is_dense_only():
    """alpha=1 should give dense-only ranking (BM25 contribution is zeroed)."""
    chunks = make_chunks(5)
    embeddings = np.eye(5, dtype="float32")
    embedder = FakeEmbedder({"unrelated text": embeddings[2]})

    hybrid = HybridRetriever(chunks, embeddings, embedder, alpha=1.0)
    results = hybrid.retrieve("unrelated text", k=1)
    # BM25 contribution gets zeroed; dense alone picks chunk 2
    assert results[0].chunk.chunk_id == "c2"
