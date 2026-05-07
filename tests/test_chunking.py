"""Unit tests for chunking strategies.

These don't need the corpus — they construct synthetic Documents.
"""
from __future__ import annotations

import pytest

from rag_culinary.chunking import (
    Chunk, fixed_size, get_chunker, overlapping, sentence_based,
)
from rag_culinary.config import ChunkingConfig
from rag_culinary.corpus import Document


def make_doc(filename: str, n_words: int) -> Document:
    text = " ".join(f"word{i}" for i in range(n_words))
    return Document(filename=filename, text=text)


# ── fixed_size ───────────────────────────────────────────────────────────────

def test_fixed_size_basic():
    docs = [make_doc("a.txt", 500)]
    chunks = fixed_size(docs, size=200)
    assert len(chunks) == 3
    assert chunks[0].token_count == 200
    assert chunks[1].token_count == 200
    assert chunks[2].token_count == 100  # remainder


def test_fixed_size_short_doc_gets_one_chunk():
    docs = [make_doc("short.txt", 50)]
    chunks = fixed_size(docs, size=200)
    assert len(chunks) == 1
    assert chunks[0].token_count == 50


def test_fixed_size_strategy_name_records_size():
    docs = [make_doc("a.txt", 50)]
    chunks = fixed_size(docs, size=200)
    assert chunks[0].strategy == "fixed_200_tokens"


# ── overlapping ──────────────────────────────────────────────────────────────

def test_overlapping_creates_overlap():
    docs = [make_doc("a.txt", 500)]
    chunks = overlapping(docs, size=200, overlap=50)
    # step = 150 → starts at 0, 150, 300, 450
    # The last window starts at 450 but only has 50 tokens; loop terminates after.
    starts = [int(c.chunk_id.rsplit("_", 1)[1]) for c in chunks]
    # Each consecutive pair of starts must differ by step=150
    for s1, s2 in zip(starts, starts[1:]):
        assert s2 - s1 == 150


def test_overlapping_rejects_overlap_geq_size():
    with pytest.raises(ValueError):
        overlapping([make_doc("a.txt", 100)], size=100, overlap=100)


def test_overlapping_short_doc_yields_one_chunk():
    docs = [make_doc("short.txt", 50)]
    chunks = overlapping(docs, size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].token_count == 50


# ── sentence_based ───────────────────────────────────────────────────────────

def test_sentence_based_basic():
    pytest.importorskip("nltk")
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
            nltk.data.find("tokenizers/punkt_tab")
        except Exception:
            pytest.skip("NLTK punkt_tab data not available (offline sandbox)")

    text = ". ".join(f"This is sentence {i}" for i in range(12)) + "."
    docs = [Document(filename="s.txt", text=text)]
    chunks = sentence_based(docs, max_sentences=5)
    # 12 sentences / 5 per chunk = 3 chunks (5, 5, 2)
    assert len(chunks) == 3


# ── registry ─────────────────────────────────────────────────────────────────

def test_get_chunker_dispatch():
    cfg = ChunkingConfig(strategy="overlapping", size=100, overlap=20)
    chunker = get_chunker(cfg)
    docs = [make_doc("a.txt", 250)]
    chunks = chunker(docs)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.token_count <= 100 for c in chunks)


def test_get_chunker_unknown_strategy():
    cfg = ChunkingConfig(strategy="random", size=100, overlap=20)
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker(cfg)


# ── round-trip serialisation ─────────────────────────────────────────────────

def test_chunk_dict_round_trip():
    chunk = Chunk(
        chunk_id="x", source="a.txt", strategy="overlap_200_50",
        text="hello world", token_count=2,
    )
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk


def test_chunk_from_dict_tolerates_legacy_keys():
    legacy = {
        "chunk_id": "x",
        "source": "a.txt",
        "strategy": "sentence_based",
        "text": "hello world",
        "sentence_count": 5,   # legacy field that no longer exists on Chunk
    }
    chunk = Chunk.from_dict(legacy)
    assert chunk.token_count == 2  # derived from text
