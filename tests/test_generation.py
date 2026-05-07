"""Tests for the generation module — prompt builders + factory dispatch.

These tests deliberately do not load any model or hit any API. They cover the
deterministic logic only: prompt formatting, context formatting, and factory
error paths.
"""
from __future__ import annotations

import pytest

from rag_culinary.chunking import Chunk
from rag_culinary.config import GenerationConfig
from rag_culinary.generation import (
    build_generator, format_context, get_prompt_fn,
)
from rag_culinary.retrieval import RetrievalResult


# ── Prompt builders ──────────────────────────────────────────────────────────

def test_basic_prompt_includes_context_and_question():
    prompt_fn = get_prompt_fn("basic")
    out = prompt_fn("What is hummus?", "Hummus is a chickpea dip.")
    assert "Hummus is a chickpea dip." in out
    assert "What is hummus?" in out


def test_instructed_prompt_says_dont_know():
    prompt_fn = get_prompt_fn("instructed")
    out = prompt_fn("Q?", "ctx")
    assert "I don't know" in out


def test_role_based_prompt_includes_cuisine():
    prompt_fn = get_prompt_fn("role_based", cuisine="Sicilian")
    out = prompt_fn("Q?", "ctx")
    assert "Sicilian cuisine" in out


def test_cot_prompt_includes_step_by_step():
    prompt_fn = get_prompt_fn("cot")
    out = prompt_fn("Q?", "ctx")
    assert "step by step" in out.lower()


def test_unknown_prompt_style_raises():
    with pytest.raises(ValueError, match="Unknown prompt style"):
        get_prompt_fn("not_a_real_style")


# ── Context formatting ──────────────────────────────────────────────────────

def test_format_context_includes_chunk_metadata():
    chunks = [
        RetrievalResult(rank=1, score=0.9, chunk=Chunk(
            chunk_id="a", source="doc1.txt", strategy="t",
            text="text one", token_count=2,
        )),
        RetrievalResult(rank=2, score=0.8, chunk=Chunk(
            chunk_id="b", source="doc2.txt", strategy="t",
            text="text two", token_count=2,
        )),
    ]
    out = format_context(chunks)
    assert "Chunk 1" in out and "Chunk 2" in out
    assert "doc1.txt" in out and "doc2.txt" in out
    assert "text one" in out and "text two" in out


# ── Factory dispatch ────────────────────────────────────────────────────────

def test_unknown_backend_raises():
    cfg = GenerationConfig(backend="not_a_backend", model="x", prompt_style="basic")
    with pytest.raises(ValueError, match="Unknown generation backend"):
        build_generator(cfg)


def test_groq_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = GenerationConfig(backend="groq", model="llama-3.3-70b-versatile",
                           prompt_style="instructed")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_generator(cfg)


def test_gemini_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = GenerationConfig(backend="gemini", model="gemini-2.5-flash",
                           prompt_style="instructed")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        build_generator(cfg)
