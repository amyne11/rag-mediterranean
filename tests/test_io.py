"""I/O tests: query parsing handles all three known input schemas."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_culinary.chunking import Chunk
from rag_culinary.io_utils import (
    load_chunks, parse_gold_answers, parse_input_queries, save_chunks,
    write_outputs,
)


# ── parse_input_queries ──────────────────────────────────────────────────────

def test_parse_gta_format(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"queries": [
        {"query_id": "0", "query": "What is hummus?"},
        {"query_id": "1", "query": "Where does paella originate?"},
    ]}))
    qs = parse_input_queries(p)
    assert len(qs) == 2
    assert qs[0].query_id == "0"
    assert qs[0].question == "What is hummus?"


def test_parse_flat_list_with_question_key(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([
        {"question": "Q1"},
        {"question": "Q2"},
    ]))
    qs = parse_input_queries(p)
    assert [q.question for q in qs] == ["Q1", "Q2"]
    # query_ids fall back to index
    assert qs[0].query_id == "0"


def test_parse_flat_list_with_query_key(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([{"query": "Q1"}]))
    qs = parse_input_queries(p)
    assert qs[0].question == "Q1"


def test_parse_nested_benchmark(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({
        "sources": [
            {"source_file": "a.txt", "questions": [
                {"question": "QA1", "answer": "A1"},
                {"question": "QA2", "answer": "A2"},
            ]},
            {"source_file": "b.txt", "questions": [
                {"question": "QB1", "answer": "B1"},
            ]},
        ]
    }))
    qs = parse_input_queries(p)
    assert len(qs) == 3
    assert qs[0].question == "QA1"
    assert qs[2].question == "QB1"


def test_parse_unknown_format_raises(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"unrecognised": []}))
    with pytest.raises(ValueError, match="Unrecognised input format"):
        parse_input_queries(p)


def test_parse_missing_question_field_raises(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([{"foo": "bar"}]))
    with pytest.raises(ValueError, match="neither 'question' nor 'query'"):
        parse_input_queries(p)


# ── parse_gold_answers ───────────────────────────────────────────────────────

def test_parse_gold_results_format(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"results": [
        {"query_id": "0", "response": "ans0"},
        {"query_id": "1", "answer": "ans1"},
    ]}))
    g = parse_gold_answers(p)
    assert g["0"] == "ans0"
    assert g["1"] == "ans1"


def test_parse_gold_list_format(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps([
        {"query_id": "0", "answer": "ans0"},
        {"answer": "ans1"},
    ]))
    g = parse_gold_answers(p)
    assert g["0"] == "ans0"
    assert g["1"] == "ans1"  # falls back to enumerate index


# ── chunk persistence round trip ─────────────────────────────────────────────

def test_save_and_load_chunks(tmp_path: Path):
    chunks = [
        Chunk(chunk_id="a", source="doc.txt", strategy="test",
              text="hello", token_count=1),
        Chunk(chunk_id="b", source="doc.txt", strategy="test",
              text="world there", token_count=2),
    ]
    path = tmp_path / "chunks.json"
    save_chunks(chunks, path)
    loaded = load_chunks(path)
    assert loaded == chunks


# ── write_outputs schema ─────────────────────────────────────────────────────

def test_write_outputs_schema(tmp_path: Path):
    results = [{
        "query_id": "0",
        "query": "Q?",
        "response": "A.",
        "retrieved_context": [{"doc_id": "000", "text": "ctx"}],
    }]
    p = tmp_path / "out.json"
    write_outputs(results, p)

    written = json.loads(p.read_text())
    assert "results" in written
    assert written["results"][0]["query_id"] == "0"
    assert written["results"][0]["retrieved_context"][0]["doc_id"] == "000"
