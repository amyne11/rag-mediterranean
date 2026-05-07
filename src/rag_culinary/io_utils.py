"""I/O utilities: chunk persistence, embedding persistence, query format parsing.

The query parser handles the three input formats the original notebooks
encountered: the GTA demo schema, a flat list, and the nested benchmark schema.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_culinary.chunking import Chunk


# ── Chunk persistence ────────────────────────────────────────────────────────

def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk.from_dict(d) for d in data]


# ── Embedding persistence ────────────────────────────────────────────────────

def save_embeddings(
    embeddings: np.ndarray,
    model_name: str,
    chunk_ids: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "model_name": model_name,
            "embeddings": embeddings,
            "chunk_ids": chunk_ids,
        }, f)


def load_embeddings(path: Path) -> dict:
    """Returns dict with keys: model_name, embeddings, chunk_ids."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Query / output schema ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Query:
    query_id: str
    question: str


def parse_input_queries(path: Path) -> list[Query]:
    """Parse a query file in any of the three known formats:

    GTA demo schema:
        {"queries": [{"query_id": "0", "query": "..."}, ...]}

    Flat list:
        [{"question": "..."}, ...]
        or
        [{"query": "..."}, ...]

    Nested benchmark:
        {"sources": [{"source_file": "...", "questions": [{"question": "...", "answer": "..."}]}]}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "queries" in raw:
        return [
            Query(query_id=str(q["query_id"]), question=q["query"])
            for q in raw["queries"]
        ]

    if isinstance(raw, list):
        out = []
        for i, q in enumerate(raw):
            text = q.get("question") or q.get("query")
            if text is None:
                raise ValueError(f"Entry {i} has neither 'question' nor 'query'")
            out.append(Query(query_id=str(q.get("query_id", i)), question=text))
        return out

    if isinstance(raw, dict) and "sources" in raw:
        out = []
        i = 0
        for src in raw["sources"]:
            for qa in src["questions"]:
                out.append(Query(query_id=str(i), question=qa["question"]))
                i += 1
        return out

    raise ValueError(f"Unrecognised input format in {path}")


def write_outputs(
    results: list[dict],
    path: Path,
) -> None:
    """Write inference outputs in the spec-required schema.

    Each result dict must have: query_id, query, response, retrieved_context.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_gold_answers(path: Path) -> dict[str, str]:
    """Parse a gold-answer file into {query_id: answer}.

    Handles the same three-format range as parse_input_queries.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "results" in raw:
        return {
            str(g["query_id"]): g.get("response") or g.get("answer", "")
            for g in raw["results"]
        }
    if isinstance(raw, dict) and "queries" in raw:
        return {
            str(g["query_id"]): g.get("answer") or g.get("response", "")
            for g in raw["queries"]
        }
    if isinstance(raw, list):
        return {
            str(g.get("query_id", i)): g.get("answer") or g.get("response", "")
            for i, g in enumerate(raw)
        }
    raise ValueError(f"Unrecognised gold-answer format in {path}")
