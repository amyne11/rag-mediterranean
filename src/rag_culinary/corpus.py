"""Load the background corpus and QA benchmark.

The corpus ships as a zip of plain-text files. The benchmark is a nested JSON
with one block per source file, each carrying multiple QA pairs.
"""
from __future__ import annotations

import json
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    filename: str
    text: str


@dataclass(frozen=True)
class QAPair:
    id: int
    question: str
    answer: str
    source_file: str


def normalise_filename(name: str) -> str:
    """Strip accents so 'provençal_cuisine.txt' matches 'provencal_cuisine.txt'.

    The benchmark and the corpus archive sometimes disagree on diacritics; this
    function makes them comparable.
    """
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")


def load_corpus(zip_path: Path) -> list[Document]:
    """Load every .txt file from the corpus zip into Document objects."""
    documents: list[Document] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = sorted(
            n for n in zf.namelist()
            if n.endswith(".txt") and not n.startswith("__MACOSX")
        )
        for entry in entries:
            text = zf.read(entry).decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
            if text:
                documents.append(Document(filename=Path(entry).name, text=text))
    return documents


def load_benchmark(path: Path) -> list[QAPair]:
    """Flatten the nested benchmark JSON into a list of QAPair objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs: list[QAPair] = []
    qa_id = 1
    for entry in raw["sources"]:
        src = normalise_filename(entry["source_file"])
        for qa in entry["questions"]:
            pairs.append(QAPair(
                id=qa_id,
                question=qa["question"],
                answer=qa["answer"],
                source_file=src,
            ))
            qa_id += 1
    return pairs


def corpus_stats(documents: Iterable[Document]) -> dict:
    """Quick summary used by build_index.py."""
    docs = list(documents)
    total_chars = sum(len(d.text) for d in docs)
    return {
        "num_documents": len(docs),
        "total_characters": total_chars,
        "avg_characters": total_chars // len(docs) if docs else 0,
    }
