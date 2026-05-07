"""Typed configuration loaded from YAML.

Centralising config in a dataclass means the rest of the package never sees raw
dict lookups, and missing/mistyped keys fail loudly at load time rather than
deep inside a pipeline run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    corpus_zip: Path
    benchmark: Path
    artifacts: Path


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str          # fixed_200 | fixed_400 | sentence | overlapping
    size: int = 200
    overlap: int = 50
    max_sentences: int = 5


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    query_prefix: str = ""
    batch_size: int = 32


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str          # dense | bm25 | hybrid
    hybrid_alpha: float = 0.6
    top_k: int = 5


@dataclass(frozen=True)
class GenerationConfig:
    backend: str           # local | groq
    model: str
    prompt_style: str      # basic | instructed | role_based | cot
    max_new_tokens: int = 150
    do_sample: bool = False


@dataclass(frozen=True)
class Config:
    paths: Paths
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    cuisine: str = "Mediterranean"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        p = raw["paths"]
        return cls(
            paths=Paths(
                corpus_zip=Path(p["corpus_zip"]),
                benchmark=Path(p["benchmark"]),
                artifacts=Path(p["artifacts"]),
            ),
            chunking=ChunkingConfig(**raw["chunking"]),
            embedding=EmbeddingConfig(**raw["embedding"]),
            retrieval=RetrievalConfig(**raw["retrieval"]),
            generation=GenerationConfig(**raw["generation"]),
            cuisine=raw.get("cuisine", "Mediterranean"),
        )


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.from_dict(raw)
