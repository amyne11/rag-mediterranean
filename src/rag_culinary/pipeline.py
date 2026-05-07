"""End-to-end RAG pipeline.

This is the user-facing class. Two entry points:

    # Build everything from a corpus zip + config (slow first run, caches to disk)
    rag = RAGPipeline.build(cfg)

    # Or load a previously-built pipeline (fast)
    rag = RAGPipeline.from_artifacts(cfg)

Then:
    answer = rag.answer("What is hummus made from?")
    result = rag.answer_with_sources("What is hummus made from?")

The Generator (LLM) is loaded lazily on first generation call, so retrieval-only
workflows (e.g. evaluate.py without --generation) never download or load the
language model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rag_culinary.chunking import get_chunker
from rag_culinary.config import Config, load_config
from rag_culinary.corpus import load_corpus
from rag_culinary.io_utils import (
    load_chunks, load_embeddings, save_chunks, save_embeddings,
)
from rag_culinary.retrieval import Retriever, RetrievalResult, build_retriever

if TYPE_CHECKING:
    from rag_culinary.generation import Generator


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    retrieved: list[RetrievalResult]


class RAGPipeline:
    """Stateful pipeline holding the loaded retriever (and a lazily-loaded generator)."""

    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        generator: Optional["Generator"] = None,
    ):
        self.config = config
        self.retriever = retriever
        self._generator = generator   # may be None until first generate() call

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "RAGPipeline":
        """Convenience: load config and build from artifacts (or build from scratch)."""
        cfg = load_config(config_path)
        try:
            return cls.from_artifacts(cfg)
        except FileNotFoundError:
            return cls.build(cfg)

    @classmethod
    def build(cls, cfg: Config) -> "RAGPipeline":
        """Build pipeline from corpus zip. Caches chunks + embeddings to disk.

        Generator is NOT loaded here — it's lazily loaded on first generation call
        so retrieval-only workflows skip the LLM download entirely.
        """
        # Heavy import deferred so this method doesn't pull in torch unless used
        from rag_culinary.embedding import Embedder

        cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)

        # 1. Corpus -> chunks
        documents = load_corpus(cfg.paths.corpus_zip)
        chunker = get_chunker(cfg.chunking)
        chunks = chunker(documents)
        save_chunks(chunks, _chunks_path(cfg))

        # 2. Chunks -> embeddings
        embedder = Embedder(cfg.embedding)
        texts = [c.text for c in chunks]
        embeddings = embedder.encode_corpus(texts, show_progress=True)
        save_embeddings(
            embeddings,
            model_name=cfg.embedding.model,
            chunk_ids=[c.chunk_id for c in chunks],
            path=_embeddings_path(cfg),
        )

        # 3. Build retriever (generator is lazy)
        retriever = build_retriever(cfg.retrieval, chunks, embeddings, embedder)
        return cls(cfg, retriever)

    @classmethod
    def from_artifacts(cls, cfg: Config) -> "RAGPipeline":
        """Load a pipeline from previously-built chunks and embeddings on disk.

        Generator is NOT loaded here — it's lazily loaded on first generation call.
        """
        from rag_culinary.embedding import Embedder

        chunks_path = _chunks_path(cfg)
        embs_path = _embeddings_path(cfg)
        if not chunks_path.exists() or not embs_path.exists():
            raise FileNotFoundError(
                f"Artifacts not found ({chunks_path}, {embs_path}). "
                f"Run scripts/build_index.py first."
            )

        chunks = load_chunks(chunks_path)
        emb_data = load_embeddings(embs_path)
        embeddings = emb_data["embeddings"]

        embedder = Embedder(cfg.embedding)
        retriever = build_retriever(cfg.retrieval, chunks, embeddings, embedder)
        return cls(cfg, retriever)

    # ── Lazy generator ───────────────────────────────────────────────────────

    @property
    def generator(self) -> "Generator":
        """Loads the configured generator on first access. Reused thereafter."""
        if self._generator is None:
            from rag_culinary.generation import build_generator
            self._generator = build_generator(self.config.generation, self.config.cuisine)
        return self._generator

    # ── Inference ────────────────────────────────────────────────────────────

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievalResult]:
        """Retrieve only — useful for inspecting retrieval without LLM cost."""
        return self.retriever.retrieve(question, k or self.config.retrieval.top_k)

    def answer(self, question: str) -> str:
        """Run the full pipeline and return just the answer string."""
        return self.answer_with_sources(question).answer

    def answer_with_sources(self, question: str) -> RAGAnswer:
        """Run the full pipeline and return answer + retrieved chunks."""
        retrieved = self.retrieve(question)
        ans = self.generator.generate(question, retrieved)
        return RAGAnswer(question=question, answer=ans, retrieved=retrieved)


# ── Path helpers (artifact filenames depend on config) ───────────────────────

def _chunks_path(cfg: Config) -> Path:
    return cfg.paths.artifacts / f"chunks_{cfg.chunking.strategy}.json"


def _embeddings_path(cfg: Config) -> Path:
    safe_model = cfg.embedding.model.replace("/", "__")
    return cfg.paths.artifacts / f"embeddings_{safe_model}.pkl"
