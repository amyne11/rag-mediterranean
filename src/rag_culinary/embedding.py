"""Thin wrapper around sentence-transformers.

Exists for two reasons:
1. To centralise the BGE query-prefix quirk — BGE retrieval models expect a
   prefix on queries but NOT on corpus passages. Forgetting this silently
   degrades retrieval quality.
2. To return L2-normalised float32 numpy arrays in a single call, which is what
   FAISS inner-product search needs.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_culinary.config import EmbeddingConfig


class Embedder:
    """Encodes corpus chunks and queries with the configured model."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_corpus(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed corpus passages. No query prefix is applied."""
        embs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embs.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query. The configured prefix (if any) is prepended."""
        text = self.cfg.query_prefix + query if self.cfg.query_prefix else query
        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return emb.astype("float32")
