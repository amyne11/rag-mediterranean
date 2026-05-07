"""Build chunks and embeddings from the corpus zip.

This is the ingestion step. Run it once after cloning, then run inference
many times against the cached artifacts.

    python scripts/build_index.py
    python scripts/build_index.py --config config.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rag_culinary.chunking import get_chunker
from rag_culinary.config import load_config
from rag_culinary.corpus import corpus_stats, load_corpus
from rag_culinary.embedding import Embedder
from rag_culinary.io_utils import save_chunks, save_embeddings


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RAG ingestion artifacts.")
    ap.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus
    print(f"[1/3] Loading corpus from {cfg.paths.corpus_zip}...")
    documents = load_corpus(cfg.paths.corpus_zip)
    stats = corpus_stats(documents)
    print(f"      {stats['num_documents']} documents | {stats['total_characters']:,} chars")

    # 2. Chunk
    print(f"[2/3] Chunking with strategy={cfg.chunking.strategy}...")
    chunker = get_chunker(cfg.chunking)
    chunks = chunker(documents)
    chunks_path = cfg.paths.artifacts / f"chunks_{cfg.chunking.strategy}.json"
    save_chunks(chunks, chunks_path)
    print(f"      {len(chunks):,} chunks -> {chunks_path}")

    # 3. Embed
    print(f"[3/3] Embedding with {cfg.embedding.model}...")
    t0 = time.time()
    embedder = Embedder(cfg.embedding)
    embeddings = embedder.encode_corpus([c.text for c in chunks], show_progress=True)

    safe_model = cfg.embedding.model.replace("/", "__")
    embs_path = cfg.paths.artifacts / f"embeddings_{safe_model}.pkl"
    save_embeddings(
        embeddings,
        model_name=cfg.embedding.model,
        chunk_ids=[c.chunk_id for c in chunks],
        path=embs_path,
    )
    print(f"      shape={embeddings.shape}  ({time.time() - t0:.1f}s)")
    print(f"      -> {embs_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
