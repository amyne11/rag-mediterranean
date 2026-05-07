"""Mediterranean Culinary RAG — retrieval-augmented Q&A package.

The Config / load_config are imported eagerly because they are cheap. The
RAGPipeline (and anything pulling in torch / sentence-transformers) is loaded
lazily on first attribute access — this keeps `import rag_culinary` fast for
tests and tools that only need the light components.
"""
from rag_culinary.config import Config, load_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "RAGPipeline"]


def __getattr__(name: str):
    if name == "RAGPipeline":
        from rag_culinary.pipeline import RAGPipeline
        return RAGPipeline
    raise AttributeError(f"module 'rag_culinary' has no attribute {name!r}")
