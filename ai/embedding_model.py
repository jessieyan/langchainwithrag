"""Embedding model factory.

Referenced by ``ingest.py``, ``workflow/retrieval.py`` and the chat
scripts. Kept thin so callers can swap providers without touching the
retrieval pipeline.
"""
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)


def get_embedding(name: str):
    if name == "dashscope":
        return DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        )
    raise ValueError(f"unknown embedding provider: {name}")
