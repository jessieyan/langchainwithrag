"""LangChain-compatible retriever over an already-populated vector store.

This module does NOT ingest documents. Ingestion is a separate, one-time
operation (see ``ai/ingest.py``). Having retriever construction embed
documents on every startup used to double application cold start time
and re-billed the embedding API for a corpus that was already indexed.
"""
from llama_index.core import VectorStoreIndex

from llamaindex_langchain_retriever import LlamaIndexLangChainRetriever


def get_llama_retriever(vector_store, embed_model):
    """Wrap an already-populated vector store as a LangChain retriever.

    Parameters
    ----------
    vector_store:
        A LlamaIndex vector store that has already been populated by
        ``ai/ingest.py``. Building a retriever over an empty store will
        return zero results at query time.
    embed_model:
        Embedding model used at query time to embed the user's question.
    """
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    llama_retriever = vector_index.as_retriever()
    return LlamaIndexLangChainRetriever(llamaindex_retriever=llama_retriever)
