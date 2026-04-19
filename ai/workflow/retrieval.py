"""Shared retriever singleton for the RAG workflow.

``rag_node`` and ``rag_video_node`` both retrieve from the same
knowledge base, so they import ``retriever`` from this module to share a
single underlying LlamaIndex index. Previously each node called
``get_llama_retriever`` independently, which built (and, prior to the
ingestion refactor, re-ingested) a separate index per node.

The vector store is expected to already be populated. Run
``python ingest.py`` once before starting the app.
"""
from embedding_model import get_embedding
from llama_retriever import get_llama_retriever
from vector_store import get_vector_store

_embed_model = get_embedding("dashscope")
_vector_store = get_vector_store("milvus")

retriever = get_llama_retriever(
    vector_store=_vector_store, embed_model=_embed_model
)
