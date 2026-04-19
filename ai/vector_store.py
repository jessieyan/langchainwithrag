"""Vector store factory.

``overwrite`` defaults to False so the query path (workflow startup)
reuses the index that ``ingest.py`` already populated instead of wiping
it. Pass ``overwrite=True`` from the ingestion script when rebuilding.
"""
from llama_index.vector_stores.milvus import MilvusVectorStore


def get_vector_store(name: str, *, overwrite: bool = False):
    if name == "milvus":
        return MilvusVectorStore(
            uri="./milvus_demo.db", dim=1536, overwrite=overwrite
        )
    if name == "chroma":
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection("quickstart")
        return ChromaVectorStore(chroma_collection=chroma_collection)
    raise ValueError(f"unknown vector store: {name}")
