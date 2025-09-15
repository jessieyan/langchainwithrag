from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def get_vector_store(name : str) -> any:
    if name == "milvus":
        vector_store = MilvusVectorStore(
            uri="./milvus_demo.db", dim=1536, overwrite=True)
    elif name == "chroma":
        # create client and a new collection
        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store