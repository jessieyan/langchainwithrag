from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

# Bridge: LlamaIndex retriever -> LangChain BaseRetriever.
#
# Contract preserved (clients depend on this):
#   - Document.page_content == node.text
#   - Document.metadata contains all ingestion-time keys from node.metadata
#     (e.g. "url", "content_type", "file_path"). rag_video_node reads
#     metadata["url"] directly.
#
# Additive provenance (underscore-prefixed to signal debug/system fields and
# avoid collisions with ingestion keys):
#   - _retriever : name of the retriever that produced this chunk
#   - _score     : similarity score from the underlying retriever, if any
#   - _node_id   : LlamaIndex node id, for tracing back to the vector store


class LlamaIndexLangChainRetriever(BaseRetriever):
    llamaindex_retriever: Any = Field(exclude=True)
    retriever_name: Optional[str] = Field(default="llamaindex_vector")

    def _get_relevant_documents(self, query: str, **_: Any) -> List[Document]:
        nodes = self.llamaindex_retriever.retrieve(query)
        documents: List[Document] = []
        for node in nodes:
            metadata = dict(node.metadata or {})
            # Do not overwrite ingestion-side keys if they happen to collide.
            metadata.setdefault("_retriever", self.retriever_name)
            score = getattr(node, "score", None)
            if score is not None:
                metadata.setdefault("_score", score)
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
            if node_id is not None:
                metadata.setdefault("_node_id", node_id)
            documents.append(Document(page_content=node.text, metadata=metadata))
        return documents
