from typing import List, Any
from langchain.schema import BaseRetriever, Document
from pydantic import Field

class LlamaIndexLangChainRetriever(BaseRetriever):
    llamaindex_retriever: Any = Field(exclude=True)

    def get_relevant_documents(self, query: str) -> List[Document]:
        nodes = self.llamaindex_retriever.retrieve(query)
        return [
            Document(page_content=node.text, metadata=node.metadata or {})
            for node in nodes
        ]
