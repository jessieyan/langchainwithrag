from langchain.schema import BaseRetriever

class LlamaIndexRetrieverWrapper(BaseRetriever):
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def get_relevant_documents(self, query):
        response = self.query_engine.query(query)
        # response might be an object; convert to list of langchain Documents
        # Here we just wrap the answer text into a Document
        from langchain.schema import Document
        return [Document(page_content=str(response))]
