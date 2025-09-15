from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader

def get_keyword_retriever(dir):
    loader = DirectoryLoader(dir, glob="**/*.*")
    docs = loader.load()
    keyword_retriever = BM25Retriever.from_documents(docs)
    return keyword_retriever