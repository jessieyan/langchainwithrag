from langchain.retrievers import EnsembleRetriever
    
def get_hybrid_retriever(retrievers, weights):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[*retrievers], weights=[*weights])
    return ensemble_retriever
