from langchain.prompts import PromptTemplate
from llama_retriever import get_llama_retriever
from embedding_model import get_embedding
from vector_store import get_vector_store
from workflow.state import State

knowledge_base = "knowledge_base/data"

# Create embeddings
embed_model = get_embedding('dashscope')

# Create vector store
vector_store = get_vector_store('milvus')

# Create llama index retriever
retriever = get_llama_retriever(vector_store=vector_store, embed_model=embed_model, dir=knowledge_base)


# RAG prompt
rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {question}
""")

def rag_video_node(state: State) -> State:
    results = retriever.get_relevant_documents(state["messages"][-1].content)

    if results:
        video_url = results[0].metadata["url"]
    else:
        video_url = None
    print(f"video_url== {video_url}")

    print(f"messages== {state["messages"]}")
    # return {**state, "video_url": video_url, "result": "Success", "messages": [{"role": "assistant", "content":  f"Here is the video we found: {video_url}"}],}
    return {**state, "video_url": video_url, "result": "Success"}

