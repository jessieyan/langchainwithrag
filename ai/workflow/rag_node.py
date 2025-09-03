from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from llama_retriever import get_llama_retriever
from embedding_model import get_embedding
from vector_store import get_vector_store
from llm.deepseek import deepseek_llm
from workflow.state import State

knowledge_base = "knowledge_base/data"

# Create embeddings
embed_model = get_embedding('dashscope')

# Create vector store
vector_store = get_vector_store('milvus')

# Create llama index retriever
retriever = get_llama_retriever(vector_store=vector_store, embed_model=embed_model, dir=knowledge_base)

template_text = """
You are a helpful assistant.

Please answer the user's question using the following combined context.

### Detailed Context (from RAG chunks)
{context}

Related Video
{video_url}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "video_url", "question"],
    template=template_text,
)

# RAG prompt
rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}
                                          
Related Video
{video_url}

Question: {question}
""")

rag_chain = RetrievalQA.from_chain_type(
    llm=deepseek_llm(),
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt}
)

def rag_node(state: State) -> State:
    response = rag_chain.run(state["messages"][-1].content)
    return {**state, "messages": response}
