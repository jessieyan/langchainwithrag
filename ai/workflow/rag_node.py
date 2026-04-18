from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from llm.deepseek import deepseek_llm
from workflow.retrieval import retriever
from workflow.state import State

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
