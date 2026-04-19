"""RAG node: retrieves relevant chunks and generates a grounded answer.

Uses an LCEL chain (retriever | prompt | llm | parser) instead of the
legacy ``RetrievalQA`` which was removed in LangChain 1.x.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm.deepseek import deepseek_llm
from workflow.retrieval import retriever
from workflow.state import State

rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {question}
""")


def _format_docs(docs):
    """Join retrieved Document.page_content into a single context string."""
    return "\n\n".join(d.page_content for d in docs)


# LCEL chain: user question -> retrieve docs -> format -> prompt -> LLM -> text
rag_chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | deepseek_llm()
    | StrOutputParser()
)


def rag_node(state: State) -> State:
    """Workflow node: run RAG chain on the latest user message."""
    response = rag_chain.invoke(state["messages"][-1].content)
    return {**state, "messages": response}
