from typing import TypedDict, List, Optional
from langchain.schema import BaseMessage
from langchain.memory.chat_message_histories import InMemoryChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import InMemoryStore, InMemorySaver
from wanx import chat_wanx
from qwen import chat_qwen
from embedding_model import get_embedding
from vector_store import get_vector_store
from llama_retriever import get_llama_retriever
from custom_yomi_tool_agent import create_react_yomi_agent
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import VectorStoreRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-6e5c573c75044deb80371b793a8b24a3"
knowledge_base = "./knowledge_base/data"
store = InMemoryStore()   # Stores memory by session key
saver = InMemorySaver(store=store)


class GraphState(TypedDict):
    input: str
    output: Optional[str]
    tool_used: Optional[str]
    error: Optional[str]
    session_id: str

# RAG
embed_model = get_embedding('dashscope')
vector_store = get_vector_store('milvus')
vector_retriever = get_llama_retriever(vector_store=vector_store, embed_mode=embed_model, dir=knowledge_base)

@tool
def ask_without_context(input: str) -> str:
    """Use this for standalone questions without any memory."""
    return f"[Stateless] You asked: {input}"

@tool
def use_wanx(input: str) -> str:
    """Use this for generating pictures."""
    return chat_wanx(input)

@tool
def use_qwen(input: str) -> str:
    """Use this for understanding pictures and images."""
    return chat_qwen(input)

@tool
def rag_tool(input: str, session_id: str) -> str:
    # Pull past messages
    history = store.get(session_id) or []
    past_context = "\n".join([m.content for m in history[-4:]])  # last 4 messages
    docs = vector_retriever.get_relevant_documents(input)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Use the context below and chat history to answer the question.

    Chat History:
    {past_context}

    Docs:
    {context}

    Question: {input}
    """
    return prompt

@tool(response_format="content_and_artifact")
def retrieve_doc(query: str):
    """Retrieve docs related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [ask_without_context, use_wanx, use_qwen, rag_tool, retrieve_doc]
tool_map = {t.name: t for t in tools}

router_llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-r1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],)

agent_executor = create_react_yomi_agent(router_llm, tools=tools)

def router_node(state: GraphState) -> GraphState:
    history = store.get(state["session_id"]) or []
    chat_history = InMemoryChatMessageHistory(messages=history)

    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True,
        memory_key="chat_history"
    )

    agent_executor = create_react_yomi_agent(router_llm, tools=tools, memory=memory)

    result = agent_executor.invoke({"input": state["input"]})
    return {
        **state,
        "tool_used": result.get("tool", None),
        "output": None,
        "error": None,
    }

# tool runner node
def tool_runner_node(state: GraphState) -> GraphState:
    try:
        tool = tool_map.get(state["tool_used"])
        if not tool:
            return {**state, "error": "Unknown tool selected"}
        result = tool.invoke(state["input"])
        return {**state, "output": result, "error": None}
    except Exception as e:
        return {**state, "error": str(e), "output": None}

# memory node
def memory_node(state: GraphState) -> GraphState:
    # Save this turn to memory
    saver.add(
        session_id=state["session_id"],
        messages=[
            HumanMessage(content=state["input"]),
            AIMessage(content=state["output"])
        ]
    )
    return state


# fall back node
def fallback_node(state: GraphState) -> GraphState:
    return {
        **state,
        "output": "Sorry, something went wrong. Please try again.",
        "tool_used": "fallback"
    }

graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("tool_runner", tool_runner_node)
graph.add_node("memory", memory_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("router")
graph.add_edge("router", "tool_runner")
graph.add_conditional_edges("tool_runner", lambda s: "fallback" if s["error"] else "memory")
graph.add_edge("memory", END)
graph.add_edge("fallback", END)

app = graph.compile()



