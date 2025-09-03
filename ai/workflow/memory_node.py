from langchain.chat_models import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

# Redis-backed memory setup
memory_llm = RunnableWithMessageHistory(
    ChatOpenAI(model="gpt-4"),
    lambda session_id: RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379"  # your Redis URI
    ),
    input_messages_key="input",
    history_messages_key="messages"
)

# Node definition
def memory_node(state: dict) -> dict:
    # use tool output or RAG as input to memory-backed LLM
    input_text = state.get("result", state["input"])  # fallback to raw input
    session_id = state["session_id"]

    # Run the memory LLM
    response = memory_llm.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )

    # Return updated state
    return {
        "messages": state.get("messages", []) + [response],
        "result": response.content
    }
