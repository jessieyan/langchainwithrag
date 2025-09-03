from langchain.memory.chat_message_histories import RedisChatMessageHistory
from pymongo import MongoClient
from workflow.state import State

def persist_node(state: State) -> State:
    redis_history = RedisChatMessageHistory(
        session_id=state["session_id"],
        url="redis://localhost:6379"
    )
    messages = redis_history.messages
    mongo = MongoClient("mongodb://localhost:27017")
    mongo["chat_logs"]["conversations"].insert_one({
        "session_id": state["session_id"],
        "messages": [{"type": m.type, "content": m.content} for m in messages]
    })
    return state
