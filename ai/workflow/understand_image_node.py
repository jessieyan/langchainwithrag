from workflow.state import State
from llm.qwen import chat_qwen

def understand_image_node(state: State) -> State:
    """Use this for standalone questions without any memory."""
    response = chat_qwen(state["messages"][-1].content, state["image_url"])
    return {**state, "result": "Success", "messages": [{"role": "assistant", "content":  response}]}