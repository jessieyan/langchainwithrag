from workflow.state import State
from llm.wanx import chat_wanx

def generate_image_node(state: State) -> State:
    response = chat_wanx(state["messages"][-1].content)
    image_url = response["output"]["results"][0]["url"]
    return {**state, "image_url": image_url, "result": "Success", "messages": [{"role": "assistant", "content":  image_url}],}