from workflow.retrieval import retriever
from workflow.state import State


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

