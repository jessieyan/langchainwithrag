from workflow.state import State
from llm.deepseek import deepseek_llm

def ask_without_context_node(state: State) -> State:
    """Use this for standalone questions without any memory."""
    llm = deepseek_llm()
    response = llm.invoke(state["messages"])
    return {**state, "messages": response}