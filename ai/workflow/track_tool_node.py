import datetime
from workflow.state import State

def track_tool_node(state: State) -> State:
    usage = state.get("tool_usage", [])
    usage.append({
        "tool": state.get("route", "unknown"),
        "timestamp": datetime.datetime.now().isoformat()
    })
    return {**state, "tool_usage": usage, "result": "Success"}