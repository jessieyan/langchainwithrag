from workflow.state import State

def fallback_node(state: State) -> State:
    return {
        **state,
        "result": "Sorry, I didn't understand your request. Can you rephrase?",
        "route": "fallback"
    }
