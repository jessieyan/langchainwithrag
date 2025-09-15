from workflow.workflow import generate_workflow

workflow = generate_workflow()

def generate_response(user_input : str, role_prompt : str, user_id : str):
    config = {"configurable": {"thread_id": user_id}}
    result = workflow.invoke({"messages": [{"role": "user", "content": user_input}, {"role": "system", "content": role_prompt}]}, config=config)
    return result["messages"][-1].content