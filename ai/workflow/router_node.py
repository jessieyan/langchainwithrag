from workflow.state import State
from llm.deepseek import deepseek_llm
import json

router_llm = deepseek_llm()

def router_node(state: State) -> State:
    prompt = (
    f"You are a task router. Analyze the user's message:\n\n"
    f"\"{state["messages"]}\"\n\n"
    f"Decide which category it falls into:\n"
    f"- ask_without_context: This task involves direct interaction using the model's inherent knowledge and conversational abilities, without requiring external tools or context.\n"
    f"- rag_tool: use contexts if there are questions related to early childhood development\n"
    f"- generate_image: generate image\n"
    f"- understand_image: understand image\n"
    f"- multimodal: input has video or audio\n"
    f"- fallback: if something is wrong\n\n"
    f"Return JSON: {{\"route\": <route>, \"reason\": <why you chose it>, \"image_url\": <if user message has imageUrl>}}"
)
    response = router_llm.invoke(prompt)
    try:
        parsed = json.loads(response.content)
        route = parsed.get("route", "fallback").lower()
        reason = parsed.get("reason", "")
        image_url = parsed.get("image_url")
    except json.JSONDecodeError:
        route = "fallback"
        reason = "Could not parse LLM response."
        image_url = "No image"
    
    return {
        **state,
        "route": route if route in {"ask_without_context", "rag_tool", "rag_video_tool", "generate_image", "understand_image", "multimodal"} else "fallback",
        "route_reason": reason,
        "image_url" : image_url
    }
