from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    input: str
    result: Optional[str]
    session_id: str
    route: Optional[str]
    tool_usage: List[dict]
    image_url: Optional[str]
    video_url: Optional[str]