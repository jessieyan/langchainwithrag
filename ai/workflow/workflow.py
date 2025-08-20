from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from workflow.router_node import router_node
from workflow.rag_node import rag_node
from workflow.rag_video_node import rag_video_node
from workflow.ask_without_context_node import ask_without_context_node
from workflow.generate_image_node import generate_image_node
from workflow.multimodal_node import multimodal_node
from workflow.fallback_node import fallback_node
from workflow.track_tool_node import track_tool_node
from workflow.understand_image_node import understand_image_node
from workflow.state import State
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from IPython.display import Image, display
from psycopg import Connection
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# uncommentfor local test
# db_path = "memory.db"
# conn = sqlite3.connect(db_path, check_same_thread=False)
# sql_memory = SqliteSaver(conn)

load_dotenv(override=True)

endpoint = "pc-wz9nv355xw5ed6129.pg.polardb.rds.aliyuncs.com:5432"
username = "ai_test"
password = "Testtest123"
database = "postgres_test"

🟢 Compose connection string
postgres_url = f"postgresql://{username}:{password}@{endpoint}:5432/{database}"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

conn = Connection.connect(postgres_url, **connection_kwargs)
checkpointer = PostgresSaver(conn)
checkpointer.setup()


# [input] → [router_node] ─┬──► [rag_node] ──► [rag_video_node] ──► [track_tool_node] ─► END
#                          ├──► [ask_without_context_node] ──► [track_tool_node] ─► END
#                          ├──► [generate_image_node] ─► [track_tool_node] ─► END
#                          ├──► [understand_image_node] ─► [track_tool_node] ─► END
#                          ├──► [multimodal_node] ─► [track_tool_node] ─► END
#                          └──► [fallback_node]


def router_condition(old_state: State) -> str:
    route = old_state['route']
    return route


def generate_workflow(): 

    # Set up Graph Builder with State
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("router", RunnableLambda(router_node))
    graph.add_node("rag_tool", RunnableLambda(rag_node))
    graph.add_node("rag_video_tool", RunnableLambda(rag_video_node))
    graph.add_node("ask_without_context", RunnableLambda(ask_without_context_node))
    graph.add_node("generate_image", RunnableLambda(generate_image_node))
    graph.add_node("understand_image", RunnableLambda(understand_image_node))
    graph.add_node("multimodal", RunnableLambda(multimodal_node))
    graph.add_node("fallback", RunnableLambda(fallback_node))
    graph.add_node("track_tool", RunnableLambda(track_tool_node))

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", router_condition, {
        "generate_image": "generate_image",
        "understand_image": "understand_image",
        "ask_without_context": "ask_without_context",
        "rag_tool": "rag_tool",
        "multimodal": "multimodal",
        "fallback": "fallback"
    })
    
    # Tool → tracking → END
    graph.add_edge("rag_tool", "rag_video_tool")
    for node in ["generate_image", "understand_image", "ask_without_context","rag_video_tool", "multimodal", "fallback"]:
        graph.add_edge(node, "track_tool")
        graph.add_edge("track_tool", END)

    # Compile the graph
    workflow = graph.compile(checkpointer=checkpointer)
    # workflow = graph.compile(checkpointer=sql_memory)

    # Display as image (if using Jupyter or IPython)
    display(Image(workflow.get_graph().draw_mermaid_png()))

    # Save image to file
    with open("graph.png", "wb") as f:
     f.write(workflow.get_graph().draw_mermaid_png())   

    return workflow

