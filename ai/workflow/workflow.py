"""LangGraph workflow definition.

Builds and compiles the conversational agent graph. The checkpointer
backend is selected at startup via the ``POSTGRES_URL`` env var
(Postgres when set, local sqlite otherwise).
"""
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

load_dotenv(override=True)


def _build_checkpointer():
    """Return a LangGraph checkpointer.

    Uses Postgres when ``POSTGRES_URL`` is set (production), otherwise a
    local sqlite file so the workflow can run without remote infra.
    """
    import os

    postgres_url = os.getenv("POSTGRES_URL")
    if postgres_url:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection

        conn = Connection.connect(
            postgres_url, autocommit=True, prepare_threshold=0
        )
        saver = PostgresSaver(conn)
        saver.setup()
        return saver

    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect("memory.db", check_same_thread=False)
    return SqliteSaver(conn)


checkpointer = _build_checkpointer()


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

    # Save graph diagram to file. Rendering uses a remote mermaid service,
    # so failures here must not block startup.
    try:
        with open("graph.png", "wb") as f:
            f.write(workflow.get_graph().draw_mermaid_png())
    except Exception as e:
        print(f"[workflow] skipped graph render: {e}")

    return workflow

