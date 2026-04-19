"""RAG video node: retrieves the best-matching video URL for the query.

Runs after ``rag_node`` in the workflow graph. Looks up the ``url``
metadata key on the first retrieved document (set during ingestion in
``ingest.py``).
"""
from workflow.retrieval import retriever
from workflow.state import State


def rag_video_node(state: State) -> State:
    """Retrieve documents and extract the top video URL, if any."""
    results = retriever.invoke(state["messages"][-1].content)

    # The "url" metadata key is guaranteed by the ingestion pipeline
    # (see SEED_DOCUMENTS in ingest.py).
    video_url = results[0].metadata["url"] if results else None
    return {**state, "video_url": video_url, "result": "Success"}

