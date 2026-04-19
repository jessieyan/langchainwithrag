"""Focused tests for LlamaIndexLangChainRetriever metadata propagation.

Scope: only the LlamaIndex -> LangChain bridge. We fake the upstream
LlamaIndex retriever with a trivial stub so the tests are hermetic (no
vector store, no embeddings, no network).

Failure modes covered (each is a separate test):

1.  Happy path: text and caller-provided metadata round-trip through the
    bridge unchanged, and `_retriever` is added.
2.  Sparse node: ``node.metadata is None`` -- the bridge must not crash
    and must still tag provenance.
3.  Empty metadata dict: ``node.metadata == {}`` -- same as above, but
    distinguishes "None" vs "empty dict" input handling.
4.  Ingestion key preservation: a key that downstream code depends on
    (``url``, consumed by ``rag_video_node``) is passed through
    byte-for-byte. Regression guard for the stated non-breaking contract.
5.  Collision on a system key: if upstream metadata already contains
    ``_retriever`` / ``_score`` / ``_node_id``, the bridge must NOT
    overwrite the caller's value. This protects callers that deliberately
    pre-tag nodes (e.g. a re-ranker upstream).
6.  Score attribute missing entirely: ``_score`` must be absent from the
    resulting metadata (not present as ``None``), so consumers that do
    ``"_score" in metadata`` get a correct answer.
7.  Score is ``None``: same guarantee -- ``None`` means "no score", not
    "score of None".
8.  Score is ``0`` or ``0.0``: must be preserved. Regression guard
    against a truthiness bug (``if score:`` would drop a legitimate
    zero score).
9.  Node id from ``id_`` only: some LlamaIndex versions expose ``id_``
    but not ``node_id``. Bridge must fall back.
10. Node id from ``node_id``: preferred attribute when present.
11. Per-node isolation: when multiple nodes are returned, each Document
    gets its own metadata dict. Mutating one must not affect another --
    this catches accidental shared-reference bugs.
12. Empty result set: upstream returns ``[]`` -- bridge returns ``[]``
    without error.
13. ``retriever_name`` constructor arg propagates to every Document, so
    ensemble consumers can distinguish which bridge produced each chunk.

Intentionally NOT covered here:
-   LangChain ``Document`` construction semantics (upstream contract).
-   LlamaIndex node construction (upstream contract).
-   ``EnsembleRetriever`` merge behavior -- that's a separate component
    and preserving metadata through it is LangChain's responsibility.
-   ``LlamaIndexRetrieverWrapper`` -- known to drop metadata entirely;
    out of scope for this change.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List

# Make `ai/` importable regardless of where pytest is invoked from.
_AI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai"))
if _AI_DIR not in sys.path:
    sys.path.insert(0, _AI_DIR)

from llamaindex_langchain_retriever import LlamaIndexLangChainRetriever  # noqa: E402


class _FakeUpstream:
    """Stands in for a LlamaIndex retriever. Returns a fixed node list."""

    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve(self, query: str):  # signature matches what the bridge calls
        return self._nodes


def _make_node(text="chunk", metadata=None, score=..., node_id=..., id_=...):
    """Build a fake node. Use ``...`` (Ellipsis) to mean 'attribute absent'."""
    ns = SimpleNamespace(text=text, metadata=metadata)
    if score is not ...:
        ns.score = score
    if node_id is not ...:
        ns.node_id = node_id
    if id_ is not ...:
        ns.id_ = id_
    return ns


def _bridge(nodes) -> List:
    retriever = LlamaIndexLangChainRetriever(llamaindex_retriever=_FakeUpstream(nodes))
    return retriever.invoke("q")


# --- 1. happy path ----------------------------------------------------------

def test_happy_path_preserves_text_and_metadata_and_tags_retriever():
    node = _make_node(text="hello", metadata={"url": "https://x", "content_type": "video"})
    [doc] = _bridge([node])
    assert doc.page_content == "hello"
    assert doc.metadata["url"] == "https://x"
    assert doc.metadata["content_type"] == "video"
    assert doc.metadata["_retriever"] == "llamaindex_vector"


# --- 2. sparse: metadata is None -------------------------------------------

def test_none_metadata_does_not_crash_and_tags_retriever():
    node = _make_node(metadata=None)
    [doc] = _bridge([node])
    assert doc.metadata["_retriever"] == "llamaindex_vector"
    # No spurious keys leaked in from nowhere.
    assert set(doc.metadata.keys()) == {"_retriever"}


# --- 3. sparse: metadata is empty dict -------------------------------------

def test_empty_metadata_dict_is_tagged():
    node = _make_node(metadata={})
    [doc] = _bridge([node])
    assert doc.metadata == {"_retriever": "llamaindex_vector"}


# --- 4. ingestion key passthrough (regression guard for rag_video_node) ----

def test_ingestion_url_key_passes_through_unchanged():
    # rag_video_node reads results[0].metadata["url"] directly.
    node = _make_node(metadata={"url": "https://youtu.be/abc"})
    [doc] = _bridge([node])
    assert doc.metadata["url"] == "https://youtu.be/abc"


# --- 5. collision: caller pre-tagged a system key --------------------------

def test_does_not_overwrite_caller_provided_system_keys():
    node = _make_node(
        metadata={"_retriever": "reranker", "_score": 0.99, "_node_id": "pre"},
        score=0.123,
        node_id="upstream",
    )
    [doc] = _bridge([node])
    assert doc.metadata["_retriever"] == "reranker"
    assert doc.metadata["_score"] == 0.99
    assert doc.metadata["_node_id"] == "pre"


# --- 6. score attribute entirely absent ------------------------------------

def test_missing_score_attribute_is_not_present_in_metadata():
    node = _make_node(metadata={})  # no score attribute at all
    [doc] = _bridge([node])
    assert "_score" not in doc.metadata


# --- 7. score is None ------------------------------------------------------

def test_none_score_is_not_present_in_metadata():
    node = _make_node(metadata={}, score=None)
    [doc] = _bridge([node])
    assert "_score" not in doc.metadata


# --- 8. score is 0 (truthiness regression guard) ---------------------------

def test_zero_score_is_preserved():
    node = _make_node(metadata={}, score=0.0)
    [doc] = _bridge([node])
    assert doc.metadata["_score"] == 0.0


# --- 9. node id fallback from id_ -----------------------------------------

def test_node_id_falls_back_to_id_attribute():
    node = _make_node(metadata={}, id_="abc-123")  # no node_id
    [doc] = _bridge([node])
    assert doc.metadata["_node_id"] == "abc-123"


# --- 10. node id from node_id ----------------------------------------------

def test_node_id_uses_node_id_attribute_when_present():
    node = _make_node(metadata={}, node_id="primary-xyz")
    [doc] = _bridge([node])
    assert doc.metadata["_node_id"] == "primary-xyz"


# --- 11. per-node metadata isolation ---------------------------------------

def test_each_document_gets_independent_metadata_dict():
    n1 = _make_node(text="a", metadata={"url": "u1"})
    n2 = _make_node(text="b", metadata={"url": "u2"})
    docs = _bridge([n1, n2])
    assert docs[0].metadata is not docs[1].metadata
    docs[0].metadata["poisoned"] = True
    assert "poisoned" not in docs[1].metadata


# --- 12. empty result set --------------------------------------------------

def test_empty_result_list():
    assert _bridge([]) == []


# --- 13. custom retriever_name propagates ----------------------------------

def test_custom_retriever_name_is_applied():
    retriever = LlamaIndexLangChainRetriever(
        llamaindex_retriever=_FakeUpstream([_make_node(metadata={})]),
        retriever_name="custom_name",
    )
    [doc] = retriever.invoke("q")
    assert doc.metadata["_retriever"] == "custom_name"


if __name__ == "__main__":
    # Allow running without pytest: `python test/test_...py`.
    # Collects test_* functions in this module and runs them.
    import traceback

    failed = 0
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok   {name}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
