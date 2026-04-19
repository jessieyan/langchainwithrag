"""One-time ingestion of the knowledge base into the vector store.

Run this whenever the knowledge base changes. Subsequent application
startups should NOT re-run ingestion -- they should wrap the already
populated vector store via ``get_llama_retriever``. Re-embedding on
every startup is slow and costs money against the hosted embedding API.

Usage::

    python ingest.py                     # default: knowledge_base/data
    python ingest.py --dir path/to/docs  # custom directory
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)

from embedding_model import get_embedding
from vector_store import get_vector_store

DEFAULT_KNOWLEDGE_BASE = "knowledge_base/data"

# Hand-curated seed documents. These previously lived inline in
# ``get_llama_retriever`` and were re-created on every import. They now
# live here so ingestion is the single source of truth for what ends up
# in the index.
SEED_DOCUMENTS: List[Document] = [
    Document(
        text="A cozy treehouse video tour with kids playing around the yard",
        metadata={"url": "https://example.com/video1.mp4", "content_type": "video"},
    ),
    Document(
        text=(
            "【拍嗝技巧】超實用~2種幫寶寶拍嗝的方法&要點！"
            "新手爸媽一看就懂｜周彥怡醫師｜禾馨怡仁婦幼中心"
        ),
        metadata={
            "url": "https://www.youtube.com/watch?v=FpMi2ee-qvs",
            "content_type": "video",
        },
    ),
    Document(
        text="A documentary about marine life in the Pacific Ocean",
        metadata={"url": "https://example.com/video2.mp4", "content_type": "video"},
    ),
]


def ingest(knowledge_base_dir: str = DEFAULT_KNOWLEDGE_BASE) -> int:
    """Embed all documents under ``knowledge_base_dir`` plus seed docs.

    Returns the number of documents written to the vector store.
    """
    if not os.path.isdir(knowledge_base_dir):
        raise FileNotFoundError(
            f"knowledge base dir not found: {knowledge_base_dir}"
        )

    embed_model = get_embedding("dashscope")
    vector_store = get_vector_store("milvus", overwrite=True)

    documents = SimpleDirectoryReader(knowledge_base_dir).load_data()
    documents.extend(SEED_DOCUMENTS)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print(f"Ingested {len(documents)} documents into the vector store.")
    return len(documents)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=DEFAULT_KNOWLEDGE_BASE,
        help="knowledge base directory to ingest",
    )
    args = parser.parse_args(argv)
    ingest(args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
