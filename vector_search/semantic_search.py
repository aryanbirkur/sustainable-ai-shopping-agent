"""
vector_search/semantic_search.py

Milestone 4 — Embeddings & Semantic Search.

The single public entry point later milestones (recommendation blending,
FastAPI) should import: given free text, return the top-k most
semantically similar products with their similarity scores.
"""

import logging
from pathlib import Path

from config.settings import CHROMA_PERSIST_DIR
from embeddings.embedding_generator import generate_embeddings
from vector_search.vector_store import get_chroma_client, get_products_collection, query_similar

logger = logging.getLogger(__name__)


def semantic_search(query: str, top_k: int = 5, persist_dir: Path = CHROMA_PERSIST_DIR):
    """
    Type: Transformer + Rule-based.
    Embed `query` with the same SBERT model used at index time, then
    return the top_k nearest products from the ChromaDB 'products'
    collection, each with a similarity score the caller can inspect.
    """
    query_embedding = generate_embeddings(query)
    client = get_chroma_client(persist_dir)
    collection = get_products_collection(client)
    results = query_similar(collection, query_embedding, top_k=top_k)
    logger.info(f"Query: '{query}' -> {len(results)} results")
    return results
