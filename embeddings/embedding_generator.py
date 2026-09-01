"""
embeddings/embedding_generator.py

Milestone 4 — Embeddings & Semantic Search.

Loads a local Sentence-BERT model and turns each product's structured +
free-text fields into a single dense vector that captures its meaning,
not just its keywords. Owns exactly two jobs: (1) build one "embedding
text" string per product row, (2) encode text into vectors using the
same model at index time and query time, so both live in the same
vector space. No CSV or ChromaDB I/O happens here — that's
vector_search's job.
"""

import logging
from typing import List, Union

import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model choice: all-MiniLM-L6-v2
#   - 22M params, 384-dim output, ~90MB download, CPU-friendly
#   - Trained on 1B+ sentence pairs, general-purpose semantic similarity
#   - Still the standard lightweight local default for sentence-transformers
#     workflows (BAAI's bge-small-en-v1.5 edges it out slightly on
#     retrieval-specific MTEB scores at a similar size, but needs
#     query-prefixing to get that edge — extra complexity not worth it
#     for this MVP; revisit if retrieval quality becomes the bottleneck)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Fields combined into one embedding text per product, in this fixed,
# documented order. Order matters only a little (SBERT pools the whole
# sequence) but is kept fixed so embeddings are reproducible run-to-run.
EMBEDDING_TEXT_FIELDS = ["product_name", "category", "subcategory", "material", "description"]

_model_cache = None


def get_embedding_model() -> SentenceTransformer:
    """Type: Transformer. Load (once) and cache the local SBERT model."""
    global _model_cache
    if _model_cache is None:
        logger.info(f"Loading Sentence-BERT model: {EMBEDDING_MODEL_NAME}")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_cache


def build_embedding_text(row: pd.Series) -> str:
    """
    Type: Rule-based. Deterministic string construction, no model involved.

    Combines EMBEDDING_TEXT_FIELDS into one string per product. Missing/NaN
    fields are skipped rather than inserted as the literal string "nan"
    (a real bug class already hit once in this project with eco_certification).
    """
    parts = []
    for field in EMBEDDING_TEXT_FIELDS:
        value = row.get(field)
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    return ". ".join(parts)


def generate_embeddings(texts: Union[str, List[str]]):
    """
    Type: Transformer.
    Encode one string or a list of strings into SBERT embedding vectors.
    The same function handles product rows (batch, at index time) and a
    live user query (single string, at search time) so both land in the
    same vector space.
    """
    model = get_embedding_model()
    single_input = isinstance(texts, str)
    inputs = [texts] if single_input else texts
    embeddings = model.encode(
        inputs,
        show_progress_bar=not single_input and len(inputs) > 20,
        convert_to_numpy=True,
    )
    return embeddings[0] if single_input else embeddings
