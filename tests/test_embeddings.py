"""
tests/test_embeddings.py

Milestone 4 tests: embedding generation + vector store correctness.
Run with: pytest tests/test_embeddings.py -v

Each test gets a unique on-disk ChromaDB path via pytest's built-in
tmp_path fixture. This matters: ChromaDB's PersistentClient caches its
internal System object in-process, keyed by path. Reusing one fixed
path across tests (delete + recreate between tests) hands later tests
a stale cached client pointing at file handles for a database that no
longer exists -> "attempt to write a readonly database". Unique paths
per test avoid the collision entirely.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedding_generator import (
    EMBEDDING_DIMENSION,
    build_embedding_text,
    generate_embeddings,
)
from vector_search.vector_store import (
    get_chroma_client,
    get_products_collection,
    query_similar,
    upsert_products,
)

SAMPLE_PRODUCTS = pd.DataFrame([
    {
        "product_id": "P001", "product_name": "Recycled Cotton Tote Bag",
        "category": "Bags", "subcategory": "Tote Bags", "material": "Organic Cotton",
        "description": "A spacious everyday tote made from 100% organic cotton, "
                        "perfect for grocery runs and light daily carrying.",
        "price": 799, "sustainability_score": 0.82,
    },
    {
        "product_id": "P002", "product_name": "Stainless Steel Water Bottle",
        "category": "Drinkware", "subcategory": "Water Bottles", "material": "Stainless Steel",
        "description": "A durable, insulated steel bottle that keeps drinks cold "
                        "for 24 hours, designed to replace single-use plastic.",
        "price": 899, "sustainability_score": 0.74,
    },
    {
        "product_id": "P003", "product_name": "Gaming Mechanical Keyboard",
        "category": "Electronics", "subcategory": "Keyboards", "material": "ABS Plastic",
        "description": "RGB backlit mechanical keyboard with hot-swappable "
                        "switches for competitive gaming.",
        "price": 4999, "sustainability_score": 0.21,
    },
])


@pytest.fixture
def collection(tmp_path):
    """
    A fresh ChromaDB collection at a unique path per test. Do NOT reuse
    one fixed path with manual delete/recreate between tests -- see the
    module docstring for why that breaks.
    """
    client = get_chroma_client(tmp_path / "chroma_store")
    return get_products_collection(client)


def _index_sample(collection):
    texts = SAMPLE_PRODUCTS.apply(build_embedding_text, axis=1).tolist()
    embeddings = generate_embeddings(texts)
    metadatas = SAMPLE_PRODUCTS[["product_id", "category", "price", "sustainability_score"]].to_dict("records")
    upsert_products(collection, SAMPLE_PRODUCTS["product_id"].tolist(), embeddings, metadatas, documents=texts)
    return texts, embeddings


def test_embedding_dimension():
    """Embeddings must have the fixed dimensionality the model promises."""
    vec = generate_embeddings("a sustainable cotton tote bag")
    assert vec.shape == (EMBEDDING_DIMENSION,)


def test_batch_embedding_shape():
    texts = SAMPLE_PRODUCTS.apply(build_embedding_text, axis=1).tolist()
    vecs = generate_embeddings(texts)
    assert vecs.shape == (len(texts), EMBEDDING_DIMENSION)


def test_embedding_text_skips_missing_fields():
    row = pd.Series({
        "product_name": "Tote Bag", "category": "Bags",
        "subcategory": None, "material": "", "description": "A bag.",
    })
    text = build_embedding_text(row)
    assert "nan" not in text.lower()
    assert "Tote Bag" in text and "A bag." in text


def test_semantically_similar_query_ranks_target_product_first(collection):
    """A query close in meaning to P001's description should return P001 near the top."""
    _index_sample(collection)
    query_vec = generate_embeddings("an organic cotton bag for carrying groceries")
    results = query_similar(collection, query_vec, top_k=3)
    assert results[0]["product_id"] == "P001"
    assert results[0]["similarity"] > 0.3


def test_irrelevant_query_returns_low_similarity(collection):
    _index_sample(collection)
    query_vec = generate_embeddings("a birthday cake recipe with chocolate frosting")
    results = query_similar(collection, query_vec, top_k=3)
    assert all(r["similarity"] < 0.5 for r in results)


def test_batch_upsert_is_idempotent(collection):
    """Running the batch upsert twice must not duplicate entries."""
    _index_sample(collection)
    _index_sample(collection)  # re-run
    assert collection.count() == len(SAMPLE_PRODUCTS)


def test_metadata_round_trips(collection):
    _index_sample(collection)
    query_vec = generate_embeddings("steel bottle")
    results = query_similar(collection, query_vec, top_k=1)
    assert results[0]["metadata"]["category"] == "Drinkware"
    assert results[0]["metadata"]["price"] == 899
