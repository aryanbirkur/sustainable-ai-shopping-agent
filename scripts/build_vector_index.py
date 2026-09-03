"""
scripts/build_vector_index.py

Milestone 4 — Embeddings & Semantic Search.

Reads data/processed/products_scored.csv, builds one embedding per
product, and upserts them into the local ChromaDB 'products' collection.
Safe to re-run: upsert overwrites existing IDs instead of duplicating.

Run:
    python scripts/build_vector_index.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PRODUCTS_SCORED_PATH, CHROMA_PERSIST_DIR
from embeddings.embedding_generator import build_embedding_text, generate_embeddings
from vector_search.vector_store import get_chroma_client, get_products_collection, upsert_products

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_METADATA_FIELDS = ["product_id", "category", "price", "sustainability_score", "sustainability_score_rule", "sustainability_score_ml", "score_explanation", "image_path", "source"]


def build_metadata(row: pd.Series) -> dict:
    """Type: Rule-based. Pull the fields needed for later metadata filtering."""
    metadata = {}
    for field in REQUIRED_METADATA_FIELDS:
        value = row.get(field)
        if pd.isna(value):
            # Empty string keeps the key present for consistent filtering
            # later, while avoiding NaN (ChromaDB's metadata validator
            # rejects it).
            value = ""
        elif not isinstance(value, (int, float, bool)):
            value = str(value)
        metadata[field] = value
    metadata["product_name"] = str(row.get("product_name", ""))
    metadata["brand"] = str(row.get("brand", ""))
    return metadata


def main():
    if not PRODUCTS_SCORED_PATH.exists():
        logger.error(
            f"{PRODUCTS_SCORED_PATH} not found. Run Milestone 3's "
            f"sustainability/batch_score.py first to generate it."
        )
        sys.exit(1)

    df = pd.read_csv(PRODUCTS_SCORED_PATH)
    logger.info(f"Loaded {len(df)} products from {PRODUCTS_SCORED_PATH}")

    missing_required = [f for f in REQUIRED_METADATA_FIELDS if f not in df.columns]
    if missing_required:
        logger.error(f"products_scored.csv is missing expected columns: {missing_required}")
        sys.exit(1)

    embedding_texts = df.apply(build_embedding_text, axis=1).tolist()
    logger.info("Generating embeddings (first run also downloads the SBERT model)...")
    embeddings = generate_embeddings(embedding_texts)

    metadatas = df.apply(build_metadata, axis=1).tolist()
    product_ids = df["product_id"].tolist()

    client = get_chroma_client(CHROMA_PERSIST_DIR)
    collection = get_products_collection(client)
    upsert_products(collection, product_ids, embeddings, metadatas, documents=embedding_texts)

    logger.info(f"Done. Collection now holds {collection.count()} products at {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
