"""
backend/services/product_detail_service.py

Type: Rule-based orchestration (calls existing Transformer + aggregation
modules; does not itself score anything).

Builds the data for a single product's detail view: full sustainability
sub-scores/explanation (already in products_scored.csv) plus a live
review-sentiment aggregation (computed on request, not precomputed --
this endpoint is only called when a user expands a specific product's
detail view in the UI, not for every search result up front).
"""

import logging
from typing import Optional

import pandas as pd

from config.settings import PRODUCTS_SCORED_PATH, CLEAN_REVIEWS_PATH
from ai_nlp.review_intelligence.sentiment_scorer import score_reviews_batch
from ai_nlp.review_intelligence.aggregator import aggregate_product_sentiment

logger = logging.getLogger(__name__)


def get_product_detail(product_id: str) -> Optional[dict]:
    """
    Type: Rule-based orchestration

    Returns None if product_id doesn't exist in products_scored.csv
    (caller maps this to a 404 -- never fabricates a detail record for
    an unknown product).
    """
    products_df = pd.read_csv(PRODUCTS_SCORED_PATH)
    row = products_df[products_df["product_id"] == product_id]
    if row.empty:
        return None
    row = row.iloc[0]

    reviews_df = pd.read_csv(CLEAN_REVIEWS_PATH)
    product_reviews = reviews_df[reviews_df["product_id"] == product_id]
    review_texts = product_reviews["review_text"].tolist()

    scored = score_reviews_batch(review_texts) if review_texts else []
    scored_non_null = [s for s in scored if s is not None]
    sentiment = aggregate_product_sentiment(scored_non_null)

    def _clean(value):
        return None if pd.isna(value) else value

    return {
        "product_id": product_id,
        "product_name": _clean(row.get("product_name")),
        "sustainability_score": _clean(row.get("sustainability_score")),
        "sustainability_score_rule": _clean(row.get("sustainability_score_rule")),
        "sustainability_score_ml": _clean(row.get("sustainability_score_ml")),
        "score_explanation": _clean(row.get("score_explanation")),
        "review_sentiment": sentiment,
    }
