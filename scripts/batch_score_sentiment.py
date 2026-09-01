import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Batch sentiment scoring — Milestone 6.

Type: Transformer (sentiment) + Rule-based (aspects) + Aggregation

Reads data/processed/reviews_clean.csv, scores every review's text with
the Transformer sentiment model, aggregates per product_id, and writes
data/processed/product_sentiment.csv — one row per product in
products_clean.csv (including products with zero reviews, which get
honest None aggregate values).

Idempotent: this script reads reviews_clean.csv and products_clean.csv
fresh each time and OVERWRITES product_sentiment.csv in full (never
appends), so re-running produces equivalent output without duplicating
or corrupting rows.

Reminder: all review text is synthetic (source="synthetic_v1"). Results
describe sentiment of synthetic review text, not real customer opinion.
"""

import logging
from collections import Counter

import pandas as pd

from ai_nlp.review_intelligence.sentiment_scorer import score_reviews_batch
from ai_nlp.review_intelligence.aspect_tagger import tag_aspects
from ai_nlp.review_intelligence.aggregator import aggregate_product_sentiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REVIEWS_PATH = "data/processed/reviews_clean.csv"
PRODUCTS_PATH = "data/processed/products_clean.csv"
OUTPUT_PATH = "data/processed/product_sentiment.csv"


def main():
    logger.info("Loading reviews from %s", REVIEWS_PATH)
    reviews_df = pd.read_csv(REVIEWS_PATH)

    logger.info("Loading product list from %s", PRODUCTS_PATH)
    products_df = pd.read_csv(PRODUCTS_PATH)
    all_product_ids = products_df["product_id"].tolist()

    logger.info(
        "Scoring %d reviews with Transformer sentiment model "
        "(first run downloads the model, ~268MB)...",
        len(reviews_df),
    )
    texts = reviews_df["review_text"].tolist()
    scores = score_reviews_batch(texts)  # index-aligned with reviews_df

    reviews_df = reviews_df.copy()
    reviews_df["_polarity"] = [s["polarity"] if s else None for s in scores]
    reviews_df["_confidence"] = [s["confidence"] if s else None for s in scores]
    reviews_df["_aspects"] = reviews_df["review_text"].apply(tag_aspects)

    logger.info("Aggregating per product...")
    rows = []
    for product_id in all_product_ids:
        product_reviews = reviews_df[reviews_df["product_id"] == product_id]

        review_scores = [
            {"polarity": row["_polarity"], "confidence": row["_confidence"]}
            for _, row in product_reviews.iterrows()
            if row["_polarity"] is not None
        ]

        agg = aggregate_product_sentiment(review_scores)

        aspect_counter = Counter()
        for aspects in product_reviews["_aspects"]:
            aspect_counter.update(aspects)
        top_aspects = (
            ", ".join(a for a, _ in aspect_counter.most_common(3))
            if aspect_counter else None
        )

        rows.append({
            "product_id": product_id,
            "avg_polarity": agg["avg_polarity"],
            "pct_positive": agg["pct_positive"],
            "pct_negative": agg["pct_negative"],
            "sentiment_confidence_avg": agg["sentiment_confidence_avg"],
            "review_count": agg["review_count"],
            "top_aspects_mentioned": top_aspects,
        })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %d product sentiment rows to %s", len(output_df), OUTPUT_PATH)

    scored_count = int((output_df["review_count"] > 0).sum())
    zero_count = int((output_df["review_count"] == 0).sum())
    logger.info(
        "%d products have >=1 review scored, %d products have zero reviews (honest None)",
        scored_count, zero_count,
    )


if __name__ == "__main__":
    main()
