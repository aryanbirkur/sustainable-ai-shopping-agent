"""
Per-product sentiment aggregation.

Type: Aggregation (not itself a model) — combines Transformer sentiment
scores (sentiment_scorer.py) into one summary per product.

Aggregation approach: simple (unweighted) mean of polarity across a
product's reviews. No recency or rating weighting — this is synthetic
data with no established real-world recency pattern to weight against,
so an unweighted mean is the most honest choice with the fewest
unverified assumptions.

Products with zero scored reviews get honest None values for every
aggregate field (never a fabricated 0.0 or neutral score), with
review_count=0 so downstream consumers can tell "no signal" apart from
"neutral signal" — same pattern as Milestone 5's CF cold start.
"""

from typing import List, Dict


def aggregate_product_sentiment(review_scores: List[Dict]) -> Dict:
    """
    Type: Aggregation (not a model)

    review_scores: list of dicts, each with "polarity" (float in [-1,1])
    and "confidence" (float in [0,1]), for reviews belonging to ONE
    product. Pass an empty list for a product with zero scored reviews.

    Returns:
        {avg_polarity, pct_positive, pct_negative,
         sentiment_confidence_avg, review_count}
    All aggregate fields are None (not 0.0) when review_count == 0.
    """
    review_count = len(review_scores)

    if review_count == 0:
        return {
            "avg_polarity": None,
            "pct_positive": None,
            "pct_negative": None,
            "sentiment_confidence_avg": None,
            "review_count": 0,
        }

    polarities = [r["polarity"] for r in review_scores]
    confidences = [r["confidence"] for r in review_scores]

    avg_polarity = sum(polarities) / review_count
    pct_positive = sum(1 for p in polarities if p > 0) / review_count
    pct_negative = sum(1 for p in polarities if p < 0) / review_count
    sentiment_confidence_avg = sum(confidences) / review_count

    return {
        "avg_polarity": round(avg_polarity, 4),
        "pct_positive": round(pct_positive, 4),
        "pct_negative": round(pct_negative, 4),
        "sentiment_confidence_avg": round(sentiment_confidence_avg, 4),
        "review_count": review_count,
    }
