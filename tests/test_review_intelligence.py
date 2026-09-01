"""
Tests for Milestone 6 — Review Sentiment Intelligence.

Uses real hand-picked examples pulled from reviews_clean.csv's actual
style (confirmed via diagnostic sampling), not synthetic test strings.
"""

import pandas as pd
import pytest

from ai_nlp.review_intelligence.sentiment_scorer import score_review_text
from ai_nlp.review_intelligence.aspect_tagger import tag_aspects
from ai_nlp.review_intelligence.aggregator import aggregate_product_sentiment

# Real examples observed in data/processed/reviews_clean.csv
REAL_POSITIVE_TEXT = "Loved the material, very comfortable for daily wear."
REAL_NEGATIVE_TEXT = "Started wearing out after a few weeks."


def test_polarity_and_confidence_in_range():
    result = score_review_text(REAL_POSITIVE_TEXT)
    assert result is not None
    assert -1.0 <= result["polarity"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["label"] in ("POSITIVE", "NEGATIVE")


def test_real_positive_scores_higher_than_real_negative():
    pos = score_review_text(REAL_POSITIVE_TEXT)
    neg = score_review_text(REAL_NEGATIVE_TEXT)
    assert pos is not None and neg is not None
    assert pos["polarity"] > neg["polarity"]


@pytest.mark.parametrize("bad_text", [None, "", "   "])
def test_missing_or_empty_text_returns_none(bad_text):
    assert score_review_text(bad_text) is None


def test_aggregate_zero_reviews_is_honest_none():
    result = aggregate_product_sentiment([])
    assert result["review_count"] == 0
    assert result["avg_polarity"] is None
    assert result["pct_positive"] is None
    assert result["pct_negative"] is None
    assert result["sentiment_confidence_avg"] is None


def test_aggregate_nonzero_reviews_computes_correctly():
    scores = [
        {"polarity": 0.9, "confidence": 0.9},
        {"polarity": -0.8, "confidence": 0.8},
        {"polarity": 0.5, "confidence": 0.5},
    ]
    result = aggregate_product_sentiment(scores)
    assert result["review_count"] == 3
    assert result["pct_positive"] == pytest.approx(2 / 3, rel=1e-3)
    assert result["pct_negative"] == pytest.approx(1 / 3, rel=1e-3)
    assert result["avg_polarity"] == pytest.approx((0.9 - 0.8 + 0.5) / 3, rel=1e-3)


def test_aggregator_is_deterministic_same_input_same_output():
    scores = [{"polarity": 0.3, "confidence": 0.6}, {"polarity": -0.2, "confidence": 0.55}]
    first = aggregate_product_sentiment(scores)
    second = aggregate_product_sentiment(scores)
    assert first == second


def test_zero_review_product_exists_in_real_data_and_is_handled_honestly():
    """
    Confirms, against the real files, that at least one product_id in
    products_clean.csv has zero rows in reviews_clean.csv, and that
    aggregating an empty list for it returns honest None fields.
    """
    products_df = pd.read_csv("data/processed/products_clean.csv")
    reviews_df = pd.read_csv("data/processed/reviews_clean.csv")

    reviewed_ids = set(reviews_df["product_id"].unique())
    zero_review_ids = [pid for pid in products_df["product_id"] if pid not in reviewed_ids]

    assert len(zero_review_ids) > 0, "Expected at least one zero-review product in real data"

    result = aggregate_product_sentiment([])
    assert result["review_count"] == 0
    assert result["avg_polarity"] is None


def test_aspect_tagger_matches_real_examples():
    assert "comfort_fit" in tag_aspects(REAL_POSITIVE_TEXT)
    assert "quality" in tag_aspects(REAL_NEGATIVE_TEXT)


def test_aspect_tagger_handles_missing_text():
    assert tag_aspects(None) == []
    assert tag_aspects("") == []
