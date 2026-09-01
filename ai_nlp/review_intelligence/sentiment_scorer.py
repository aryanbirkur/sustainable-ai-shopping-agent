"""
Sentiment scoring for product reviews.

Type: Transformer
Model: distilbert-base-uncased-finetuned-sst-2-english (Hugging Face, local, CPU)

Scores each review's text as positive/negative and converts the model's
confidence into a signed polarity score in [-1, 1]:
    polarity = +confidence  if label == POSITIVE
    polarity = -confidence  if label == NEGATIVE

Real Transformer inference on every review — never a fabricated or
hard-coded score. Reviews in this project are synthetic
(source="synthetic_v1"); scores reflect sentiment of synthetic review
text, not real customer opinion.
"""

from pathlib import Path
from typing import List, Dict, Optional

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "sentiment_model_cache"

_sentiment_pipeline = None


def _get_pipeline():
    """
    Lazily load and cache the HF sentiment-analysis pipeline (loaded once per process).

    Loads model + tokenizer explicitly with cache_dir (a valid from_pretrained
    argument) rather than passing cache_dir to pipeline() directly -- in this
    transformers version, pipeline(cache_dir=...) gets incorrectly re-forwarded
    into the tokenizer's per-call kwargs, causing a TypeError at inference time.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        SENTIMENT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL_NAME, cache_dir=str(SENTIMENT_MODEL_CACHE_DIR)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            SENTIMENT_MODEL_NAME, cache_dir=str(SENTIMENT_MODEL_CACHE_DIR)
        )
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
        )
    return _sentiment_pipeline


def score_review_text(text: str) -> Optional[Dict]:
    """
    Type: Transformer

    Score a single review's text. Returns None (never a fabricated 0.0)
    for missing/empty/whitespace-only text, so callers can distinguish
    "no signal" from "neutral signal".

    Returns:
        {"label": "POSITIVE"|"NEGATIVE", "confidence": float in [0,1],
         "polarity": float in [-1,1]}
        or None if text is missing/empty.
    """
    if text is None or not isinstance(text, str) or text.strip() == "":
        return None

    clf = _get_pipeline()
    result = clf(text.strip(), truncation=True)[0]
    label = result["label"]
    confidence = float(result["score"])
    polarity = confidence if label == "POSITIVE" else -confidence

    return {"label": label, "confidence": confidence, "polarity": polarity}


def score_reviews_batch(review_texts: List[str]) -> List[Optional[Dict]]:
    """
    Type: Transformer

    Batch-score a list of review texts using the HF pipeline's native
    batching (much faster than looping score_review_text for 2000+ rows).
    Preserves order; returns None in-place for missing/empty text so the
    output list stays index-aligned with the input.
    """
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(review_texts):
        if text is not None and isinstance(text, str) and text.strip() != "":
            valid_indices.append(i)
            valid_texts.append(text.strip())

    results: List[Optional[Dict]] = [None] * len(review_texts)

    if valid_texts:
        clf = _get_pipeline()
        raw_results = clf(valid_texts, truncation=True)
        for idx, raw in zip(valid_indices, raw_results):
            label = raw["label"]
            confidence = float(raw["score"])
            polarity = confidence if label == "POSITIVE" else -confidence
            results[idx] = {"label": label, "confidence": confidence, "polarity": polarity}

    return results
