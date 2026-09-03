"""
recommendation/content_based/content_scorer.py

Type: Recommendation-algorithm (thin packaging layer, not a model itself)

Wraps Milestone 4's Transformer-based semantic search
(vector_search.semantic_search) and repackages its output as a normalized
0-1 content score per candidate product for the hybrid blender.

Does NOT re-embed products or reimplement vector search -- that logic stays
in embeddings/ and vector_search/ from Milestone 4.

Also adds one thing Milestone 4 didn't have: an honest "out_of_domain" flag.
This catalog is apparel-only (Shoes/Bags/T-Shirts/Jeans/Jackets/Shirts/Dresses).
A query like "wireless bluetooth headphones" will still get top_k results back
from Chroma (it always returns *something*), but their similarity scores will
be far lower than a genuine match (~0.10-0.19 vs ~0.68+ for a real match, based
on manual verification against this catalog). Rather than silently presenting
those low-similarity results as confident recommendations, this wrapper flags them.

NOTE: semantic_search() nests per-product fields under a "metadata" dict, e.g.
{"product_id": ..., "similarity": ..., "metadata": {"product_name": ..., "price": ...}}
-- confirmed against a live call on 2026-08-31.
"""

import logging
from typing import Dict, List, Optional, Tuple

from vector_search.semantic_search import semantic_search
from config import settings

logger = logging.getLogger(__name__)


def get_content_scores(
    query: str,
    top_k: Optional[int] = None,
    min_similarity_threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, dict], bool, float]:
    """
    Run semantic search for `query` and package results for the hybrid blender.

    Returns:
        scores: {product_id: content_score in [0, 1]}
        metadata: {product_id: {product_name, category, brand, price, sustainability_score}}
        is_out_of_domain: True if even the best match is below the threshold --
            signals "this catalog probably doesn't have what you're looking for"
            instead of silently returning low-relevance products as confident matches.
        best_similarity: the top raw similarity score seen (for logging/debugging).
    """
    top_k = top_k or settings.RECOMMENDATION_CANDIDATE_POOL_SIZE
    min_similarity_threshold = (
        min_similarity_threshold
        if min_similarity_threshold is not None
        else settings.CONTENT_MIN_SIMILARITY_THRESHOLD
    )

    try:
        results = semantic_search(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Semantic search failed for query='{query}': {e}")
        return {}, {}, True, 0.0

    if not results:
        logger.warning(f"No semantic search results for query='{query}'")
        return {}, {}, True, 0.0

    scores: Dict[str, float] = {}
    metadata: Dict[str, dict] = {}
    best_similarity = 0.0

    for r in results:
        pid = r["product_id"]
        similarity = float(r["similarity"])
        content_score = max(0.0, min(1.0, similarity))  # defensive clip to [0,1]
        scores[pid] = content_score

        m = r.get("metadata", {})
        raw_price = m.get("price")
        # Chroma stores a missing/NaN price as "" (see build_vector_index.py's
        # build_metadata()), since its metadata validator rejects real NaN/None.
        # Normalize back to a real None here -- the single point everything
        # downstream (filtering, the API response, the frontend) reads from --
        # instead of special-casing "" separately in each consumer.
        price = raw_price if isinstance(raw_price, (int, float)) and not isinstance(raw_price, bool) else None
        source = m.get("source") or None

        # Currency handling: the catalog mixes real USD (Amazon), real INR
        # (synthetic), and no price at all (H&M). `price`/`currency` are the
        # HONEST native values -- shown to the user exactly as-is, never
        # converted for display. `price_inr_equiv` is an internal-only
        # approximation (documented static rate, config.settings.USD_TO_INR_RATE)
        # used for cross-currency filtering/sorting and aggregate stats --
        # it is exposed in the API for that purpose but must never be shown
        # to the user as if it were a real per-product price.
        currency = settings.SOURCE_CURRENCY.get(source) if price is not None else None
        if price is None or currency is None:
            price_inr_equiv = None
        elif currency == "INR":
            price_inr_equiv = price
        elif currency == "USD":
            price_inr_equiv = round(price * settings.USD_TO_INR_RATE, 2)
        else:
            price_inr_equiv = None

        metadata[pid] = {
            "product_name": m.get("product_name"),
            "category": m.get("category"),
            "brand": m.get("brand"),
            "price": price,
            "currency": currency,
            "price_inr_equiv": price_inr_equiv,
            "source": source,
            "sustainability_score": m.get("sustainability_score"),
            "sustainability_score_rule": m.get("sustainability_score_rule"),
            "sustainability_score_ml": m.get("sustainability_score_ml"),
            "score_explanation": m.get("score_explanation") or None,
            "image_path": m.get("image_path") or None,
        }
        best_similarity = max(best_similarity, similarity)

    is_out_of_domain = best_similarity < min_similarity_threshold
    if is_out_of_domain:
        logger.warning(
            f"Query '{query}' looks out-of-domain for this catalog "
            f"(best similarity={best_similarity:.4f} < threshold={min_similarity_threshold}). "
            f"Flagging results as low-confidence rather than presenting them as strong matches."
        )

    return scores, metadata, is_out_of_domain, best_similarity
