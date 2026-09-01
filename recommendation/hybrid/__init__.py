"""
recommendation/hybrid/__init__.py -- Public entry point for Milestone 5.
"""

import logging
from typing import Dict, List, Optional

from recommendation.hybrid.blender import blend
from recommendation.ranking.ranker import rank

logger = logging.getLogger(__name__)


def recommend(
    query: str,
    user_id: Optional[str] = None,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    category: Optional[str] = None,
    sustainability_tilt: bool = False,
) -> List[dict]:
    """
    Blend content, collaborative, and sustainability signals for `query`
    (and `user_id` if known) and return the top_k ranked results.

    Milestone 8 additions (all optional, default to prior behavior):
        price_min, price_max, category, sustainability_tilt -- see blend().

    Each result dict contains: product_id, final_score, score_breakdown,
    rank, weights_used, cold_start, out_of_domain_query, filtering,
    raw_signals, product_name, category, brand, price, sustainability_score.
    """
    candidates = blend(
        query=query,
        user_id=user_id,
        weights=weights,
        price_min=price_min,
        price_max=price_max,
        category=category,
        sustainability_tilt=sustainability_tilt,
    )
    if not candidates:
        logger.warning(f"recommend(): no candidates for query='{query}'")
        return []
    return rank(candidates, top_k=top_k)


def recommend_with_intent(
    query: str,
    user_id: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Type: Aggregation (Milestone 8)

    Wires Milestone 7's rule-based extract_intent() into the existing
    recommend() pipeline -- extracts price/category/sustainability signals
    from `query`, applies them as filtering/weight-tilt parameters, and
    reshapes the response so both honest failure signals (out-of-catalog
    category from intent extraction, out_of_domain_query from content-based
    search) surface side by side rather than one masking the other.

    Does not modify recommend()'s signature or behavior for existing callers.
    """
    from ai_nlp.intent_extraction.intent_parser import extract_intent

    intent = extract_intent(query)

    results = recommend(
        query=query,
        user_id=user_id,
        top_k=top_k,
        price_min=intent["price_min"],
        price_max=intent["price_max"],
        category=intent["category"],
        sustainability_tilt=intent["sustainability_emphasis"],
    )

    out_of_catalog_category = any(
        "not a category in this catalog" in note
        for note in intent["unparsed_confidence_notes"]
    )
    out_of_domain_query = bool(results[0]["out_of_domain_query"]) if results else True
    filtering_info = results[0]["filtering"] if results else None
    weights_used = results[0]["weights_used"] if results else None

    return {
        "results": results,
        "intent": intent,
        "filtering": filtering_info,
        "weights_used": weights_used,
        "warnings": {
            "out_of_catalog_category": out_of_catalog_category,
            "out_of_domain_query": out_of_domain_query,
        },
    }
