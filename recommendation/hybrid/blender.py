"""
recommendation/hybrid/blender.py

Type: Recommendation-algorithm (the blending math itself is NOT a model --
it combines three already-labeled signals:
  - content: Transformer-based (Milestone 4 semantic search)
  - collaborative: ML (Milestone 5 item-item CF)
  - sustainability: ML/Rule-based blend (Milestone 3, already scored)

When the collaborative signal is unavailable (cold start), its weight is
redistributed proportionally across the remaining signals so weights still
sum to 1 -- never silently treated as a zero score.
"""

import logging
from typing import Dict, List, Optional, Tuple

from config import settings
from recommendation.content_based.content_scorer import get_content_scores
from recommendation.collaborative.cf_scorer import CollaborativeFilteringScorer

logger = logging.getLogger(__name__)

_cf_scorer = CollaborativeFilteringScorer()


def _renormalize_weights(weights: Dict[str, float], available_signals: List[str]) -> Dict[str, float]:
    """Drop unavailable signals and rescale the remaining weights to sum to 1."""
    available_weights = {k: v for k, v in weights.items() if k in available_signals}
    total = sum(available_weights.values())
    if total <= 0:
        n = len(available_signals)
        return {k: 1.0 / n for k in available_signals} if n else {}
    return {k: v / total for k, v in available_weights.items()}


def blend(
    query: str,
    user_id: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    category: Optional[str] = None,
    sustainability_tilt: bool = False,
) -> List[dict]:
    """
    Score all candidates for `query` (and optionally `user_id`). Returns an
    UNRANKED list of dicts -- ranking/sorting happens in recommendation/ranking/.

    Milestone 8 additions (all optional, default to prior behavior):
        price_min, price_max, category: hard-filter candidates before CF scoring.
        sustainability_tilt: shift weight toward sustainability after renormalization.
    """
    weights = dict(weights) if weights else dict(settings.HYBRID_WEIGHTS_DEFAULT)

    content_scores, metadata, is_out_of_domain, best_similarity = get_content_scores(query)
    if not content_scores:
        logger.warning(f"No candidates returned for query='{query}'")
        return []

    candidates_before_filter = len(content_scores)
    content_scores, metadata, filter_relaxed = _filter_candidates(
        content_scores, metadata, price_min, price_max, category
    )
    candidates_after_filter = len(content_scores)

    filtering_info = {
        "price_filter_applied": price_min is not None or price_max is not None,
        "category_filter_applied": category is not None,
        "candidates_before_filter": candidates_before_filter,
        "candidates_after_filter": candidates_after_filter,
        "filter_relaxed": filter_relaxed,
    }

    candidate_ids = list(content_scores.keys())

    cf_raw = _cf_scorer.score(user_id, candidate_ids)
    cf_available = any(v is not None for v in cf_raw.values())

    available_signals = ["content", "sustainability"] + (["collaborative"] if cf_available else [])
    final_weights = _renormalize_weights(weights, available_signals)

    if sustainability_tilt:
        final_weights = _apply_sustainability_tilt(final_weights)

    results = []
    for pid in candidate_ids:
        content_s = content_scores[pid]
        sustainability_raw = metadata[pid].get("sustainability_score")
        sustainability_s = float(sustainability_raw) if sustainability_raw is not None else 0.0
        cf_s = cf_raw.get(pid)

        breakdown = {
            "content": round(content_s * final_weights.get("content", 0.0), 6),
            "collaborative": (
                round((cf_s or 0.0) * final_weights.get("collaborative", 0.0), 6)
                if cf_available else None
            ),
            "sustainability": round(sustainability_s * final_weights.get("sustainability", 0.0), 6),
        }
        final_score = round(sum(v for v in breakdown.values() if v is not None), 6)

        results.append({
            "product_id": pid,
            "final_score": final_score,
            "score_breakdown": breakdown,
            "weights_used": final_weights,
            "cold_start": not cf_available,
            "out_of_domain_query": is_out_of_domain,
            "filtering": filtering_info,
            "raw_signals": {
                "content": round(content_s, 4),
                "collaborative": round(cf_s, 4) if cf_s is not None else None,
                "sustainability": round(sustainability_s, 4),
            },
            **metadata[pid],
        })

    return results


def _apply_sustainability_tilt(
    weights: Dict[str, float], tilt_amount: Optional[float] = None
) -> Dict[str, float]:
    """
    Type: Aggregation

    Shift `tilt_amount` of total weight from content/collaborative into
    sustainability, proportional to their relative sizes. Runs AFTER
    cold-start renormalization, so it only redistributes from signals
    actually present in the dict it's given.
    """
    tilt_amount = (
        tilt_amount if tilt_amount is not None else settings.SUSTAINABILITY_TILT_AMOUNT
    )
    donors = [k for k in ("content", "collaborative") if k in weights]
    donor_total = sum(weights[k] for k in donors)
    if not donors or donor_total <= 0:
        return weights
    actual_tilt = min(tilt_amount, donor_total)
    tilted = dict(weights)
    for k in donors:
        share = weights[k] / donor_total
        tilted[k] = weights[k] - (actual_tilt * share)
    tilted["sustainability"] = weights.get("sustainability", 0.0) + actual_tilt
    return tilted


def _filter_candidates(
    content_scores: Dict[str, float],
    metadata: Dict[str, dict],
    price_min: Optional[float],
    price_max: Optional[float],
    category: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, dict], bool]:
    """
    Type: Rule-based

    Hard-filters candidates by price_min/price_max/category. If the filter
    would empty the pool entirely, falls back to the unfiltered candidates
    and returns filter_relaxed=True instead of returning nothing.
    """
    if price_min is None and price_max is None and category is None:
        return content_scores, metadata, False

    filtered_ids = []
    for pid in content_scores:
        m = metadata[pid]
        price = m.get("price")
        # Chroma metadata stores a missing/NaN price as "" (not None) --
        # see build_vector_index.py's build_metadata(). Treat anything
        # that isn't a real number as "unknown", same as None, rather
        # than comparing a string to a float and crashing.
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            price = None
        cat = m.get("category")
        if price_min is not None and (price is None or price < price_min):
            continue
        if price_max is not None and (price is None or price > price_max):
            continue
        if category is not None and cat != category:
            continue
        filtered_ids.append(pid)

    if not filtered_ids:
        logger.warning(
            "Structured filter matched zero candidates -- relaxing filter."
        )
        return content_scores, metadata, True

    return (
        {pid: content_scores[pid] for pid in filtered_ids},
        {pid: metadata[pid] for pid in filtered_ids},
        False,
    )
