"""
recommendation/ranking/ranker.py

Type: Recommendation-algorithm

Sorts blended candidates by final_score and assigns rank. Kept separate
from the blender (single responsibility) so a future learned re-ranker can
be swapped in here later without touching blending logic.
"""

from typing import List


def rank(candidates: List[dict], top_k: int = 10) -> List[dict]:
    """Sort by final_score descending, assign 1-indexed rank, truncate to top_k."""
    sorted_candidates = sorted(candidates, key=lambda c: c["final_score"], reverse=True)
    for i, c in enumerate(sorted_candidates, start=1):
        c["rank"] = i
    return sorted_candidates[:top_k]
