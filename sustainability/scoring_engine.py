"""
sustainability/scoring_engine.py

Type: Rule-based (no ML, no learned parameters — pure documented formula).

Combines eight sustainability-related product attributes into a single
0.0-1.0 `sustainability_score` using fixed, hand-chosen weights and
normalization ranges (see docs/milestone3_sustainability.md for the full
reasoning behind every weight and bound).

Every score this module produces is an ESTIMATE derived from synthetic
(source="synthetic_v1") data, NOT a verified real-world environmental
measurement. Nothing here should ever be presented to a user as a
certified or lab-measured fact — see docs/dataset_sourcing.md.

Missing-data handling: if an attribute is missing (NaN, empty, or an
unrecognized eco_certification string) for a product, that component is
excluded from the score entirely and its weight is redistributed
proportionally across the remaining available attributes. A missing
attribute is never silently treated as zero — that would unfairly punish
a product just because a data field wasn't collected.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import (
    CARBON_FOOTPRINT_BOUNDS,
    CATEGORY_SUSTAINABILITY_BASELINE,
    ECO_CERTIFICATION_SCORES,
    PRODUCT_LIFETIME_BOUNDS,
    SUSTAINABILITY_COMPONENT_WEIGHTS,
    WATER_USAGE_BOUNDS,
)

logger = logging.getLogger(__name__)

# Sanity check at import time: weights must always sum to 1.0.
_weight_sum = sum(SUSTAINABILITY_COMPONENT_WEIGHTS.values())
if abs(_weight_sum - 1.0) > 1e-9:
    raise ValueError(
        f"SUSTAINABILITY_COMPONENT_WEIGHTS must sum to 1.0, got {_weight_sum}"
    )

# Components where a HIGHER raw value is WORSE (so we invert after normalizing).
_INVERTED_COMPONENTS = {"carbon_footprint_kg", "water_usage_liters"}

# Components already stored as literal percentages (0-100 -> 0-1).
_PERCENTAGE_COMPONENTS = {"recycled_material_percentage", "organic_material_percentage"}

# Components already stored as a 0.0-1.0 score, just clipped for safety.
_ALREADY_NORMALIZED_COMPONENTS = {"recyclability_score", "repairability_score"}


def _clip_normalize(value: float, low: float, high: float, invert: bool = False) -> float:
    """Rule-based helper: linearly map `value` into [0, 1] given [low, high], clipping outliers."""
    if high == low:
        return 0.5
    v = max(low, min(high, value))
    norm = (v - low) / (high - low)
    return 1.0 - norm if invert else norm


def _safe_get(row: pd.Series, col: str):
    """Return the raw value for `col`, or None if it's missing/NaN/blank."""
    val = row.get(col, None)
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, str) and val.strip() == "":
        return None
    return val


def score_component(row: pd.Series, component: str) -> Optional[float]:
    """
    Rule-based: return the normalized [0.0, 1.0] sub-score for a single
    component of one product row, or None if that attribute is missing
    or unusable for this product.
    """
    raw = _safe_get(row, component)
    if raw is None:
        return None

    if component == "carbon_footprint_kg":
        return _clip_normalize(float(raw), *CARBON_FOOTPRINT_BOUNDS, invert=True)
    if component == "water_usage_liters":
        return _clip_normalize(float(raw), *WATER_USAGE_BOUNDS, invert=True)
    if component in _PERCENTAGE_COMPONENTS:
        return _clip_normalize(float(raw), 0.0, 100.0)
    if component == "product_lifetime_years":
        return _clip_normalize(float(raw), *PRODUCT_LIFETIME_BOUNDS)
    if component in _ALREADY_NORMALIZED_COMPONENTS:
        return max(0.0, min(1.0, float(raw)))
    if component == "eco_certification":
        cert = str(raw).strip()
        if cert not in ECO_CERTIFICATION_SCORES:
            logger.warning(
                "Unknown eco_certification value '%s' for product %s — "
                "treating as missing rather than guessing.",
                cert, row.get("product_id", "<unknown>"),
            )
            return None
        return ECO_CERTIFICATION_SCORES[cert]

    raise ValueError(f"scoring_engine: unrecognized component '{component}'")


def compute_sustainability_score(row: pd.Series) -> Tuple[float, Dict[str, Optional[float]]]:
    """
    Rule-based: compute the 0.0-1.0 sustainability_score for one product.

    Returns:
        (score, subscores) where `subscores` maps each of the 8 attribute
        names to its normalized [0,1] value, or None if it was missing for
        this product (its weight was redistributed among the rest).
    """
    subscores: Dict[str, Optional[float]] = {}
    weighted_sum = 0.0
    used_weight = 0.0

    for component, weight in SUSTAINABILITY_COMPONENT_WEIGHTS.items():
        sub = score_component(row, component)
        subscores[component] = sub
        if sub is not None:
            weighted_sum += sub * weight
            used_weight += weight

    if used_weight == 0.0:
        category = row.get("category")
        baseline = CATEGORY_SUSTAINABILITY_BASELINE.get(category, 0.5)
        logger.warning(
            "Product %s has no usable sustainability attributes; "
            "falling back to the '%s' category baseline (%.2f) instead of "
            "a universal constant -- see docs/dataset_sourcing.md.",
            row.get("product_id", "<unknown>"), category, baseline,
        )
        return baseline, subscores

    score = weighted_sum / used_weight  # renormalize over whatever data WAS available
    score = round(min(1.0, max(0.0, score)), 4)
    return score, subscores
