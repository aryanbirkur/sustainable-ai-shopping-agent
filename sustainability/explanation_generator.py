"""
sustainability/explanation_generator.py

Type: Rule-based (template-based text generation — no ML, no LLM).

Turns a product's sustainability_score plus its component sub-scores
(produced by scoring_engine.compute_sustainability_score) into a short,
human-readable explanation sentence.

Design notes:
- Every phrase is generated from an actual input attribute of the
  product being explained — nothing is invented or hallucinated.
- Only synthetic/estimated data is available, so explanations use
  hedged, dataset-relative language rather than absolute real-world claims.
- If a component was missing for this product, it's simply left out of
  the explanation (never described as if it were known).
"""

from typing import Dict, Optional

import pandas as pd

STRONG_THRESHOLD = 0.7
WEAK_THRESHOLD = 0.3

_POSITIVE_PHRASES = {
    "carbon_footprint_kg": "a low carbon footprint ({value:.1f} kg CO2e est.)",
    "water_usage_liters": "low water usage ({value:.0f} L est.)",
    "recycled_material_percentage": "{value:.0f}% recycled material",
    "organic_material_percentage": "{value:.0f}% organic material",
    "recyclability_score": "strong recyclability ({value:.2f}/1.0)",
    "repairability_score": "strong repairability ({value:.2f}/1.0)",
    "product_lifetime_years": "a long estimated lifespan ({value:.1f} years)",
    "eco_certification": "{value} certification",
}

_NEGATIVE_PHRASES = {
    "carbon_footprint_kg": "carbon footprint is above the typical range for this dataset",
    "water_usage_liters": "water usage is above the typical range for this dataset",
    "recycled_material_percentage": "very little recycled material ({value:.0f}%)",
    "organic_material_percentage": "very little organic material ({value:.0f}%)",
    "recyclability_score": "low recyclability ({value:.2f}/1.0)",
    "repairability_score": "low repairability ({value:.2f}/1.0)",
    "product_lifetime_years": "a short estimated lifespan ({value:.1f} years)",
    "eco_certification": "no eco-certification",
}


def _raw_value(row: pd.Series, component: str):
    return row.get(component, None)


def generate_explanation(
    row: pd.Series,
    score: float,
    subscores: Dict[str, Optional[float]],
) -> str:
    """
    Rule-based / template-based: build a one-sentence explanation for a
    product's sustainability_score from its component sub-scores.
    """
    strengths = []
    weaknesses = []

    for component, sub in subscores.items():
        if sub is None:
            continue
        raw = _raw_value(row, component)
        if sub >= STRONG_THRESHOLD:
            phrase = _POSITIVE_PHRASES[component].format(value=raw)
            strengths.append((sub, phrase))
        elif sub <= WEAK_THRESHOLD:
            phrase = _NEGATIVE_PHRASES[component].format(value=raw)
            weaknesses.append((sub, phrase))

    strengths.sort(key=lambda t: -t[0])
    weaknesses.sort(key=lambda t: t[0])

    top_strengths = [p for _, p in strengths[:2]]
    top_weakness = [p for _, p in weaknesses[:1]]

    n_missing = sum(1 for v in subscores.values() if v is None)
    data_caveat = ""
    if n_missing >= 3:
        data_caveat = " (score is based on partial data — several attributes were unavailable)"

    if top_strengths and top_weakness:
        sentence = (
            f"Scored {_score_bucket(score)} due to {_join(top_strengths)}, "
            f"though {top_weakness[0]}."
        )
    elif top_strengths:
        sentence = f"Scored {_score_bucket(score)} due to {_join(top_strengths)}."
    elif top_weakness:
        sentence = f"Scored {_score_bucket(score)}, dragged down by {top_weakness[0]}."
    else:
        sentence = (
            f"Scored {_score_bucket(score)} overall, with no single "
            f"attribute standing out as a particular strength or weakness."
        )

    return sentence + data_caveat


def _score_bucket(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "moderately"
    if score >= 0.25:
        return "low"
    return "very low"


def _join(phrases) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return f"{phrases[0]} and {phrases[1]}"
