"""
sustainability/batch_score.py

Type: Script (orchestrates the rule-based scorer, ML scorer, and
explanation generator; combines their outputs).

Reads data/processed/products_clean.csv, computes sustainability_score_rule,
sustainability_score_ml, sustainability_score (blended), and
score_explanation for every product, and writes
data/processed/products_scored.csv.

If no trained ML model is found, falls back to rule-based score alone
with a warning instead of crashing.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import CLEAN_PRODUCTS_PATH, ML_BLEND_WEIGHT, SCORED_PRODUCTS_PATH
from sustainability.explanation_generator import generate_explanation
from sustainability.ml_scorer import load_model, predict_ml_score
from sustainability.scoring_engine import compute_sustainability_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def score_products(df: pd.DataFrame) -> pd.DataFrame:
    """Score every product in `df`; returns a new DataFrame with the added columns."""
    ml_model = load_model()
    effective_ml_weight = ML_BLEND_WEIGHT if ml_model is not None else 0.0
    if ml_model is None:
        logger.warning(
            "Proceeding with rule-based scores only (ML model not found). "
            "Run scripts/train_sustainability_model.py to enable blending."
        )

    rule_scores, ml_scores, blended_scores, explanations = [], [], [], []

    for _, row in df.iterrows():
        rule_score, subscores = compute_sustainability_score(row)
        rule_scores.append(rule_score)

        if ml_model is not None:
            ml_score = predict_ml_score(ml_model, row)
        else:
            ml_score = None
        ml_scores.append(ml_score)

        blended = (
            (1 - effective_ml_weight) * rule_score + effective_ml_weight * ml_score
            if ml_score is not None
            else rule_score
        )
        blended_scores.append(round(blended, 4))

        explanations.append(generate_explanation(row, rule_score, subscores))

    out = df.copy()
    out["sustainability_score_rule"] = rule_scores
    out["sustainability_score_ml"] = ml_scores
    out["sustainability_score"] = blended_scores
    out["score_explanation"] = explanations
    return out


def main() -> None:
    if not CLEAN_PRODUCTS_PATH.exists():
        logger.error(
            "Could not find %s. Run scripts/run_pipeline.py first (Milestone 2).",
            CLEAN_PRODUCTS_PATH,
        )
        sys.exit(1)

    df = pd.read_csv(CLEAN_PRODUCTS_PATH)
    logger.info("Loaded %d products from %s", len(df), CLEAN_PRODUCTS_PATH)

    scored = score_products(df)

    SCORED_PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_PRODUCTS_PATH, index=False)
    logger.info("Wrote %d scored products to %s", len(scored), SCORED_PRODUCTS_PATH)

    print("\n--- Sample output ---")
    sample_cols = [
        "product_id", "product_name",
        "sustainability_score_rule", "sustainability_score_ml",
        "sustainability_score", "score_explanation",
    ]
    print(scored[sample_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
