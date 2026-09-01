"""
tests/test_sustainability.py

Tests for Milestone 3 (sustainability/). Run with:
    pytest tests/test_sustainability.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from sustainability.explanation_generator import generate_explanation
from sustainability.ml_scorer import build_feature_frame, compute_proxy_target, train_model
from sustainability.scoring_engine import compute_sustainability_score


def make_row(**overrides) -> pd.Series:
    """Baseline mid-range product row; override specific fields per test."""
    base = {
        "product_id": "TEST0001",
        "category": "T-Shirts",
        "carbon_footprint_kg": 5.0,
        "water_usage_liters": 2500.0,
        "recycled_material_percentage": 50.0,
        "organic_material_percentage": 50.0,
        "eco_certification": "Fair Trade",
        "recyclability_score": 0.5,
        "repairability_score": 0.5,
        "product_lifetime_years": 4.0,
    }
    base.update(overrides)
    return pd.Series(base)


class TestRuleBasedScorer:
    def test_high_sustainability_product_scores_high(self):
        row = make_row(
            carbon_footprint_kg=0.5,
            water_usage_liters=200.0,
            recycled_material_percentage=95.0,
            organic_material_percentage=95.0,
            eco_certification="GOTS",
            recyclability_score=0.95,
            repairability_score=0.9,
            product_lifetime_years=7.5,
        )
        score, _ = compute_sustainability_score(row)
        assert score >= 0.75, f"expected a high score, got {score}"

    def test_low_sustainability_product_scores_low(self):
        row = make_row(
            carbon_footprint_kg=9.8,
            water_usage_liters=4900.0,
            recycled_material_percentage=0.0,
            organic_material_percentage=0.0,
            eco_certification="No Certification",
            recyclability_score=0.1,
            repairability_score=0.1,
            product_lifetime_years=0.5,
        )
        score, _ = compute_sustainability_score(row)
        assert score <= 0.25, f"expected a low score, got {score}"

    @pytest.mark.parametrize("seed_val", range(20))
    def test_score_always_in_valid_range(self, seed_val):
        rng = np.random.default_rng(seed_val)
        row = make_row(
            carbon_footprint_kg=float(rng.uniform(0, 15)),
            water_usage_liters=float(rng.uniform(0, 6000)),
            recycled_material_percentage=float(rng.uniform(0, 100)),
            organic_material_percentage=float(rng.uniform(0, 100)),
            eco_certification=rng.choice(
                ["GOTS", "Fair Trade", "OEKO-TEX", "Global Recycled Standard", "No Certification"]
            ),
            recyclability_score=float(rng.uniform(0, 1)),
            repairability_score=float(rng.uniform(0, 1)),
            product_lifetime_years=float(rng.uniform(0, 10)),
        )
        score, _ = compute_sustainability_score(row)
        assert 0.0 <= score <= 1.0

    def test_missing_attributes_handled_gracefully_not_crashed(self):
        row = make_row(
            carbon_footprint_kg=np.nan,
            water_usage_liters=np.nan,
            eco_certification="",
        )
        score, subscores = compute_sustainability_score(row)
        assert 0.0 <= score <= 1.0
        assert subscores["carbon_footprint_kg"] is None
        assert subscores["water_usage_liters"] is None
        assert subscores["eco_certification"] is None
        assert subscores["recyclability_score"] is not None

    def test_missing_attribute_not_silently_treated_as_zero(self):
        strong_but_missing_carbon = make_row(
            carbon_footprint_kg=np.nan,
            water_usage_liters=100.0,
            recycled_material_percentage=95.0,
            organic_material_percentage=95.0,
            eco_certification="GOTS",
            recyclability_score=0.95,
            repairability_score=0.9,
            product_lifetime_years=7.0,
        )
        strong_with_worst_carbon = make_row(
            carbon_footprint_kg=10.0,
            water_usage_liters=100.0,
            recycled_material_percentage=95.0,
            organic_material_percentage=95.0,
            eco_certification="GOTS",
            recyclability_score=0.95,
            repairability_score=0.9,
            product_lifetime_years=7.0,
        )
        score_missing, _ = compute_sustainability_score(strong_but_missing_carbon)
        score_worst, _ = compute_sustainability_score(strong_with_worst_carbon)
        assert score_missing > score_worst

    def test_all_attributes_missing_falls_back_to_neutral_not_crash(self):
        row = pd.Series({"product_id": "TEST_EMPTY", "category": "Shoes"})
        score, subscores = compute_sustainability_score(row)
        assert score == 0.5
        assert all(v is None for v in subscores.values())

    def test_unknown_certification_string_treated_as_missing(self):
        row = make_row(eco_certification="Some Made Up Cert")
        score, subscores = compute_sustainability_score(row)
        assert subscores["eco_certification"] is None
        assert 0.0 <= score <= 1.0


class TestExplanationGenerator:
    def test_explanation_is_non_empty(self):
        row = make_row()
        score, subscores = compute_sustainability_score(row)
        explanation = generate_explanation(row, score, subscores)
        assert isinstance(explanation, str)
        assert len(explanation.strip()) > 0

    def test_explanation_mentions_a_real_input_attribute(self):
        row = make_row(
            recycled_material_percentage=95.0,
            eco_certification="GOTS",
        )
        score, subscores = compute_sustainability_score(row)
        explanation = generate_explanation(row, score, subscores)
        assert "95" in explanation or "GOTS" in explanation

    def test_explanation_does_not_mention_missing_attributes(self):
        row = make_row(carbon_footprint_kg=np.nan)
        score, subscores = compute_sustainability_score(row)
        explanation = generate_explanation(row, score, subscores)
        assert "co2e" not in explanation.lower()


class TestMLScorer:
    @pytest.fixture(scope="class")
    def sample_df(self):
        rng = np.random.default_rng(42)
        n = 60
        return pd.DataFrame({
            "product_id": [f"P{i:04d}" for i in range(n)],
            "carbon_footprint_kg": rng.uniform(0.5, 9.5, n),
            "water_usage_liters": rng.uniform(100, 4900, n),
            "recycled_material_percentage": rng.uniform(0, 100, n),
            "organic_material_percentage": rng.uniform(0, 100, n),
            "eco_certification": rng.choice(
                ["GOTS", "Fair Trade", "OEKO-TEX", "Global Recycled Standard", "No Certification"], n
            ),
            "recyclability_score": rng.uniform(0.1, 0.95, n),
            "repairability_score": rng.uniform(0.1, 0.95, n),
            "product_lifetime_years": rng.uniform(0.5, 7.5, n),
        })

    def test_feature_frame_shape(self, sample_df):
        features = build_feature_frame(sample_df)
        assert features.shape[0] == len(sample_df)
        assert features.isna().sum().sum() == 0

    def test_proxy_target_in_valid_range(self, sample_df):
        features = build_feature_frame(sample_df)
        target = compute_proxy_target(features, add_noise=True, seed=1)
        assert (target >= 0.0).all() and (target <= 1.0).all()

    def test_model_trains_and_predicts_in_range(self, sample_df):
        model, metrics = train_model(sample_df)
        assert "mae" in metrics and "r2" in metrics
        features = build_feature_frame(sample_df)
        preds = model.predict(features)
        assert (preds >= 0.0).all() and (preds <= 1.05).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
