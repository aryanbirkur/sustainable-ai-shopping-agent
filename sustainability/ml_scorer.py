"""
sustainability/ml_scorer.py

Type: ML (scikit-learn RandomForestRegressor — a genuinely trained model,
not a hard-coded or fake result).

WHY THIS MODULE EXISTS: we have no real ground-truth sustainability
score for any product (all data is synthetic). To still build a real ML
pipeline now, we train against a PROXY target computed with DIFFERENT
linear weights than scoring_engine.py, plus two nonlinear interaction
terms, plus injected Gaussian noise — so the model has a genuinely
different target to fit, not just the rule-based formula. A shallow
RandomForest is used deliberately so it doesn't perfectly memorize the
proxy either.

LIMITATION: this model is trained against a synthetic proxy, not real
ground truth. Treat it as a working pipeline placeholder to be retrained
against real labels once they exist.
"""

import logging
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from config.settings import (
    ML_PROXY_NOISE_STD,
    ML_TRAINING_RANDOM_SEED,
    SUSTAINABILITY_COMPONENT_WEIGHTS,
    SUSTAINABILITY_ML_MODEL_PATH,
)
from sustainability.scoring_engine import score_component

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = list(SUSTAINABILITY_COMPONENT_WEIGHTS.keys())

PROXY_LINEAR_WEIGHTS = {
    "carbon_footprint_kg": 0.10,
    "water_usage_liters": 0.10,
    "recycled_material_percentage": 0.20,
    "organic_material_percentage": 0.05,
    "eco_certification": 0.15,
    "recyclability_score": 0.15,
    "repairability_score": 0.20,
    "product_lifetime_years": 0.05,
}
assert abs(sum(PROXY_LINEAR_WEIGHTS.values()) - 1.0) < 1e-9

SYNERGY_BONUS_WEIGHT = 0.05
PENALTY_WEIGHT = 0.05


def _row_subscores(row: pd.Series) -> Dict[str, float]:
    """
    Build the 8-feature vector for one row. Missing attributes are
    imputed as 0.5 (neutral) for the ML feature matrix — a different
    missing-data strategy than the rule-based scorer, documented here.
    """
    features = {}
    for component in FEATURE_COLUMNS:
        sub = score_component(row, component)
        features[component] = 0.5 if sub is None else sub
    return features


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """ML: build the numeric feature matrix (one row per product) used for train/predict."""
    rows = [_row_subscores(row) for _, row in df.iterrows()]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def compute_proxy_target(features: pd.DataFrame, add_noise: bool, seed: int) -> np.ndarray:
    """
    Generate the synthetic proxy training label: different linear weights
    + nonlinear interaction terms + optional Gaussian noise. Used ONLY to
    create a target to train against — never shown to users as a real score.
    """
    linear = np.zeros(len(features))
    for component, weight in PROXY_LINEAR_WEIGHTS.items():
        linear += features[component].to_numpy() * weight

    recycled = features["recycled_material_percentage"].to_numpy()
    cert = features["eco_certification"].to_numpy()
    synergy = SYNERGY_BONUS_WEIGHT * (recycled * cert)

    carbon_bad = 1.0 - features["carbon_footprint_kg"].to_numpy()
    water_bad = 1.0 - features["water_usage_liters"].to_numpy()
    penalty = PENALTY_WEIGHT * (carbon_bad * water_bad)

    target = linear + synergy - penalty

    if add_noise:
        rng = np.random.default_rng(seed)
        target = target + rng.normal(0, ML_PROXY_NOISE_STD, size=len(target))

    return np.clip(target, 0.0, 1.0)


def train_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    ML: train the RandomForestRegressor against the synthetic proxy
    target and return (fitted_model, evaluation_metrics).
    """
    X = build_feature_frame(df)
    y = compute_proxy_target(X, add_noise=True, seed=ML_TRAINING_RANDOM_SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=ML_TRAINING_RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=ML_TRAINING_RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    logger.info("Trained sustainability ML model: %s", metrics)
    return model, metrics


def save_model(model: RandomForestRegressor, path=SUSTAINABILITY_ML_MODEL_PATH) -> None:
    """ML: persist the trained model to disk (gitignored artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, path)
    logger.info("Saved sustainability ML model to %s", path)


def load_model(path=SUSTAINABILITY_ML_MODEL_PATH) -> Optional[RandomForestRegressor]:
    """ML: load a previously trained model. Returns None if no artifact exists yet."""
    if not path.exists():
        logger.warning(
            "No trained ML model found at %s. Run "
            "scripts/train_sustainability_model.py first.", path,
        )
        return None
    payload = joblib.load(path)
    return payload["model"]


def predict_ml_score(model: RandomForestRegressor, row: pd.Series) -> float:
    """ML: predict the sustainability sub-score (0.0-1.0) for a single product row."""
    features = pd.DataFrame([_row_subscores(row)], columns=FEATURE_COLUMNS)
    pred = float(model.predict(features)[0])
    return round(min(1.0, max(0.0, pred)), 4)
