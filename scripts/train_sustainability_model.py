"""
scripts/train_sustainability_model.py

Type: Script (invokes the ML training in sustainability/ml_scorer.py).

Trains the sustainability ML regressor on data/processed/products_clean.csv
against the synthetic proxy target (see sustainability/ml_scorer.py
docstring for why a proxy is used) and saves the fitted model to
models/sustainability_ml_model.joblib.

Run this once before running sustainability/batch_score.py, and again
any time products_clean.csv changes materially.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import CLEAN_PRODUCTS_PATH
from sustainability.ml_scorer import save_model, train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    if not CLEAN_PRODUCTS_PATH.exists():
        logger.error(
            "Could not find %s. Run scripts/run_pipeline.py first "
            "(Milestone 2) to generate the cleaned products file.",
            CLEAN_PRODUCTS_PATH,
        )
        sys.exit(1)

    df = pd.read_csv(CLEAN_PRODUCTS_PATH)
    logger.info("Loaded %d products from %s", len(df), CLEAN_PRODUCTS_PATH)

    model, metrics = train_model(df)
    save_model(model)

    print("\n--- Sustainability ML model training complete ---")
    print(f"Train rows: {metrics['n_train']}  Test rows: {metrics['n_test']}")
    print(f"Test MAE:  {metrics['mae']:.4f}  (0 = perfect, scale is 0-1)")
    print(f"Test R^2:  {metrics['r2']:.4f}  (1 = perfect fit to the proxy target)")
    print(
        "\nReminder: R^2/MAE here measure fit to the SYNTHETIC PROXY target, "
        "not real-world sustainability accuracy. See ml_scorer.py docstring."
    )


if __name__ == "__main__":
    main()
