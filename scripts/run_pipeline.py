"""
scripts/run_pipeline.py

Orchestrates the data pipeline:
    raw CSVs -> validation report -> cleaning -> clean CSVs

Run this after scripts/generate_synthetic_data.py (or after dropping in a
real dataset in the same raw CSV format).

Module type: Rule-based orchestration script. No ML/AI here.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import (
    RAW_PRODUCTS_PATH, RAW_REVIEWS_PATH, RAW_USERS_PATH, RAW_INTERACTIONS_PATH,
    CLEAN_PRODUCTS_PATH, CLEAN_REVIEWS_PATH, CLEAN_USERS_PATH, CLEAN_INTERACTIONS_PATH,
    VALIDATION_LOG_PATH, PROCESSED_DATA_DIR,
)
from backend.services.validation import run_all_validations
from backend.services.preprocessing import (
    clean_products, clean_reviews, clean_users, clean_interactions,
)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected raw data file not found: {path}\n"
            f"Run 'python scripts/generate_synthetic_data.py' first, or place "
            f"a real dataset at this path in the same column format."
        )
    return pd.read_csv(path)


def main():
    print("=== Sustainable AI Shopping Agent: Data Pipeline ===\n")

    print("[1/4] Loading raw data...")
    products_df = _load_csv(RAW_PRODUCTS_PATH)
    reviews_df = _load_csv(RAW_REVIEWS_PATH)
    users_df = _load_csv(RAW_USERS_PATH)
    interactions_df = _load_csv(RAW_INTERACTIONS_PATH)
    print(f"  products={len(products_df)}, reviews={len(reviews_df)}, "
          f"users={len(users_df)}, interactions={len(interactions_df)}")

    print("\n[2/4] Validating raw data...")
    validation_report = run_all_validations(products_df, reviews_df, users_df, interactions_df)
    total_issues = sum(len(v) for v in validation_report.values())
    for dataset, issues in validation_report.items():
        if issues:
            print(f"  {dataset}: {len(issues)} issue type(s) found")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  {dataset}: no issues found")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_LOG_PATH, "w") as f:
        json.dump(validation_report, f, indent=2)
    print(f"  validation report saved -> {VALIDATION_LOG_PATH}")
    print(f"  (found {total_issues} issue type(s) total across all datasets — "
          f"this is expected on first run against synthetic test data; "
          f"cleaning step below will fix or drop the offending rows)")

    print("\n[3/4] Cleaning data...")
    products_df, products_log = clean_products(products_df)
    valid_product_ids = set(products_df["product_id"])

    reviews_df, reviews_log = clean_reviews(reviews_df, valid_product_ids)

    users_df, users_log = clean_users(users_df)
    valid_user_ids = set(users_df["user_id"])

    interactions_df, interactions_log = clean_interactions(interactions_df, valid_product_ids, valid_user_ids)

    for log in (products_log, reviews_log, users_log, interactions_log):
        for line in log:
            print(f"  - {line}")

    print("\n[4/4] Saving clean data...")
    products_df.to_csv(CLEAN_PRODUCTS_PATH, index=False)
    reviews_df.to_csv(CLEAN_REVIEWS_PATH, index=False)
    users_df.to_csv(CLEAN_USERS_PATH, index=False)
    interactions_df.to_csv(CLEAN_INTERACTIONS_PATH, index=False)
    print(f"  products  -> {CLEAN_PRODUCTS_PATH} ({len(products_df)} rows)")
    print(f"  reviews   -> {CLEAN_REVIEWS_PATH} ({len(reviews_df)} rows)")
    print(f"  users     -> {CLEAN_USERS_PATH} ({len(users_df)} rows)")
    print(f"  interactions -> {CLEAN_INTERACTIONS_PATH} ({len(interactions_df)} rows)")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
