"""
scripts/data_quality_report.py

Produces a summary data-quality report from the CLEANED dataset
(run this after scripts/run_pipeline.py). Saves JSON to
data/processed/data_quality_report.json.

Module type: Rule-based reporting script. No ML/AI here.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import (
    CLEAN_PRODUCTS_PATH, CLEAN_REVIEWS_PATH, CLEAN_USERS_PATH, CLEAN_INTERACTIONS_PATH,
    QUALITY_REPORT_PATH,
)

SUSTAINABILITY_FIELDS = [
    "recycled_material_percentage", "organic_material_percentage",
    "eco_certification", "carbon_footprint_kg", "water_usage_liters",
    "recyclability_score", "repairability_score", "product_lifetime_years",
]


def _missing_by_column(df: pd.DataFrame) -> dict:
    return {col: int(df[col].isna().sum()) for col in df.columns}


def _numeric_ranges(df: pd.DataFrame, columns: list[str]) -> dict:
    ranges = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            ranges[col] = {
                "min": float(df[col].min()) if df[col].notna().any() else None,
                "max": float(df[col].max()) if df[col].notna().any() else None,
                "mean": round(float(df[col].mean()), 2) if df[col].notna().any() else None,
            }
    return ranges


def build_report(products_df, reviews_df, users_df, interactions_df) -> dict:
    report = {
        "row_counts": {
            "products": len(products_df),
            "reviews": len(reviews_df),
            "users": len(users_df),
            "interactions": len(interactions_df),
        },
        "duplicate_counts": {
            "products_by_product_id": int(products_df.duplicated(subset=["product_id"]).sum()),
            "reviews_by_review_id": int(reviews_df.duplicated(subset=["review_id"]).sum()),
            "users_by_user_id": int(users_df.duplicated(subset=["user_id"]).sum()),
        },
        "missing_values": {
            "products": _missing_by_column(products_df),
            "reviews": _missing_by_column(reviews_df),
            "users": _missing_by_column(users_df),
            "interactions": _missing_by_column(interactions_df),
        },
        "numeric_ranges": {
            "products": _numeric_ranges(products_df, [
                "price", "rating", "review_count", "weight_grams",
                "recycled_material_percentage", "organic_material_percentage",
                "carbon_footprint_kg", "water_usage_liters",
                "recyclability_score", "repairability_score", "product_lifetime_years",
            ]),
        },
        "category_distribution": products_df["category"].value_counts().to_dict()
            if "category" in products_df.columns else {},
        "average_rating": round(float(products_df["rating"].mean()), 2)
            if "rating" in products_df.columns and products_df["rating"].notna().any() else None,
        "sustainability_field_availability_pct": {
            field: round(100 * products_df[field].notna().mean(), 1)
            for field in SUSTAINABILITY_FIELDS if field in products_df.columns
        },
        "data_source_breakdown": products_df["source"].value_counts().to_dict()
            if "source" in products_df.columns else {},
    }
    return report


def main():
    products_df = pd.read_csv(CLEAN_PRODUCTS_PATH)
    reviews_df = pd.read_csv(CLEAN_REVIEWS_PATH)
    users_df = pd.read_csv(CLEAN_USERS_PATH)
    interactions_df = pd.read_csv(CLEAN_INTERACTIONS_PATH)

    report = build_report(products_df, reviews_df, users_df, interactions_df)

    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUALITY_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Data quality report saved -> {QUALITY_REPORT_PATH}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
