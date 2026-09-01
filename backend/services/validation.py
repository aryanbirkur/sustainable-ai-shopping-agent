"""
backend/services/validation.py

Validation checks for the raw dataset, run BEFORE cleaning so we know
exactly what's wrong with the source data (cleaning then fixes what it can).

Module type: Rule-based. No ML/AI here — these are explicit business rules.

Each `validate_*` function takes a pandas DataFrame and returns a list of
human-readable issue strings (empty list = no issues found at that check).
Nothing raises by default; scripts/run_pipeline.py decides whether to abort.
"""

import pandas as pd

from config.settings import (
    VALID_RATING_RANGE,
    VALID_PERCENTAGE_RANGE,
    VALID_SCORE_RANGE,
)


def _check_range(df, column, low, high, issues):
    if column not in df.columns:
        return
    out_of_range = df[(df[column].notna()) & ((df[column] < low) | (df[column] > high))]
    if not out_of_range.empty:
        issues.append(
            f"{column}: {len(out_of_range)} row(s) outside valid range [{low}, {high}] "
            f"(e.g. row id(s): {list(out_of_range.index[:5])})"
        )


def _check_non_negative(df, column, issues):
    if column not in df.columns:
        return
    negative = df[(df[column].notna()) & (df[column] < 0)]
    if not negative.empty:
        issues.append(
            f"{column}: {len(negative)} row(s) with negative values "
            f"(e.g. row id(s): {list(negative.index[:5])})"
        )


def _check_required_not_missing(df, column, issues):
    if column not in df.columns:
        issues.append(f"{column}: column is missing entirely from the dataset")
        return
    missing = df[df[column].isna() | (df[column].astype(str).str.strip() == "")]
    if not missing.empty:
        issues.append(
            f"{column}: {len(missing)} row(s) missing a required value "
            f"(e.g. row id(s): {list(missing.index[:5])})"
        )


def _check_unique(df, column, issues):
    if column not in df.columns:
        return
    dupes = df[df.duplicated(subset=[column], keep=False)]
    if not dupes.empty:
        n_unique_dupe_values = dupes[column].nunique()
        issues.append(
            f"{column}: {len(dupes)} row(s) share a duplicate value across "
            f"{n_unique_dupe_values} distinct {column}(s)"
        )


def validate_products(df: pd.DataFrame) -> list[str]:
    issues = []
    _check_required_not_missing(df, "product_id", issues)
    _check_required_not_missing(df, "product_name", issues)
    _check_required_not_missing(df, "category", issues)
    _check_unique(df, "product_id", issues)
    _check_non_negative(df, "price", issues)
    _check_range(df, "rating", *VALID_RATING_RANGE, issues)
    _check_non_negative(df, "carbon_footprint_kg", issues)
    _check_non_negative(df, "water_usage_liters", issues)
    _check_range(df, "recycled_material_percentage", *VALID_PERCENTAGE_RANGE, issues)
    _check_range(df, "organic_material_percentage", *VALID_PERCENTAGE_RANGE, issues)
    _check_range(df, "recyclability_score", *VALID_SCORE_RANGE, issues)
    _check_range(df, "repairability_score", *VALID_SCORE_RANGE, issues)
    return issues


def validate_reviews(df: pd.DataFrame) -> list[str]:
    issues = []
    _check_required_not_missing(df, "review_id", issues)
    _check_required_not_missing(df, "product_id", issues)
    _check_unique(df, "review_id", issues)
    _check_range(df, "rating", *VALID_RATING_RANGE, issues)
    return issues


def validate_users(df: pd.DataFrame) -> list[str]:
    issues = []
    _check_required_not_missing(df, "user_id", issues)
    _check_unique(df, "user_id", issues)
    return issues


def validate_interactions(df: pd.DataFrame) -> list[str]:
    issues = []
    _check_required_not_missing(df, "user_id", issues)
    _check_required_not_missing(df, "product_id", issues)
    _check_required_not_missing(df, "interaction_type", issues)
    return issues


def run_all_validations(products_df, reviews_df, users_df, interactions_df) -> dict:
    """Runs every validator and returns a dict report, e.g. for JSON logging."""
    return {
        "products": validate_products(products_df),
        "reviews": validate_reviews(reviews_df),
        "users": validate_users(users_df),
        "interactions": validate_interactions(interactions_df),
    }
