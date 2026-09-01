"""
backend/services/preprocessing.py

Cleaning and standardization pipeline. Runs AFTER validation has reported
what's wrong, and turns the raw dataset into the clean dataset saved under
data/processed/.

Module type: Rule-based / data engineering. No ML or AI here.

Design choice: cleaning is deliberately conservative.
- Rows with an unusable required field (e.g. missing product_id or a
  negative price we can't infer) are DROPPED, and the drop is logged.
- Rows with a fixable problem (out-of-range value, inconsistent casing,
  whitespace) are CORRECTED, and the correction is logged.
We never silently invent a "real" value for missing sustainability data —
if it's missing, it stays missing (NaN) and downstream sustainability
scoring is responsible for deciding how to handle unknown data.
"""

import numpy as np
import pandas as pd

from config.settings import (
    VALID_RATING_RANGE,
    VALID_PERCENTAGE_RANGE,
    VALID_SCORE_RANGE,
)

CATEGORY_STANDARD_MAP = {
    "t-shirts": "T-Shirts", "tshirts": "T-Shirts", "t shirts": "T-Shirts",
    "shirts": "Shirts",
    "jeans": "Jeans",
    "jackets": "Jackets",
    "shoes": "Shoes",
    "dresses": "Dresses",
    "bags": "Bags",
}

MATERIAL_STANDARD_MAP = {
    "organic cotton": "Organic Cotton",
    "recycled polyester": "Recycled Polyester",
    "hemp blend": "Hemp Blend",
    "tencel lyocell": "Tencel Lyocell",
    "recycled nylon": "Recycled Nylon",
    "linen": "Linen",
    "bamboo fabric": "Bamboo Fabric",
    "conventional cotton": "Conventional Cotton",
    "polyester": "Polyester",
}


def _standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in df.columns:
        return df
    key = df["category"].astype(str).str.strip().str.lower()
    df["category"] = key.map(CATEGORY_STANDARD_MAP).fillna(df["category"].astype(str).str.strip())
    return df


def _standardize_material(df: pd.DataFrame) -> pd.DataFrame:
    if "material" not in df.columns:
        return df
    key = df["material"].astype(str).str.strip().str.lower()
    df["material"] = key.map(MATERIAL_STANDARD_MAP).fillna(df["material"].astype(str).str.strip())
    return df


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clip_to_range(df: pd.DataFrame, column: str, low: float, high: float, log: list[str]) -> pd.DataFrame:
    if column not in df.columns:
        return df
    mask = df[column].notna() & ((df[column] < low) | (df[column] > high))
    n = int(mask.sum())
    if n:
        log.append(f"{column}: clipped {n} out-of-range value(s) to [{low}, {high}]")
        df.loc[mask, column] = df.loc[mask, column].clip(lower=low, upper=high)
    return df


def clean_products(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    before = len(df)

    # Drop rows missing a truly required field we cannot reconstruct
    required = ["product_id", "product_name", "category"]
    for col in required:
        if col in df.columns:
            missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            n_missing = int(missing_mask.sum())
            if n_missing:
                log.append(f"Dropped {n_missing} row(s) missing required field '{col}'")
                df = df[~missing_mask]

    # Remove exact duplicate products (same product_id)
    dupe_mask = df.duplicated(subset=["product_id"], keep="first")
    n_dupes = int(dupe_mask.sum())
    if n_dupes:
        log.append(f"Dropped {n_dupes} duplicate product_id row(s)")
        df = df[~dupe_mask]

    # Standardize text fields
    df = _standardize_category(df)
    df = _standardize_material(df)
    if "color" in df.columns:
        df["color"] = df["color"].astype(str).str.strip().str.title()
    if "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.strip()

    # Coerce numeric types
    numeric_cols = [
        "price", "rating", "review_count", "weight_grams",
        "recycled_material_percentage", "organic_material_percentage",
        "carbon_footprint_kg", "water_usage_liters",
        "recyclability_score", "repairability_score", "product_lifetime_years",
    ]
    df = _coerce_numeric(df, numeric_cols)

    # Negative price cannot be fixed by clipping (a negative price is not
    # "a slightly-too-low price", it's invalid data) -> drop those rows.
    if "price" in df.columns:
        bad_price_mask = df["price"].notna() & (df["price"] < 0)
        n_bad_price = int(bad_price_mask.sum())
        if n_bad_price:
            log.append(f"Dropped {n_bad_price} row(s) with negative price")
            df = df[~bad_price_mask]

    # Fixable out-of-range numeric values -> clip
    df = _clip_to_range(df, "rating", *VALID_RATING_RANGE, log)
    df = _clip_to_range(df, "recycled_material_percentage", *VALID_PERCENTAGE_RANGE, log)
    df = _clip_to_range(df, "organic_material_percentage", *VALID_PERCENTAGE_RANGE, log)
    df = _clip_to_range(df, "recyclability_score", *VALID_SCORE_RANGE, log)
    df = _clip_to_range(df, "repairability_score", *VALID_SCORE_RANGE, log)
    df = _clip_to_range(df, "water_usage_liters", 0, np.inf, log)
    df = _clip_to_range(df, "carbon_footprint_kg", 0, np.inf, log)

    df = df.reset_index(drop=True)
    log.append(f"Products: {before} -> {len(df)} rows after cleaning")
    return df, log


def clean_reviews(df: pd.DataFrame, valid_product_ids: set) -> tuple[pd.DataFrame, list[str]]:
    log = []
    before = len(df)

    # Drop reviews missing required fields
    for col in ["review_id", "product_id"]:
        if col in df.columns:
            missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            n = int(missing_mask.sum())
            if n:
                log.append(f"Dropped {n} review row(s) missing required field '{col}'")
                df = df[~missing_mask]

    # Drop reviews pointing at a product_id that doesn't exist after product cleaning
    orphan_mask = ~df["product_id"].isin(valid_product_ids)
    n_orphan = int(orphan_mask.sum())
    if n_orphan:
        log.append(f"Dropped {n_orphan} review row(s) referencing an unknown product_id")
        df = df[~orphan_mask]

    df = _coerce_numeric(df, ["rating"])
    df = _clip_to_range(df, "rating", *VALID_RATING_RANGE, log)

    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["review_id"], keep="first").reset_index(drop=True)
    log.append(f"Reviews: {before} -> {len(df)} rows after cleaning")
    return df, log


def clean_users(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    before = len(df)
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    missing_mask = df["user_id"].isna() | (df["user_id"].astype(str).str.strip() == "")
    n = int(missing_mask.sum())
    if n:
        log.append(f"Dropped {n} user row(s) missing user_id")
        df = df[~missing_mask]
    df = df.reset_index(drop=True)
    log.append(f"Users: {before} -> {len(df)} rows after cleaning")
    return df, log


def clean_interactions(df: pd.DataFrame, valid_product_ids: set, valid_user_ids: set) -> tuple[pd.DataFrame, list[str]]:
    log = []
    before = len(df)

    orphan_products = ~df["product_id"].isin(valid_product_ids)
    orphan_users = ~df["user_id"].isin(valid_user_ids)
    orphan_mask = orphan_products | orphan_users
    n_orphan = int(orphan_mask.sum())
    if n_orphan:
        log.append(f"Dropped {n_orphan} interaction row(s) referencing an unknown user_id or product_id")
        df = df[~orphan_mask]

    df = _coerce_numeric(df, ["interaction_value"])
    df = df.reset_index(drop=True)
    log.append(f"Interactions: {before} -> {len(df)} rows after cleaning")
    return df, log
