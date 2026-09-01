"""
tests/test_data_pipeline.py

Basic tests for validation.py and preprocessing.py.
Run with:  pytest tests/test_data_pipeline.py -v

These tests build tiny in-memory DataFrames rather than depending on the
generated dataset, so they run the same way regardless of random seed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backend.services.validation import (
    validate_products, validate_reviews, validate_users, validate_interactions,
)
from backend.services.preprocessing import clean_products, clean_reviews


def make_product_row(**overrides):
    row = {
        "product_id": "P0001",
        "product_name": "Test Tee",
        "category": "T-Shirts",
        "subcategory": "Basic Tee",
        "brand": "EcoWeave",
        "price": 999.0,
        "rating": 4.2,
        "review_count": 3,
        "description": "A test product.",
        "material": "Organic Cotton",
        "color": "Black",
        "weight_grams": 200.0,
        "manufacturing_location": "India",
        "recycled_material_percentage": 20.0,
        "organic_material_percentage": 80.0,
        "eco_certification": "GOTS",
        "carbon_footprint_kg": 2.5,
        "water_usage_liters": 900.0,
        "recyclability_score": 0.7,
        "repairability_score": 0.6,
        "product_lifetime_years": 3.0,
        "packaging_type": "Recycled Cardboard",
        "source": "synthetic_v1",
        "created_at": "2026-01-01",
    }
    row.update(overrides)
    return row


# --- Validation tests -------------------------------------------------

def test_valid_product_row_has_no_issues():
    df = pd.DataFrame([make_product_row()])
    issues = validate_products(df)
    assert issues == []


def test_negative_price_is_flagged():
    df = pd.DataFrame([make_product_row(price=-100.0)])
    issues = validate_products(df)
    assert any("price" in issue for issue in issues)


def test_invalid_rating_is_flagged():
    df = pd.DataFrame([make_product_row(rating=9.9)])
    issues = validate_products(df)
    assert any("rating" in issue for issue in issues)


def test_invalid_recycled_percentage_is_flagged():
    df = pd.DataFrame([make_product_row(recycled_material_percentage=150.0)])
    issues = validate_products(df)
    assert any("recycled_material_percentage" in issue for issue in issues)


def test_missing_required_field_is_flagged():
    df = pd.DataFrame([make_product_row(product_name="")])
    issues = validate_products(df)
    assert any("product_name" in issue for issue in issues)


def test_duplicate_product_id_is_flagged():
    df = pd.DataFrame([make_product_row(), make_product_row()])
    issues = validate_products(df)
    assert any("product_id" in issue for issue in issues)


def test_successful_loading_of_valid_data_passes_all_checks():
    products = pd.DataFrame([make_product_row(product_id="P0001"), make_product_row(product_id="P0002")])
    reviews = pd.DataFrame([{
        "review_id": "R0001", "product_id": "P0001", "user_id": "U0001",
        "rating": 5, "review_text": "Great!", "review_date": "2026-01-01", "source": "synthetic_v1",
    }])
    users = pd.DataFrame([{"user_id": "U0001", "age_group": "25-34"}])
    interactions = pd.DataFrame([{
        "user_id": "U0001", "product_id": "P0001", "interaction_type": "view",
        "interaction_value": 1, "timestamp": "2026-01-01T00:00:00",
    }])

    assert validate_products(products) == []
    assert validate_reviews(reviews) == []
    assert validate_users(users) == []
    assert validate_interactions(interactions) == []


# --- Cleaning tests -----------------------------------------------------

def test_clean_products_drops_negative_price_row():
    df = pd.DataFrame([make_product_row(product_id="P0001"), make_product_row(product_id="P0002", price=-50.0)])
    cleaned, log = clean_products(df)
    assert len(cleaned) == 1
    assert (cleaned["price"] >= 0).all()


def test_clean_products_clips_out_of_range_rating():
    df = pd.DataFrame([make_product_row(rating=7.0)])
    cleaned, log = clean_products(df)
    assert cleaned.loc[0, "rating"] == 5.0


def test_clean_products_removes_exact_duplicates():
    df = pd.DataFrame([make_product_row(product_id="P0001"), make_product_row(product_id="P0001")])
    cleaned, log = clean_products(df)
    assert len(cleaned) == 1


def test_clean_products_standardizes_category_casing():
    df = pd.DataFrame([make_product_row(category="t-shirts")])
    cleaned, log = clean_products(df)
    assert cleaned.loc[0, "category"] == "T-Shirts"


def test_clean_reviews_drops_orphan_product_id():
    products = pd.DataFrame([make_product_row(product_id="P0001")])
    reviews = pd.DataFrame([
        {"review_id": "R0001", "product_id": "P0001", "user_id": "U0001",
         "rating": 5, "review_text": "Nice", "review_date": "2026-01-01", "source": "synthetic_v1"},
        {"review_id": "R0002", "product_id": "P9999", "user_id": "U0002",
         "rating": 4, "review_text": "N/A", "review_date": "2026-01-01", "source": "synthetic_v1"},
    ])
    cleaned, log = clean_reviews(reviews, valid_product_ids={"P0001"})
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["review_id"] == "R0001"
