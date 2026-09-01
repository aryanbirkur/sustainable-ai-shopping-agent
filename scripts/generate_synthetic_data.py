"""
scripts/generate_synthetic_data.py

Generates a clearly-labeled SYNTHETIC/DEMO dataset for the Sustainable AI
Shopping Agent MVP: products, reviews, users, and interactions.

Why synthetic data (see docs/dataset_sourcing.md for the full writeup):
No free, ready-to-use public dataset combines realistic fashion product
catalogs with per-product environmental attributes (carbon footprint,
water usage, recycled-material %, etc.) at the field granularity this
project needs. Real per-product environmental data is normally either
proprietary (brand LCA reports) or estimated by specialized paid tools.
So for MVP development we generate synthetic data that is:
  - structurally realistic (plausible ranges, correlated fields)
  - clearly labeled as synthetic in every row (`source` column)
  - never presented as measured real-world environmental fact

Every generated product's `source` column is set to
config.settings.DATA_SOURCE_LABEL so downstream code (and humans) can
always tell synthetic rows apart from any real data imported later.

Module type: Rule-based / procedural generation (NOT machine learning,
NOT LLM). No function here should be mistaken for an AI component.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import random
from datetime import datetime, timedelta

from config.settings import (
    RAW_DATA_DIR,
    RAW_PRODUCTS_PATH,
    RAW_REVIEWS_PATH,
    RAW_USERS_PATH,
    RAW_INTERACTIONS_PATH,
    DATA_SOURCE_LABEL,
    RANDOM_SEED,
    NUM_PRODUCTS,
    NUM_USERS,
    MIN_REVIEWS_PER_PRODUCT,
    MAX_REVIEWS_PER_PRODUCT,
    MIN_INTERACTIONS_PER_USER,
    MAX_INTERACTIONS_PER_USER,
)

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Vocabulary tables (kept simple and readable — no external dependency
# like Faker, so requirements.txt stays minimal for this milestone)
# ---------------------------------------------------------------------------

CATEGORY_SUBCATEGORIES = {
    "T-Shirts": ["Basic Tee", "Graphic Tee", "Oversized Tee"],
    "Shirts": ["Formal Shirt", "Casual Shirt", "Linen Shirt"],
    "Jeans": ["Slim Fit", "Straight Fit", "Wide Leg"],
    "Jackets": ["Denim Jacket", "Windbreaker", "Puffer Jacket"],
    "Shoes": ["Running Shoes", "Casual Sneakers", "Sandals"],
    "Dresses": ["Casual Dress", "Maxi Dress", "Wrap Dress"],
    "Bags": ["Tote Bag", "Backpack", "Sling Bag"],
}

BRANDS = [
    "EcoWeave", "GreenThread", "TerraFit", "PureLoom", "Reforma",
    "CircleWear", "Rootwear", "Bambu & Co", "Renu Apparel", "Solstice Studio",
]

MATERIALS = [
    "Organic Cotton", "Recycled Polyester", "Hemp Blend", "Tencel Lyocell",
    "Recycled Nylon", "Linen", "Bamboo Fabric", "Conventional Cotton", "Polyester",
]

COLORS = ["Black", "White", "Olive", "Navy", "Beige", "Terracotta", "Charcoal", "Sage Green"]

MANUFACTURING_LOCATIONS = ["India", "Bangladesh", "Vietnam", "Portugal", "Turkey", "China"]

ECO_CERTIFICATIONS = [
    "GOTS", "Fair Trade", "OEKO-TEX", "Global Recycled Standard",
    "No Certification", "No Certification", "No Certification",
]
# "No Certification" weighted higher on purpose: certification is realistically
# the exception, not the norm. NOTE: deliberately not using the string "None" —
# pandas' read_csv silently parses a literal "None" value as NaN/missing,
# which would make every uncertified product look like it has *unknown*
# certification status instead of *explicitly no* certification. Those are
# different facts and the sustainability engine (Milestone 3) needs to tell
# them apart.

PACKAGING_TYPES = ["Recycled Cardboard", "Plastic Poly Bag", "Compostable Mailer", "Minimal/No Packaging"]

REVIEW_SNIPPETS_POSITIVE = [
    "Great fit and the fabric feels durable.",
    "Good quality for the price, would buy again.",
    "Loved the material, very comfortable for daily wear.",
    "True to size and holds up well after multiple washes.",
    "Nice sustainable option that doesn't compromise on comfort.",
]

REVIEW_SNIPPETS_NEUTRAL = [
    "It's okay, nothing special but does the job.",
    "Decent product, delivery took a while.",
    "Average quality, expected a bit more for the price.",
]

REVIEW_SNIPPETS_NEGATIVE = [
    "Fabric felt thinner than expected.",
    "Started wearing out after a few weeks.",
    "Sizing ran small, had to return it.",
    "Not worth the price, quality was disappointing.",
]

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
BUDGET_RANGES = ["under_1500", "1500_4000", "4000_8000", "8000_plus"]
SUSTAINABILITY_PREFERENCES = ["low", "medium", "high"]
INTERACTION_TYPES = ["view", "click", "wishlist", "purchase", "rating", "dislike"]


def _rand_float(low, high, decimals=1):
    return round(random.uniform(low, high), decimals)


def generate_products(n=NUM_PRODUCTS):
    """Generate a synthetic product catalog. Returns a list of dicts."""
    products = []
    for i in range(1, n + 1):
        category = random.choice(list(CATEGORY_SUBCATEGORIES.keys()))
        subcategory = random.choice(CATEGORY_SUBCATEGORIES[category])
        brand = random.choice(BRANDS)
        material = random.choice(MATERIALS)
        is_recycled_material = "Recycled" in material
        is_organic_material = "Organic" in material

        # Correlate sustainability-ish fields loosely with material choice,
        # so the dataset isn't pure random noise (useful for later EDA/modeling).
        recycled_pct = _rand_float(40, 95) if is_recycled_material else _rand_float(0, 15)
        organic_pct = _rand_float(60, 100) if is_organic_material else _rand_float(0, 10)
        carbon_footprint_kg = round(random.uniform(1.5, 3.5) if (is_recycled_material or is_organic_material)
                                     else random.uniform(3.0, 9.0), 2)
        water_usage_liters = round(random.uniform(300, 1200) if (is_recycled_material or is_organic_material)
                                    else random.uniform(1500, 4500), 1)
        recyclability_score = round(random.uniform(0.5, 0.95) if is_recycled_material
                                     else random.uniform(0.1, 0.6), 2)
        repairability_score = round(random.uniform(0.3, 0.9), 2)
        product_lifetime_years = round(random.uniform(1.0, 6.0), 1)

        price = round(random.uniform(499, 7999), 2)
        rating = round(random.uniform(2.5, 5.0), 1)

        product = {
            "product_id": f"P{i:04d}",
            "product_name": f"{brand} {subcategory}",
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "price": price,
            "rating": rating,
            "review_count": 0,  # filled in after reviews are generated
            "description": (
                f"{subcategory} made from {material.lower()}, designed for everyday wear. "
                f"Manufactured in {random.choice(MANUFACTURING_LOCATIONS)}."
            ),
            "material": material,
            "color": random.choice(COLORS),
            "weight_grams": round(random.uniform(120, 900), 1),
            "manufacturing_location": random.choice(MANUFACTURING_LOCATIONS),
            "recycled_material_percentage": recycled_pct,
            "organic_material_percentage": organic_pct,
            "eco_certification": random.choice(ECO_CERTIFICATIONS),
            "carbon_footprint_kg": carbon_footprint_kg,
            "water_usage_liters": water_usage_liters,
            "recyclability_score": recyclability_score,
            "repairability_score": repairability_score,
            "product_lifetime_years": product_lifetime_years,
            "packaging_type": random.choice(PACKAGING_TYPES),
            "source": DATA_SOURCE_LABEL,
            "created_at": (datetime.now() - timedelta(days=random.randint(0, 400))).date().isoformat(),
        }
        products.append(product)

    # Inject a small number of deliberately messy rows so the cleaning
    # pipeline in Milestone 2 Part 4/5 has real problems to catch.
    if len(products) >= 10:
        products[3]["price"] = -50.0                       # invalid: negative price
        products[7]["rating"] = 7.2                          # invalid: out of range
        products[11]["recycled_material_percentage"] = 140.0  # invalid: out of range
        products[15]["material"] = "  organic cotton "       # needs standardization
        products[15]["category"] = "t-shirts"                 # needs standardization
        products[19]["product_name"] = ""                     # missing required field
        products[23] = dict(products[2])                      # exact duplicate product_id/content
        products[23]["product_id"] = products[2]["product_id"]

    return products


def generate_reviews(products):
    """Generate synthetic reviews tied to product_ids. Also back-fills review_count."""
    reviews = []
    review_id = 1
    for product in products:
        n_reviews = random.randint(MIN_REVIEWS_PER_PRODUCT, MAX_REVIEWS_PER_PRODUCT)
        for _ in range(n_reviews):
            rating = random.randint(1, 5)
            if rating >= 4:
                text = random.choice(REVIEW_SNIPPETS_POSITIVE)
            elif rating == 3:
                text = random.choice(REVIEW_SNIPPETS_NEUTRAL)
            else:
                text = random.choice(REVIEW_SNIPPETS_NEGATIVE)

            reviews.append({
                "review_id": f"R{review_id:05d}",
                "product_id": product["product_id"],
                "user_id": f"U{random.randint(1, NUM_USERS):04d}",
                "rating": rating,
                "review_text": text,
                "review_date": (datetime.now() - timedelta(days=random.randint(0, 365))).date().isoformat(),
                "source": DATA_SOURCE_LABEL,
            })
            review_id += 1
        product["review_count"] = n_reviews

    # Inject a couple of messy review rows for the pipeline to catch.
    if reviews:
        reviews[0]["rating"] = 9          # invalid: out of range
        reviews[min(4, len(reviews) - 1)]["product_id"] = ""  # missing required field

    return reviews


def generate_users(n=NUM_USERS):
    users = []
    for i in range(1, n + 1):
        users.append({
            "user_id": f"U{i:04d}",
            "age_group": random.choice(AGE_GROUPS),
            "preferred_categories": random.choice(list(CATEGORY_SUBCATEGORIES.keys())),
            "preferred_materials": random.choice(MATERIALS),
            "budget_range": random.choice(BUDGET_RANGES),
            "sustainability_preference": random.choice(SUSTAINABILITY_PREFERENCES),
            "source": DATA_SOURCE_LABEL,
        })
    return users


def generate_interactions(users, products):
    interactions = []
    product_ids = [p["product_id"] for p in products]
    for user in users:
        n_interactions = random.randint(MIN_INTERACTIONS_PER_USER, MAX_INTERACTIONS_PER_USER)
        for _ in range(n_interactions):
            interaction_type = random.choice(INTERACTION_TYPES)
            # interaction_value gives a numeric strength signal used later for
            # weighting implicit feedback in collaborative filtering (see docs/data_dictionary.md)
            value_by_type = {
                "view": 1, "click": 2, "wishlist": 3,
                "purchase": 5, "rating": 4, "dislike": -2,
            }
            interactions.append({
                "user_id": user["user_id"],
                "product_id": random.choice(product_ids),
                "interaction_type": interaction_type,
                "interaction_value": value_by_type[interaction_type],
                "timestamp": (datetime.now() - timedelta(
                    days=random.randint(0, 180), hours=random.randint(0, 23)
                )).isoformat(),
                "source": DATA_SOURCE_LABEL,
            })
    return interactions


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def main():
    print(f"Generating synthetic dataset (seed={RANDOM_SEED}, source label='{DATA_SOURCE_LABEL}')")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    products = generate_products()
    reviews = generate_reviews(products)   # also fills review_count on products
    users = generate_users()
    interactions = generate_interactions(users, products)

    _write_csv(RAW_PRODUCTS_PATH, products, list(products[0].keys()))
    _write_csv(RAW_REVIEWS_PATH, reviews, list(reviews[0].keys()))
    _write_csv(RAW_USERS_PATH, users, list(users[0].keys()))
    _write_csv(RAW_INTERACTIONS_PATH, interactions, list(interactions[0].keys()))

    print("Done. Raw synthetic data written to data/raw/.")
    print("NOTE: This data is SYNTHETIC (source='synthetic_v1'), not real-world "
          "environmental measurements. See docs/dataset_sourcing.md.")


if __name__ == "__main__":
    main()
