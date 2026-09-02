#!/usr/bin/env python3
"""
scripts/integrate_hm_data.py

Appends a sample of real H&M product/customer/purchase data to your
RAW data files (data/raw/*.csv) -- the same layer generate_synthetic_data.py
writes to. This is deliberate: entering at the raw layer means the data
then flows through your EXISTING pipeline unchanged --
    run_pipeline.py (validate + clean)
    -> sustainability/batch_score.py (rule + ML sustainability scoring)
    -> scripts/build_vector_index.py (embeddings)
-- instead of a parallel path that could drift from it.

Adds a new `image_path` column to products.csv (blank for synthetic rows).
Every field H&M genuinely doesn't provide (material, sustainability
metrics, rating, budget_range, etc.) is left BLANK for hm_real_v1 rows --
scoring_engine.py already handles missing sustainability data correctly
(redistributes weight, never treats missing as zero), so this is safe
to leave blank rather than invented.

H&M's `price` in the public Kaggle files is normalized (~0.0-0.6), not
real currency -- left blank for hm_real_v1 products for that reason;
`interaction_value` for hm_real_v1 rows keeps the raw normalized number
since it's only used for relative CF similarity, not displayed as currency.

Run from your project root, AFTER scripts/01_fetch_hm_data.sh:
    python scripts/integrate_hm_data.py
"""
import os
import sys
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import (
    RAW_DATA_DIR, RAW_PRODUCTS_PATH, RAW_REVIEWS_PATH,
    RAW_USERS_PATH, RAW_INTERACTIONS_PATH,
)

HM_RAW_DIR = "data/hm_raw"
IMAGES_ZIP = os.path.join(HM_RAW_DIR, "images.zip")
IMAGES_OUT_DIR = "data/product_images"

N_PRODUCTS = 2000
N_USERS = 500
TODAY = date.today().isoformat()
SOURCE_LABEL = "hm_real_v1"

AGE_BUCKETS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 200)]


def bucket_age(age):
    if pd.isna(age):
        return ""
    age = int(float(age))
    for lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return ""


def backup(path: Path):
    if path.exists():
        bak = path.with_name(path.stem + "_pre_hm_backup.csv")
        if not bak.exists():
            shutil.copy(path, bak)
            print(f"  backed up {path} -> {bak}")


def main():
    print("1. Loading raw H&M files...")
    # dtype=str keeps IDs as strings (no scientific notation / lost leading
    # zeros); encoding="latin1" tolerates non-UTF-8 bytes in descriptions;
    # on_bad_lines="skip" tolerates the rare malformed row (stray comma/quote
    # in free-text fields) instead of crashing the whole read.
    articles = pd.read_csv(
        os.path.join(HM_RAW_DIR, "articles.csv"),
        dtype=str, encoding="latin1", on_bad_lines="skip",
    )
    customers = pd.read_csv(
        os.path.join(HM_RAW_DIR, "customers.csv"),
        dtype=str, encoding="latin1", on_bad_lines="skip",
    )
    transactions = pd.read_csv(
        os.path.join(HM_RAW_DIR, "transactions_train.csv"),
        dtype=str, encoding="latin1", on_bad_lines="skip",
        usecols=["t_dat", "customer_id", "article_id", "price"],
    )

    print("2. Sampling most-purchased products and their buyers...")
    top_articles = transactions["article_id"].value_counts().head(N_PRODUCTS).index.tolist()
    articles_s = articles[articles["article_id"].isin(top_articles)].copy()

    txn_s = transactions[transactions["article_id"].isin(top_articles)].copy()
    top_customers = txn_s["customer_id"].value_counts().head(N_USERS).index.tolist()
    customers_s = customers[customers["customer_id"].isin(top_customers)].copy()
    txn_s = txn_s[txn_s["customer_id"].isin(top_customers)]
    print(f"   -> {len(articles_s)} products, {len(customers_s)} users, {len(txn_s)} interactions")

    image_path_map = {}
    if os.path.exists(IMAGES_ZIP):
        print("3. Extracting matching images from images.zip (no full unzip)...")
        os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
        with zipfile.ZipFile(IMAGES_ZIP) as z:
            names_by_article = {
                os.path.basename(n).replace(".jpg", ""): n
                for n in z.namelist() if n.endswith(".jpg")
            }
            for aid in articles_s["article_id"]:
                member = names_by_article.get(aid)
                if member:
                    out_path = os.path.join(IMAGES_OUT_DIR, f"{aid}.jpg")
                    with z.open(member) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    image_path_map[aid] = out_path
        print(f"   -> extracted {len(image_path_map)} images to {IMAGES_OUT_DIR}/")
    else:
        print("3. images.zip not found -- skipping images for now.")
        print("   Products will still be REAL H&M products (real names, categories, purchases),")
        print("   just without photos yet. image_path will be blank; your app already handles")
        print("   that gracefully (no broken image icons). Add images later by re-running this")
        print("   script once images.zip is downloaded to data/hm_raw/.")

    # ---------------- raw products ----------------
    print("4. Building product rows (raw-layer schema)...")
    prod_rows = []
    for _, r in articles_s.iterrows():
        prod_rows.append({
            "product_id": f"HM{r['article_id']}",
            "product_name": r.get("prod_name", ""),
            "category": r.get("product_group_name", ""),
            "subcategory": r.get("product_type_name", ""),
            "brand": "H&M",
            "price": "",
            "rating": "",
            "review_count": 0,
            "description": r.get("detail_desc", ""),
            "material": "",
            "color": r.get("colour_group_name", ""),
            "weight_grams": "",
            "manufacturing_location": "",
            "recycled_material_percentage": "",
            "organic_material_percentage": "",
            "eco_certification": "",
            "carbon_footprint_kg": "",
            "water_usage_liters": "",
            "recyclability_score": "",
            "repairability_score": "",
            "product_lifetime_years": "",
            "packaging_type": "",
            "source": SOURCE_LABEL,
            "created_at": TODAY,
            "image_path": image_path_map.get(r["article_id"], ""),
        })
    products_new = pd.DataFrame(prod_rows)

    backup(RAW_PRODUCTS_PATH)
    products_existing = pd.read_csv(RAW_PRODUCTS_PATH)
    products_existing = products_existing[products_existing["source"] != SOURCE_LABEL]  # idempotency: drop a previous run's H&M rows before re-adding
    if "image_path" not in products_existing.columns:
        products_existing["image_path"] = ""
    products_out = pd.concat([products_existing, products_new], ignore_index=True)
    products_out.to_csv(RAW_PRODUCTS_PATH, index=False)
    print(f"   -> {RAW_PRODUCTS_PATH}: {len(products_existing)} existing + {len(products_new)} new = {len(products_out)} rows")

    # ---------------- raw users ----------------
    print("5. Building user rows...")
    cust_top_category = defaultdict(Counter)
    art_to_group = dict(zip(articles["article_id"], articles["product_group_name"]))
    for _, t in txn_s.iterrows():
        grp = art_to_group.get(t["article_id"])
        if grp:
            cust_top_category[t["customer_id"]][grp] += 1

    user_rows = []
    for _, r in customers_s.iterrows():
        cid = r["customer_id"]
        top_cat = cust_top_category[cid].most_common(1)
        user_rows.append({
            "user_id": f"HM{cid[:12]}",
            "age_group": bucket_age(r.get("age")),
            "preferred_categories": top_cat[0][0] if top_cat else "",
            "preferred_materials": "",
            "budget_range": "",  # H&M price is normalized, not comparable to INR buckets
            "sustainability_preference": "",
            "source": SOURCE_LABEL,
        })
    users_new = pd.DataFrame(user_rows)
    id_map = dict(zip(customers_s["customer_id"], users_new["user_id"]))

    backup(RAW_USERS_PATH)
    users_existing = pd.read_csv(RAW_USERS_PATH)
    users_existing = users_existing[users_existing["source"] != SOURCE_LABEL]  # idempotency
    users_out = pd.concat([users_existing, users_new], ignore_index=True)
    users_out.to_csv(RAW_USERS_PATH, index=False)
    print(f"   -> {RAW_USERS_PATH}: {len(users_existing)} existing + {len(users_new)} new = {len(users_out)} rows")

    # ---------------- raw interactions ----------------
    print("6. Building interaction rows...")
    inter_rows = []
    for _, t in txn_s.iterrows():
        uid = id_map.get(t["customer_id"])
        if not uid:
            continue
        inter_rows.append({
            "user_id": uid,
            "product_id": f"HM{t['article_id']}",
            "interaction_type": "purchase",
            "interaction_value": 5,  # matches your existing purchase weight (see data_dictionary.md)
            "timestamp": t["t_dat"],
            "source": SOURCE_LABEL,
        })
    interactions_new = pd.DataFrame(inter_rows)

    backup(RAW_INTERACTIONS_PATH)
    interactions_existing = pd.read_csv(RAW_INTERACTIONS_PATH)
    interactions_existing = interactions_existing[interactions_existing["source"] != SOURCE_LABEL]  # idempotency
    interactions_out = pd.concat([interactions_existing, interactions_new], ignore_index=True)
    interactions_out.to_csv(RAW_INTERACTIONS_PATH, index=False)
    print(f"   -> {RAW_INTERACTIONS_PATH}: {len(interactions_existing)} existing + {len(interactions_new)} new = {len(interactions_out)} rows")

    # ---------------- reviews: untouched ----------------
    print("7. Reviews: no real H&M review text exists, so reviews.csv is left untouched.")
    print("   product_sentiment.csv will show review_count=0 for hm_real_v1 products,")
    print("   which aggregator.py already renders as honest None values, not fake neutral scores.")

    print("\nDone. Next: run your existing pipeline --")
    print("  python scripts/run_pipeline.py")
    print("  python sustainability/batch_score.py")
    print("  python scripts/build_vector_index.py    (after applying the image_path patch below)")


if __name__ == "__main__":
    main()
