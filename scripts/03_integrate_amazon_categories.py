"""
scripts/03_integrate_amazon_categories.py

Integrates REAL products from the Amazon Reviews 2023 dataset
(McAuley Lab, Hugging Face: McAuley-Lab/Amazon-Reviews-2023) into
data/raw/products.csv, at the raw layer, so they flow through the
existing validation/cleaning pipeline unchanged -- same pattern as
scripts/02_integrate_hm_data.py.

Categories added: Electronics, Cell_Phones_and_Accessories,
Home_and_Kitchen, Beauty_and_Personal_Care.

HONESTY NOTE: this dataset is a static snapshot collected at a fixed
point in time. It does NOT contain current-year new releases. Fields
we CAN get real values for: product_name, category, price (when
Amazon provides one), rating, review_count, image_path (hosted CDN
URL). Fields left genuinely blank, same as H&M: material,
sustainability inputs, weight_grams, manufacturing_location,
packaging_type.

Idempotent by design: re-running this script deletes any existing
rows with source == SOURCE_LABEL before re-appending.

Requires: pip install datasets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import load_dataset

from config.settings import RAW_PRODUCTS_PATH

SOURCE_LABEL = "amazon_real_v1"

CATEGORY_CONFIGS = {
    "Electronics": "Electronics",
    "Cell_Phones_and_Accessories": "Cell Phones & Accessories",
    "Home_and_Kitchen": "Home & Kitchen",
    "Beauty_and_Personal_Care": "Beauty & Personal Care",
}

TOP_N_PER_CATEGORY = 500


def fetch_category_products(hf_config: str, our_category: str, top_n: int) -> pd.DataFrame:
    print(f"  streaming meta_{hf_config} ...")
    # McAuley-Lab/Amazon-Reviews-2023 has been converted to Parquet on the
    # Hub; the config-name/loading-script path is deprecated and rejected
    # by datasets>=4.0. Load the Parquet files directly instead.
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw_meta_{hf_config}/*.parquet",
        split="train",
        streaming=True,
    )

    rows = []
    for item in ds:
        title = (item.get("title") or "").strip()
        if not title:
            continue  # never fabricate a name for a nameless row

        images = item.get("images") or {}
        image_urls = images.get("large") or images.get("hi_res") or images.get("thumb") or []
        image_path = image_urls[0] if image_urls else None

        price_raw = item.get("price")
        try:
            price = float(price_raw) if price_raw not in (None, "", "None") else None
        except (TypeError, ValueError):
            price = None  # never fabricate a numeric price from unparseable text

        rows.append({
            "product_id": item.get("parent_asin"),
            "product_name": title,
            "category": our_category,
            "subcategory": None,
            "brand": item.get("store"),
            "price": price,
            "rating": item.get("average_rating"),
            "review_count": item.get("rating_number"),
            "description": " ".join(item.get("description") or [])[:500] or None,
            "material": None,
            "color": None,
            "weight_grams": None,
            "manufacturing_location": None,
            "recycled_material_percentage": None,
            "organic_material_percentage": None,
            "eco_certification": None,
            "carbon_footprint_kg": None,
            "water_usage_liters": None,
            "recyclability_score": None,
            "repairability_score": None,
            "product_lifetime_years": None,
            "packaging_type": None,
            "image_path": image_path,
            "source": SOURCE_LABEL,
        })

        if len(rows) >= top_n * 3:
            break

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["product_id"])
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df = df.sort_values("review_count", ascending=False).drop_duplicates(subset=["product_id"])
    df = df.head(top_n)
    print(f"    -> kept {len(df)} products for {our_category}")
    return df


def main():
    print(f"Integrating Amazon Reviews 2023 categories (source='{SOURCE_LABEL}')")
    print("NOTE: this is a historical snapshot dataset, not a live feed -- "
          "it will not contain this-year new product releases. See module "
          "docstring for what's real vs blank in each row.\n")

    all_new = []
    skipped = []
    for hf_config, our_category in CATEGORY_CONFIGS.items():
        try:
            df = fetch_category_products(hf_config, our_category, TOP_N_PER_CATEGORY)
            all_new.append(df)
        except ValueError as e:
            print(f"  SKIPPED {our_category}: not available as Parquet yet ({e})")
            skipped.append(our_category)

    if not all_new:
        print("No categories were fetchable -- nothing to write. Try again later.")
        return

    new_products = pd.concat(all_new, ignore_index=True)
    if skipped:
        print(f"\nNote: skipped {len(skipped)} categorie(s) not yet available: {skipped}")
        print("Re-run this script later to add them once the Hub finishes converting them.")

    existing = pd.read_csv(RAW_PRODUCTS_PATH)

    before = len(existing)
    existing = existing[existing.get("source") != SOURCE_LABEL]
    dropped = before - len(existing)
    if dropped:
        print(f"Removed {dropped} existing '{SOURCE_LABEL}' row(s) before re-appending (idempotent re-run).")

    for col in new_products.columns:
        if col not in existing.columns:
            existing[col] = None
    for col in existing.columns:
        if col not in new_products.columns:
            new_products[col] = None

    combined = pd.concat([existing, new_products[existing.columns]], ignore_index=True)
    combined.to_csv(RAW_PRODUCTS_PATH, index=False)
    print(f"\nWrote {len(combined)} total product rows -> {RAW_PRODUCTS_PATH} "
          f"({len(new_products)} new '{SOURCE_LABEL}' rows)")

    print("\nDone. Next steps (run in order):")
    print("  python scripts/run_pipeline.py")
    print("  python scripts/train_sustainability_model.py")
    print("  python scripts/build_vector_index.py")
    print("  pytest -v")


if __name__ == "__main__":
    main()
