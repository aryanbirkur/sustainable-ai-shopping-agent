# Data Dictionary

Legend: **Raw** = present in source data as-is. **Derived** = computed by
the pipeline or by later ML stages, not present in raw source data.

## Products (`data/raw/products.csv` → `data/processed/products_clean.csv`)

| Field | Type | Meaning | Example | Range/Allowed | Required | Raw/Derived |
|---|---|---|---|---|---|---|
| product_id | string | Unique product identifier | P0001 | — | Yes | Raw |
| product_name | string | Display name | EcoWeave Basic Tee | — | Yes | Raw |
| category | string | Top-level category | T-Shirts | one of 7 fixed categories | Yes | Raw |
| subcategory | string | Finer-grained type | Basic Tee | — | No | Raw |
| brand | string | Brand name | EcoWeave | — | No | Raw |
| price | float | Price in INR | 1899.0 | ≥ 0 | Yes | Raw |
| rating | float | Average rating | 4.2 | 0.0–5.0 | No | Raw (would be Derived if computed live from reviews) |
| review_count | int | Number of reviews | 11 | ≥ 0 | No | Derived (filled from reviews dataset at generation time) |
| description | text | Free-text product description | "Basic Tee made from organic cotton..." | — | Yes | Raw (this is the field embeddings will be built from in Milestone 5) |
| material | string | Primary material | Organic Cotton | one of 9 standardized values | No | Raw |
| color | string | Color | Black | — | No | Raw |
| weight_grams | float | Product weight | 220.5 | > 0 | No | Raw |
| manufacturing_location | string | Country of manufacture | India | — | No | Raw |
| recycled_material_percentage | float | % of material that is recycled | 45.0 | 0–100 | No | Raw (synthetic — see dataset_sourcing.md) |
| organic_material_percentage | float | % of material that is organic | 80.0 | 0–100 | No | Raw (synthetic) |
| eco_certification | string | Certification label | GOTS | GOTS / Fair Trade / OEKO-TEX / Global Recycled Standard / "No Certification" | No | Raw (synthetic) |
| carbon_footprint_kg | float | Estimated CO2e per unit | 2.35 | ≥ 0 | No | Raw (synthetic estimate, NOT a measured value) |
| water_usage_liters | float | Estimated water use per unit | 950.0 | ≥ 0 | No | Raw (synthetic estimate) |
| recyclability_score | float | How recyclable the product is | 0.72 | 0.0–1.0 | No | Raw (synthetic) |
| repairability_score | float | How repairable the product is | 0.55 | 0.0–1.0 | No | Raw (synthetic) |
| product_lifetime_years | float | Estimated usable lifetime | 3.2 | > 0 | No | Raw (synthetic estimate) |
| packaging_type | string | Packaging material | Recycled Cardboard | — | No | Raw |
| source | string | Data provenance marker | synthetic_v1 | "synthetic_v1" (real data would use a different label) | Yes | Raw — **critical field**, see dataset_sourcing.md |
| created_at | date | Date record was created | 2026-01-15 | — | No | Raw |

**Fields considered and deliberately excluded from the original brainstormed
list** (see Milestone 2 prompt): `size` (variant-level, would require a
separate size/SKU table — deferred past MVP), `energy_usage` (would
duplicate the signal already carried by `carbon_footprint_kg` without a
distinct real data source to separate them — revisit if a real data source
distinguishes them), `shipping_region` (affects logistics emissions, not
core product sustainability — deferred past MVP).

## Reviews (`data/raw/reviews.csv` → `data/processed/reviews_clean.csv`)

| Field | Type | Meaning | Example | Range | Required | Raw/Derived |
|---|---|---|---|---|---|---|
| review_id | string | Unique review identifier | R00001 | — | Yes | Raw |
| product_id | string | FK to Products | P0001 | must exist in Products | Yes | Raw |
| user_id | string | FK to Users | U0001 | must exist in Users | No | Raw |
| rating | int | Star rating given | 5 | 0–5 | Yes | Raw |
| review_text | text | Review content | "Great fit and durable." | — | No | Raw — source for review intelligence (Milestone 7) |
| review_date | date | When posted | 2026-02-01 | — | No | Raw |
| source | string | Data provenance marker | synthetic_v1 | — | Yes | Raw |

**Future (Milestone 7) derived fields** (not yet present): `sentiment_score`
(float, from a transformer sentiment model), `topics` (list of strings,
from topic extraction). Added as columns once that milestone is built —
not fabricated now.

## Users (`data/raw/users.csv` → `data/processed/users_clean.csv`)

| Field | Type | Meaning | Example | Range | Required | Raw/Derived |
|---|---|---|---|---|---|---|
| user_id | string | Unique user identifier | U0001 | — | Yes | Raw |
| age_group | string | Age bracket (not exact age) | 25-34 | one of 5 fixed brackets | No | Raw |
| preferred_categories | string | Stated category preference | Shoes | one of 7 fixed categories | No | Raw |
| preferred_materials | string | Stated material preference | Organic Cotton | one of 9 standardized values | No | Raw |
| budget_range | string | Stated budget bracket | 1500_4000 | one of 4 fixed brackets | No | Raw |
| sustainability_preference | string | Stated sustainability priority | high | low / medium / high | No | Raw |
| source | string | Data provenance marker | synthetic_v1 | — | Yes | Raw |

## Interactions (`data/raw/interactions.csv` → `data/processed/interactions_clean.csv`)

| Field | Type | Meaning | Example | Range | Required | Raw/Derived |
|---|---|---|---|---|---|---|
| user_id | string | FK to Users | U0001 | must exist in Users | Yes | Raw |
| product_id | string | FK to Products | P0001 | must exist in Products | Yes | Raw |
| interaction_type | string | Kind of interaction | click | view / click / wishlist / purchase / rating / dislike | Yes | Raw |
| interaction_value | int | Implicit-feedback strength | 2 | view=1, click=2, wishlist=3, rating=4, purchase=5, dislike=-2 | No | Derived (assigned by generator; a real system would assign this same way) |
| timestamp | datetime | When it happened | 2026-05-01T14:22:00 | — | No | Raw |
| source | string | Data provenance marker | synthetic_v1 | — | Yes | Raw |

**How interactions will be used later:**
- **Collaborative filtering (Milestone 8):** `interaction_value` becomes the
  implicit-feedback strength in a user-item matrix (e.g. for matrix
  factorization) — purchases and wishlists count far more than passive
  views, dislikes subtract.
- **Personalization (Milestone 9):** a user's positively-weighted
  interaction history is used to build/update their preference vector,
  which re-ranks future recommendations toward products similar to what
  they've engaged with positively.
- **Evaluation (Milestone 12):** held-out interactions (e.g. purchases) are
  used as ground truth for precision@k / recall@k style offline metrics.
