# Dataset Sourcing — Real Data vs. Synthetic Data

## Decision: synthetic/demo data for MVP development, clearly labeled

### What I checked

I looked for a public dataset that combines a realistic fashion product
catalog with **per-product environmental attributes** (carbon footprint,
water usage, recycled-material %, certifications, etc.) at the field level
this project needs.

What exists publicly:
- Several Kaggle datasets tagged "sustainable fashion" (e.g. *Sustainable
  Fashion: Eco-Friendly Trends*) exist, but based on their own descriptions
  and community discussion they are **themselves synthetically generated**
  for demo/analysis purposes — not measured real-world data. Using one
  wouldn't get us closer to "real" data, just someone else's synthetic data
  under a different license.
- Large real fashion catalogs (e.g. retailer product feeds, Amazon product
  datasets) exist and are real, but **do not include environmental/
  sustainability attributes** — brands rarely publish per-SKU carbon
  footprint or water usage figures publicly, and where they do (some
  sustainability reports), it's at the brand or product-line level, not
  per individual SKU, and not in a structured, bulk-downloadable format.
- Genuine per-product Life Cycle Assessment (LCA) data (real carbon/water
  figures) is generally either proprietary (paid LCA databases/tools) or
  published as PDF sustainability reports at brand level — not something
  we can legally bulk-scrape or redistribute for free.

**Conclusion:** there is no free, ready-to-use, properly licensed dataset
that gives us real per-product sustainability figures for fashion items.
This isn't a shortcut we're taking — it reflects that this data is
genuinely hard to get for free anywhere, including for real companies
building similar tools.

### What we're doing instead

We generate a **clearly labeled synthetic dataset** for development:

- Every row (products, reviews, users, interactions) has a `source` column
  set to `synthetic_v1` (see `config/settings.DATA_SOURCE_LABEL`).
- Product names, brands, and descriptions are template-generated, not real
  brand data.
- Sustainability figures (carbon footprint, water usage, recyclability,
  etc.) are **randomly generated within plausible ranges**, loosely
  correlated with material type (e.g. recycled/organic materials get lower
  carbon/water numbers) so the data has realistic *structure* for building
  and testing models — but the actual numbers are **not derived from any
  real measurement** and must never be presented to an end user as fact.
- Prices are in INR (₹), in ranges typical for the categories chosen.

### Fields we could NOT get from any public source (all of them, currently)

Every sustainability-related field in the schema (`carbon_footprint_kg`,
`water_usage_liters`, `recycled_material_percentage`,
`organic_material_percentage`, `eco_certification`, `recyclability_score`,
`repairability_score`, `product_lifetime_years`) is synthetic for this
milestone, for the reasons above.

### Path to real data later (not part of this milestone)

If/when this project moves beyond MVP:
1. Real product catalogs (e.g. via a retailer partnership, public product
   feed, or manual curation of a small real dataset) could replace
   `product_name`, `brand`, `price`, `description`, `category`.
2. Real sustainability signals could be approximated by:
   - Mapping stated materials/certifications (if a real catalog includes
     them) to published *category-average* impact factors from public
     LCA studies (e.g. Higg Index-style category benchmarks) rather than
     claiming per-product measured values.
   - Clearly labeling these as "estimated from category averages," never
     as measured fact — same honesty principle as today, just with a
     better estimation method.
3. Real reviews could come from a real e-commerce review export (subject
   to its own license terms) instead of generated text.

Until any of that happens, **every environmental claim this system makes
is either clearly labeled synthetic (now) or clearly labeled as an
estimate (later) — never presented as a verified real-world measurement.**
