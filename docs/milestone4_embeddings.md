# Milestone 4 — Embeddings & Semantic Search

## Model
`all-MiniLM-L6-v2` via `sentence-transformers`. 384-dim output, 22M
params, local/CPU, no API key. Chosen for being the standard lightweight
default for local semantic search as of 2026; `bge-small-en-v1.5` scores
marginally higher on retrieval MTEB but requires query-prefixing —
not worth the added complexity for this MVP.

## Embedding text construction
Per product, fields are joined with ". " in this fixed order:
`product_name, category, subcategory, material, description`.
Missing/NaN fields are skipped (never inserted as the string "nan").

## Vector store
Local, persistent ChromaDB at `data/chroma_store/`, one collection
named `products`, cosine distance space. Stores `product_id`, `category`,
`price`, `sustainability_score`, `product_name`, `brand` as metadata
alongside each vector, so later milestones can combine metadata
filtering (e.g. "under Rs.4,000") with semantic ranking.

## Re-running
`scripts/build_vector_index.py` reads `data/processed/products_scored.csv`
and upserts every product — safe to re-run any time the scored data
changes; it will not create duplicate entries.

## Honesty note
Per `docs/dataset_sourcing.md`: all product descriptions embedded here
are synthetic (`source="synthetic_v1"`). Semantic search results should
never be presented as reflecting real customer language or real reviews.
