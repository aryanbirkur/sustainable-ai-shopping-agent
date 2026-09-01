"""
Category extraction from free-text queries.

Type: Rule-based

Matches only against the 7 confirmed real categories in this catalog
(Shoes, Bags, T-Shirts, Jeans, Jackets, Shirts, Dresses). Uses exact
(case-insensitive) matching on the category name and a small set of
known singular/plural variants -- deliberately NOT fuzzy string
matching, since fuzzy matching risks silently matching the wrong
category. An unrecognized category-like term (e.g. "electronics",
"headphones") is honestly flagged as unparsed, never guessed.
"""

import re
from typing import Dict, List, Optional

# Real categories confirmed in products_clean.csv, plus their known
# singular/plural/spacing variants a user might naturally type.
CATEGORY_VARIANTS = {
    "Shoes": ["shoes", "shoe"],
    "Bags": ["bags", "bag"],
    "T-Shirts": ["t-shirts", "t-shirt", "tshirts", "tshirt", "t shirts", "t shirt"],
    "Jeans": ["jeans", "jean"],
    "Jackets": ["jackets", "jacket"],
    "Shirts": ["shirts", "shirt"],
    "Dresses": ["dresses", "dress"],
}

# Common out-of-catalog terms worth explicitly recognizing so the
# "flagged, not silently guessed" behavior is visible and testable.
KNOWN_OUT_OF_CATALOG_TERMS = [
    "electronics", "headphones", "laptop", "phone", "watch", "furniture",
]


def extract_category(query: str) -> Dict[str, Optional[object]]:
    """
    Type: Rule-based

    Returns:
        {"category": str | None, "unparsed_confidence_notes": list[str]}
    category is one of the 7 confirmed catalog category strings, or
    None if no match is found (never a guessed/closest category).
    """
    query_lower = query.lower()
    notes: List[str] = []

    for canonical_name, variants in CATEGORY_VARIANTS.items():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", query_lower):
                return {"category": canonical_name, "unparsed_confidence_notes": notes}

    for term in KNOWN_OUT_OF_CATALOG_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", query_lower):
            notes.append(
                f"Query mentions '{term}', which is not a category in this "
                "catalog (catalog is 100% apparel) -- category not extracted."
            )
            return {"category": None, "unparsed_confidence_notes": notes}

    return {"category": None, "unparsed_confidence_notes": notes}
