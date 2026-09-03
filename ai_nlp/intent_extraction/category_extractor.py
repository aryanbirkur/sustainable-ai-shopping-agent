"""
Category extraction from free-text queries.

Type: Rule-based

Matches only against the 9 confirmed real categories in this catalog
(Shoes, Bags, T-Shirts, Jeans, Jackets, Shirts, Dresses, Electronics,
Cell Phones & Accessories -- the last two added when Amazon Reviews
2023 categories were integrated, see scripts/03_integrate_amazon_categories.py).
Uses exact (case-insensitive), word-boundary matching on the category
name and a small set of known singular/plural variants -- deliberately
NOT fuzzy string matching, since fuzzy matching risks silently matching
the wrong category. An unrecognized category-like term (e.g. "headphones",
"laptop") is honestly flagged as unparsed, never guessed.
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
    "Electronics": ["electronics", "electronic"],
    "Cell Phones & Accessories": ["phone", "phones", "cell phone", "cell phones", "smartphone", "smartphones"],
    "Beauty & Personal Care": ["beauty", "makeup", "cosmetics", "skincare", "skin care", "personal care"],
    "Arts, Crafts & Sewing": ["arts and crafts", "craft supplies", "crafts", "craft", "sewing"],
    "Toys & Games": ["toys", "toy", "games", "board game", "board games"],
    "Musical Instruments": ["musical instrument", "musical instruments", "guitar", "keyboard instrument"],
    "Handmade Products": ["handmade"],
    "Industrial & Scientific": ["industrial supplies", "scientific equipment", "industrial and scientific"],
}

# Common out-of-catalog terms worth explicitly recognizing so the
# "flagged, not silently guessed" behavior is visible and testable.
KNOWN_OUT_OF_CATALOG_TERMS = [
    "headphones", "laptop", "watch", "furniture",
]
# "electronics" and "phone" moved to CATEGORY_VARIANTS above now that
# Electronics and Cell Phones & Accessories are real catalog categories
# (see scripts/03_integrate_amazon_categories.py). "headphones", "laptop",
# "watch", "furniture" stay flagged out-of-catalog: they're plausible
# sub-types that MIGHT exist inside the broad "Electronics" bucket, but
# there's no dedicated category for them, so matching them to "Electronics"
# would be a guess this module deliberately never makes.


def extract_category(query: str) -> Dict[str, Optional[object]]:
    """
    Type: Rule-based

    Returns:
        {"category": str | None, "unparsed_confidence_notes": list[str]}
    category is one of the confirmed catalog category strings (see
    CATEGORY_VARIANTS keys), or None if no match is found (never a
    guessed/closest category).
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
                "catalog -- category not extracted."
            )
            return {"category": None, "unparsed_confidence_notes": notes}

    return {"category": None, "unparsed_confidence_notes": notes}
