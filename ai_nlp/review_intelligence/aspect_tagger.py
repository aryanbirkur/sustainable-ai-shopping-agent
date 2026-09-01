"""
Aspect tagging for product reviews.

Type: Rule-based (keyword matching) — NOT AI. This module never claims
to be a model; it only flags which of a small set of aspect categories a
review's text plausibly touches on, based on keyword presence. Used
alongside sentiment_scorer.py's Transformer output, never as a
substitute for it.

Aspects covered: quality, value, comfort_fit, sustainability_materials.
A review can match zero, one, or multiple aspects.
"""

from typing import List

ASPECT_KEYWORDS = {
    "quality": ["quality", "durable", "wear out", "wearing out", "worn out",
                "stitching", "sturdy", "fell apart", "well made", "cheaply made"],
    "value": ["price", "value", "worth", "expensive", "cheap", "overpriced", "affordable"],
    "comfort_fit": ["comfort", "comfortable", "fit", "sizing", "size", "tight", "loose", "snug"],
    "sustainability_materials": ["sustainable", "material", "recycled", "organic", "eco", "environment"],
}


def tag_aspects(text: str) -> List[str]:
    """
    Type: Rule-based

    Return the list of aspect categories whose keywords appear in the
    review text (case-insensitive substring match). Returns an empty
    list for missing/empty text or no keyword matches — never fabricates
    a match.
    """
    if text is None or not isinstance(text, str) or text.strip() == "":
        return []

    text_lower = text.lower()
    matched = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(aspect)
    return matched
