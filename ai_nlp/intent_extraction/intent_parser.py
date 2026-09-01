"""
Public entry point for intent extraction -- Milestone 7.

Type: Rule-based (combines price_extractor, category_extractor,
material_extractor -- each independently Type: Rule-based)

Combines the three independent extractors into one structured intent
dict. Each sub-extractor's honest None/empty-list/notes behavior is
preserved and merged, never overwritten with a guess.

NOT wired into Milestone 4's semantic_search() or Milestone 5's
hybrid blender in this milestone -- this is deliberately a later,
separate wiring step, per the project's independent-testability rule.
"""

from typing import Dict, List, Optional

from ai_nlp.intent_extraction.price_extractor import extract_price
from ai_nlp.intent_extraction.category_extractor import extract_category
from ai_nlp.intent_extraction.material_extractor import extract_material_signals


def extract_intent(query: str) -> Dict[str, object]:
    """
    Type: Rule-based

    Returns a structured intent dict:
        {
            "raw_query": str,
            "price_min": float | None,
            "price_max": float | None,
            "category": str | None,
            "material_signals": list[str],
            "sustainability_emphasis": bool,
            "unparsed_confidence_notes": list[str],
        }
    Deterministic: same input always produces the same output.
    """
    if query is None or not isinstance(query, str) or query.strip() == "":
        return {
            "raw_query": query,
            "price_min": None,
            "price_max": None,
            "category": None,
            "material_signals": [],
            "sustainability_emphasis": False,
            "unparsed_confidence_notes": ["Empty or missing query -- nothing extracted."],
        }

    price_result = extract_price(query)
    category_result = extract_category(query)
    material_result = extract_material_signals(query)

    notes: List[str] = []
    notes.extend(price_result["unparsed_confidence_notes"])
    notes.extend(category_result["unparsed_confidence_notes"])

    return {
        "raw_query": query,
        "price_min": price_result["price_min"],
        "price_max": price_result["price_max"],
        "category": category_result["category"],
        "material_signals": material_result["material_signals"],
        "sustainability_emphasis": material_result["sustainability_emphasis"],
        "unparsed_confidence_notes": notes,
    }
