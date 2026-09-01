"""
Material/attribute and sustainability-emphasis signal extraction.

Type: Rule-based

Extracts keyword-based signals from the query text:
  - material_signals: attribute keywords mentioned (e.g. "recycled",
    "organic", "lightweight") -- an empty list if none found, never
    fabricated.
  - sustainability_emphasis: a boolean SIGNAL (not a numeric weight)
    indicating the query expresses an eco/sustainability preference.
    This is deliberately NOT wired into the Milestone 5 hybrid
    blender's weights in this milestone -- that wiring is a later,
    deliberate step.
"""

import re
from typing import Dict, List

MATERIAL_KEYWORDS = [
    "recycled", "organic", "lightweight", "tencel", "lyocell",
    "polyester", "nylon", "cotton", "durable", "waterproof",
]

SUSTAINABILITY_KEYWORDS = [
    "sustainable", "sustainability", "eco-friendly", "eco friendly",
    "environmentally responsible", "environment friendly", "green",
    "ethical", "eco",
]


def extract_material_signals(query: str) -> Dict[str, object]:
    """
    Type: Rule-based

    Returns:
        {"material_signals": list[str], "sustainability_emphasis": bool}
    """
    query_lower = query.lower()

    matched_materials: List[str] = [
        kw for kw in MATERIAL_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", query_lower)
    ]

    sustainability_emphasis = any(
        re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in SUSTAINABILITY_KEYWORDS
    )

    return {
        "material_signals": matched_materials,
        "sustainability_emphasis": sustainability_emphasis,
    }
