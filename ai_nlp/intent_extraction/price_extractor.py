"""
Price constraint extraction from free-text queries.

Type: Rule-based

Recognizes common phrasings for a price ceiling, floor, or range:
  "under 4000", "under Rs 4000", "below rupees 4,000"
  "over 2000", "above 2000"
  "between 2000 and 5000", "2000 to 5000", "2000-5000"

Never fabricates a price constraint when none is stated -- returns
None for price_min/price_max in that case. When a number is present
but the direction is ambiguous, returns None and adds a note to
unparsed_confidence_notes rather than guessing a direction.
"""

import re
from typing import Dict, Optional

_NUMBER = r"(?:rs\.?|rupees?|₹)?\s*([\d,]+(?:\.\d+)?)"

_UNDER_PATTERN = re.compile(
    rf"\b(?:under|below|less than|cheaper than|up to)\s*{_NUMBER}", re.IGNORECASE
)
_OVER_PATTERN = re.compile(
    rf"\b(?:over|above|more than|greater than)\s*{_NUMBER}", re.IGNORECASE
)
_BETWEEN_PATTERN = re.compile(
    rf"\bbetween\s*{_NUMBER}\s*(?:and|to|-)\s*{_NUMBER}", re.IGNORECASE
)
_RANGE_DASH_PATTERN = re.compile(
    rf"{_NUMBER}\s*(?:to|-)\s*{_NUMBER}", re.IGNORECASE
)


def _parse_number(raw: str) -> float:
    """Strip commas and convert a matched number string to float."""
    return float(raw.replace(",", ""))


def extract_price(query: str) -> Dict[str, Optional[object]]:
    """
    Type: Rule-based

    Returns:
        {"price_min": float | None, "price_max": float | None,
         "unparsed_confidence_notes": list[str]}
    price_min/price_max are None (never a fabricated default) when
    no price constraint is stated or the direction is ambiguous.
    """
    notes = []

    between_match = _BETWEEN_PATTERN.search(query)
    if between_match:
        low = _parse_number(between_match.group(1))
        high = _parse_number(between_match.group(2))
        return {
            "price_min": min(low, high),
            "price_max": max(low, high),
            "unparsed_confidence_notes": notes,
        }

    range_dash_match = _RANGE_DASH_PATTERN.search(query)
    if range_dash_match:
        low = _parse_number(range_dash_match.group(1))
        high = _parse_number(range_dash_match.group(2))
        return {
            "price_min": min(low, high),
            "price_max": max(low, high),
            "unparsed_confidence_notes": notes,
        }

    under_match = _UNDER_PATTERN.search(query)
    over_match = _OVER_PATTERN.search(query)

    price_min = None
    price_max = None

    if under_match:
        price_max = _parse_number(under_match.group(1))
    if over_match:
        price_min = _parse_number(over_match.group(1))

    if not under_match and not over_match:
        # Check if a bare number exists with no direction word at all --
        # honestly flag it rather than guessing under/over.
        bare_number = re.search(r"\d[\d,]*", query)
        if bare_number and re.search(r"(?:rs\.?|rupees?|₹)", query, re.IGNORECASE):
            notes.append(
                f"Found a price-like number ('{bare_number.group(0)}') "
                "but no clear direction word (under/over/between) -- not extracted."
            )

    return {
        "price_min": price_min,
        "price_max": price_max,
        "unparsed_confidence_notes": notes,
    }
