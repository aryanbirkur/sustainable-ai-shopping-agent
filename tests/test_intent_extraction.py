"""
Tests for Milestone 7 -- Intent Extraction.

Test cases use real price/category data confirmed against
products_clean.csv (price range ~509 to ~7970 across all 7 categories).
"""

import pytest

from ai_nlp.intent_extraction.intent_parser import extract_intent
from ai_nlp.intent_extraction.price_extractor import extract_price
from ai_nlp.intent_extraction.category_extractor import extract_category


def test_under_price_extracted_correctly():
    result = extract_price("running shoes under 4000 rupees")
    assert result["price_max"] == 4000.0
    assert result["price_min"] is None


def test_between_price_extracted_correctly():
    result = extract_price("jeans between 2000 and 5000")
    assert result["price_min"] == 2000.0
    assert result["price_max"] == 5000.0


def test_no_price_mention_returns_none_never_fabricated():
    result = extract_price("a comfortable pair of shoes")
    assert result["price_min"] is None
    assert result["price_max"] is None


def test_ambiguous_bare_number_is_flagged_not_guessed():
    result = extract_price("shoes around Rs 4000 or so")
    assert result["price_max"] is None
    assert result["price_min"] is None
    assert len(result["unparsed_confidence_notes"]) == 1


def test_real_category_matched_case_insensitive():
    result = extract_category("Looking for JACKETS for winter")
    assert result["category"] == "Jackets"


def test_real_category_singular_variant_matched():
    result = extract_category("need a new shoe")
    assert result["category"] == "Shoes"


def test_out_of_catalog_term_is_honestly_flagged_not_guessed():
    result = extract_category("wireless bluetooth headphones")
    assert result["category"] is None
    assert len(result["unparsed_confidence_notes"]) == 1


def test_no_category_mention_returns_none():
    result = extract_category("something under 3000 rupees")
    assert result["category"] is None
    assert result["unparsed_confidence_notes"] == []


def test_full_intent_combines_all_signals():
    result = extract_intent(
        "running shoes under 4000 rupees, lightweight and environmentally responsible"
    )
    assert result["price_max"] == 4000.0
    assert result["category"] == "Shoes"
    assert "lightweight" in result["material_signals"]
    assert result["sustainability_emphasis"] is True


def test_material_signals_empty_list_when_none_mentioned():
    result = extract_intent("shoes under 3000")
    assert result["material_signals"] == []
    assert result["sustainability_emphasis"] is False


def test_empty_query_handled_honestly():
    result = extract_intent("")
    assert result["price_max"] is None
    assert result["category"] is None
    assert len(result["unparsed_confidence_notes"]) == 1


def test_extraction_is_deterministic():
    query = "recycled cotton bag below Rs 3000"
    first = extract_intent(query)
    second = extract_intent(query)
    assert first == second


def test_real_price_range_ceiling_within_catalog_bounds():
    """
    Sanity check against the real confirmed price range (509.23 to
    7970.71) -- a stated ceiling within that range should extract
    cleanly with no ambiguity flags.
    """
    result = extract_price("dresses under 6000")
    assert result["price_max"] == 6000.0
    assert result["unparsed_confidence_notes"] == []
