"""
tests/test_recommendation.py

Tests for Milestone 5. Content-based scores are monkeypatched so these
run fast and don't depend on a live ChromaDB index -- they test the
CF / blending / ranking logic in isolation, mirroring the tmp_path
isolation lesson learned in Milestone 4's test suite.
"""

import pandas as pd
import pytest

from recommendation.collaborative.cf_scorer import CollaborativeFilteringScorer
from recommendation.hybrid import blender
from recommendation.ranking.ranker import rank


@pytest.fixture
def small_interactions_csv(tmp_path):
    path = tmp_path / "interactions_clean.csv"
    rows = [
        ("U1", "P1", "purchase", 5),
        ("U1", "P2", "wishlist", 3),
        ("U2", "P3", "purchase", 5),
        ("U2", "P4", "view", 1),
        ("U3", "P1", "view", 1),
        ("U3", "P3", "purchase", 5),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "product_id", "interaction_type", "interaction_value"])
    df.to_csv(path, index=False)
    return str(path)


def test_cf_cold_start_returns_none_never_fabricates(small_interactions_csv):
    scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)
    assert all(v is None for v in scorer.score(None, ["P1", "P2", "P3"]).values())
    assert all(v is None for v in scorer.score("GHOST", ["P1", "P2"]).values())


def test_cf_known_user_returns_scores_in_0_1(small_interactions_csv):
    scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)
    result = scorer.score("U1", ["P1", "P2", "P3", "P4"])
    assert all(v is not None and 0.0 <= v <= 1.0 for v in result.values())


def test_weight_renormalization_sums_to_one():
    weights = {"content": 0.5, "collaborative": 0.3, "sustainability": 0.2}
    renorm = blender._renormalize_weights(weights, available_signals=["content", "sustainability"])
    assert "collaborative" not in renorm
    assert abs(sum(renorm.values()) - 1.0) < 1e-9


def test_blend_cold_start_never_raises_and_flags_cold_start(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.9, "P2": 0.4}
    fake_meta = {
        "P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 100, "sustainability_score": 0.6},
        "P2": {"product_name": "B", "category": "Shoes", "brand": "Y", "price": 200, "sustainability_score": 0.3},
    }
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.9))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    results = blender.blend(query="anything", user_id=None)
    assert len(results) == 2
    for r in results:
        assert r["cold_start"] is True
        assert r["score_breakdown"]["collaborative"] is None
        assert r["raw_signals"]["collaborative"] is None


def test_blend_known_user_differs_from_cold_start(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.5, "P3": 0.5}
    fake_meta = {
        "P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 100, "sustainability_score": 0.5},
        "P3": {"product_name": "C", "category": "Shoes", "brand": "Z", "price": 150, "sustainability_score": 0.5},
    }
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.5))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    cold = {r["product_id"]: r["final_score"] for r in blender.blend(query="q", user_id=None)}
    warm = {r["product_id"]: r["final_score"] for r in blender.blend(query="q", user_id="U1")}

    assert cold["P1"] == cold["P3"]        # identical content/sustainability -> tie, cold start
    assert warm["P1"] != warm["P3"]        # U1's history breaks the tie


def test_custom_weights_are_respected(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 1.0}
    fake_meta = {"P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 100, "sustainability_score": 0.0}}
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 1.0))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    results = blender.blend(query="q", user_id=None, weights={"content": 1.0, "collaborative": 0.0, "sustainability": 0.0})
    assert abs(results[0]["final_score"] - 1.0) < 1e-6


def test_score_breakdown_sums_to_final_score(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.7, "P2": 0.3}
    fake_meta = {
        "P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 100, "sustainability_score": 0.4},
        "P2": {"product_name": "B", "category": "Shoes", "brand": "Y", "price": 200, "sustainability_score": 0.8},
    }
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.7))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    for r in blender.blend(query="q", user_id="U1"):
        parts = [v for v in r["score_breakdown"].values() if v is not None]
        assert abs(sum(parts) - r["final_score"]) < 1e-6


def test_ranker_assigns_correct_order():
    candidates = [
        {"product_id": "A", "final_score": 0.2},
        {"product_id": "B", "final_score": 0.9},
        {"product_id": "C", "final_score": 0.5},
    ]
    ranked = rank(candidates, top_k=2)
    assert [c["product_id"] for c in ranked] == ["B", "C"]
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2


# ==================== MILESTONE 8: INTENT WIRING TESTS ====================

from config import settings
from recommendation.hybrid import recommend, recommend_with_intent
from recommendation.hybrid.blender import _apply_sustainability_tilt, _filter_candidates


def test_price_filter_excludes_products_above_max(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.9, "P2": 0.8}
    fake_meta = {
        "P1": {"product_name": "Cheap", "category": "Shoes", "brand": "X", "price": 1000, "sustainability_score": 0.5},
        "P2": {"product_name": "Pricey", "category": "Shoes", "brand": "Y", "price": 9000, "sustainability_score": 0.5},
    }
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.9))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    results = blender.blend(query="q", user_id=None, price_max=4000)
    ids = [r["product_id"] for r in results]
    assert "P1" in ids
    assert "P2" not in ids
    assert results[0]["filtering"]["price_filter_applied"] is True
    assert results[0]["filtering"]["filter_relaxed"] is False


def test_price_filter_relaxes_when_it_would_empty_pool(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.9}
    fake_meta = {
        "P1": {"product_name": "Only", "category": "Shoes", "brand": "X", "price": 9000, "sustainability_score": 0.5},
    }
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.9))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    results = blender.blend(query="q", user_id=None, price_max=100)
    assert len(results) == 1  # fell back to unfiltered pool
    assert results[0]["filtering"]["filter_relaxed"] is True


def test_category_filter_excludes_non_matching_category(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.9, "P2": 0.8}
    fake_meta = {
        "P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 1000, "sustainability_score": 0.5},
        "P2": {"product_name": "B", "category": "Bags", "brand": "Y", "price": 1000, "sustainability_score": 0.5},
    }
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.9))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    results = blender.blend(query="q", user_id=None, category="Shoes")
    assert [r["product_id"] for r in results] == ["P1"]


def test_sustainability_tilt_shifts_weight_toward_sustainability():
    weights = {"content": 0.5, "collaborative": 0.3, "sustainability": 0.2}
    tilted = _apply_sustainability_tilt(weights, tilt_amount=0.15)
    assert tilted["sustainability"] > weights["sustainability"]
    assert tilted["content"] < weights["content"]
    assert tilted["collaborative"] < weights["collaborative"]
    assert abs(sum(tilted.values()) - 1.0) < 1e-9


def test_sustainability_tilt_composes_with_cold_start_renormalization(monkeypatch, small_interactions_csv):
    """Cold start drops collaborative first; tilt then redistributes only
    from content, since collaborative is already gone."""
    fake_scores = {"P1": 1.0}
    fake_meta = {"P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 100, "sustainability_score": 0.5}}
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 1.0))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    results = blender.blend(query="q", user_id=None, sustainability_tilt=True)
    w = results[0]["weights_used"]
    assert "collaborative" not in w  # cold start dropped it first
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # sustainability got the full default tilt amount taken only from content
    assert w["sustainability"] > settings.HYBRID_WEIGHTS_DEFAULT["sustainability"]


def test_out_of_catalog_and_out_of_domain_both_surface(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.05}  # low similarity -> out_of_domain
    fake_meta = {"P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 100, "sustainability_score": 0.5}}
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, True, 0.05))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    output = recommend_with_intent(query="electronics under 500", user_id=None)
    assert output["warnings"]["out_of_catalog_category"] is True
    assert output["warnings"]["out_of_domain_query"] is True
    assert output["intent"]["category"] is None


def test_recommend_with_intent_applies_price_ceiling(monkeypatch, small_interactions_csv):
    fake_scores = {"P1": 0.9, "P2": 0.8}
    fake_meta = {
        "P1": {"product_name": "Cheap", "category": "Shoes", "brand": "X", "price": 1000, "sustainability_score": 0.5},
        "P2": {"product_name": "Pricey", "category": "Shoes", "brand": "Y", "price": 9000, "sustainability_score": 0.5},
    }
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.9))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    output = recommend_with_intent(query="shoes under 4000 rupees", user_id=None)
    ids = [r["product_id"] for r in output["results"]]
    assert "P2" not in ids


def test_existing_recommend_unaffected_by_new_optional_params(monkeypatch, small_interactions_csv):
    """Milestone 5 backward-compatibility guard: calling recommend() the old
    way (no new params) must produce identical shape/behavior to before."""
    fake_scores = {"P1": 0.9}
    fake_meta = {"P1": {"product_name": "A", "category": "Shoes", "brand": "X", "price": 100, "sustainability_score": 0.5}}
    monkeypatch.setattr(blender, "get_content_scores", lambda query, **kw: (fake_scores, fake_meta, False, 0.9))
    blender._cf_scorer = CollaborativeFilteringScorer(interactions_path=small_interactions_csv)

    results = recommend(query="q", user_id=None, top_k=5)
    assert results[0]["filtering"]["price_filter_applied"] is False
    assert results[0]["filtering"]["category_filter_applied"] is False
    assert results[0]["filtering"]["filter_relaxed"] is False
