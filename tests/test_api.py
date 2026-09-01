"""
tests/test_api.py -- Milestone 9 API layer tests.

Mocks recommend_with_intent()/recommend() at their app.routes import site,
mirroring how test_recommendation.py mocks get_content_scores(). These
tests cover routing, validation, and error-handling only -- the underlying
recommendation logic is already covered by test_recommendation.py's 16 tests.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _fake_result_item(product_id="P001"):
    return {
        "product_id": product_id, "final_score": 0.75, "rank": 1,
        "score_breakdown": {"content": 0.8, "collaborative": 0.6, "sustainability": 0.7},
        "weights_used": {"content": 0.5, "collaborative": 0.3, "sustainability": 0.2},
        "cold_start": False, "out_of_domain_query": False,
        "filtering": {"price_filter_applied": False, "category_filter_applied": False,
                      "candidates_before_filter": 397, "candidates_after_filter": 397,
                      "filter_relaxed": False},
        "raw_signals": {"content": 0.8, "collaborative": 0.6, "sustainability": 0.7},
        "product_name": "Test Shoe", "category": "Shoes", "brand": "TestBrand",
        "price": 2999.0, "sustainability_score": 0.7,
    }


def _fake_intent(query, category=None):
    return {
        "raw_query": query, "price_min": None, "price_max": None, "category": category,
        "material_signals": [], "sustainability_emphasis": False, "unparsed_confidence_notes": [],
    }


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_normal_query(monkeypatch):
    def fake_recommend_with_intent(query, user_id=None, top_k=10):
        item = _fake_result_item()
        return {
            "results": [item],
            "intent": _fake_intent(query, category="Shoes"),
            "filtering": item["filtering"],
            "weights_used": item["weights_used"],
            "warnings": {"out_of_catalog_category": False, "out_of_domain_query": False},
        }

    monkeypatch.setattr("app.routes.recommend_with_intent", fake_recommend_with_intent)

    response = client.get("/recommend", params={"query": "running shoes under 4000"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["product_id"] == "P001"
    assert body["warnings"]["out_of_domain_query"] is False


def test_recommend_out_of_catalog_query_returns_200(monkeypatch):
    def fake_recommend_with_intent(query, user_id=None, top_k=10):
        return {
            "results": [],
            "intent": _fake_intent(query, category=None),
            "filtering": None,
            "weights_used": None,
            "warnings": {"out_of_catalog_category": True, "out_of_domain_query": True},
        }

    monkeypatch.setattr("app.routes.recommend_with_intent", fake_recommend_with_intent)

    response = client.get("/recommend", params={"query": "laptop charger"})
    assert response.status_code == 200
    body = response.json()
    assert body["warnings"]["out_of_catalog_category"] is True
    assert body["warnings"]["out_of_domain_query"] is True
    assert body["results"] == []


def test_recommend_missing_query_returns_422():
    response = client.get("/recommend")
    assert response.status_code == 422
    assert "error" in response.json()


def test_recommend_empty_query_returns_422():
    response = client.get("/recommend", params={"query": "   "})
    assert response.status_code == 422


def test_recommend_top_k_too_high_returns_422():
    response = client.get("/recommend", params={"query": "shoes", "top_k": 999})
    assert response.status_code == 422


def test_recommend_top_k_zero_returns_422():
    response = client.get("/recommend", params={"query": "shoes", "top_k": 0})
    assert response.status_code == 422


def test_recommend_manual_endpoint(monkeypatch):
    def fake_recommend(query, user_id=None, top_k=10, weights=None,
                        price_min=None, price_max=None, category=None,
                        sustainability_tilt=False):
        return [_fake_result_item()]

    monkeypatch.setattr("app.routes.recommend", fake_recommend)

    response = client.get("/recommend/manual", params={
        "query": "jeans", "price_min": 1000, "price_max": 3000, "category": "Jeans",
    })
    assert response.status_code == 200
    assert response.json()["results"][0]["product_id"] == "P001"


def test_recommend_manual_invalid_price_range_returns_422():
    response = client.get("/recommend/manual", params={
        "query": "jeans", "price_min": 5000, "price_max": 1000,
    })
    assert response.status_code == 422


def test_recommend_internal_error_returns_clean_500(monkeypatch):
    def fake_recommend_with_intent(query, user_id=None, top_k=10):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.routes.recommend_with_intent", fake_recommend_with_intent)

    response = client.get("/recommend", params={"query": "shoes"})
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "RECOMMENDATION_FAILED"
    assert "boom" not in body["error"]["message"]
