"""
frontend/api_client.py -- thin HTTP client wrapping calls to the
Milestone 9 FastAPI backend. No AI/ML logic here.
"""

import requests
from config import settings


def check_health():
    try:
        r = requests.get(f"{settings.API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _handle_response(r):
    if r.status_code == 200:
        return r.json(), None
    try:
        err = r.json().get("error", {})
        return None, err.get("message", f"API returned status {r.status_code}")
    except ValueError:
        return None, f"API returned status {r.status_code}"


def get_recommendations(query, user_id=None, top_k=10):
    """Calls GET /recommend. Returns (data, error) tuple."""
    params = {"query": query, "top_k": top_k}
    if user_id:
        params["user_id"] = user_id
    try:
        r = requests.get(f"{settings.API_BASE_URL}/recommend", params=params, timeout=15)
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to the API. Is it running on {settings.API_BASE_URL}?"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"
    return _handle_response(r)


def get_recommendations_manual(query, price_min=None, price_max=None, category=None,
                                sustainability_tilt=False, user_id=None, top_k=10):
    """Calls GET /recommend/manual. Returns (data, error) tuple."""
    params = {"query": query, "top_k": top_k, "sustainability_tilt": sustainability_tilt}
    if price_min is not None:
        params["price_min"] = price_min
    if price_max is not None:
        params["price_max"] = price_max
    if category:
        params["category"] = category
    if user_id:
        params["user_id"] = user_id
    try:
        r = requests.get(f"{settings.API_BASE_URL}/recommend/manual", params=params, timeout=15)
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to the API. Is it running on {settings.API_BASE_URL}?"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"
    return _handle_response(r)
