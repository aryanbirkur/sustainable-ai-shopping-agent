"""
app/routes.py -- HTTP routes wiring the existing recommendation pipeline.

Type: N/A (routing/orchestration only). Calls recommend_with_intent() and
recommend() from recommendation.hybrid exactly as they already exist --
this file does not reimplement or reinterpret their logic.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.schemas import RecommendResponse, RecommendManualResponse, ProductDetailResponse
from recommendation.hybrid import recommend, recommend_with_intent
from backend.services.product_detail_service import get_product_detail

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_TOP_K = 50


@router.get("/health")
def health_check():
    """Simple liveness check -- does not touch the recommendation pipeline."""
    return {"status": "ok"}


@router.get("/recommend", response_model=RecommendResponse)
def get_recommendations(
    query: str = Query(..., min_length=1, description="Natural-language shopping query"),
    user_id: str = Query(None, description="Optional known user id for collaborative filtering"),
    top_k: int = Query(10, ge=1, le=MAX_TOP_K, description=f"Number of results, 1-{MAX_TOP_K}"),
):
    """
    Primary endpoint. Runs the full intent-extraction + hybrid
    recommendation pipeline via recommend_with_intent() and returns its
    dict shape unchanged.
    """
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail={
            "code": "EMPTY_QUERY",
            "message": "query must not be empty or whitespace-only.",
        })

    try:
        result = recommend_with_intent(query=query, user_id=user_id, top_k=top_k)
    except Exception:
        logger.exception(f"recommend_with_intent failed for query='{query}'")
        raise HTTPException(status_code=500, detail={
            "code": "RECOMMENDATION_FAILED",
            "message": "Something went wrong while generating recommendations.",
        })

    return result


@router.get("/recommend/manual", response_model=RecommendManualResponse)
def get_recommendations_manual(
    query: str = Query(..., min_length=1),
    user_id: str = Query(None),
    top_k: int = Query(10, ge=1, le=MAX_TOP_K),
    price_min: float = Query(None, ge=0),
    price_max: float = Query(None, ge=0),
    category: str = Query(None),
    sustainability_tilt: bool = Query(False),
):
    """
    Manual-control endpoint. Skips intent extraction; calls recommend()
    directly so callers can set filtering/tilt explicitly themselves.
    """
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail={
            "code": "EMPTY_QUERY",
            "message": "query must not be empty or whitespace-only.",
        })
    if price_min is not None and price_max is not None and price_min > price_max:
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_PRICE_RANGE",
            "message": "price_min must not be greater than price_max.",
        })

    try:
        results = recommend(
            query=query,
            user_id=user_id,
            top_k=top_k,
            price_min=price_min,
            price_max=price_max,
            category=category,
            sustainability_tilt=sustainability_tilt,
        )
    except Exception:
        logger.exception(f"recommend failed for query='{query}'")
        raise HTTPException(status_code=500, detail={
            "code": "RECOMMENDATION_FAILED",
            "message": "Something went wrong while generating recommendations.",
        })

    return {"results": results}


@router.get("/products/{product_id}/detail", response_model=ProductDetailResponse)
def get_product_detail_route(product_id: str):
    """
    Lazily computes review sentiment for one product (live inference,
    not precomputed) -- called only when the frontend expands a specific
    product's detail view, not for every search result.
    """
    detail = get_product_detail(product_id)
    if detail is None:
        raise HTTPException(status_code=404, detail={
            "code": "PRODUCT_NOT_FOUND",
            "message": f"No product found with id '{product_id}'.",
        })
    return detail
