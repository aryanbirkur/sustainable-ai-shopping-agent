"""
app/schemas.py -- Pydantic request/response models for Milestone 9's FastAPI
layer.

Type: N/A (validation/serialization only, no AI/ML logic). These models
mirror the return shape of recommendation.hybrid.recommend_with_intent()
and recommend() as documented in the Milestone 9 spec -- they validate and
serialize; they do not reshape or reinterpret the underlying data.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    content: float
    collaborative: Optional[float] = None
    sustainability: float


class RawSignals(BaseModel):
    content: float
    collaborative: Optional[float] = None
    sustainability: float


class WeightsUsed(BaseModel):
    content: float
    collaborative: Optional[float] = None
    sustainability: float


class FilteringInfo(BaseModel):
    price_filter_applied: bool
    category_filter_applied: bool
    candidates_before_filter: int
    candidates_after_filter: int
    filter_relaxed: bool


class ResultItem(BaseModel):
    product_id: str
    final_score: float
    rank: int
    score_breakdown: ScoreBreakdown
    weights_used: WeightsUsed
    cold_start: bool
    out_of_domain_query: bool
    filtering: FilteringInfo
    raw_signals: RawSignals
    product_name: str
    category: str
    brand: str
    price: float
    sustainability_score: float
    image_path: Optional[str] = None


class IntentInfo(BaseModel):
    raw_query: str
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category: Optional[str] = None
    material_signals: List[str] = Field(default_factory=list)
    sustainability_emphasis: bool
    unparsed_confidence_notes: List[str] = Field(default_factory=list)


class Warnings(BaseModel):
    out_of_catalog_category: bool
    out_of_domain_query: bool


class RecommendResponse(BaseModel):
    results: List[ResultItem]
    intent: IntentInfo
    filtering: Optional[FilteringInfo] = None
    weights_used: Optional[WeightsUsed] = None
    warnings: Warnings


class RecommendManualResponse(BaseModel):
    results: List[ResultItem]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
