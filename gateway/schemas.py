from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum

class QueryType(str, Enum):
    CODING="coding"
    SUMMARIZATION="summarization"
    REASONING="reasoning"
    CREATIVE="creative"
    SIMPLE="simple"
    UNKNOWN="unknown"

class RoutingStrategy(str, Enum):
    COST_OPTIMIZED="cost_optimized"
    PERFORMANCE="performance"
    BALANCED="balanced"

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    max_tokens: int = Field(default=1000, ge=1, le=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    preferred_model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    request_id: Optional[str] = None
    content: str
    model_used: str
    provider: str
    query_type: str
    complexity_score: float
    tokens_used: int
    estimated_cost_usd: float
    latency_ms: Optional[float] = None
    routing_reason: str
    fallback_used: bool = False
    original_tokens: Optional[int] = None
    compressed_tokens: Optional[int] = None
    compression_ratio: Optional[float] = None
    compression_savings_usd: Optional[float] = None
    compression_applied: bool = False

class HealthResponse(BaseModel):
    status: str
    version: str
