"""Pydantic models shared by the engine, API, and desktop shell."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Stance(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    CONFLICT = "CONFLICT"


class SourceStatus(str, Enum):
    LIVE = "LIVE"
    NO_DATA = "NO_DATA"
    NEEDS_AUTH = "NEEDS_AUTH"
    ERROR = "ERROR"
    STALE = "STALE"


class FeedStatus(str, Enum):
    LIVE = "LIVE"
    PARTIAL = "PARTIAL"
    STALE = "STALE"
    NO_DATA = "NO_DATA"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    INVALID = "INVALID"
    DEMO = "DEMO"


class GateState(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class Post(BaseModel):
    id: str
    platform: str
    author: str = "anon"
    text: str
    created_at: float
    url: Optional[str] = None
    user_sentiment: Optional[Stance] = None
    source_name: Optional[str] = None


class SourceResult(BaseModel):
    id: str
    name: str
    icon: str
    kind: str
    status: SourceStatus
    reason: Optional[str] = None
    stance: Stance = Stance.WAIT
    intensity: int = Field(default=0, ge=0, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sample_count: int = Field(default=0, ge=0)
    analyzed_count: int = Field(default=0, ge=0)
    bullish_count: int = Field(default=0, ge=0)
    bearish_count: int = Field(default=0, ge=0)
    neutral_count: int = Field(default=0, ge=0)
    keywords: List[str] = Field(default_factory=list)
    evidence: Optional[str] = None
    evidence_url: Optional[str] = None
    latest_post_at: Optional[float] = None
    fetched_at: float = 0.0
    provenance: Optional[str] = None
    posts: List[Post] = Field(default_factory=list)
    latency_ms: int = 0
    cached: bool = False
    symbol_verified: bool = True
    low_sample: bool = False
    mixed_signals: bool = False


class MarketContext(BaseModel):
    ticker: str
    label: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    currency: str = "USD"
    source: str = "—"
    fear_greed: Optional[int] = None
    fear_greed_label: Optional[str] = None
    fetched_at: float = 0.0
    error: Optional[str] = None


class GateCheck(BaseModel):
    id: str
    name: str
    state: GateState
    detail: str = ""


class ScanResult(BaseModel):
    ticker: str
    label: str
    timestamp: float
    feed_status: FeedStatus
    consensus: str
    conflict_detected: bool = False
    intensity: int = Field(default=0, ge=0, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: List[SourceResult] = Field(default_factory=list)
    gates: List[GateCheck] = Field(default_factory=list)
    market: Optional[MarketContext] = None
    total_posts: int = 0
    analyzed_posts: int = 0
    live_sources: int = 0
    latency_ms: int = 0
    cached: bool = False
    is_demo: bool = False
    next_refresh_in: int = 0


class AnalysisResult(BaseModel):
    input_text: str
    sanitized_text: Optional[str] = None
    is_relevant: bool = False
    reason: str = ""
    stance: Stance = Stance.WAIT
    intensity: int = Field(default=0, ge=0, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_terms: List[str] = Field(default_factory=list)
    evidence_quote: Optional[str] = None
    is_ambiguous: bool = False
    engine: str = "lexicon"
    model: Optional[str] = None
    reasoning: Optional[str] = None
    latency_ms: int = 0
    scores: Dict[str, int] = Field(default_factory=dict)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    gates: List[GateCheck] = Field(default_factory=list)
    model_error: Optional[str] = None
    evidence_verified: bool = True


class HistoryEntry(BaseModel):
    timestamp: float
    kind: str
    ticker: Optional[str] = None
    stance: str = "WAIT"
    intensity: int = 0
    text: str = ""
    engine: str = "lexicon"
    model: Optional[str] = None


class ModelStatus(BaseModel):
    available: bool
    version: str = "SLM-1"
    architecture: str = ""
    accuracy: float = 0.0
    samples: int = 0
    trained_at: Optional[str] = None
    labels: List[str] = Field(default_factory=lambda: ["BUY", "SELL", "WAIT"])
    feature_dim: int = 2048
    error: Optional[str] = None


class ModelBrief(BaseModel):
    ticker: str
    model: str
    stance: Stance
    intensity: int = Field(default=50, ge=0, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    key_drivers: List[str] = Field(default_factory=list)
    evidence_quote: Optional[str] = None
    evidence_url: Optional[str] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)
    latency_ms: int = 0
    timestamp: float = 0.0
    sample_size: int = 0
    disclaimer: str = ""
