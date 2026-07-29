from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    id: int
    question: str
    answer: str
    embedding: list[float]
    provider: str
    model: str
    created_at: str
    expires_at: Optional[str]
    access_count: int
    last_accessed_at: Optional[str]


@dataclass
class CacheLookupResult:
    hit: bool
    answer: Optional[str] = None
    matched_question: Optional[str] = None
    similarity: float = 0.0
    entry_id: Optional[int] = None


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


@dataclass
class QueryResult:
    question: str
    answer: str
    cache_hit: bool
    similarity: float
    matched_question: Optional[str]
    latency_ms: float
    provider: str
    model: str
    estimated_cost_usd: Optional[float]
