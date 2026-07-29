from __future__ import annotations

from time import perf_counter

from src.cache_store import SQLiteSemanticCache
from src.embeddings import EmbeddingService
from src.llm_providers import LLMProvider
from src.models import QueryResult


class SemanticCacheService:
    """Coordinates embedding, cache lookup, LLM generation, and metrics."""

    def __init__(
        self,
        cache: SQLiteSemanticCache,
        embedding_service: EmbeddingService,
    ) -> None:
        self.cache = cache
        self.embedding_service = embedding_service

    def answer(
        self,
        question: str,
        provider: LLMProvider,
        threshold: float,
        ttl_hours: int,
        isolate_by_model: bool = True,
    ) -> QueryResult:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        started_at = perf_counter()
        query_embedding = self.embedding_service.encode(cleaned_question)

        lookup = self.cache.lookup(
            question=cleaned_question,
            query_embedding=query_embedding,
            threshold=threshold,
            provider=provider.name if isolate_by_model else None,
            model=provider.model if isolate_by_model else None,
        )

        if lookup.hit and lookup.answer is not None:
            latency_ms = (perf_counter() - started_at) * 1000
            avoided_cost = self._default_avoided_cost(provider.name)

            self.cache.record_query_event(
                question=cleaned_question,
                cache_hit=True,
                similarity=lookup.similarity,
                latency_ms=latency_ms,
                provider=provider.name,
                model=provider.model,
                estimated_cost_usd=avoided_cost,
            )

            return QueryResult(
                question=cleaned_question,
                answer=lookup.answer,
                cache_hit=True,
                similarity=lookup.similarity,
                matched_question=lookup.matched_question,
                latency_ms=latency_ms,
                provider=provider.name,
                model=provider.model,
                estimated_cost_usd=avoided_cost,
            )

        llm_response = provider.generate(cleaned_question)

        self.cache.add(
            question=cleaned_question,
            answer=llm_response.text,
            embedding=query_embedding,
            provider=llm_response.provider,
            model=llm_response.model,
            ttl_hours=ttl_hours,
        )

        latency_ms = (perf_counter() - started_at) * 1000
        actual_or_estimated_cost = (
            llm_response.estimated_cost_usd
            if llm_response.estimated_cost_usd is not None
            else self._default_avoided_cost(provider.name)
        )

        self.cache.record_query_event(
            question=cleaned_question,
            cache_hit=False,
            similarity=lookup.similarity,
            latency_ms=latency_ms,
            provider=provider.name,
            model=provider.model,
            estimated_cost_usd=actual_or_estimated_cost,
        )

        return QueryResult(
            question=cleaned_question,
            answer=llm_response.text,
            cache_hit=False,
            similarity=lookup.similarity,
            matched_question=lookup.matched_question,
            latency_ms=latency_ms,
            provider=llm_response.provider,
            model=llm_response.model,
            estimated_cost_usd=actual_or_estimated_cost,
        )

    @staticmethod
    def _default_avoided_cost(provider_name: str) -> float:
        estimates = {
            "OpenAI": 0.002,
            "Ollama": 0.0,
            "Demo": 0.002,
        }
        return estimates.get(provider_name, 0.0)
