"""Framework-neutral operations shared by future frontends.

Credentials and provider URLs come only from server configuration. No provider
objects, environment dictionaries, or raw exception messages cross this boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
from typing import Callable

from src.cache_store import SQLiteSemanticCache
from src.config import Settings
from src.embeddings import EmbeddingService
from src.evaluation import COMPARISON_THRESHOLDS, build_evaluation_report
from src.llm_providers import ProviderConfigurationError, build_provider
from src.semantic_cache import SemanticCacheService


class ApplicationError(Exception):
    """An allowlisted public error; never contains upstream exception text."""

    MESSAGES = {
        "service_warming": "The semantic engine is warming. Please try again when ready.",
        "model_initialization_failed": "The semantic engine could not initialize. Restart the service to retry.",
        "invalid_request": "Check the request fields and supported values.",
        "provider_configuration": "The provider is disabled or not configured on the server.",
        "operation_failed": "The operation failed. Check the server configuration and try again.",
    }

    def __init__(self, code: str):
        self.code = code
        super().__init__(self.MESSAGES[code])


def safe_operation(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except ApplicationError:
            raise
        except ProviderConfigurationError:
            raise ApplicationError("provider_configuration") from None
        except Exception:
            raise ApplicationError("operation_failed") from None
    return wrapped


@dataclass(frozen=True)
class QueryCommand:
    question: str
    provider: str = "Demo"
    model: str | None = None
    threshold: float | None = None
    ttl_hours: int | None = None
    isolate_by_model: bool = True


class CacheApplication:
    def __init__(self, settings: Settings, cache: SQLiteSemanticCache,
                 embeddings: EmbeddingService, provider_factory: Callable = build_provider):
        self.settings = settings
        self.cache = cache
        self.embeddings = embeddings
        self.pipeline = SemanticCacheService(cache, embeddings)
        self.provider_factory = provider_factory

    def capabilities(self) -> dict:
        return self.describe_capabilities(self.settings)

    @staticmethod
    def describe_capabilities(settings: Settings) -> dict:
        models = {"Demo": "demo-rule-based", "OpenAI": settings.openai_model,
                  "Claude": settings.claude_model, "Gemini": settings.gemini_model,
                  "Ollama": settings.ollama_model}
        names = ["Demo"] if settings.app_mode == "demo" else list(models)
        return {
            "app_mode": settings.app_mode,
            "providers": [{"name": name, "default_model": models[name]} for name in names],
            "defaults": {"provider": "Demo", "threshold": settings.default_threshold,
                         "ttl_hours": settings.default_ttl_hours, "isolate_by_model": True},
            "evaluation_thresholds": list(COMPARISON_THRESHOLDS),
            "credentials": "server_environment_only",
        }

    @staticmethod
    def _validate_threshold(threshold):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ApplicationError("invalid_request")

    @safe_operation
    def query(self, command: QueryCommand) -> dict:
        threshold = self.settings.default_threshold if command.threshold is None else command.threshold
        ttl = self.settings.default_ttl_hours if command.ttl_hours is None else command.ttl_hours
        self._validate_threshold(threshold)
        if (not isinstance(command.question, str) or not command.question.strip()
                or type(ttl) is not int or type(command.isolate_by_model) is not bool
                or (command.model is not None and
                    (not isinstance(command.model, str) or not command.model.strip()))):
            raise ApplicationError("invalid_request")
        if command.provider not in {p["name"] for p in self.capabilities()["providers"]}:
            raise ApplicationError("provider_configuration")
        if command.provider == "Demo" and command.model not in (None, "demo-rule-based"):
            raise ApplicationError("invalid_request")
        model_args = {"openai_model": self.settings.openai_model, "claude_model": self.settings.claude_model,
                      "gemini_model": self.settings.gemini_model, "ollama_model": self.settings.ollama_model}
        if command.model is not None and command.provider != "Demo":
            model_args[f"{command.provider.lower()}_model"] = command.model
        provider = self.provider_factory(provider_name=command.provider,
                                         ollama_base_url=self.settings.ollama_base_url,
                                         app_mode=self.settings.app_mode, **model_args)
        result = self.pipeline.answer(command.question, provider, threshold, ttl, command.isolate_by_model)
        return {
            "question": result.question, "answer": result.answer, "cache_hit": result.cache_hit,
            "hit_type": result.hit_type,
            # Exact equality has a legacy score of 1, but no measured cosine.
            "similarity": result.cosine_similarity,
            "matched_question": result.matched_question,
            "threshold": threshold, "ttl_hours": ttl, "isolate_by_model": command.isolate_by_model,
            "provider": result.provider, "model": result.model, "latency_ms": result.latency_ms,
            "estimated_cost_usd": result.estimated_cost_usd if result.cost_basis else None,
            "cost_basis": result.cost_basis,
            "cost_kind": "avoided" if result.cache_hit else "incurred",
        }

    @safe_operation
    def cache_snapshot(self, limit: int = 100) -> dict:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ApplicationError("invalid_request")
        metrics = self.cache.metrics()
        miss_latency = metrics["average_miss_latency_ms"]
        reduction = ((miss_latency - metrics["average_hit_latency_ms"]) / miss_latency * 100
                     if metrics["cache_hits"] and metrics["cache_misses"] and miss_latency > 0 else None)
        return {"entries": self.cache.list_entries(limit), "metrics": metrics,
                "latency_reduction_percentage": reduction,
                "cost_metrics_basis": "legacy_estimates_not_billing"}

    @safe_operation
    def clear_cache(self) -> dict:
        self.cache.clear()
        return {"cleared": True}

    @safe_operation
    def evaluate(self, threshold: float | None = None) -> dict:
        threshold = self.settings.default_threshold if threshold is None else threshold
        self._validate_threshold(threshold)
        return asdict(build_evaluation_report(self.embeddings, threshold))


def create_application(settings: Settings) -> CacheApplication:
    embeddings = EmbeddingService(settings.embedding_model, lazy=True)
    cache = SQLiteSemanticCache(settings.database_path, embeddings)
    return CacheApplication(settings, cache, embeddings)
