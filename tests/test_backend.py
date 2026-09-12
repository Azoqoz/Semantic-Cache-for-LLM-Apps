"""Service and HTTP integration tests: real cache, deterministic vectors, no network."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api import create_app
from src.application import ApplicationError, CacheApplication, QueryCommand
from src.cache_store import SQLiteSemanticCache
from src.config import Settings
from src.embeddings import EmbeddingService
from src.llm_providers import ProviderConfigurationError
from src.models import LLMResponse


class BackendFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Settings(app_mode="local", database_path=Path(temporary.name) / "test.sqlite3")
        self.embeddings = Mock(spec=EmbeddingService)
        self.embeddings.encode.return_value = [1.0, 0.0]
        self.embeddings.cosine_similarity.side_effect = EmbeddingService.cosine_similarity
        self.cache = SQLiteSemanticCache(self.settings.database_path, self.embeddings)
        self.factory = Mock(side_effect=self.provider)
        self.service = CacheApplication(self.settings, self.cache, self.embeddings, self.factory)

    @staticmethod
    def provider(provider_name, **kwargs):
        provider = Mock()
        provider.name = provider_name
        provider.model = "demo-rule-based" if provider_name == "Demo" else kwargs[f"{provider_name.lower()}_model"]
        provider.generate.return_value = LLMResponse(" answer ", provider.name, provider.model,
                                                    estimated_cost_usd=.006 if provider.name == "Demo" else None)
        return provider


class ApplicationTests(BackendFixture):
    def test_hit_types_are_from_lookup_not_score_and_do_not_add_accesses(self):
        miss = self.service.query(QueryCommand("  hello  "))
        exact = self.service.query(QueryCommand(" HELLO "))
        semantic = self.service.query(QueryCommand("paraphrase"))
        self.assertEqual([r["hit_type"] for r in (miss, exact, semantic)], ["miss", "exact", "semantic"])
        self.assertEqual([r["similarity"] for r in (miss, exact, semantic)], [None, None, 1.0])
        self.assertEqual(miss["answer"], " answer ")
        self.assertEqual(exact["answer"], "answer")
        self.assertEqual(self.cache.metrics()["total_reuses"], 2)
        self.assertEqual(self.cache.metrics()["total_queries"], 3)
        self.assertEqual(self.embeddings.encode.call_count, 3)

    def test_signed_miss_similarity_preserves_legacy_event_score(self):
        self.cache.add("negative", "answer", [-1.0, 0.0], "Demo", "demo-rule-based", 0)
        result = self.service.query(QueryCommand("other", threshold=0))
        self.assertEqual(result["hit_type"], "miss")
        self.assertEqual(result["similarity"], -1.0)
        self.assertIsNone(result["matched_question"])

    def test_threshold_boundary_ttl_and_isolation_pass_through(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with patch("src.cache_store.utc_now", return_value=start):
            self.cache.add("stored", "answer", [3., 4.], "OpenAI", "one", 1)
        threshold = EmbeddingService.cosine_similarity([1., 0.], [3., 4.])
        with patch("src.cache_store.utc_now", return_value=start + timedelta(minutes=59)):
            result = self.service.query(QueryCommand("other", "OpenAI", "one", threshold))
            self.assertEqual(result["hit_type"], "semantic")
            isolated = self.service.query(QueryCommand("stored", "OpenAI", "two", threshold))
            self.assertEqual(isolated["hit_type"], "miss")
            shared = self.service.query(QueryCommand("stored", isolate_by_model=False))
            self.assertEqual(shared["hit_type"], "exact")
        with patch("src.cache_store.utc_now", return_value=start + timedelta(hours=1)):
            expired = self.service.query(QueryCommand("stored", "OpenAI", "one", ttl_hours=-1))
            self.assertEqual(expired["hit_type"], "miss")
        self.assertIsNone(self.cache.list_entries()[0]["expires_at"])

    def test_defaults_zero_ttl_and_provider_model_forwarding(self):
        result = self.service.query(QueryCommand("q", "Ollama", "custom", ttl_hours=0))
        self.assertEqual(result["threshold"], .84)
        self.assertEqual(result["ttl_hours"], 0)
        self.assertIsNone(self.cache.list_entries()[0]["expires_at"])
        self.assertEqual(self.factory.call_args.kwargs["ollama_model"], "custom")
        self.assertEqual(self.factory.call_args.kwargs["ollama_base_url"], self.settings.ollama_base_url)

    def test_cost_and_latency_availability(self):
        self.assertIsNone(self.service.cache_snapshot()["latency_reduction_percentage"])
        with patch("src.semantic_cache.perf_counter", side_effect=[0, .5, 1, 1.01]):
            miss = self.service.query(QueryCommand("q"))
            self.assertIsNone(self.service.cache_snapshot()["latency_reduction_percentage"])
            hit = self.service.query(QueryCommand("q"))
        self.assertEqual((miss["estimated_cost_usd"], miss["cost_basis"], miss["cost_kind"]),
                         (.006, "provider_estimate", "incurred"))
        self.assertEqual((hit["estimated_cost_usd"], hit["cost_basis"], hit["cost_kind"]),
                         (.002, "default_estimate", "avoided"))
        snapshot = self.service.cache_snapshot()
        self.assertAlmostEqual(snapshot["latency_reduction_percentage"], 98)
        self.assertEqual(snapshot["metrics"], self.cache.metrics())
        for provider in ("Claude", "Gemini"):
            result = self.service.query(QueryCommand("q", provider))
            self.assertIsNone(result["estimated_cost_usd"])
            self.assertIsNone(result["cost_basis"])

    def test_validation_and_demo_gate_work_without_fastapi(self):
        for command in (QueryCommand(" "), QueryCommand("q", threshold=1.1),
                        QueryCommand("q", threshold=float("nan")), QueryCommand("q", ttl_hours=True),
                        QueryCommand("q", model="wrong-demo-model")):
            with self.assertRaises(ApplicationError) as error:
                self.service.query(command)
            self.assertEqual(error.exception.code, "invalid_request")
        self.service.settings = replace(self.settings, app_mode="demo")
        for provider in ("OpenAI", "Claude", "Gemini", "Ollama"):
            with self.assertRaises(ApplicationError):
                self.service.query(QueryCommand("q", provider))
        self.factory.assert_not_called()
        self.assertEqual(self.cache.metrics()["total_queries"], 0)

    def test_provider_exception_details_are_not_public(self):
        self.factory.side_effect = ProviderConfigurationError("secret-value")
        with self.assertRaises(ApplicationError) as error:
            self.service.query(QueryCommand("q"))
        self.assertEqual(error.exception.code, "provider_configuration")
        self.assertNotIn("secret-value", str(error.exception))

    def test_evaluation_reuses_production_report_without_changing_cache(self):
        self.service.query(QueryCommand("q"))
        before = self.cache.metrics()
        report = self.service.evaluate(.9)
        self.assertEqual(len(report["selected_rows"]), 36)
        self.assertEqual(report["selected_metrics"]["true_positives"], 18)
        self.assertEqual(report["selected_metrics"]["false_positives"], 18)
        self.assertEqual(report["recommended_threshold"], .9)
        self.assertEqual(self.cache.metrics(), before)
        self.assertEqual(self.settings.default_threshold, .84)


class APITests(BackendFixture):
    def setUp(self):
        super().setUp()
        self.client = TestClient(create_app(self.service))
        self.addCleanup(self.client.close)

    def test_http_query_cache_metrics_and_clear(self):
        first = self.client.post("/query", json={"question": "hello"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["hit_type"], "miss")
        self.assertIsNone(first.json()["similarity"])
        self.assertEqual(self.client.post("/query", json={"question": "HELLO"}).json()["hit_type"], "exact")
        self.assertEqual(self.client.post("/query", json={"question": "paraphrase"}).json()["hit_type"], "semantic")
        snapshot = self.client.get("/cache?limit=1").json()
        self.assertEqual(len(snapshot["entries"]), 1)
        self.assertNotIn("embedding_json", snapshot["entries"][0])
        self.assertEqual(snapshot["metrics"]["cache_hits"], 2)
        self.assertEqual(self.client.delete("/cache").json(), {"cleared": True})
        self.assertEqual(self.client.get("/cache").json()["metrics"]["total_queries"], 0)

    def test_health_and_capabilities_do_not_load_model_or_database(self):
        with patch("src.api.create_application") as factory:
            with TestClient(create_app(settings=replace(self.settings, app_mode="demo"))) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok", "check": "liveness"})
                result = client.get("/capabilities").json()
                self.assertEqual(result["providers"], [{"name": "Demo", "default_model": "demo-rule-based"}])
                self.assertNotIn("database_path", result)
                factory.assert_not_called()

    def test_lazy_initialization_once_and_failure_is_safe(self):
        with patch("src.api.create_application", return_value=self.service) as factory:
            with TestClient(create_app(settings=self.settings)) as client:
                self.assertEqual(client.get("/cache").status_code, 200)
                self.assertEqual(client.get("/cache").status_code, 200)
            factory.assert_called_once_with(self.settings)
        with patch("src.api.create_application", side_effect=RuntimeError("secret-value")):
            with TestClient(create_app(settings=self.settings)) as client:
                response = client.get("/cache")
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("secret-value", response.text)

    def test_validation_rejects_credentials_and_never_echoes_inputs(self):
        for body in ({"question": "q", "api_key": "secret-value"},
                     {"question": "q", "threshold": "secret-value"},
                     {"question": "q", "provider": "secret-value"},
                     {"question": "q", "ollama_base_url": "secret-value"},
                     {"question": "q", "threshold": 1.1}, {"question": "q", "ttl_hours": 1.5},
                     {"question": " "}, {}):
            with self.subTest(body=body):
                response = self.client.post("/query", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "invalid_request")
                self.assertNotIn("secret-value", response.text)
        self.assertEqual(self.cache.metrics()["total_queries"], 0)
        self.factory.assert_not_called()

    def test_provider_failures_never_return_log_or_persist_server_credentials(self):
        secret = "test-provider-secret-never-expose"
        self.factory.side_effect = RuntimeError(secret)
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}), self.assertNoLogs(level="WARNING"):
            response = self.client.post("/query", json={"question": "q", "provider": "OpenAI"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, self.client.get("/capabilities").text)
        self.assertNotIn(secret.encode(), self.settings.database_path.read_bytes())
        self.assertEqual(self.cache.metrics()["total_queries"], 0)

    def test_demo_restriction_and_configuration_error_status(self):
        self.service.settings = replace(self.settings, app_mode="demo")
        response = self.client.post("/query", json={"question": "q", "provider": "OpenAI"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "provider_configuration")
        self.factory.assert_not_called()

    def test_evaluation_contract_and_validation(self):
        result = self.client.post("/evaluation", json={"threshold": .84})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(set(result.json()), {"selected_rows", "selected_metrics", "difficulty_rows",
                                              "comparison_rows", "recommended_threshold"})
        self.assertEqual(len(result.json()["selected_rows"]), 36)
        self.assertEqual(self.client.post("/evaluation", json={"threshold": -1}).status_code, 422)
        self.assertEqual(self.client.get("/cache?limit=0").status_code, 422)
        self.assertEqual(self.client.get("/cache?limit=1001").status_code, 422)

    def test_unexpected_errors_and_http_errors_are_structured(self):
        with patch.object(self.service, "cache_snapshot", side_effect=RuntimeError("secret-value")):
            response = self.client.get("/cache")
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("secret-value", response.text)
        self.assertEqual(self.client.get("/missing").json()["error"]["code"], "http_error")
        self.assertEqual(self.client.post("/cache").status_code, 405)

    def test_openapi_exposes_only_requested_operations_and_no_credentials(self):
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(set(schema["paths"]), {"/health", "/capabilities", "/query", "/cache", "/evaluation"})
        properties = schema["components"]["schemas"]["QueryBody"]["properties"]
        self.assertNotIn("api_key", properties)
        self.assertFalse(schema["components"]["schemas"]["QueryBody"]["additionalProperties"])
