"""Characterize current behavior, including append-only inserts and retained TTL rows.

Uses real SQLite and cosine math, deterministic embeddings, and no live providers.
"""

import ast
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src.cache_store import SQLiteSemanticCache
from src.embeddings import EmbeddingService
from src.models import LLMResponse
from src.semantic_cache import SemanticCacheService


class CacheParity(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.sqlite3"
        self.embedding = Mock(spec=EmbeddingService)
        self.embedding.encode.return_value = [1.0, 0.0]
        self.embedding.cosine_similarity.side_effect = EmbeddingService.cosine_similarity
        self.cache = SQLiteSemanticCache(self.path, self.embedding)
        self.service = SemanticCacheService(self.cache, self.embedding)
        self.provider = Mock(name="provider")
        self.provider.name = "Demo"
        self.provider.model = "demo-rule-based"
        self.provider.generate.return_value = LLMResponse(
            "  fresh answer  ", "Demo", "demo-rule-based", estimated_cost_usd=0.006
        )

    def add(self, question="Stored question", answer="Stored answer", vector=None,
            provider="Demo", model="demo-rule-based", ttl=0):
        return self.cache.add(question, answer, vector or [1.0, 0.0], provider, model, ttl)

    def lookup(self, question="different question", vector=None, threshold=0.84, **filters):
        return self.cache.lookup(question, vector or [1.0, 0.0], threshold, **filters)

    def answer(self, question="new question", threshold=0.84, isolate=True):
        return self.service.answer(question, self.provider, threshold, 24, isolate)

    def test_exact_normalization_newest_duplicate_and_access_update(self):
        old = self.add("  HELLO   world  ", "old")
        newest = self.add("hello world", "  newest  ", [0.0, 1.0])
        hit = self.lookup("\tHello\n WORLD ", vector=[-1.0, 0.0], threshold=1.0)
        self.assertEqual((hit.hit, hit.answer, hit.similarity, hit.entry_id),
                         (True, "newest", 1.0, newest))
        self.embedding.cosine_similarity.assert_not_called()
        rows = self.cache.list_entries()
        self.assertEqual([r["id"] for r in rows], [newest, old])
        self.assertEqual([r["access_count"] for r in rows], [1, 0])
        self.assertIsNotNone(rows[0]["last_accessed_at"])
        self.assertIsNone(rows[1]["last_accessed_at"])
        self.assertEqual(rows[1]["question"], "HELLO   world")
        self.assertEqual(self.cache.metrics()["total_queries"], 0)

    def test_semantic_best_match_inclusive_threshold_and_newest_tie(self):
        self.add("weak", vector=[0.0, 1.0])
        self.add("older best", vector=[3.0, 4.0])
        newest = self.add("newer best", vector=[3.0, 4.0])
        boundary = EmbeddingService.cosine_similarity([1.0, 0.0], [3.0, 4.0])
        hit = self.lookup(threshold=boundary)
        self.assertEqual((hit.hit, hit.entry_id, hit.matched_question),
                         (True, newest, "newer best"))
        miss = self.lookup(threshold=boundary + 0.000001)
        self.assertFalse(miss.hit)
        self.assertEqual(miss.similarity, boundary)
        self.assertEqual(miss.matched_question, "newer best")
        self.assertIsNone(miss.answer)
        self.assertIsNone(miss.entry_id)
        self.assertEqual(self.cache.metrics()["total_reuses"], 1)

    def test_empty_negative_and_zero_similarity_behavior(self):
        empty = self.lookup()
        self.assertEqual((empty.hit, empty.similarity, empty.matched_question), (False, 0.0, None))
        self.add(vector=[-1.0, 0.0])
        # Exactly -1 never beats the initial best score and has no matched question.
        self.assertIsNone(self.lookup(threshold=0).matched_question)
        self.add("negative", vector=[-3.0, 4.0])
        miss = self.lookup(threshold=0)
        self.assertEqual((miss.hit, miss.similarity, miss.matched_question), (False, 0.0, "negative"))
        self.add("zero", vector=[0.0, 0.0])
        self.assertTrue(self.lookup(threshold=0).hit)

    def test_ttl_exact_boundary_filters_both_paths_but_retains_rows(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with patch("src.cache_store.utc_now", return_value=start):
            self.add(ttl=1)
        with patch("src.cache_store.utc_now", return_value=start + timedelta(hours=1, microseconds=-1)):
            self.assertTrue(self.lookup("Stored question").hit)
        with patch("src.cache_store.utc_now", return_value=start + timedelta(hours=1)):
            self.assertFalse(self.lookup("Stored question").hit)
            self.assertFalse(self.lookup().hit)
        self.assertEqual(len(self.cache.list_entries()), 1)
        self.assertEqual(self.cache.metrics()["cache_entries"], 1)

    def test_none_zero_and_negative_ttl_never_expire(self):
        for ttl in (None, 0, -1):
            with self.subTest(ttl=ttl):
                self.add(question=str(ttl), ttl=ttl)
        with patch("src.cache_store.utc_now", return_value=datetime(2100, 1, 1, tzinfo=timezone.utc)):
            for row in self.cache.list_entries():
                self.assertIsNone(row["expires_at"])
                self.assertTrue(self.lookup(row["question"]).hit)

    def test_provider_and_model_filter_exact_and_semantic_candidates(self):
        target = self.add(provider="OpenAI", model="a")
        self.add(answer="wrong model", provider="OpenAI", model="b")
        self.add(answer="wrong provider", provider="Demo", model="a")
        for question in ("Stored question", "paraphrase"):
            with self.subTest(question=question):
                self.assertEqual(self.lookup(question, provider="OpenAI", model="a").entry_id, target)
                self.assertFalse(self.lookup(question, provider="Other", model="a").hit)
                self.assertFalse(self.lookup(question, provider="OpenAI", model="missing").hit)
                self.assertEqual(self.lookup(question, provider="OpenAI").answer, "wrong model")
                self.assertEqual(self.lookup(question, model="a").answer, "wrong provider")

    def test_service_isolation_can_be_disabled_and_reports_request_provider_on_hit(self):
        self.add(provider="Other", model="other")
        hit = self.answer("Stored question", isolate=False)
        self.assertTrue(hit.cache_hit)
        self.assertEqual((hit.provider, hit.model), ("Demo", "demo-rule-based"))
        self.provider.generate.assert_not_called()
        self.assertFalse(self.answer("Stored question").cache_hit)

    def test_miss_then_exact_and_semantic_hits_embed_each_time_and_record_cost_latency(self):
        with patch("src.semantic_cache.perf_counter", side_effect=[1, 1.5, 2, 2.01, 3, 3.02]):
            miss = self.answer("  new question  ")
            exact = self.answer("new question")
            semantic = self.answer("paraphrase")
        self.assertFalse(miss.cache_hit)
        self.assertEqual(miss.answer, "  fresh answer  ")
        self.assertEqual(miss.question, "new question")
        self.assertTrue(exact.cache_hit)
        self.assertTrue(semantic.cache_hit)
        self.assertEqual(exact.answer, "fresh answer")
        self.assertEqual(semantic.matched_question, "new question")
        self.provider.generate.assert_called_once_with("new question")
        self.assertEqual(self.embedding.encode.call_count, 3)
        self.assertAlmostEqual(miss.latency_ms, 500)
        self.assertAlmostEqual(exact.latency_ms, 10)
        self.assertAlmostEqual(semantic.latency_ms, 20)
        metrics = self.cache.metrics()
        for key, value in {"total_queries": 3, "cache_hits": 2, "cache_misses": 1,
                           "cache_entries": 1, "total_reuses": 2, "hit_rate": 2/3,
                           "average_latency_ms": 530/3, "average_hit_latency_ms": 15,
                           "average_miss_latency_ms": 500, "llm_cost_usd": 0.006,
                           "avoided_cost_usd": 0.004, "total_cost_without_cache_usd": 0.01,
                           "savings_percentage": 40}.items():
            self.assertAlmostEqual(metrics[key], value, msg=key)

    def test_miss_preserves_nearest_candidate_and_response_identity(self):
        self.add("nearest", vector=[3.0, 4.0])
        self.provider.generate.return_value = LLMResponse("answer", "actual", "actual-model")
        result = self.answer()
        self.assertFalse(result.cache_hit)
        self.assertAlmostEqual(result.similarity, 0.6)
        self.assertEqual(result.matched_question, "nearest")
        self.assertEqual((result.provider, result.model), ("actual", "actual-model"))
        self.assertEqual(self.cache.list_entries()[0]["provider"], "actual")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            self.assertEqual(connection.execute("SELECT provider, model FROM query_events").fetchone(),
                             ("Demo", "demo-rule-based"))

    def test_fallback_costs_and_explicit_zero(self):
        for name, cost in (("OpenAI", .002), ("Demo", .002), ("Ollama", 0),
                           ("Claude", 0), ("Gemini", 0), ("Unknown", 0)):
            with self.subTest(provider=name):
                self.cache.clear()
                self.provider.name = name
                self.provider.generate.return_value = LLMResponse("answer", name, self.provider.model)
                self.assertEqual(self.answer().estimated_cost_usd, cost)
                self.assertEqual(self.answer().estimated_cost_usd, cost)
        self.cache.clear()
        self.provider.name = "OpenAI"
        self.provider.generate.return_value = LLMResponse("answer", "OpenAI", self.provider.model,
                                                         estimated_cost_usd=0.0)
        self.assertEqual(self.answer().estimated_cost_usd, 0.0)

    def test_validation_and_embedding_provider_errors_have_no_writes_or_fallback(self):
        for question, threshold in ((" \n", .84), ("q", -.1), ("q", 1.1), ("q", float("nan"))):
            with self.subTest(question=question, threshold=threshold), self.assertRaises(ValueError):
                self.answer(question, threshold)
        self.embedding.encode.assert_not_called()
        self.embedding.encode.side_effect = RuntimeError("embedding unavailable")
        with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
            self.answer()
        self.provider.generate.assert_not_called()
        self.embedding.encode.side_effect = None
        self.provider.generate.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.answer()
        self.assertEqual(self.cache.metrics()["total_queries"], 0)
        self.assertEqual(self.cache.list_entries(), [])

    def test_reopen_persists_embeddings_entries_access_and_events(self):
        identifier = self.add()
        self.answer("Stored question")
        before = self.cache.metrics()
        reopened = SQLiteSemanticCache(self.path, self.embedding)
        self.assertEqual(reopened.metrics(), before)
        self.assertEqual(reopened.list_entries(), self.cache.list_entries())
        hit = reopened.lookup("paraphrase", [1.0, 0.0], 1.0)
        self.assertEqual(hit.entry_id, identifier)

    def test_clear_removes_entries_and_events_but_does_not_reset_ids(self):
        old = self.add()
        self.answer()
        self.cache.clear()
        self.assertEqual(self.cache.list_entries(), [])
        self.assertTrue(all(value == 0 for value in self.cache.metrics().values()))
        self.assertFalse(self.lookup().hit)
        self.assertGreater(self.add(), old)

    def test_selective_delete_preserves_query_events(self):
        old = self.add()
        keep = self.add("keep")
        self.answer()
        before = self.cache.metrics()["total_queries"]
        self.cache.delete_entries([])
        self.cache.delete_entries([old, 9999])
        self.assertEqual([r["id"] for r in self.cache.list_entries()], [keep])
        self.assertEqual(self.cache.metrics()["total_queries"], before)

    def test_legacy_event_schema_is_migrated_without_losing_rows(self):
        path = Path(self.directory.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE query_events (id INTEGER PRIMARY KEY)")
            connection.execute("INSERT INTO query_events VALUES (7)")
        cache = SQLiteSemanticCache(path, self.embedding)
        self.assertEqual(cache.metrics()["total_queries"], 1)
        self.assertEqual(cache.metrics()["cache_misses"], 1)
        with closing(sqlite3.connect(path)) as connection, connection:
            row = connection.execute("SELECT id, question, provider, model, estimated_cost_usd FROM query_events").fetchone()
        self.assertEqual(row, (7, "", "Unknown", "Unknown", None))
        self.assertEqual(SQLiteSemanticCache(path, self.embedding).metrics(), cache.metrics())

    def test_empty_and_null_cost_metrics(self):
        self.assertTrue(all(value == 0 for value in self.cache.metrics().values()))
        self.cache.record_query_event("q", True, 1, 20, "Demo", "m", None)
        metrics = self.cache.metrics()
        self.assertEqual(metrics["hit_rate"], 1)
        self.assertEqual(metrics["average_miss_latency_ms"], 0)
        self.assertEqual(metrics["savings_percentage"], 0)


class DashboardCalculationParity(unittest.TestCase):
    def test_actual_dashboard_latency_expression_including_negative_and_zero(self):
        # Execute only the existing calculation, without importing/running the UI.
        source = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "latency_reduction" for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        expression = compile(ast.Expression(assignments[0].value), str(source), "eval")
        for miss, hit, expected in ((500, 15, 97), (0, 20, 0), (10, 20, -100)):
            with self.subTest(miss=miss, hit=hit):
                self.assertEqual(eval(expression, {"__builtins__": {}},
                                      {"average_miss_latency": miss, "average_hit_latency": hit}), expected)
