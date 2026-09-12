"""Public restrictions and real ONNX scenario decisions; no cloud providers."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import create_app
from src.application import CacheApplication, QueryCommand, create_application
from src.config import Settings, settings, read_app_mode
from src.embeddings import model_cache_path
from src.evaluation import EVALUATION_DATASET
from src.public_demo import CANONICAL_PROMPTS, DEMO_SAMPLES, DEMO_THRESHOLD, seed_demo


class ModeCapabilitiesTests(unittest.TestCase):
    def test_default_and_explicit_local_preserve_full_controls(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_app_mode(), "local")
            self.assertEqual(Settings().app_mode, "local")
        result = CacheApplication.describe_capabilities(Settings(app_mode="local"))
        self.assertEqual([p["name"] for p in result["providers"]], ["Demo", "OpenAI", "Claude", "Gemini", "Ollama"])
        self.assertTrue(all(result["controls"].values()))
        self.assertNotIn("demo_samples", result)

    def test_public_capabilities_are_curated_and_do_not_expose_configuration(self):
        client = TestClient(create_app(settings=Settings(app_mode="demo")))
        self.addCleanup(client.close)
        caps = client.get("/capabilities")
        self.assertEqual(caps.json()["app_mode"], "demo")
        self.assertEqual(caps.json()["providers"], [{"name": "Demo", "default_model": "demo-rule-based"}])
        self.assertEqual(caps.json()["controls"], {"free_form": False, "settings_editable": False, "clear_cache": False, "evaluation": True})
        self.assertEqual(caps.json()["evaluation_thresholds"], [.84])
        self.assertEqual(len(caps.json()["demo_samples"]), 4)
        for text in ("OpenAI", "Claude", "Gemini", "Ollama", "database_path", "base_url"):
            self.assertNotIn(text, caps.text)
            self.assertNotIn(text, client.get("/openapi.json").text)
        dataset_questions = {text for pair in EVALUATION_DATASET for text in (pair.original_question, pair.similar_question)}
        self.assertTrue({s.question for s in DEMO_SAMPLES}.issubset(dataset_questions))


@unittest.skipUnless((model_cache_path(settings.embedding_model) / "onnx/model.onnx").is_file(),
                     "Run python -m src.prefetch_model for real public demo tests")
class PublicDemoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Settings(app_mode="demo", database_path=Path(temporary.name) / "local.sqlite3")
        self.sleep = patch("src.llm_providers.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.app = create_app(settings=self.config)
        self.client = self.enterContext(TestClient(self.app))
        self.assertTrue(self.app.state.warmup_complete.wait(10))
        self.assertEqual(self.client.get("/ready").json(), {"status": "ready"})

    def test_every_sample_has_its_verified_real_outcome(self):
        expected = [("exact", None, CANONICAL_PROMPTS[0]),
                    ("semantic", .944134, CANONICAL_PROMPTS[0]),
                    ("miss", None, None),
                    ("semantic", .916593, CANONICAL_PROMPTS[1])]
        results = []
        for sample, (kind, similarity, canonical) in zip(DEMO_SAMPLES, expected):
            self.client.cookies.clear()  # A new visitor gets only the canonical seed.
            response = self.client.post("/query", json={"question": sample.question})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            results.append(result)
            self.assertEqual(result["hit_type"], kind)
            self.assertEqual(result["threshold"], DEMO_THRESHOLD)
            self.assertEqual(result["ttl_hours"], 0)
            self.assertTrue(result["isolate_by_model"])
            if canonical:
                self.assertEqual(result["matched_question"], canonical)
            if similarity:
                self.assertAlmostEqual(result["similarity"], similarity, places=5)
                self.assertGreater(result["similarity"], DEMO_THRESHOLD)
                self.assertGreater(result["estimated_cost_usd"], 0)
                self.assertEqual(result["cost_kind"], "avoided")
            elif kind == "exact":
                self.assertIsNone(result["similarity"])
            else:
                self.assertLess(result["similarity"], DEMO_THRESHOLD)
                self.assertEqual(result["cost_kind"], "incurred")
        print("Public scenarios:", [(r["hit_type"], r["similarity"]) for r in results])

    def test_new_intent_then_exact_reuse_is_isolated_and_persisted_per_visitor(self):
        question = DEMO_SAMPLES[2].question
        first = self.client.post("/query", json={"question": question})
        token = self.client.cookies.get("cache_flow_demo")
        self.assertEqual(first.json()["hit_type"], "miss")
        self.assertEqual(self.client.post("/query", json={"question": question}).json()["hit_type"], "exact")
        snapshot = self.client.get("/cache").json()
        self.assertEqual(snapshot["metrics"]["cache_entries"], 3)
        self.assertEqual(snapshot["metrics"]["total_queries"], 2)
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/query", json={"question": question}).json()["hit_type"], "miss")
        self.client.cookies.clear()
        self.client.cookies.set("cache_flow_demo", token)
        self.assertEqual(self.client.get("/cache").json()["metrics"]["total_queries"], 2)

    def test_arbitrary_inputs_settings_providers_and_clear_are_rejected(self):
        self.client.get("/cache")
        before = self.client.get("/cache").json()
        for body in ({"question": "arbitrary secret prompt"},
                     *({"question": DEMO_SAMPLES[0].question, "provider": name} for name in ("OpenAI", "Claude", "Gemini", "Ollama")),
                     *({"question": DEMO_SAMPLES[0].question, **extra} for extra in (
                         {"model": "custom"}, {"threshold": .1}, {"ttl_hours": 168}, {"isolate_by_model": False},
                         {"api_key": "secret"}, {"base_url": "secret"}))):
            response = self.client.post("/query", json=body)
            self.assertIn(response.status_code, (403, 422), response.text)
            self.assertNotIn("secret", response.text)
        self.assertEqual(self.client.delete("/cache").status_code, 403)
        self.assertEqual(self.client.get("/cache").json(), before)

    def test_seeding_is_idempotent_and_never_touches_local_data(self):
        self.assertFalse(self.config.database_path.exists())
        self.config.database_path.write_bytes(b"local-data-marker")
        service = create_application(self.config)
        before = service.cache.list_entries()
        seed_demo(service)
        seed_demo(service)
        self.assertEqual(service.cache.list_entries(), before)
        self.assertEqual(service.cache.metrics()["cache_entries"], 2)
        self.assertEqual(service.cache.metrics()["total_queries"], 0)
        self.assertEqual(self.config.database_path.read_bytes(), b"local-data-marker")

    def test_local_is_unseeded_and_all_settings_and_clearing_remain_available(self):
        local = create_application(Settings(app_mode="local", database_path=self.config.database_path))
        seed_demo(local)
        self.assertEqual(local.cache.metrics()["cache_entries"], 0)
        result = local.query(QueryCommand("Free-form local question", threshold=.9, ttl_hours=12, isolate_by_model=False))
        self.assertEqual(result["hit_type"], "miss")
        self.assertEqual((result["threshold"], result["ttl_hours"], result["isolate_by_model"]), (.9, 12, False))
        self.assertEqual(local.clear_cache(), {"cleared": True})

    def test_public_evaluation_uses_fixed_dataset_and_threshold(self):
        self.assertEqual(self.client.post("/evaluation", json={"threshold": .1}).status_code, 403)
        self.assertEqual(self.client.post("/evaluation", json={"pairs": ["arbitrary"]}).status_code, 422)
        response = self.client.post("/evaluation", json={})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["selected_rows"]), 36)
