"""Startup must not import the inference stack or download a model."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api import create_app
from src.config import Settings
from src.embeddings import EmbeddingService


class StartupTests(unittest.TestCase):
    def test_cold_api_import_creation_health_capabilities_and_cache_skip_inference(self):
        # A separate interpreter prevents earlier test imports masking eager loads.
        script = textwrap.dedent('''
            import importlib.abc
            import sys
            import tempfile
            from pathlib import Path
            from time import perf_counter

            class BlockInference(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'sentence_transformers', 'torch', 'transformers'}:
                        raise AssertionError('Inference imported during startup: ' + fullname)
            sys.meta_path.insert(0, BlockInference())
            started = perf_counter()
            from src.api import app, create_app
            from src.config import Settings
            from fastapi.testclient import TestClient
            with tempfile.TemporaryDirectory() as directory:
                config = Settings(app_mode='demo', database_path=Path(directory) / 'cache.sqlite3')
                with TestClient(create_app(settings=config)) as client:
                    assert not config.database_path.exists()
                    assert client.get('/health').json() == {'status': 'ok', 'check': 'liveness'}
                    assert client.get('/capabilities').json()['app_mode'] == 'demo'
                    assert not config.database_path.exists()
                    assert client.get('/cache').status_code == 200
                    assert client.delete('/cache').json() == {'cleared': True}
            assert not any(name in sys.modules for name in ('sentence_transformers', 'torch', 'transformers'))
            print('Cold import and lightweight endpoints passed in %.2fs' % (perf_counter() - started))
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).resolve().parents[1],
                                env={**os.environ, "APP_MODE": "demo"},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def setUp(self):
        EmbeddingService._load_model.cache_clear()
        self.addCleanup(EmbeddingService._load_model.cache_clear)

    def test_query_and_evaluation_initialize_once_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.embeddings.SentenceTransformer") as constructor:
            constructor.return_value.encode.return_value = [1.0, 0.0]
            config = Settings(app_mode="demo", database_path=Path(directory) / "cache.sqlite3")
            with TestClient(create_app(settings=config)) as client:
                client.get("/health")
                client.get("/capabilities")
                client.get("/cache")
                constructor.assert_not_called()
                with patch("src.llm_providers.time.sleep"):
                    response = client.post("/query", json={"question": "What is semantic caching?"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["hit_type"], "miss")
                self.assertEqual(client.post("/query", json={"question": "What is semantic caching?"}).json()["hit_type"], "exact")
                self.assertEqual(client.post("/evaluation", json={"threshold": .84}).status_code, 200)
                constructor.assert_called_once_with(config.embedding_model)

    def test_evaluation_can_be_first_model_user(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.embeddings.SentenceTransformer") as constructor:
            constructor.return_value.encode.return_value = [1.0, 0.0]
            config = Settings(app_mode="demo", database_path=Path(directory) / "cache.sqlite3")
            with TestClient(create_app(settings=config)) as client:
                constructor.assert_not_called()
                response = client.post("/evaluation", json={})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()["selected_rows"]), 36)
                constructor.assert_called_once_with(config.embedding_model)

    def test_concurrent_first_encodes_share_one_model(self):
        gate = Barrier(8)
        with patch("src.embeddings.SentenceTransformer") as constructor:
            constructor.return_value.encode.return_value = [1.0, 0.0]
            service = EmbeddingService("test-model", lazy=True)
            constructor.assert_not_called()

            def encode(_):
                gate.wait(timeout=5)
                return service.encode("question")

            with ThreadPoolExecutor(max_workers=8) as executor:
                self.assertEqual(list(executor.map(encode, range(8))), [[1.0, 0.0]] * 8)
            constructor.assert_called_once_with("test-model")

    def test_failed_load_is_retryable_without_fallback(self):
        model = Mock()
        model.encode.return_value = [1.0, 0.0]
        with patch("src.embeddings.SentenceTransformer", side_effect=[RuntimeError("unavailable"), model]) as constructor:
            service = EmbeddingService("test-model", lazy=True)
            with self.assertRaises(ValueError):
                service.encode(" ")
            constructor.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                service.encode("question")
            self.assertIsNone(service.model)
            self.assertEqual(service.encode("question"), [1.0, 0.0])
            self.assertEqual(service.encode("question"), [1.0, 0.0])
            self.assertEqual(constructor.call_count, 2)
