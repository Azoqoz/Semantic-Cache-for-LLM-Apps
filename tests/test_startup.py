"""Background startup readiness without live downloads or provider calls."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from threading import Event, current_thread
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from src.api import create_app
from src.config import Settings
from src.embeddings import EmbeddingService


class StartupTests(unittest.TestCase):
    def setUp(self):
        EmbeddingService._load_model.cache_clear()
        self.addCleanup(EmbeddingService._load_model.cache_clear)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = Settings(app_mode="demo", database_path=Path(directory.name) / "cache.sqlite3")

    def test_cold_import_and_api_creation_do_not_load_model(self):
        script = textwrap.dedent('''
            import importlib.abc
            import sys
            class BlockInference(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'sentence_transformers', 'torch', 'transformers'}:
                        raise AssertionError('Eager inference import: ' + fullname)
            sys.meta_path.insert(0, BlockInference())
            from src.api import app, create_app
            from fastapi.testclient import TestClient
            with_model_not_started = TestClient(create_app())
            assert with_model_not_started.get('/health').status_code == 200
            assert with_model_not_started.get('/ready').json() == {'status': 'warming'}
            assert with_model_not_started.get('/capabilities').status_code == 200
            assert not any(name in sys.modules for name in ('sentence_transformers', 'torch', 'transformers'))
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).resolve().parents[1],
                                env={**os.environ, "APP_MODE": "demo"},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_health_live_while_warming_then_queries_and_evaluation_reuse_ready_model(self):
        entered, release = Event(), Event()
        model = Mock()
        model.encode.return_value = [1.0, 0.0]
        threads = []

        def load(_):
            threads.append(current_thread().name)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release timed out")
            return model

        with patch("src.embeddings.SentenceTransformer", side_effect=load) as constructor:
            app = create_app(settings=self.config)
            constructor.assert_not_called()
            with TestClient(app) as client:
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(client.get('/health').json(), {'status': 'ok', 'check': 'liveness'})
                    self.assertEqual(client.get('/ready').json(), {'status': 'warming'})
                    self.assertEqual(client.get('/capabilities').status_code, 200)
                    for path, body in (('/query', {'question': 'What is semantic caching?'}), ('/evaluation', {})):
                        response = client.post(path, json=body)
                        self.assertEqual(response.status_code, 503)
                        self.assertEqual(response.json()['error']['code'], 'service_warming')
                    model.encode.assert_not_called()
                    constructor.assert_called_once_with(self.config.embedding_model)
                finally:
                    release.set()
                    self.assertTrue(app.state.warmup_complete.wait(5))
                self.assertEqual(client.get('/ready').json(), {'status': 'ready'})
                model.encode.assert_called_once_with("Semantic cache warm-up.", normalize_embeddings=True, show_progress_bar=False)
                with patch('src.llm_providers.time.sleep'):
                    result = client.post('/query', json={'question': 'What is semantic caching?'})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['hit_type'], 'miss')
                self.assertEqual(client.post('/query', json={'question': 'What is semantic caching?'}).json()['hit_type'], 'exact')
                self.assertEqual(client.post('/evaluation', json={}).status_code, 200)
                constructor.assert_called_once_with(self.config.embedding_model)
                self.assertEqual(threads, ['semantic-engine-warmup'])
            # Re-entering lifespan does not create a second startup attempt.
            with TestClient(app) as client:
                self.assertEqual(client.get('/ready').json(), {'status': 'ready'})
            constructor.assert_called_once()

    def test_requests_cannot_start_initialization_without_startup(self):
        with patch('src.embeddings.SentenceTransformer') as constructor:
            client = TestClient(create_app(settings=self.config))
            self.addCleanup(client.close)
            for _ in range(3):
                self.assertEqual(client.post('/query', json={'question': 'q'}).status_code, 503)
                self.assertEqual(client.post('/evaluation', json={}).status_code, 503)
            constructor.assert_not_called()

    def test_failed_warmup_is_safe_and_never_automatically_retried(self):
        with patch('src.embeddings.SentenceTransformer', side_effect=RuntimeError('secret-detail')) as constructor:
            app = create_app(settings=self.config)
            with TestClient(app) as client:
                self.assertTrue(app.state.warmup_complete.wait(5))
                for _ in range(3):
                    ready = client.get('/ready')
                    self.assertEqual(ready.json()['status'], 'error')
                    self.assertNotIn('secret-detail', ready.text)
                    self.assertEqual(client.get('/health').status_code, 200)
                    for path, body in (('/query', {'question': 'q'}), ('/evaluation', {})):
                        response = client.post(path, json=body)
                        self.assertEqual(response.status_code, 503)
                        self.assertEqual(response.json()['error']['code'], 'model_initialization_failed')
                        self.assertNotIn('secret-detail', response.text)
                constructor.assert_called_once()

    def test_timeout_during_loading_or_first_encode_is_terminal_without_polling(self):
        for stall in ("construction", "encode"):
            with self.subTest(stall=stall):
                EmbeddingService._load_model.cache_clear()
                entered, release = Event(), Event()
                model = Mock()

                def block():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test release timed out")

                def construct(_):
                    if stall == "construction":
                        block()
                    return model

                def encode(*args, **kwargs):
                    if stall == "encode":
                        block()
                    return [1.0, 0.0]

                model.encode.side_effect = encode
                with patch("src.embeddings.SentenceTransformer", side_effect=construct) as constructor:
                    app = create_app(settings=self.config, warmup_timeout_seconds=0.5)
                    with TestClient(app) as client:
                        try:
                            self.assertTrue(entered.wait(5))
                            self.assertEqual(client.get("/health").status_code, 200)
                            self.assertTrue(app.state.warmup_complete.wait(5))
                            ready = client.get("/ready").json()
                            self.assertEqual(ready["status"], "error")
                            self.assertIn("timed out", ready["message"])
                            for path, body in (("/query", {"question": "q"}), ("/evaluation", {})):
                                result = client.post(path, json=body)
                                self.assertEqual(result.status_code, 503)
                                self.assertEqual(result.json()["error"]["code"], "model_initialization_failed")
                        finally:
                            release.set()
                            app.state.warmup_thread.join(5)
                        self.assertFalse(app.state.warmup_thread.is_alive())
                        self.assertEqual(client.get("/ready").json(), ready)
                        constructor.assert_called_once()
                    with TestClient(app) as client:
                        self.assertEqual(client.get("/ready").json(), ready)
                    constructor.assert_called_once()

    def test_first_encode_failure_is_sanitized_in_logs_and_readiness(self):
        model = Mock()
        model.encode.side_effect = RuntimeError("secret-detail")
        with patch("src.embeddings.SentenceTransformer", return_value=model), self.assertLogs("uvicorn.error", level="INFO") as logs:
            app = create_app(settings=self.config)
            with TestClient(app) as client:
                self.assertTrue(app.state.warmup_complete.wait(5))
                self.assertEqual(client.get("/ready").json()["status"], "error")
                self.assertNotIn("secret-detail", client.get("/ready").text)
        output = "\n".join(logs.output)
        for stage in ("thread_started", "application_created", "first_test_encode", "initialization_failure"):
            self.assertIn("stage=" + stage, output)
        self.assertIn("elapsed_seconds=", output)
        self.assertIn("rss_mib=", output)
        self.assertNotIn("secret-detail", output)
        self.assertNotIn("stage=model_ready", output)
