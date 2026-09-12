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
