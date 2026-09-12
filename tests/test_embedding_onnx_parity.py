"""Real offline migration parity. See BACKEND.md for artifact preparation.

Set SENTENCE_TRANSFORMER_REFERENCE to a local copy of the pinned reference model.
Without reference artifacts, ordinary unit tests need neither PyTorch nor downloads.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.cache_store import SQLiteSemanticCache
from src.config import settings
from src.embeddings import EmbeddingService, model_cache_path
from src.evaluation import EVALUATION_DATASET, COMPARISON_THRESHOLDS, build_evaluation_report
from src.onnx_embeddings import OnnxSentenceEncoder


class OnnxPoolingTests(unittest.TestCase):
    def test_mean_pooling_ignores_padding_and_normalizes_float32(self):
        encoder = OnnxSentenceEncoder.__new__(OnnxSentenceEncoder)
        encoder._inference_lock = Lock()
        encoder.tokenizer = Mock()
        encoder.tokenizer.encode.return_value = SimpleNamespace(ids=[101, 102, 0], attention_mask=[1, 1, 0], type_ids=[0, 0, 0])
        hidden = np.zeros((1, 3, 384), dtype=np.float32)
        hidden[0, 0, 0], hidden[0, 1, 1], hidden[0, 2, :] = 6, 8, 1000
        encoder.session = Mock()
        encoder.session.run.return_value = [hidden]
        result = encoder.encode(" example ")
        np.testing.assert_allclose(result[:2], [.6, .8], atol=1e-7)
        self.assertTrue(np.all(result[2:] == 0))
        self.assertEqual(result.dtype, np.float32)
        encoder.tokenizer.encode.assert_called_once_with("example", add_special_tokens=True)


@unittest.skipUnless((model_cache_path(settings.embedding_model) / "onnx/model.onnx").is_file(),
                     "Run python -m src.prefetch_model for real runtime tests")
class OnnxStartupTests(unittest.TestCase):
    def test_real_api_startup_and_query_without_torch_transformers_or_network(self):
        script = textwrap.dedent('''
            import importlib.abc, sys, socket, tempfile, time
            from pathlib import Path
            class BlockHeavy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'torch','sentence_transformers','transformers','huggingface_hub'}:
                        raise AssertionError('Forbidden runtime import: ' + fullname)
            sys.meta_path.insert(0, BlockHeavy())
            def no_network(*args, **kwargs):
                raise AssertionError('Runtime network access')
            original_connect = socket.socket.connect
            def local_connect(sock, address):
                # Windows asyncio uses a loopback socket pair for its wakeup pipe.
                if isinstance(address, tuple) and address[0] in ('127.0.0.1', '::1'):
                    return original_connect(sock, address)
                return no_network()
            socket.socket.connect = local_connect
            socket.create_connection = no_network
            start = time.monotonic()
            from fastapi.testclient import TestClient
            from src.api import create_app
            from src.config import Settings
            from unittest.mock import patch
            with tempfile.TemporaryDirectory() as directory:
                app = create_app(settings=Settings(app_mode='local', database_path=Path(directory)/'cache.sqlite3'))
                with TestClient(app) as client:
                    assert client.get('/health').status_code == 200
                    assert app.state.warmup_complete.wait(30)
                    assert client.get('/ready').json() == {'status':'ready'}
                    with patch('src.llm_providers.time.sleep'):
                        response = client.post('/query',json={'question':'What is semantic caching?'})
                    assert response.status_code == 200, response.text
                    assert response.json()['hit_type'] == 'miss'
                    assert client.post('/query',json={'question':'What is semantic caching?'}).json()['hit_type'] == 'exact'
            print('ONNX API startup + two queries seconds:', round(time.monotonic()-start,3))
            assert not any(name in sys.modules for name in ('torch','sentence_transformers','transformers','huggingface_hub'))
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip())


@unittest.skipUnless(os.getenv("SENTENCE_TRANSFORMER_REFERENCE"), "Set local reference path for real PyTorch parity")
class EmbeddingOnnxParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sentence_transformers import SentenceTransformer
        cls.reference = SentenceTransformer(os.environ["SENTENCE_TRANSFORMER_REFERENCE"],
                                            local_files_only=True, device="cpu")
        cls.onnx = OnnxSentenceEncoder(settings.embedding_model)
        cls.texts = list(dict.fromkeys(
            [text for pair in EVALUATION_DATASET for text in (pair.original_question, pair.similar_question)]
            + ["  Hello café 世界 👋  ", "CACHE   HIT!", "cache " * 300, "cache " * 254 + "totally different tail", "x"]))
        cls.old = np.asarray(cls.reference.encode([t.strip() for t in cls.texts], normalize_embeddings=True,
                                                  show_progress_bar=False))
        cls.new = np.asarray([cls.onnx.encode(text) for text in cls.texts])
        cls.index = {text: i for i, text in enumerate(cls.texts)}
        print(f"Real parity: {len(cls.texts)} prompts, max component error={np.max(np.abs(cls.old-cls.new)):.3g}")

    def test_dimensions_normalization_and_components(self):
        self.assertEqual(self.new.shape, (len(self.texts), 384))
        self.assertEqual(self.new.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(self.new, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(self.new, self.old, atol=2e-5, rtol=2e-4)

    def test_similarity_ordering_and_all_evaluation_threshold_decisions(self):
        old_scores, new_scores = self.old @ self.old.T, self.new @ self.new.T
        np.testing.assert_allclose(new_scores, old_scores, atol=2e-5, rtol=2e-4)
        candidates = [self.index[t] for t in ("What is semantic caching?", "What is an embedding?", "What is SQLite?")]
        for row in range(len(self.texts)):
            np.testing.assert_array_equal(np.argsort(old_scores[row, candidates]), np.argsort(new_scores[row, candidates]))
        for pair in EVALUATION_DATASET:
            a, b = self.index[pair.original_question], self.index[pair.similar_question]
            for threshold in COMPARISON_THRESHOLDS:
                self.assertEqual(old_scores[a,b] >= threshold, new_scores[a,b] >= threshold)

    def test_existing_persisted_vectors_keep_exact_semantic_and_miss_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            embeddings = EmbeddingService(settings.embedding_model, lazy=True)
            database = Path(directory) / "existing.sqlite3"
            cache = SQLiteSemanticCache(database, embeddings)
            stored = ["What is semantic caching?", "What is an embedding?", "What is SQLite?"]
            for text in stored:
                cache.add(text, "stored answer", self.old[self.index[text]].tolist(), "Demo", "demo-rule-based", 168)
            # Reopen the database to exercise cross-runtime persisted compatibility.
            cache = SQLiteSemanticCache(database, embeddings)
            seen = set()
            for text in self.texts:
                for threshold in COMPARISON_THRESHOLDS:
                    baseline = cache.lookup(text, self.old[self.index[text]].tolist(), threshold, "Demo", "demo-rule-based")
                    migrated = cache.lookup(text, self.new[self.index[text]].tolist(), threshold, "Demo", "demo-rule-based")
                    self.assertEqual((baseline.hit_type, baseline.entry_id, baseline.matched_question),
                                     (migrated.hit_type, migrated.entry_id, migrated.matched_question))
                    seen.add(migrated.hit_type)
            self.assertEqual(seen, {"exact", "semantic", "miss"})

    def test_evaluation_report_metrics_and_recommendation(self):
        class Vectors:
            cosine_similarity = staticmethod(EmbeddingService.cosine_similarity)
            def __init__(inner, values):
                inner.values = values
            def encode(inner, text):
                return inner.values[self.index[text]].tolist()
        old = build_evaluation_report(Vectors(self.old), settings.default_threshold)
        new = build_evaluation_report(Vectors(self.new), settings.default_threshold)
        for field in ("selected_metrics", "difficulty_rows", "comparison_rows", "recommended_threshold"):
            self.assertEqual(getattr(old, field), getattr(new, field))
