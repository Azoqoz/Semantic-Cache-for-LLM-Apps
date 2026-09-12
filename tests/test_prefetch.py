"""Build/runtime artifact contract without downloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.config import settings
from src.embeddings import model_cache_path
from src.onnx_embeddings import MODEL_FILES, MODEL_REVISION, OnnxSentenceEncoder
from src.prefetch_model import prefetch


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch("src.embeddings.PROJECT_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_prefetch_uses_configured_model_revision_and_shared_local_path(self):
        download = Mock()
        with patch.dict("sys.modules", {"huggingface_hub": Mock(snapshot_download=download)}), \
                patch("src.prefetch_model.OnnxSentenceEncoder") as constructor:
            prefetch()
        path = model_cache_path(settings.embedding_model)
        self.assertEqual(path, self.root / ".model-cache" / "sentence-transformers--all-MiniLM-L6-v2" / ("onnx-" + MODEL_REVISION))
        download.assert_called_once_with(repo_id=settings.embedding_model, revision=MODEL_REVISION,
                                         local_dir=str(path), allow_patterns=list(MODEL_FILES))
        constructor.assert_called_once_with(settings.embedding_model)
        constructor.return_value.encode.assert_called_once_with("Semantic cache warm-up.")
        self.assertNotIn("model.safetensors", MODEL_FILES)

    def test_missing_artifact_fails_before_runtime_import_or_network(self):
        with patch.dict("sys.modules", {"onnxruntime": None, "torch": None, "sentence_transformers": None,
                                         "huggingface_hub": None}):
            with self.assertRaisesRegex(RuntimeError, "build artifact missing"):
                OnnxSentenceEncoder(settings.embedding_model)

    def test_invalid_inference_fails_prefetch(self):
        with patch.dict("sys.modules", {"huggingface_hub": Mock()}), \
                patch("src.prefetch_model.OnnxSentenceEncoder") as constructor:
            constructor.return_value.encode.side_effect = RuntimeError("bad output")
            with self.assertRaisesRegex(RuntimeError, "bad output"):
                prefetch()

    def test_unsupported_model_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "Unsupported embedding model"):
            OnnxSentenceEncoder("different-model")
