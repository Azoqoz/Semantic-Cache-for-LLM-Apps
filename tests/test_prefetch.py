"""Build/runtime artifact contract without network downloads."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.config import settings
from src.embeddings import SentenceTransformer, model_cache_path
from src.prefetch_model import prefetch


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch("src.embeddings.PROJECT_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.model = Mock()
        self.model.save_pretrained.side_effect = lambda path: Path(path).mkdir(parents=True)
        self.constructor = Mock(return_value=self.model)
        module_patch = patch.dict("sys.modules", {"sentence_transformers": Mock(SentenceTransformer=self.constructor)})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def test_prefetch_and_runtime_share_configured_model_and_offline_artifact(self):
        with patch.dict(os.environ, {"RENDER": "true"}):
            prefetch()
            path = model_cache_path(settings.embedding_model)
            self.assertEqual(path, self.root / ".model-cache" / "sentence-transformers--all-MiniLM-L6-v2")
            self.model.save_pretrained.assert_called_once_with(str(path))
            self.assertEqual(self.constructor.call_args_list[0].args, (settings.embedding_model,))
            self.constructor.assert_called_with(str(path), local_files_only=True)
            self.constructor.reset_mock()
            with self.assertLogs("uvicorn.error", level="INFO") as logs:
                SentenceTransformer(settings.embedding_model)
            self.constructor.assert_called_once_with(str(path), local_files_only=True)
            output = "\n".join(logs.output)
            for stage in ("importing_sentence_transformers", "sentence_transformers_imported", "locating_cached_model", "constructing_sentence_transformer", "sentence_transformer_constructed"):
                self.assertIn("stage=" + stage, output)
            self.assertIn(str(path), output)
            self.assertIn("local_files_only=True", output)

    def test_render_missing_artifact_never_downloads(self):
        with patch.dict(os.environ, {"RENDER": "true"}):
            with self.assertRaisesRegex(RuntimeError, "build artifact missing"):
                SentenceTransformer(settings.embedding_model)
        self.constructor.assert_not_called()

    def test_existing_invalid_artifact_fails_build_without_download_retry(self):
        model_cache_path(settings.embedding_model).mkdir(parents=True)
        self.constructor.side_effect = RuntimeError("incomplete artifact")
        with self.assertRaisesRegex(RuntimeError, "incomplete artifact"):
            prefetch()
        self.constructor.assert_called_once_with(str(model_cache_path(settings.embedding_model)), local_files_only=True)

    def test_local_without_artifact_preserves_original_loading(self):
        with patch.dict(os.environ, {"RENDER": ""}):
            SentenceTransformer(settings.embedding_model)
        self.constructor.assert_called_once_with(settings.embedding_model)
