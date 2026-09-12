"""Embedding/provider contracts without model downloads, credentials, or network calls."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

from src.embeddings import EmbeddingService
from src import llm_providers as providers


class EmbeddingParity(unittest.TestCase):
    def test_encoding_trims_requests_normalization_and_returns_float32_list(self):
        with patch("src.embeddings.SentenceTransformer") as constructor:
            EmbeddingService._load_model.cache_clear()
            self.addCleanup(EmbeddingService._load_model.cache_clear)
            constructor.return_value.encode.return_value = np.array([0.123456789, 1], dtype=np.float64)
            service = EmbeddingService("test-model")
            result = service.encode("  Hello  world \n")
            constructor.assert_called_once_with("test-model")
            constructor.return_value.encode.assert_called_once_with(
                "Hello  world", normalize_embeddings=True, show_progress_bar=False)
            self.assertEqual(result, [float(np.float32(.123456789)), 1.0])
            with self.assertRaisesRegex(ValueError, "Text cannot be empty"):
                service.encode(" \n")
            self.assertEqual(constructor.return_value.encode.call_count, 1)

    def test_model_loading_reuses_names_and_evicts_after_two_models(self):
        with patch("src.embeddings.SentenceTransformer") as constructor:
            EmbeddingService._load_model.cache_clear()
            self.addCleanup(EmbeddingService._load_model.cache_clear)
            first = EmbeddingService("a")
            self.assertIs(EmbeddingService("a").model, first.model)
            EmbeddingService("b")
            EmbeddingService("c")
            EmbeddingService("a")
            self.assertEqual([call.args[0] for call in constructor.call_args_list], ["a", "b", "c", "a"])

    def test_cosine_is_scale_invariant_signed_and_zero_safe(self):
        for a, b, expected in (([2, 0], [5, 0], 1), ([1, 0], [0, 1], 0),
                               ([1, 0], [-1, 0], -1), ([0, 0], [1, 0], 0),
                               ([1, 0], [3, 4], .6), ([], [], 0)):
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(EmbeddingService.cosine_similarity(a, b), expected)


class ProviderParity(unittest.TestCase):
    def build(self, name, mode="local"):
        return providers.build_provider(name, "openai-model", "claude-model", "gemini-model",
                                        "http://localhost:11434/", "ollama-model", app_mode=mode)

    def test_demo_mode_rejects_all_external_providers_before_construction(self):
        for name in ("OpenAI", "Claude", "Gemini", "Ollama", "Unknown"):
            with self.subTest(name=name), patch.object(providers, name + "Provider", create=True) as constructor:
                with self.assertRaisesRegex(providers.ProviderConfigurationError, "External providers are disabled"):
                    self.build(name, "demo")
                constructor.assert_not_called()
        self.assertIsInstance(self.build("Demo", "demo"), providers.DemoProvider)

    def test_local_dispatch_passes_model_and_url_and_rejects_unknown_inputs(self):
        for name, args in (("OpenAI", ("openai-model",)), ("Claude", ("claude-model",)),
                           ("Gemini", ("gemini-model",)),
                           ("Ollama", ("http://localhost:11434/", "ollama-model"))):
            with self.subTest(name=name), patch.object(providers, name + "Provider") as constructor:
                self.assertIs(self.build(name), constructor.return_value)
                constructor.assert_called_once_with(*args)
        self.assertIsInstance(self.build("Demo"), providers.DemoProvider)
        for name, mode in (("Unknown", "local"), ("Demo", "invalid")):
            with self.assertRaises(providers.ProviderConfigurationError):
                self.build(name, mode)

    def test_cloud_providers_require_environment_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            for name, key in (("OpenAI", "OPENAI_API_KEY"), ("Claude", "ANTHROPIC_API_KEY"),
                              ("Gemini", "GEMINI_API_KEY")):
                with self.subTest(name=name), self.assertRaisesRegex(providers.ProviderConfigurationError, key):
                    self.build(name)

    def test_demo_rules_precedence_latency_and_unknown_topic_fallback(self):
        demo = providers.DemoProvider()
        with patch("src.llm_providers.time.sleep") as sleep:
            for prompt, prefix in ((" THRESHOLD   tuning and cache hit ", "Threshold tuning balances"),
                                    ("cache hit", "A cache hit occurs"), ("cache miss", "A cache miss occurs"),
                                    ("embedding", "An embedding is"), ("cosine similarity", "Cosine similarity measures"),
                                    ("save cost", "Semantic caching reduces"), ("reduce cost", "Semantic caching reduces"),
                                    ("how semantic caching works", "Semantic caching embeds"),
                                    ("semantic caching", "Semantic caching stores"),
                                    ("unrecognized topic", "This offline demo uses simple topic rules")):
                with self.subTest(prompt=prompt):
                    result = demo.generate(prompt)
                    self.assertTrue(result.text.startswith(prefix))
                    self.assertEqual((result.provider, result.model, result.estimated_cost_usd),
                                     ("Demo", "demo-rule-based", .002))
                    sleep.assert_called_with(1.2)
            self.assertEqual(sleep.call_count, 10)

    def test_openai_request_response_tokens_and_error_propagation(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}), patch("src.llm_providers.OpenAI") as sdk:
            provider = providers.OpenAIProvider("model")
        request = sdk.return_value.responses.create
        request.return_value = SimpleNamespace(output_text=" answer ", usage=SimpleNamespace(input_tokens=7, output_tokens=9))
        result = provider.generate(" prompt ")
        request.assert_called_once_with(model="model", input="prompt")
        self.assertEqual((result.text, result.input_tokens, result.output_tokens, result.estimated_cost_usd),
                         (" answer ", 7, 9, None))
        request.return_value = SimpleNamespace(output_text="answer")
        self.assertIsNone(provider.generate("prompt").input_tokens)
        request.side_effect = RuntimeError("upstream failure")
        with self.assertRaisesRegex(RuntimeError, "upstream failure"):
            provider.generate("prompt")

    def test_claude_request_joins_text_blocks_and_handles_missing_usage(self):
        # Bypass SDK construction only; execute the actual adapter method.
        provider = providers.ClaudeProvider.__new__(providers.ClaudeProvider)
        provider.model = "model"
        provider.client = Mock()
        request = provider.client.messages.create
        request.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text=" first"),
                                                        SimpleNamespace(type="tool_use"),
                                                        SimpleNamespace(type="text", text=" second ")],
                                               usage=SimpleNamespace(input_tokens=2, output_tokens=3))
        result = provider.generate(" prompt ")
        request.assert_called_once_with(model="model", max_tokens=1024,
                                        messages=[{"role": "user", "content": "prompt"}])
        self.assertEqual((result.text, result.input_tokens, result.output_tokens), ("first second", 2, 3))
        request.return_value = SimpleNamespace(content=[])
        result = provider.generate("prompt")
        self.assertEqual(result.text, "")
        self.assertIsNone(result.input_tokens)
        request.side_effect = RuntimeError("upstream failure")
        with self.assertRaisesRegex(RuntimeError, "upstream failure"):
            provider.generate("prompt")

    def test_gemini_request_tokens_and_empty_text(self):
        provider = providers.GeminiProvider.__new__(providers.GeminiProvider)
        provider.model = "model"
        provider.client = Mock()
        request = provider.client.models.generate_content
        request.return_value = SimpleNamespace(text=" answer ", usage_metadata=SimpleNamespace(
            prompt_token_count=4, candidates_token_count=5))
        result = provider.generate(" prompt ")
        request.assert_called_once_with(model="model", contents="prompt")
        self.assertEqual((result.text, result.input_tokens, result.output_tokens), ("answer", 4, 5))
        request.return_value = SimpleNamespace(text=None)
        result = provider.generate("prompt")
        self.assertEqual(result.text, "")
        self.assertIsNone(result.input_tokens)
        request.side_effect = RuntimeError("upstream failure")
        with self.assertRaisesRegex(RuntimeError, "upstream failure"):
            provider.generate("prompt")

    def test_ollama_http_contract_empty_payload_and_connection_http_errors(self):
        provider = providers.OllamaProvider("http://localhost:11434///", "model")
        with patch("src.llm_providers.requests.post") as post:
            post.return_value.json.return_value = {"response": " answer ", "prompt_eval_count": 2, "eval_count": 3}
            result = provider.generate(" prompt ")
            post.assert_called_once_with("http://localhost:11434/api/generate",
                                         json={"model": "model", "prompt": "prompt", "stream": False}, timeout=120)
            post.return_value.raise_for_status.assert_called_once_with()
            self.assertEqual((result.text, result.input_tokens, result.output_tokens, result.estimated_cost_usd),
                             ("answer", 2, 3, 0.0))
            post.return_value.json.return_value = {}
            self.assertEqual(provider.generate("prompt").text, "")
            post.return_value.raise_for_status.side_effect = requests.HTTPError("bad status")
            with self.assertRaisesRegex(RuntimeError, "Could not connect to Ollama"):
                provider.generate("prompt")
            post.side_effect = requests.ConnectionError("offline")
            with self.assertRaisesRegex(RuntimeError, "Could not connect to Ollama") as error:
                provider.generate("prompt")
            self.assertIsInstance(error.exception.__cause__, requests.ConnectionError)

    def test_app_mode_normalization_default_and_validation(self):
        # Prevent dotenv from reading real credentials on this test's first config import.
        with patch.dict(os.environ, {"APP_MODE": "local"}), patch("dotenv.load_dotenv"):
            from src.config import read_app_mode, Settings
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_app_mode(), "local")
            for raw, expected in ((" DEMO ", "demo"), (" LOCAL ", "local")):
                os.environ["APP_MODE"] = raw
                self.assertEqual(Settings().app_mode, expected)
            os.environ["APP_MODE"] = "invalid"
            with self.assertRaisesRegex(ValueError, "Invalid APP_MODE"):
                read_app_mode()
