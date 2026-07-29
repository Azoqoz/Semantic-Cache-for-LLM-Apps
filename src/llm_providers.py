from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import requests
from openai import OpenAI

from src.models import LLMResponse


class ProviderConfigurationError(ValueError):
    """A safe, user-facing provider configuration error."""


class LLMProvider(ABC):
    """Interface implemented by every supported completion provider."""

    name: str
    model: str

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for a prompt."""
        raise NotImplementedError


class DemoProvider(LLMProvider):
    """Realistic offline provider for demonstrating semantic-cache behavior."""

    name = "Demo"

    def __init__(self) -> None:
        self.model = "demo-rule-based"

    def generate(self, prompt: str) -> LLMResponse:
        """Return a rule-based answer after simulating normal LLM latency."""
        time.sleep(1.2)
        question = " ".join(prompt.lower().strip().split())
        rules: list[tuple[tuple[str, ...], str]] = [
            (("threshold", "tuning"), "Threshold tuning balances precision and recall. A threshold that is too low may reuse answers for different intents, while one that is too high misses useful reuse opportunities. Evaluate representative question pairs to choose a value that fits the application's risk tolerance."),
            (("cache hit",), "A cache hit occurs when a valid stored question is sufficiently similar to the incoming question. The application returns the stored answer instead of calling the LLM, reducing both response time and estimated model cost."),
            (("cache miss",), "A cache miss occurs when no valid cached question reaches the selected similarity threshold. The application calls the configured LLM provider, returns its answer, and stores the new question and embedding for future reuse."),
            (("embedding",), "An embedding is a numeric vector that represents the meaning of text. Questions with similar intent tend to have vectors near one another, enabling semantic comparison even when their wording differs."),
            (("cosine", "similarity"), "Cosine similarity measures the angle between two vectors. For normalized text embeddings, a score closer to 1 indicates stronger semantic similarity, while lower scores indicate less related meanings."),
            (("cost", "save"), "Semantic caching reduces LLM cost by serving answers for repeated or rephrased requests from local storage. Every safe cache hit avoids another provider call, so savings grow with request volume and repeated intent."),
            (("reduce", "cost"), "Semantic caching reduces LLM cost by serving answers for repeated or rephrased requests from local storage. Every safe cache hit avoids another provider call, so savings grow with request volume and repeated intent."),
            (("how", "semantic", "cach"), "Semantic caching embeds an incoming question, compares that vector with cached question embeddings, and reuses the closest valid answer when its cosine similarity meets the configured threshold. Otherwise, it calls the LLM and caches the new result."),
            (("semantic", "cach"), "Semantic caching stores LLM answers together with vector embeddings of their questions. Unlike exact-key caching, it can reuse an answer for a differently worded question when both questions have sufficiently similar meaning."),
        ]
        answer = next(
            (response for keywords, response in rules if all(word in question for word in keywords)),
            "This offline demo uses simple topic rules rather than a live LLM. I do not have a predefined answer for that question yet, but the response can still be cached and reused for a semantically similar follow-up.",
        )
        return LLMResponse(
            text=answer,
            provider=self.name,
            model=self.model,
            estimated_cost_usd=0.002,
        )


class OpenAIProvider(LLMProvider):
    """OpenAI Responses API provider."""

    name = "OpenAI"

    def __init__(self, model: str) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.responses.create(model=self.model, input=prompt.strip())
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        return LLMResponse(
            text=response.output_text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,
        )


class ClaudeProvider(LLMProvider):
    """Anthropic Messages API provider."""

    name = "Claude"

    def __init__(self, model: str) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "ANTHROPIC_API_KEY is missing. Add it to your .env file."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The Anthropic SDK is not installed. Run pip install -r requirements.txt."
            ) from exc
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt.strip()}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            estimated_cost_usd=None,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini Generate Content API provider."""

    name = "Gemini"

    def __init__(self, model: str) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The Google Gen AI SDK is not installed. Run pip install -r requirements.txt."
            ) from exc
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt.strip(),
        )
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=(response.text or "").strip(),
            provider=self.name,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            estimated_cost_usd=None,
        )


class OllamaProvider(LLMProvider):
    """Provider for a locally running Ollama server."""

    name = "Ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt.strip(), "stream": False},
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Confirm that Ollama is running "
                f"at {self.base_url} and that model '{self.model}' is installed."
            ) from exc
        payload = response.json()
        return LLMResponse(
            text=payload.get("response", "").strip(),
            provider=self.name,
            model=self.model,
            input_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
            estimated_cost_usd=0.0,
        )


def build_provider(
    provider_name: str,
    openai_model: str,
    claude_model: str,
    gemini_model: str,
    ollama_base_url: str,
    ollama_model: str,
    app_mode: str = "local",
) -> LLMProvider:
    """Build the provider selected in the UI."""
    if app_mode == "demo":
        if provider_name != "Demo":
            raise ProviderConfigurationError(
                "External providers are disabled in Hosted Demo mode."
            )
        return DemoProvider()
    if app_mode != "local":
        raise ProviderConfigurationError(
            f"Unsupported application mode: {app_mode}."
        )
    if provider_name == "Demo":
        return DemoProvider()
    if provider_name == "OpenAI":
        return OpenAIProvider(openai_model)
    if provider_name == "Claude":
        return ClaudeProvider(claude_model)
    if provider_name == "Gemini":
        return GeminiProvider(gemini_model)
    if provider_name == "Ollama":
        return OllamaProvider(ollama_base_url, ollama_model)
    raise ProviderConfigurationError(f"Unsupported provider: {provider_name}.")
