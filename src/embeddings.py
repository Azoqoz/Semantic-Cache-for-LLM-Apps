from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Creates normalized sentence embeddings using a local model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = self._load_model(model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        return SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty.")

        vector = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32).tolist()

    @staticmethod
    def cosine_similarity(
        first: list[float],
        second: list[float],
    ) -> float:
        vector_a = np.asarray(first, dtype=np.float32)
        vector_b = np.asarray(second, dtype=np.float32)

        denominator = float(
            np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        )
        if denominator == 0:
            return 0.0

        return float(np.dot(vector_a, vector_b) / denominator)
