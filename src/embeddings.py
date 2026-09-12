from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from src.config import PROJECT_ROOT


def model_cache_path(model_name: str) -> Path:
    """Build artifact location shared by prefetch and runtime, independent of cwd."""
    return PROJECT_ROOT / ".model-cache" / model_name.replace("/", "--")

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerModel


def SentenceTransformer(model_name: str) -> SentenceTransformerModel:
    """Import the heavy inference runtime only when a model is requested."""
    from sentence_transformers import SentenceTransformer as Model

    path = model_cache_path(model_name)
    if path.is_dir():
        return Model(str(path), local_files_only=True)
    if os.getenv("RENDER") == "true":
        raise RuntimeError("Embedding build artifact missing; run python -m src.prefetch_model during build.")
    return Model(model_name)


class EmbeddingService:
    """Creates normalized sentence embeddings using a local model."""

    def __init__(self, model_name: str, *, lazy: bool = False) -> None:
        self.model_name = model_name
        self._initialization_lock = Lock()
        self.model = None if lazy else self._load_model(model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformerModel:
        return SentenceTransformer(model_name)

    def warm_up(self) -> None:
        """Initialize once without making a query or recording cache activity."""
        with self._initialization_lock:
            if self.model is None:
                self.model = self._load_model(self.model_name)

    def encode(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty.")

        # API requests share this service. A failed load remains retryable, and
        # concurrent first requests cannot construct multiple models here.
        self.warm_up()
        model = self.model

        vector = model.encode(
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
