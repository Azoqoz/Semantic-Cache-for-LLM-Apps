from __future__ import annotations

from functools import lru_cache
import os
import logging
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np

from src.config import PROJECT_ROOT


def log_warmup(stage: str, started_at: float, **details) -> None:
    """Only caller-supplied diagnostics, never user input or exception messages."""
    memory = "rss_mib=unavailable"
    try:
        # Linux/Render: current resident memory, without importing a monitor.
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                memory = f"rss_mib={int(line.split()[1]) / 1024:.1f}"
                break
    except (OSError, ValueError):
        pass
    logging.getLogger("uvicorn.error").info(
        "semantic_warmup stage=%s elapsed_seconds=%.3f %s %s",
        stage, monotonic() - started_at, memory,
        " ".join(f"{key}={value}" for key, value in details.items()),
    )


def model_cache_path(model_name: str) -> Path:
    """Build artifact location shared by prefetch and runtime, independent of cwd."""
    return PROJECT_ROOT / ".model-cache" / model_name.replace("/", "--")

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerModel


def SentenceTransformer(model_name: str) -> SentenceTransformerModel:
    """Import the heavy inference runtime only when a model is requested."""
    started_at = monotonic()
    log_warmup("importing_sentence_transformers", started_at)
    from sentence_transformers import SentenceTransformer as Model
    log_warmup("sentence_transformers_imported", started_at)

    path = model_cache_path(model_name)
    log_warmup("locating_cached_model", started_at, path=path)
    if path.is_dir():
        source, offline = str(path), True
    else:
        if os.getenv("RENDER") == "true":
            raise RuntimeError("Embedding build artifact missing; run python -m src.prefetch_model during build.")
        source, offline = model_name, False
    log_warmup("constructing_sentence_transformer", started_at, source=source, local_files_only=offline)
    model = Model(source, local_files_only=True) if offline else Model(source)
    log_warmup("sentence_transformer_constructed", started_at)
    return model


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
