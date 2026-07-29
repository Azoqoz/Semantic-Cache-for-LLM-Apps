from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_APP_MODES = {"demo", "local"}


def read_app_mode() -> str:
    """Read and validate the application mode from the environment."""
    app_mode = os.getenv("APP_MODE", "local").strip().lower()
    if app_mode not in SUPPORTED_APP_MODES:
        supported = ", ".join(sorted(SUPPORTED_APP_MODES))
        raise ValueError(
            f"Invalid APP_MODE '{app_mode}'. Supported values are: {supported}."
        )
    return app_mode


@dataclass(frozen=True)
class Settings:
    app_mode: str = field(default_factory=read_app_mode)
    database_path: Path = DATA_DIR / "semantic_cache.sqlite3"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_threshold: float = 0.84
    default_ttl_hours: int = 168
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    ).rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")


settings = Settings()
