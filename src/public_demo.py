"""Curated public scenarios and isolated, bounded visitor SQLite sandboxes."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import re
import sqlite3
from threading import Lock
from time import time

from src.cache_store import SQLiteSemanticCache
from src.llm_providers import DemoProvider

DEMO_THRESHOLD = 0.84
DEMO_TTL_HOURS = 0
DEMO_MODEL = "demo-rule-based"
CANONICAL_PROMPTS = (
    "What is semantic caching?",
    "How does semantic caching reduce LLM costs?",
)


@dataclass(frozen=True)
class DemoSample:
    id: str
    label: str
    question: str


DEMO_SAMPLES = (
    DemoSample("exact", "Exact reuse", CANONICAL_PROMPTS[0]),
    DemoSample("semantic", "Semantic reuse", "Explain semantic caching."),
    DemoSample("miss", "New intent", "What is a vector database?"),
    DemoSample("savings", "Reuse / cost saving", "How does a semantic cache lower LLM API costs?"),
)
DEMO_QUESTIONS = frozenset(sample.question for sample in DEMO_SAMPLES)
_seed_lock = Lock()


def seed_demo(service) -> None:
    """Only add absent demo canonicals; never clear, update or overwrite rows."""
    if service.settings.app_mode != "demo":
        return
    with _seed_lock:
        existing = {(row["question"], row["provider"], row["model"])
                    for row in service.cache.list_entries(limit=1000)}
        for question in CANONICAL_PROMPTS:
            if (question, "Demo", DEMO_MODEL) not in existing:
                vector = service.embeddings.encode(question)
                answer = DemoProvider().generate(question)
                service.cache.add(question, answer.text, vector, "Demo", DEMO_MODEL, DEMO_TTL_HOURS)


class DemoSessions:
    """A seed template and isolated visitor databases; idle retention is one hour."""
    def __init__(self, template):
        self.template = template
        self.directory = template.cache.database_path.with_suffix(".sessions")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def for_visitor(self, token: str):
        from src.application import CacheApplication, ApplicationError

        if not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ApplicationError("invalid_request")
        with self.lock:
            now = time()
            files = [path for path in self.directory.glob("*.sqlite3")
                     if re.fullmatch(r"[0-9a-f]{32}\.sqlite3", path.name)]
            for path in files:
                if now - path.stat().st_mtime > 3600:
                    path.unlink()
            path = self.directory / f"{token}.sqlite3"
            if not path.exists():
                if len(list(self.directory.glob("*.sqlite3"))) >= 128:
                    raise ApplicationError("demo_capacity")
                with closing(sqlite3.connect(self.template.cache.database_path)) as source, \
                        closing(sqlite3.connect(path)) as target:
                    source.backup(target)
            path.touch()
            cache = SQLiteSemanticCache(path, self.template.embeddings)
            return CacheApplication(self.template.settings, cache, self.template.embeddings)
