from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from src.embeddings import EmbeddingService
from src.models import CacheEntry, CacheLookupResult


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteSemanticCache:
    """Persistent SQLite cache that performs semantic nearest-match lookup."""

    def __init__(
        self,
        database_path: Path,
        embedding_service: EmbeddingService,
    ) -> None:
        self.database_path = database_path
        self.embedding_service = embedding_service
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_cache_normalized_question
                ON cache_entries(normalized_question);

                CREATE INDEX IF NOT EXISTS idx_cache_created_at
                ON cache_entries(created_at);

                CREATE TABLE IF NOT EXISTS query_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    similarity REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    estimated_cost_usd REAL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._migrate_query_events(connection)

    @staticmethod
    def _migrate_query_events(connection: sqlite3.Connection) -> None:
        """Add required analytics columns when opening an older database."""
        required_columns = {
            "question": "TEXT NOT NULL DEFAULT ''",
            "cache_hit": "INTEGER NOT NULL DEFAULT 0",
            "similarity": "REAL NOT NULL DEFAULT 0",
            "latency_ms": "REAL NOT NULL DEFAULT 0",
            "provider": "TEXT NOT NULL DEFAULT 'Unknown'",
            "model": "TEXT NOT NULL DEFAULT 'Unknown'",
            "estimated_cost_usd": "REAL",
            "created_at": "TEXT NOT NULL DEFAULT ''",
        }
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(query_events)").fetchall()
        }
        for column, definition in required_columns.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE query_events ADD COLUMN {column} {definition}"
                )

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.lower().strip().split())

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            embedding=json.loads(row["embedding_json"]),
            provider=row["provider"],
            model=row["model"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            access_count=row["access_count"],
            last_accessed_at=row["last_accessed_at"],
        )

    def add(
        self,
        question: str,
        answer: str,
        embedding: list[float],
        provider: str,
        model: str,
        ttl_hours: Optional[int],
    ) -> int:
        created_at = utc_now()
        expires_at = (
            created_at + timedelta(hours=ttl_hours)
            if ttl_hours and ttl_hours > 0
            else None
        )

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO cache_entries (
                    question,
                    normalized_question,
                    answer,
                    embedding_json,
                    provider,
                    model,
                    created_at,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question.strip(),
                    self._normalize_question(question),
                    answer.strip(),
                    json.dumps(embedding),
                    provider,
                    model,
                    created_at.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                ),
            )
            return int(cursor.lastrowid)

    def lookup(
        self,
        question: str,
        query_embedding: list[float],
        threshold: float,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> CacheLookupResult:
        normalized_question = self._normalize_question(question)
        now_iso = utc_now().isoformat()

        sql = """
            SELECT *
            FROM cache_entries
            WHERE (expires_at IS NULL OR expires_at > ?)
        """
        parameters: list[object] = [now_iso]

        if provider:
            sql += " AND provider = ?"
            parameters.append(provider)
        if model:
            sql += " AND model = ?"
            parameters.append(model)

        with closing(self._connect()) as connection, connection:
            exact_row = connection.execute(
                sql + " AND normalized_question = ? ORDER BY id DESC LIMIT 1",
                [*parameters, normalized_question],
            ).fetchone()

            if exact_row:
                entry = self._row_to_entry(exact_row)
                self._record_access(entry.id)
                return CacheLookupResult(
                    hit=True,
                    answer=entry.answer,
                    matched_question=entry.question,
                    similarity=1.0,
                    entry_id=entry.id,
                )

            rows = connection.execute(
                sql + " ORDER BY id DESC",
                parameters,
            ).fetchall()

        best_entry: Optional[CacheEntry] = None
        best_similarity = -1.0

        for row in rows:
            entry = self._row_to_entry(row)
            similarity = self.embedding_service.cosine_similarity(
                query_embedding,
                entry.embedding,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry and best_similarity >= threshold:
            self._record_access(best_entry.id)
            return CacheLookupResult(
                hit=True,
                answer=best_entry.answer,
                matched_question=best_entry.question,
                similarity=best_similarity,
                entry_id=best_entry.id,
            )

        return CacheLookupResult(
            hit=False,
            similarity=max(best_similarity, 0.0),
            matched_question=best_entry.question if best_entry else None,
        )

    def _record_access(self, entry_id: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE cache_entries
                SET access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE id = ?
                """,
                (utc_now().isoformat(), entry_id),
            )

    def record_query_event(
        self,
        question: str,
        cache_hit: bool,
        similarity: float,
        latency_ms: float,
        provider: str,
        model: str,
        estimated_cost_usd: Optional[float],
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO query_events (
                    question,
                    cache_hit,
                    similarity,
                    latency_ms,
                    provider,
                    model,
                    estimated_cost_usd,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question.strip(),
                    int(cache_hit),
                    similarity,
                    latency_ms,
                    provider,
                    model,
                    estimated_cost_usd,
                    utc_now().isoformat(),
                ),
            )

    def list_entries(self, limit: int = 100) -> list[dict[str, object]]:
        """Return the most recent cache entries for the explorer UI."""
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    question,
                    answer,
                    provider,
                    model,
                    created_at,
                    expires_at,
                    access_count,
                    last_accessed_at
                FROM cache_entries
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def metrics(self) -> dict[str, float | int]:
        """Aggregate cache usage, latency, and estimated cost metrics."""
        with closing(self._connect()) as connection, connection:
            event_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_queries,
                    COALESCE(SUM(cache_hit), 0) AS cache_hits,
                    COALESCE(AVG(latency_ms), 0) AS average_latency_ms,
                    COALESCE(AVG(CASE WHEN cache_hit = 1 THEN latency_ms END), 0)
                        AS average_hit_latency_ms,
                    COALESCE(AVG(CASE WHEN cache_hit = 0 THEN latency_ms END), 0)
                        AS average_miss_latency_ms,
                    COALESCE(
                        SUM(CASE WHEN cache_hit = 0 THEN estimated_cost_usd ELSE 0 END),
                        0
                    ) AS llm_cost_usd,
                    COALESCE(
                        SUM(CASE WHEN cache_hit = 1 THEN estimated_cost_usd ELSE 0 END),
                        0
                    ) AS avoided_cost_usd
                FROM query_events
                """
            ).fetchone()

            cache_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS cache_entries,
                    COALESCE(SUM(access_count), 0) AS total_reuses
                FROM cache_entries
                """
            ).fetchone()

        total_queries = int(event_row["total_queries"])
        cache_hits = int(event_row["cache_hits"])
        hit_rate = cache_hits / total_queries if total_queries else 0.0
        llm_cost = float(event_row["llm_cost_usd"])
        avoided_cost = float(event_row["avoided_cost_usd"])
        total_cost_without_cache = llm_cost + avoided_cost
        savings_percentage = (
            avoided_cost / total_cost_without_cache * 100
            if total_cost_without_cache
            else 0.0
        )

        return {
            "total_queries": total_queries,
            "cache_hits": cache_hits,
            "cache_misses": total_queries - cache_hits,
            "hit_rate": hit_rate,
            "average_latency_ms": float(event_row["average_latency_ms"]),
            "average_hit_latency_ms": float(event_row["average_hit_latency_ms"]),
            "average_miss_latency_ms": float(event_row["average_miss_latency_ms"]),
            "llm_cost_usd": llm_cost,
            "avoided_cost_usd": avoided_cost,
            "total_cost_without_cache_usd": total_cost_without_cache,
            "savings_percentage": savings_percentage,
            "cache_entries": int(cache_row["cache_entries"]),
            "total_reuses": int(cache_row["total_reuses"]),
        }

    def clear(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM cache_entries")
            connection.execute("DELETE FROM query_events")

    def delete_entries(self, entry_ids: Iterable[int]) -> None:
        ids = list(entry_ids)
        if not ids:
            return

        placeholders = ",".join("?" for _ in ids)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                f"DELETE FROM cache_entries WHERE id IN ({placeholders})",
                ids,
            )
