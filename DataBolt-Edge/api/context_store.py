"""DuckDB context store for DataBolt Edge.

Stores analysis records (parsed summaries + model responses) for history
and multi-turn context retrieval.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

import duckdb

DB_PATH = os.getenv(
    "DATABOLT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "databolt.duckdb"),
)


def _get_conn() -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    return duckdb.connect(DB_PATH)


def init_schema() -> None:
    """Create tables if they do not already exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id    VARCHAR PRIMARY KEY,
            log_type       VARCHAR,
            raw_content    TEXT,
            question       TEXT,
            parsed_json    JSON,
            model_response TEXT,
            optimized_sql  TEXT,
            backend        VARCHAR,
            latency_ms     INTEGER,
            tokens         INTEGER,
            created_at     TIMESTAMP
        )
    """)
    conn.close()


def save_analysis(
    *,
    log_type: str,
    raw_content: str,
    question: str,
    parsed_json: dict,
    model_response: str,
    optimized_sql: str | None,
    backend: str,
    latency_ms: int,
    tokens: int,
) -> str:
    """Persist one analysis record and return its analysis_id."""
    import json

    analysis_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO analyses (
            analysis_id, log_type, raw_content, question,
            parsed_json, model_response, optimized_sql,
            backend, latency_ms, tokens, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            analysis_id,
            log_type,
            raw_content,
            question,
            json.dumps(parsed_json),
            model_response,
            optimized_sql,
            backend,
            latency_ms,
            tokens,
            datetime.utcnow(),
        ],
    )
    conn.close()
    return analysis_id


def get_analysis(analysis_id: str) -> dict | None:
    """Return one analysis record by ID, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT analysis_id, log_type, raw_content, question,
               parsed_json, model_response, optimized_sql,
               backend, latency_ms, tokens, created_at
        FROM analyses WHERE analysis_id = ?
        """,
        [analysis_id],
    ).fetchone()
    conn.close()
    if row is None:
        return None
    keys = [
        "analysis_id", "log_type", "raw_content", "question",
        "parsed_json", "model_response", "optimized_sql",
        "backend", "latency_ms", "tokens", "created_at",
    ]
    return dict(zip(keys, row))


def list_analyses(limit: int = 100) -> list[dict]:
    """Return recent analysis records, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT analysis_id, log_type, question, backend,
               latency_ms, tokens, created_at
        FROM analyses
        ORDER BY created_at DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    conn.close()
    keys = ["analysis_id", "log_type", "question", "backend", "latency_ms", "tokens", "created_at"]
    return [dict(zip(keys, r)) for r in rows]


def table_counts() -> dict[str, int]:
    """Return row counts for all tables."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()
        count = row[0] if row else 0
    except Exception:
        count = 0
    conn.close()
    return {"analyses": count}
