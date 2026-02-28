"""Tests for AgentOp-Studio database layer.

conftest.py adds AgentOp-Studio/ to sys.path, so we import from ``backend``
directly (the hyphen in the directory name prevents dotted-import via the
parent package name).
"""

import importlib
import os
import uuid
from datetime import datetime, timezone

import pytest


@pytest.fixture()
def db_path(tmp_path):
    """Provide a temporary DuckDB path and reload the db module to use it."""
    path = str(tmp_path / "test_agentops.duckdb")
    os.environ["AGENTOPS_DB_PATH"] = path

    import backend.db as db_mod  # noqa: PLC0415

    importlib.reload(db_mod)
    yield db_mod
    os.environ.pop("AGENTOPS_DB_PATH", None)


def test_init_schema_creates_all_tables(db_path):
    db = db_path
    db.init_schema()
    conn = db.get_conn()
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    conn.close()
    assert {"runs", "messages", "tool_calls", "memory_snapshots", "diffs", "evaluations"} <= tables


def test_runs_insert_and_query(db_path):
    db = db_path
    db.init_schema()
    run_id = str(uuid.uuid4())
    conn = db.get_conn()
    conn.execute(
        """
        INSERT INTO runs (run_id, agent_id, user_id, start_time, status, config, initial_prompt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [run_id, "test-agent", "test-user", datetime.now(timezone.utc), "running", "{}", "Hello"],
    )
    row = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", [run_id]).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == run_id


def test_messages_insert_and_query(db_path):
    db = db_path
    db.init_schema()
    run_id = str(uuid.uuid4())
    msg_id = str(uuid.uuid4())
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO runs (run_id, agent_id, user_id, start_time, status, config, initial_prompt) "
        "VALUES (?, 'a', 'u', ?, 'running', '{}', 'hi')",
        [run_id, datetime.now(timezone.utc)],
    )
    conn.execute(
        "INSERT INTO messages (message_id, run_id, role, content, timestamp) VALUES (?, ?, 'user', 'Hello', ?)",
        [msg_id, run_id, datetime.now(timezone.utc)],
    )
    row = conn.execute(
        "SELECT role, content FROM messages WHERE message_id = ?", [msg_id]
    ).fetchone()
    conn.close()
    assert row == ("user", "Hello")


def test_tool_calls_insert_and_query(db_path):
    db = db_path
    db.init_schema()
    run_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO runs (run_id, agent_id, user_id, start_time, status, config, initial_prompt) "
        "VALUES (?, 'a', 'u', ?, 'running', '{}', 'hi')",
        [run_id, datetime.now(timezone.utc)],
    )
    conn.execute(
        "INSERT INTO tool_calls (call_id, run_id, tool_name, args, return_value, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [call_id, run_id, "hf_list_models", '{"query": "mistral"}', '["a", "b"]', 42],
    )
    row = conn.execute(
        "SELECT tool_name, latency_ms FROM tool_calls WHERE call_id = ?", [call_id]
    ).fetchone()
    conn.close()
    assert row == ("hf_list_models", 42)


def test_diffs_insert_and_query(db_path):
    db = db_path
    db.init_schema()
    diff_id = str(uuid.uuid4())
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO diffs (diff_id, run_id, parameter_changed, before_output, after_output, diff_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [diff_id, "run-1", "TEMPERATURE", "old", "new", '{"diff": ""}'],
    )
    row = conn.execute(
        "SELECT parameter_changed FROM diffs WHERE diff_id = ?", [diff_id]
    ).fetchone()
    conn.close()
    assert row[0] == "TEMPERATURE"


def test_evaluations_insert_and_query(db_path):
    db = db_path
    db.init_schema()
    eval_id = str(uuid.uuid4())
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO evaluations (eval_id, run_id, metric_name, metric_value) VALUES (?, ?, ?, ?)",
        [eval_id, "run-1", "accuracy", 0.95],
    )
    row = conn.execute(
        "SELECT metric_value FROM evaluations WHERE eval_id = ?", [eval_id]
    ).fetchone()
    conn.close()
    assert abs(row[0] - 0.95) < 1e-9
