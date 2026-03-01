"""Tests for evaluation endpoints and memory snapshot DB operations.

Covers:
    - POST /eval  (log metrics for a run)
    - GET /evals  (list all evaluations)
    - GET /evals/{run_id}  (evaluations for a specific run)
    - GET /runs/{run_id}/memory  (memory snapshots)
    - memory_snapshots table insert/query
"""

import importlib
import json
import os
import uuid
from datetime import datetime, timezone

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path):
    """Provide a temporary DuckDB and reload db module to use it."""
    path = str(tmp_path / "test_eval.duckdb")
    os.environ["AGENTOPS_DB_PATH"] = path

    import backend.db as db_mod  # noqa: PLC0415

    importlib.reload(db_mod)
    db_mod.init_schema()
    yield db_mod
    os.environ.pop("AGENTOPS_DB_PATH", None)


@pytest.fixture()
def seeded_run(db_path):
    """Insert a run row and return its run_id."""
    run_id = str(uuid.uuid4())
    conn = db_path.get_conn()
    conn.execute(
        """
        INSERT INTO runs (run_id, agent_id, user_id, start_time, status, config, initial_prompt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [run_id, "test-agent", "test-user", datetime.now(timezone.utc), "success", "{}", "Hello"],
    )
    conn.close()
    return run_id


@pytest.fixture()
def api_client(db_path):
    """Return a FastAPI TestClient wired to the temp DB."""
    import importlib  # noqa: PLC0415

    import backend.main as main_mod  # noqa: PLC0415

    importlib.reload(main_mod)
    from fastapi.testclient import TestClient  # noqa: PLC0415

    return TestClient(main_mod.app)


# ---------------------------------------------------------------------------
# Memory snapshot DB tests
# ---------------------------------------------------------------------------


def test_memory_snapshots_insert_and_query(db_path, seeded_run):
    memory_id = str(uuid.uuid4())
    mem_state = [{"role": "user", "content": "Hello"}]
    conn = db_path.get_conn()
    conn.execute(
        "INSERT INTO memory_snapshots (memory_id, run_id, timestamp, memory_json) VALUES (?, ?, ?, ?)",
        [memory_id, seeded_run, datetime.now(timezone.utc), json.dumps(mem_state)],
    )
    row = conn.execute(
        "SELECT memory_json FROM memory_snapshots WHERE memory_id = ?", [memory_id]
    ).fetchone()
    conn.close()
    assert row is not None
    parsed = json.loads(row[0])
    assert parsed[0]["role"] == "user"


def test_memory_snapshots_multiple_per_run(db_path, seeded_run):
    conn = db_path.get_conn()
    for i in range(3):
        conn.execute(
            "INSERT INTO memory_snapshots (memory_id, run_id, timestamp, memory_json) VALUES (?, ?, ?, ?)",
            [str(uuid.uuid4()), seeded_run, datetime.now(timezone.utc), json.dumps([{"round": i}])],
        )
    count = conn.execute(
        "SELECT COUNT(*) FROM memory_snapshots WHERE run_id = ?", [seeded_run]
    ).fetchone()[0]
    conn.close()
    assert count == 3


# ---------------------------------------------------------------------------
# POST /eval endpoint tests
# ---------------------------------------------------------------------------


def test_post_eval_logs_metrics(api_client, seeded_run):
    resp = api_client.post(
        "/eval",
        json={"run_id": seeded_run, "metrics": {"accuracy": 0.9, "latency_ok": 1.0}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == seeded_run
    assert data["metrics_logged"] == 2
    assert len(data["eval_ids"]) == 2


def test_post_eval_unknown_run_returns_404(api_client):
    resp = api_client.post(
        "/eval",
        json={"run_id": "nonexistent-run-id", "metrics": {"score": 1.0}},
    )
    assert resp.status_code == 404


def test_post_eval_single_metric(api_client, seeded_run):
    resp = api_client.post(
        "/eval",
        json={"run_id": seeded_run, "metrics": {"success": 1.0}},
    )
    assert resp.status_code == 200
    assert resp.json()["metrics_logged"] == 1


# ---------------------------------------------------------------------------
# GET /evals endpoint tests
# ---------------------------------------------------------------------------


def test_get_evals_empty(api_client):
    resp = api_client.get("/evals")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_evals_after_logging(api_client, seeded_run):
    api_client.post(
        "/eval",
        json={"run_id": seeded_run, "metrics": {"precision": 0.85}},
    )
    resp = api_client.get("/evals")
    assert resp.status_code == 200
    records = resp.json()
    assert any(r["metric_name"] == "precision" for r in records)


# ---------------------------------------------------------------------------
# GET /evals/{run_id} endpoint tests
# ---------------------------------------------------------------------------


def test_get_evals_for_run(api_client, seeded_run):
    api_client.post(
        "/eval",
        json={"run_id": seeded_run, "metrics": {"f1": 0.75, "recall": 0.80}},
    )
    resp = api_client.get(f"/evals/{seeded_run}")
    assert resp.status_code == 200
    records = resp.json()
    assert len(records) == 2
    names = {r["metric_name"] for r in records}
    assert names == {"f1", "recall"}


def test_get_evals_for_run_returns_empty_list_if_none(api_client, seeded_run):
    resp = api_client.get(f"/evals/{seeded_run}")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/memory endpoint tests
# ---------------------------------------------------------------------------


def test_get_run_memory_empty(api_client, seeded_run):
    resp = api_client.get(f"/runs/{seeded_run}/memory")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_run_memory_after_insert(api_client, db_path, seeded_run):
    # Directly insert a memory snapshot
    memory_id = str(uuid.uuid4())
    mem = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    conn = db_path.get_conn()
    conn.execute(
        "INSERT INTO memory_snapshots (memory_id, run_id, timestamp, memory_json) VALUES (?, ?, ?, ?)",
        [memory_id, seeded_run, datetime.now(timezone.utc), json.dumps(mem)],
    )
    conn.close()

    resp = api_client.get(f"/runs/{seeded_run}/memory")
    assert resp.status_code == 200
    snapshots = resp.json()
    assert len(snapshots) == 1
    assert snapshots[0]["memory_id"] == memory_id
