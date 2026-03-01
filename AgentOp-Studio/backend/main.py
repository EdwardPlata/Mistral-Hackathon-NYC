"""FastAPI backend for AgentOps-Studio.

Endpoints:
    POST /run                   Run the agent with a prompt.
    POST /replay                Replay a run with optional parameter overrides.
    GET  /runs                  List all runs.
    GET  /runs/{run_id}         Full run detail (messages + tool calls).
    GET  /runs/{run_id}/memory  Memory snapshots for a run.
    POST /eval                  Log evaluation metrics for a run.
    GET  /evals                 List all evaluation records.
    GET  /evals/{run_id}        Evaluations for a specific run.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .db import get_conn, init_schema
from .instrumented_agent import run_instrumented
from .replay import replay_run


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    init_schema()
    yield


app = FastAPI(title="AgentOps-Studio", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    prompt: str
    user_id: str = "default"


class ReplayRequest(BaseModel):
    run_id: str
    override_params: dict[str, Any] | None = None


class EvalRequest(BaseModel):
    run_id: str
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/run")
def post_run(req: RunRequest) -> dict:
    """Run the instrumented agent and return run metadata."""
    result = run_instrumented(req.prompt, user_id=req.user_id)
    return result


@app.post("/replay")
def post_replay(req: ReplayRequest) -> dict:
    """Replay a run with optional parameter overrides."""
    try:
        result = replay_run(req.run_id, req.override_params)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@app.get("/runs")
def list_runs() -> list[dict]:
    """Return a summary list of all runs."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT run_id, agent_id, user_id, start_time, end_time,
               status, total_tokens, total_cost
        FROM runs
        ORDER BY start_time DESC
        """
    ).fetchall()
    conn.close()
    keys = [
        "run_id", "agent_id", "user_id", "start_time", "end_time",
        "status", "total_tokens", "total_cost",
    ]
    return [dict(zip(keys, row)) for row in rows]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Return full run detail including messages and tool calls."""
    conn = get_conn()

    run_row = conn.execute(
        """
        SELECT run_id, agent_id, user_id, start_time, end_time, status,
               total_tokens, total_cost, config, initial_prompt, final_response
        FROM runs WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()

    if run_row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    run_keys = [
        "run_id", "agent_id", "user_id", "start_time", "end_time", "status",
        "total_tokens", "total_cost", "config", "initial_prompt", "final_response",
    ]
    run = dict(zip(run_keys, run_row))

    msg_rows = conn.execute(
        """
        SELECT message_id, role, content, timestamp, token_count, finish_reason
        FROM messages WHERE run_id = ?
        ORDER BY timestamp
        """,
        [run_id],
    ).fetchall()
    msg_keys = ["message_id", "role", "content", "timestamp", "token_count", "finish_reason"]
    run["messages"] = [dict(zip(msg_keys, r)) for r in msg_rows]

    tc_rows = conn.execute(
        """
        SELECT call_id, tool_name, args, return_value, latency_ms
        FROM tool_calls WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()
    tc_keys = ["call_id", "tool_name", "args", "return_value", "latency_ms"]
    run["tool_calls"] = [dict(zip(tc_keys, r)) for r in tc_rows]

    conn.close()
    return run


@app.get("/runs/{run_id}/memory")
def get_run_memory(run_id: str) -> list[dict]:
    """Return memory snapshots for a run, ordered by timestamp."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT memory_id, run_id, timestamp, memory_json
        FROM memory_snapshots WHERE run_id = ?
        ORDER BY timestamp
        """,
        [run_id],
    ).fetchall()
    conn.close()
    keys = ["memory_id", "run_id", "timestamp", "memory_json"]
    return [dict(zip(keys, r)) for r in rows]


@app.post("/eval")
def post_eval(req: EvalRequest) -> dict:
    """Log evaluation metrics for a run.

    Accepts a dict of {metric_name: metric_value} pairs and persists each
    as a separate row in the evaluations table.
    """
    conn = get_conn()

    # Verify the run exists
    row = conn.execute(
        "SELECT run_id FROM runs WHERE run_id = ?", [req.run_id]
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {req.run_id!r} not found")

    eval_ids = []
    for metric_name, metric_value in req.metrics.items():
        eval_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO evaluations (eval_id, run_id, metric_name, metric_value)
            VALUES (?, ?, ?, ?)
            """,
            [eval_id, req.run_id, metric_name, float(metric_value)],
        )
        eval_ids.append(eval_id)

    conn.close()
    return {"run_id": req.run_id, "eval_ids": eval_ids, "metrics_logged": len(eval_ids)}


@app.get("/evals")
def list_evals() -> list[dict]:
    """Return all evaluation records, newest runs first."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT e.eval_id, e.run_id, e.metric_name, e.metric_value,
               r.start_time, r.agent_id
        FROM evaluations e
        LEFT JOIN runs r ON e.run_id = r.run_id
        ORDER BY r.start_time DESC
        """
    ).fetchall()
    conn.close()
    keys = ["eval_id", "run_id", "metric_name", "metric_value", "start_time", "agent_id"]
    return [dict(zip(keys, r)) for r in rows]


@app.get("/evals/{run_id}")
def get_evals_for_run(run_id: str) -> list[dict]:
    """Return evaluation records for a specific run."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT eval_id, run_id, metric_name, metric_value
        FROM evaluations WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()
    conn.close()
    keys = ["eval_id", "run_id", "metric_name", "metric_value"]
    return [dict(zip(keys, r)) for r in rows]
