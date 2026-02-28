"""FastAPI backend for AgentOps-Studio.

Endpoints:
    POST /run           Run the agent with a prompt.
    POST /replay        Replay a run with optional parameter overrides.
    GET  /runs          List all runs.
    GET  /runs/{run_id} Full run detail (messages + tool calls).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .db import get_conn, init_schema
from .instrumented_agent import run_instrumented
from .replay import replay_run

app = FastAPI(title="AgentOps-Studio", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    init_schema()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    prompt: str
    user_id: str = "default"


class ReplayRequest(BaseModel):
    run_id: str
    override_params: dict[str, Any] | None = None


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
