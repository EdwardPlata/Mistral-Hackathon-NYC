"""FastAPI backend for AgentOps-Studio.

Endpoints:
    POST /run                    Run the agent with a prompt.
    POST /replay                 Replay a run with optional parameter overrides.
    GET  /runs                   List all runs.
    GET  /runs/{run_id}          Full run detail (messages + tool calls).
    GET  /runs/{run_id}/memory   Memory snapshots for a run.
    POST /eval                   Log evaluation metrics for a run.
    GET  /evals                  List all evaluation records.
    GET  /evals/{run_id}         Evaluations for a specific run.
    POST /wandb/training-run     Launch a Mistral training run in W&B with HF data.
    GET  /wandb/status           Check W&B connectivity.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .databricks_agents import PREDEFINED_ERRORS, DebugAgent, KnowledgeBase
from .db import get_conn, init_schema
from .instrumented_agent import run_instrumented
from .replay import replay_run
from .wandb_analysis import TrainingConfig, WandbTrainingRun
from .workflow_renderer import (
    get_mock_failed_jobs,
    get_pipeline_code,
    parse_bundle,
    parse_dlt_pipeline,
)


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


class TrainingRunRequest(BaseModel):
    model_name: str = "mistral-7b-instruct-v0.2"
    dataset_name: str = "tatsu-lab/alpaca"
    num_steps: int = 50
    sample_size: int = 20
    run_name: str | None = None


class InvestigateRequest(BaseModel):
    error_message: str
    task_key: str = ""
    job_name: str = ""
    extra_context: str = ""
    include_pipeline_code: bool = True


class SaveKnowledgeRequest(BaseModel):
    error_category: str
    error_description: str
    job_name: str = ""
    task_key: str = ""
    analysis: dict
    log_to_wandb: bool = True


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


# ---------------------------------------------------------------------------
# W&B training analysis routes
# ---------------------------------------------------------------------------


def _fetch_hf_samples(dataset_name: str, sample_size: int) -> list[dict]:
    """Stream a small sample from a HuggingFace dataset (best-effort)."""
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=False)
        samples = []
        for row in ds:
            samples.append(dict(row))
            if len(samples) >= sample_size:
                break
        return samples
    except Exception:
        return [
            {
                "instruction": f"Explain concept #{i} in machine learning.",
                "input": "",
                "output": (
                    f"Concept #{i} is a foundational principle in ML training "
                    "related to iterative optimization of a loss function."
                ),
            }
            for i in range(1, sample_size + 1)
        ]


@app.post("/wandb/training-run")
def post_wandb_training_run(req: TrainingRunRequest) -> dict:
    """Launch a Mistral fine-tuning training run in W&B with HuggingFace data.

    Fetches a sample from the requested HF dataset, runs a simulated
    Mistral training loop logging step-level metrics, and also uploads
    existing AgentOps run records as agent-analysis metrics.
    """
    import os  # noqa: PLC0415

    if not os.getenv("WANDB_API_KEY"):
        raise HTTPException(status_code=503, detail="WANDB_API_KEY is not configured")

    samples = _fetch_hf_samples(req.dataset_name, req.sample_size)

    config = TrainingConfig(
        model_name=req.model_name,
        dataset_name=req.dataset_name,
        num_steps=req.num_steps,
        eval_every=max(1, req.num_steps // 10),
        warmup_steps=max(1, req.num_steps // 10),
    )

    # Pull recent AgentOps runs for agent-analysis section
    conn = get_conn()
    rows = conn.execute(
        "SELECT run_id, status, total_tokens, total_cost FROM runs LIMIT 20"
    ).fetchall()
    conn.close()
    agent_records = [
        {"run_id": r[0], "status": r[1], "total_tokens": r[2], "total_cost": r[3]}
        for r in rows
    ]

    result: dict = {}
    with WandbTrainingRun(config=config, run_name=req.run_name) as run:
        run.log_dataset(samples)
        metrics = run.run_training_loop()
        if agent_records:
            run.log_agent_analysis(agent_records)
        result = {
            "run_url": run.run_url,
            "run_name": run.run_name,
            "dataset_samples": len(samples),
            **metrics,
        }

    return result


@app.get("/wandb/status")
def get_wandb_status() -> dict:
    """Check W&B connectivity and return project configuration."""
    import os  # noqa: PLC0415

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        return {"connected": False, "reason": "WANDB_API_KEY not set"}

    try:
        import wandb  # noqa: PLC0415

        api = wandb.Api()
        project = os.getenv("WANDB_PROJECT", "agentops-studio")
        entity = api.default_entity
        return {
            "connected": True,
            "project": project,
            "entity": entity,
            "key_prefix": api_key[:6] + "…",
        }
    except Exception as exc:
        return {"connected": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Databricks workflow + debug agent routes
# ---------------------------------------------------------------------------


@app.get("/databricks/workflow")
def get_databricks_workflow() -> dict:
    """Return parsed bundle structure + mock/real failed jobs."""
    bundle = parse_bundle()
    pipeline = parse_dlt_pipeline()
    failed_jobs = get_mock_failed_jobs()
    return {
        "bundle": bundle,
        "pipeline": pipeline,
        "failed_jobs": failed_jobs,
        "predefined_errors": PREDEFINED_ERRORS,
    }


@app.get("/databricks/pipeline")
def get_databricks_pipeline() -> dict:
    """Return the parsed DLT table DAG for visualization."""
    return parse_dlt_pipeline()


@app.get("/databricks/pipeline/code")
def get_pipeline_source() -> dict:
    """Return the raw DLT pipeline source code."""
    code = get_pipeline_code()
    return {"code": code, "lines": len(code.splitlines())}


@app.post("/databricks/investigate")
def post_investigate(req: InvestigateRequest) -> dict:
    """Use Mistral to investigate a Databricks job failure.

    Returns root cause, recommended fix, code snippet, and confidence score.
    """
    pipeline_code = get_pipeline_code() if req.include_pipeline_code else ""
    agent = DebugAgent()
    analysis = agent.investigate(
        error_message=req.error_message,
        task_key=req.task_key,
        pipeline_code=pipeline_code,
        extra_context=req.extra_context,
    )
    return analysis


@app.post("/databricks/knowledge")
def post_save_knowledge(req: SaveKnowledgeRequest) -> dict:
    """Save an error→fix record to the knowledge base (DuckDB + W&B)."""
    kb = KnowledgeBase()
    kb_id = kb.save(
        error_category=req.error_category,
        error_description=req.error_description,
        job_name=req.job_name,
        task_key=req.task_key,
        analysis=req.analysis,
        log_to_wandb=req.log_to_wandb,
    )
    return {"kb_id": kb_id, "saved": True}


@app.get("/databricks/knowledge")
def list_knowledge() -> list[dict]:
    """Return all knowledge base entries, newest first."""
    kb = KnowledgeBase()
    return kb.list_entries()
