"""FastAPI backend for DataBolt Edge.

Endpoints:
    POST /analyze              Parse a log/SQL and run LLM analysis.
    GET  /results/{id}         Retrieve a stored analysis by ID.
    GET  /analyses             List recent analyses.
    GET  /health               Backend status and DB counts.
"""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .context_store import init_schema, list_analyses, get_analysis, save_analysis, table_counts
from .prompt_builder import build_prompt
from .telemetry import run_with_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    init_schema()
    yield


app = FastAPI(title="DataBolt Edge", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    log_type: str          # "spark" | "airflow" | "sql"
    content: str           # raw log text or SQL EXPLAIN output
    question: str = "What is wrong and how do I fix it?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(log_type: str, content: str) -> tuple[dict[str, Any], str]:
    """Dispatch to the right parser and return (summary_dict, prompt_context_str)."""
    if log_type == "spark":
        from parsers.spark import parse_spark  # noqa: PLC0415
        result = parse_spark(content)
        return dataclasses.asdict(result), result.to_prompt_context()

    if log_type == "airflow":
        from parsers.airflow import parse_airflow  # noqa: PLC0415
        result = parse_airflow(content)
        return dataclasses.asdict(result), result.to_prompt_context()

    if log_type == "sql":
        from parsers.sql_plan import parse_sql_plan  # noqa: PLC0415
        result = parse_sql_plan(content)
        return dataclasses.asdict(result), result.to_prompt_context()

    raise ValueError(f"Unknown log_type: {log_type!r}")


def _extract_sql_rewrite(model_response: str) -> str | None:
    """Pull the first ```sql ... ``` block from the model response, if any."""
    import re
    m = re.search(r"```sql\s*(.*?)```", model_response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/analyze")
def post_analyze(req: AnalyzeRequest) -> dict:
    """Parse the submitted log/SQL and return an LLM-generated analysis."""
    if req.log_type not in ("spark", "airflow", "sql"):
        raise HTTPException(status_code=400, detail="log_type must be 'spark', 'airflow', or 'sql'")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")

    # 1. Parse
    try:
        parsed, context_str = _parse(req.log_type, req.content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Parse error: {exc}") from exc

    # 2. Build prompt
    system_prompt, user_prompt = build_prompt(req.log_type, context_str, req.question)

    # 3. Run inference
    try:
        from inference.factory import get_backend  # noqa: PLC0415
        from inference.base import InferenceRequest  # noqa: PLC0415

        backend = get_backend()
        infer_req = InferenceRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.15,
        )
        response, telemetry = run_with_telemetry(backend, infer_req)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference error: {exc}") from exc

    model_response = response.text
    optimized_sql = _extract_sql_rewrite(model_response) if req.log_type == "sql" else None

    # 4. Persist
    analysis_id = save_analysis(
        log_type=req.log_type,
        raw_content=req.content,
        question=req.question,
        parsed_json=parsed,
        model_response=model_response,
        optimized_sql=optimized_sql,
        backend=telemetry.backend,
        latency_ms=telemetry.latency_ms,
        tokens=telemetry.total_tokens,
    )

    return {
        "analysis_id": analysis_id,
        "log_type": req.log_type,
        "parsed_summary": parsed,
        "model_response": model_response,
        "optimized_sql": optimized_sql,
        "telemetry": {
            "backend": telemetry.backend,
            "latency_ms": telemetry.latency_ms,
            "prompt_tokens": telemetry.prompt_tokens,
            "completion_tokens": telemetry.completion_tokens,
            "total_tokens": telemetry.total_tokens,
            "tokens_per_second": telemetry.tokens_per_second,
            **telemetry.gpu_stats,
        },
    }


@app.get("/results/{analysis_id}")
def get_result(analysis_id: str) -> dict:
    """Return a stored analysis record."""
    record = get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id!r} not found")
    return record


@app.get("/analyses")
def list_all_analyses(limit: int = 100) -> list[dict]:
    """Return recent analyses, newest first."""
    return list_analyses(limit=limit)


@app.get("/health")
def health() -> dict:
    """Return backend status, inference backend availability, and DB counts."""
    import os  # noqa: PLC0415

    # Check inference backend
    backend_status: dict[str, Any] = {}
    try:
        from inference.factory import get_backend  # noqa: PLC0415
        b = get_backend()
        backend_status = b.health_check()
    except Exception as exc:
        backend_status = {"available": False, "reason": str(exc)}

    return {
        "status": "ok",
        "inference": backend_status,
        "db": table_counts(),
        "env": {
            "INFERENCE_BACKEND": os.getenv("INFERENCE_BACKEND", "auto"),
            "TRT_ENGINE_PATH": os.getenv("TRT_ENGINE_PATH", "(not set)"),
            "NVIDIA_API_KEY_SET": bool(
                os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_BEARER_TOKEN")
            ),
        },
    }
