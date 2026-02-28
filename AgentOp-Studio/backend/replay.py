"""Replay a previous run with optional parameter overrides and diff generation."""

import difflib
import json
import os
import uuid
from datetime import datetime, timezone

from .db import get_conn
from .instrumented_agent import run_instrumented


def replay_run(run_id: str, override_params: dict | None = None) -> dict:
    """Replay a run, optionally with parameter overrides, and record a diff.

    Args:
        run_id:          ID of the run to replay.
        override_params: Optional dict of env-level overrides, e.g.
                         {"MISTRAL_MODEL": "mistral-small-latest",
                          "TEMPERATURE": "0.3"}.

    Returns:
        dict with keys: diff_id, run_id, new_run_id, original_response,
                        new_response, diff.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT initial_prompt, config, final_response FROM runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"Run {run_id!r} not found")

    initial_prompt, config_json, original_response = row
    original_response = original_response or ""

    # Temporarily apply env overrides
    old_env: dict[str, str | None] = {}
    if override_params:
        for key, val in override_params.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = str(val)

    try:
        result = run_instrumented(initial_prompt)
    finally:
        # Restore original env
        for key, old_val in old_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val

    new_run_id = result["run_id"]
    new_response = result["response"]

    diff_lines = list(
        difflib.unified_diff(
            original_response.splitlines(keepends=True),
            new_response.splitlines(keepends=True),
            fromfile=f"run/{run_id}",
            tofile=f"run/{new_run_id}",
        )
    )
    diff_text = "".join(diff_lines)
    diff_id = str(uuid.uuid4())

    parameter_changed = (
        json.dumps(override_params) if override_params else "none"
    )

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO diffs
            (diff_id, run_id, parameter_changed, before_output, after_output, diff_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            diff_id,
            run_id,
            parameter_changed,
            original_response,
            new_response,
            json.dumps({"diff": diff_text}),
        ],
    )
    conn.close()

    return {
        "diff_id": diff_id,
        "run_id": run_id,
        "new_run_id": new_run_id,
        "original_response": original_response,
        "new_response": new_response,
        "diff": diff_text,
    }
