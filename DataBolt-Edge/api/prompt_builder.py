"""Prompt builder for DataBolt Edge.

Converts the structured context string from parser.to_prompt_context()
into Mistral chat prompts for each log type.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts per log type
# ---------------------------------------------------------------------------

_SPARK_SYSTEM = (
    "You are DataBolt Edge, an expert Spark performance engineer. "
    "A user has uploaded a Spark application log. "
    "You will receive a structured summary of errors, exceptions, and stage/task info. "
    "Provide a concise diagnosis (2-4 sentences) and 3-5 specific actionable fixes "
    "with exact Spark config property names where applicable."
)

_AIRFLOW_SYSTEM = (
    "You are DataBolt Edge, an expert Apache Airflow engineer. "
    "A user has uploaded an Airflow task log. "
    "You will receive a structured summary of errors, tracebacks, and DAG metadata. "
    "Identify the root cause concisely and provide 3-5 actionable remediation steps."
)

_SQL_SYSTEM = (
    "You are DataBolt Edge, an expert SQL query optimizer. "
    "A user has provided a SQL EXPLAIN plan. "
    "You will receive a structured summary of full scans, join types, and cost estimates. "
    "Explain the bottleneck concisely, then provide the optimized SQL rewrite "
    "inside a ```sql ... ``` code block."
)

_SYSTEMS = {
    "spark": _SPARK_SYSTEM,
    "airflow": _AIRFLOW_SYSTEM,
    "sql": _SQL_SYSTEM,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompt(log_type: str, context_str: str, question: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for InferenceRequest.

    Args:
        log_type:    'spark', 'airflow', or 'sql'
        context_str: Output of ParseResult.to_prompt_context()
        question:    User's natural-language question

    Returns:
        (system_prompt, user_prompt)
    """
    if log_type not in _SYSTEMS:
        raise ValueError(f"Unknown log_type: {log_type!r}. Must be 'spark', 'airflow', or 'sql'.")

    system_prompt = _SYSTEMS[log_type]
    user_prompt = (
        f"Log analysis context:\n{context_str}\n\n"
        f"User question: {question}"
    )
    return system_prompt, user_prompt
