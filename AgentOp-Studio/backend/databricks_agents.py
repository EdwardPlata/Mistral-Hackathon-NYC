"""Databricks self-improvement agent suite.

DebugAgent  — uses Mistral to investigate a job failure, generate a root-cause
              analysis, and propose a code fix.

KnowledgeBase — persists error→analysis→fix triples to DuckDB and optionally
                uploads them as a W&B dataset artifact for future model fine-tuning.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Predefined error categories with context hints
# ---------------------------------------------------------------------------

PREDEFINED_ERRORS: list[dict] = [
    {
        "id": "schema_mismatch",
        "label": "Schema Mismatch",
        "prompt": (
            "Column not found or type mismatch error in DLT pipeline. "
            "The pipeline failed with AnalysisException or schema evolution issue."
        ),
        "icon": "🔀",
    },
    {
        "id": "oom_shuffle",
        "label": "OOM / Shuffle",
        "prompt": (
            "Out of memory error or GC overhead limit exceeded during shuffle stage. "
            "Possible shuffle skew or insufficient partitioning."
        ),
        "icon": "💥",
    },
    {
        "id": "data_quality",
        "label": "Data Quality",
        "prompt": (
            "DLT expectation violation rate exceeded threshold. "
            "Too many rows dropped by expect_or_drop rules."
        ),
        "icon": "⚠️",
    },
    {
        "id": "null_pointer",
        "label": "Null Pointer",
        "prompt": (
            "NullPointerException or null values in non-nullable columns. "
            "Upstream data contains unexpected nulls."
        ),
        "icon": "🚫",
    },
    {
        "id": "partition_imbalance",
        "label": "Partition Imbalance",
        "prompt": (
            "Slow stage or skewed partition detected. "
            "One executor is handling most of the data load."
        ),
        "icon": "⚖️",
    },
]


# ---------------------------------------------------------------------------
# DebugAgent
# ---------------------------------------------------------------------------

_DEBUG_SYSTEM_PROMPT = """\
You are an expert Databricks Delta Live Tables (DLT) and Apache Spark engineer.
A Databricks job has failed. Given the error message, the failed task name, and
the relevant pipeline source code, produce a structured analysis.

Respond with ONLY valid JSON — no markdown fences, no extra text:
{
  "root_cause": "<1–2 sentence root cause>",
  "severity": "critical|high|medium|low",
  "category": "schema_mismatch|oom_shuffle|data_quality|null_pointer|partition_imbalance|other",
  "recommended_fix": "<clear prose explanation of the fix>",
  "fix_code": "<Python/SQL code snippet showing the specific change; empty string if no code change needed>",
  "fix_location": "<file path and approximate line range where fix applies>",
  "prevention": "<how to prevent this class of error in future>",
  "confidence": <float 0.0–1.0>
}
"""


class DebugAgent:
    """Investigates a Databricks job failure with Mistral AI.

    Usage::

        agent = DebugAgent()
        result = agent.investigate(
            error_message="AssertionError: silver_trips_clean has 0 rows",
            task_key="validate_output",
            pipeline_code=code_str,
        )
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("MISTRAL_MODEL", "mistral-large-latest")

    def investigate(
        self,
        error_message: str,
        task_key: str = "",
        pipeline_code: str = "",
        extra_context: str = "",
    ) -> dict:
        """Run a Mistral investigation against the provided error.

        Returns a dict with keys: root_cause, severity, category,
        recommended_fix, fix_code, fix_location, prevention, confidence,
        model, tokens_used.
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return {
                "error": "MISTRAL_API_KEY not set",
                "root_cause": "",
                "recommended_fix": "",
                "fix_code": "",
                "confidence": 0.0,
                "tokens_used": 0,
            }

        # Build a concise user message — pipeline code is trimmed to keep tokens low
        code_snippet = pipeline_code[:4000] if pipeline_code else "(not provided)"
        user_message = (
            f"**Failed task:** {task_key or 'unknown'}\n\n"
            f"**Error message:**\n{error_message}\n\n"
            f"**Pipeline source code (excerpt):**\n```python\n{code_snippet}\n```"
        )
        if extra_context.strip():
            user_message += f"\n\n**Additional context:**\n{extra_context}"

        try:
            from mistralai import Mistral  # noqa: PLC0415

            client = Mistral(api_key=api_key)
            resp = client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": _DEBUG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1024,
                temperature=0.3,
            )

            content = resp.choices[0].message.content or ""
            tokens_used = resp.usage.total_tokens if resp.usage else 0

            # Strip any accidental markdown fences
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                analysis = {
                    "root_cause": content[:500],
                    "recommended_fix": "",
                    "fix_code": "",
                    "severity": "unknown",
                    "category": "other",
                    "fix_location": "",
                    "prevention": "",
                    "confidence": 0.5,
                }

            analysis["model"] = self.model
            analysis["tokens_used"] = tokens_used
            return analysis

        except Exception as exc:
            return {
                "error": str(exc),
                "root_cause": f"Agent error: {exc}",
                "recommended_fix": "",
                "fix_code": "",
                "severity": "unknown",
                "category": "other",
                "fix_location": "",
                "prevention": "",
                "confidence": 0.0,
                "model": self.model,
                "tokens_used": 0,
            }


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Persists Databricks error→fix triples for future model fine-tuning.

    Stores records in the AgentOps DuckDB ``knowledge_base`` table and
    optionally uploads a snapshot as a W&B dataset artifact so it can be
    used as fine-tuning data.
    """

    def save(
        self,
        error_category: str,
        error_description: str,
        job_name: str,
        task_key: str,
        analysis: dict,
        log_to_wandb: bool = True,
    ) -> str:
        """Persist an error/fix record and return its kb_id."""
        from .db import get_conn  # noqa: PLC0415

        kb_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc).isoformat()

        conn = get_conn()
        conn.execute(
            """
            INSERT INTO knowledge_base (
                kb_id, created_at, error_category, error_description,
                job_name, task_key, root_cause, recommended_fix, fix_code,
                fix_location, prevention, confidence, mistral_model, tokens_used,
                logged_to_wandb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                kb_id,
                now,
                error_category,
                error_description,
                job_name,
                task_key,
                analysis.get("root_cause", ""),
                analysis.get("recommended_fix", ""),
                analysis.get("fix_code", ""),
                analysis.get("fix_location", ""),
                analysis.get("prevention", ""),
                float(analysis.get("confidence", 0.0)),
                analysis.get("model", ""),
                int(analysis.get("tokens_used", 0)),
                False,
            ],
        )
        conn.close()

        if log_to_wandb:
            self._log_to_wandb(kb_id, error_category, error_description, analysis)
            from .db import get_conn as _gc  # noqa: PLC0415

            c = _gc()
            c.execute(
                "UPDATE knowledge_base SET logged_to_wandb = true WHERE kb_id = ?",
                [kb_id],
            )
            c.close()

        return kb_id

    def list_entries(self, limit: int = 50) -> list[dict]:
        """Return the most recent knowledge base entries."""
        from .db import get_conn  # noqa: PLC0415

        conn = get_conn()
        rows = conn.execute(
            """
            SELECT kb_id, created_at, error_category, job_name, task_key,
                   root_cause, recommended_fix, fix_code, confidence,
                   mistral_model, logged_to_wandb
            FROM knowledge_base
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        conn.close()
        keys = [
            "kb_id", "created_at", "error_category", "job_name", "task_key",
            "root_cause", "recommended_fix", "fix_code", "confidence",
            "mistral_model", "logged_to_wandb",
        ]
        return [dict(zip(keys, r)) for r in rows]

    def _log_to_wandb(
        self,
        kb_id: str,
        error_category: str,
        error_description: str,
        analysis: dict,
    ) -> None:
        """Upload this knowledge base entry as a W&B dataset artifact (best-effort)."""
        if not os.getenv("WANDB_API_KEY"):
            return
        try:
            import json as _json  # noqa: PLC0415
            import tempfile  # noqa: PLC0415

            import wandb  # noqa: PLC0415

            project = os.getenv("WANDB_PROJECT", "agentops-studio")
            if wandb.run is None:
                wandb.init(project=project, job_type="knowledge-base", reinit=True)

            record = {
                "kb_id": kb_id,
                "error_category": error_category,
                "error_description": error_description,
                **analysis,
            }

            # Log as a W&B Table row for browsing
            table = wandb.Table(
                columns=["kb_id", "category", "error", "root_cause", "fix_code", "confidence"]
            )
            table.add_data(
                kb_id[:8],
                error_category,
                error_description[:200],
                analysis.get("root_cause", "")[:300],
                analysis.get("fix_code", "")[:300],
                analysis.get("confidence", 0.0),
            )
            wandb.log({"knowledge_base/new_entry": table})

            # Also upload as a JSON artifact (future fine-tuning dataset)
            artifact = wandb.Artifact(
                name=f"kb-entry-{kb_id[:8]}",
                type="knowledge-base",
                description=f"Databricks error fix: {error_category}",
                metadata={"category": error_category, "confidence": analysis.get("confidence", 0.0)},
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                _json.dump(record, f, indent=2)
                tmp = f.name
            artifact.add_file(tmp, name="entry.json")
            wandb.log_artifact(artifact)
            wandb.finish()
        except Exception:
            pass  # W&B is best-effort
