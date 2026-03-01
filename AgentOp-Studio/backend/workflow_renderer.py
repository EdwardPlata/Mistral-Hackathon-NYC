"""Databricks workflow and DLT pipeline renderer.

Parses databricks.yml and dlt_pipeline.py into structured dicts suitable
for UI display and agent analysis.

Functions:
    parse_bundle(bundle_path)     — parse databricks.yml
    parse_dlt_pipeline(path)      — parse dlt_pipeline.py into table DAG
    get_mock_failed_jobs()        — realistic sample failed-job records
    get_pipeline_code(path)       — read raw source for agent context
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# Bundle YAML parser
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

_DEFAULT_BUNDLE_DIR = os.path.join(
    _REPO_ROOT, "DataBolt-Edge", "databricks"
)
_DEFAULT_PIPELINE_PATH = os.path.join(
    _DEFAULT_BUNDLE_DIR, "src", "nyc_taxi", "dlt_pipeline.py"
)
_DEFAULT_BUNDLE_YAML = os.path.join(_DEFAULT_BUNDLE_DIR, "databricks.yml")


def parse_bundle(bundle_path: str | None = None) -> dict:
    """Parse a databricks.yml bundle file into a structured dict.

    Returns:
        {
          "bundle_name": str,
          "variables": [{name, description, default}],
          "targets": [{name, mode, is_default, variables}],
          "pipelines": [{name, serverless, source_notebook, target_schema}],
          "jobs": [{name, tasks: [{key, type, depends_on, description}]}],
          "task_graph": [{from, to}],   # edges for DAG rendering
        }
    """
    yaml_path = bundle_path or _DEFAULT_BUNDLE_YAML
    try:
        import yaml  # noqa: PLC0415

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return {"error": f"Bundle file not found: {yaml_path}"}
    except Exception as exc:
        return {"error": str(exc)}

    bundle_name = data.get("bundle", {}).get("name", "unknown")

    variables = [
        {
            "name": k,
            "description": v.get("description", ""),
            "default": v.get("default", ""),
        }
        for k, v in (data.get("variables") or {}).items()
    ]

    targets = [
        {
            "name": k,
            "mode": v.get("mode", "development"),
            "is_default": v.get("default", False),
            "variables": v.get("variables", {}),
        }
        for k, v in (data.get("targets") or {}).items()
    ]

    resources = data.get("resources", {})

    pipelines = [
        {
            "key": k,
            "name": v.get("name", k),
            "serverless": v.get("serverless", False),
            "source_notebook": (
                v.get("libraries", [{}])[0].get("notebook", {}).get("path", "")
                if v.get("libraries")
                else ""
            ),
            "target_schema": v.get("target", ""),
            "catalog": v.get("catalog", ""),
            "continuous": v.get("continuous", False),
        }
        for k, v in (resources.get("pipelines") or {}).items()
    ]

    jobs = []
    task_graph = []
    for job_key, job_val in (resources.get("jobs") or {}).items():
        tasks = []
        for t in job_val.get("tasks", []):
            task_type = "notebook"
            if "pipeline_task" in t:
                task_type = "pipeline"
            elif "notebook_task" in t:
                task_type = "notebook"
            tasks.append(
                {
                    "key": t.get("task_key", ""),
                    "type": task_type,
                    "description": t.get("description", ""),
                    "depends_on": [d["task_key"] for d in t.get("depends_on", [])],
                }
            )
            for dep in t.get("depends_on", []):
                task_graph.append(
                    {"from": dep["task_key"], "to": t.get("task_key", "")}
                )
        schedule = job_val.get("schedule", {})
        jobs.append(
            {
                "key": job_key,
                "name": job_val.get("name", job_key),
                "max_concurrent_runs": job_val.get("max_concurrent_runs", 1),
                "schedule_cron": schedule.get("quartz_cron_expression", ""),
                "schedule_tz": schedule.get("timezone_id", ""),
                "schedule_paused": schedule.get("pause_status", "PAUSED") == "PAUSED",
                "tasks": tasks,
            }
        )

    return {
        "bundle_name": bundle_name,
        "variables": variables,
        "targets": targets,
        "pipelines": pipelines,
        "jobs": jobs,
        "task_graph": task_graph,
    }


# ---------------------------------------------------------------------------
# DLT pipeline parser
# ---------------------------------------------------------------------------

_LAYER_ORDER = {"bronze": 0, "silver": 1, "gold": 2, "features": 3}


def _detect_layer(table_name: str, quality_prop: str | None) -> str:
    """Infer medallion layer from name prefix or quality table_property."""
    if quality_prop:
        return quality_prop
    for prefix in ("bronze", "silver", "gold", "features"):
        if table_name.lower().startswith(prefix):
            return prefix
    return "unknown"


def parse_dlt_pipeline(pipeline_path: str | None = None) -> dict:
    """Parse a DLT pipeline Python file into a structured table DAG.

    Returns:
        {
          "tables": [{name, layer, comment, quality_rules, depends_on}],
          "dag_edges": [{source, target}],
          "layers": {layer_name: [table_names]},
          "quality_rule_count": int,
          "total_tables": int,
        }
    """
    path = pipeline_path or _DEFAULT_PIPELINE_PATH
    try:
        with open(path) as f:
            source = f.read()
    except FileNotFoundError:
        return {"error": f"Pipeline file not found: {path}"}

    tables = {}  # name → dict

    # --- extract @dlt.table blocks ---
    table_pattern = re.compile(
        r'@dlt\.table\s*\(\s*(?:[^)]*?name\s*=\s*["\']([^"\']+)["\'][^)]*?|)'
        r'(?:[^)]*?comment\s*=\s*["\']([^"\']*)["\'][^)]*?|)'
        r'(?:[^)]*?table_properties\s*=\s*\{([^}]*)\}[^)]*?|)',
        re.DOTALL,
    )

    # Simpler approach: scan for @dlt.table(name=...) blocks
    for block in re.finditer(
        r'@dlt\.table\s*\(([^)]+)\)',
        source,
        re.DOTALL,
    ):
        attrs = block.group(1)
        name_m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', attrs)
        comment_m = re.search(r'comment\s*=\s*["\']([^"\']+)["\']', attrs, re.DOTALL)
        quality_m = re.search(r'"quality"\s*:\s*"([^"]+)"', attrs)
        if name_m:
            name = name_m.group(1)
            tables[name] = {
                "name": name,
                "comment": (comment_m.group(1).strip() if comment_m else ""),
                "quality_prop": quality_m.group(1) if quality_m else None,
                "quality_rules": [],
                "depends_on": [],
            }

    # --- extract @dlt.expect_or_drop rules (associate with the next table def) ---
    # Walk sequentially: collect expect rules before each dlt.table decorator
    expect_pattern = re.compile(
        r'@dlt\.expect_or_drop\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\)'
    )
    table_def_pattern = re.compile(r'def\s+(\w+)\s*\(')

    # Build an ordered list of (pos, event_type, data)
    events = []
    for m in expect_pattern.finditer(source):
        events.append((m.start(), "expect", m.group(1), m.group(2)))
    for m in re.finditer(r'@dlt\.table\s*\(', source):
        # find name in subsequent attrs
        block_end = source.find(")", m.end()) + 1
        attrs = source[m.end():block_end]
        name_m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', attrs)
        if name_m:
            events.append((m.start(), "table", name_m.group(1)))
    events.sort(key=lambda e: e[0])

    pending_expects = []
    for ev in events:
        if ev[1] == "expect":
            pending_expects.append({"rule_name": ev[2], "expression": ev[3]})
        elif ev[1] == "table":
            table_name = ev[2]
            if table_name in tables and pending_expects:
                tables[table_name]["quality_rules"] = pending_expects.copy()
                pending_expects = []

    # --- extract dlt.read("xxx") dependencies ---
    for table_name in list(tables.keys()):
        # Find the function body for this table
        func_m = re.search(
            rf'def\s+{table_name.replace("-", "_")}\s*\(\s*\)(.*?)(?=\n\n@|\Z)',
            source,
            re.DOTALL,
        )
        if func_m:
            body = func_m.group(1)
            deps = re.findall(r'dlt\.read\(["\']([^"\']+)["\']\)', body)
            tables[table_name]["depends_on"] = deps

    # --- build layer metadata ---
    for t in tables.values():
        t["layer"] = _detect_layer(t["name"], t.get("quality_prop"))

    dag_edges = [
        {"source": dep, "target": t["name"]}
        for t in tables.values()
        for dep in t["depends_on"]
    ]

    layers: dict[str, list[str]] = {}
    for t in tables.values():
        layers.setdefault(t["layer"], []).append(t["name"])

    return {
        "tables": list(tables.values()),
        "dag_edges": dag_edges,
        "layers": layers,
        "quality_rule_count": sum(len(t["quality_rules"]) for t in tables.values()),
        "total_tables": len(tables),
        "layer_order": _LAYER_ORDER,
    }


# ---------------------------------------------------------------------------
# Mock failed jobs  (realistic sample data)
# ---------------------------------------------------------------------------

_NOW = datetime.now(tz=timezone.utc)


def get_mock_failed_jobs() -> list[dict]:
    """Return realistic mock failed Databricks job records.

    When DATABRICKS_HOST is configured this would be replaced with real
    API calls; for demo purposes these samples cover the most common
    DLT/job failure categories.
    """
    return [
        {
            "run_id": "982341",
            "job_name": "NYC Taxi Workflow [dev]",
            "task_key": "validate_output",
            "state": "FAILED",
            "result_state": "FAILED",
            "error_message": (
                "AssertionError: silver_trips_clean has 0 rows. "
                "Expected ≥1000 but got 0. "
                "Possible cause: source path dbfs:/tmp/nyc-taxi-raw/yellow is empty "
                "or DLT pipeline ran with schema mismatch on bronze ingestion."
            ),
            "started_at": (_NOW - timedelta(hours=2, minutes=14)).isoformat(),
            "duration_seconds": 127,
            "run_url": "https://adb-example.azuredatabricks.net/jobs/12/runs/982341",
            "category": "Schema Mismatch",
        },
        {
            "run_id": "981100",
            "job_name": "NYC Taxi Workflow [dev]",
            "task_key": "run_dlt_pipeline",
            "state": "FAILED",
            "result_state": "FAILED",
            "error_message": (
                "com.databricks.backend.daemon.driver.DriverClient$DriverException: "
                "AnalysisException: Column 'tpep_pickup_datetime' not found in schema. "
                "Bronze schema evolution may have renamed columns in source parquet files."
            ),
            "started_at": (_NOW - timedelta(hours=14)).isoformat(),
            "duration_seconds": 342,
            "run_url": "https://adb-example.azuredatabricks.net/jobs/12/runs/981100",
            "category": "Schema Mismatch",
        },
        {
            "run_id": "979800",
            "job_name": "NYC Taxi Workflow [dev]",
            "task_key": "run_dlt_pipeline",
            "state": "FAILED",
            "result_state": "FAILED",
            "error_message": (
                "SparkException: Job aborted due to stage failure: "
                "Task 14 in stage 8.0 failed 4 times. "
                "OOM: GC overhead limit exceeded in shuffle stage. "
                "silver_trips_clean groupBy on PULocationID causing shuffle spill — "
                "consider repartitioning before aggregation."
            ),
            "started_at": (_NOW - timedelta(days=1, hours=3)).isoformat(),
            "duration_seconds": 891,
            "run_url": "https://adb-example.azuredatabricks.net/jobs/12/runs/979800",
            "category": "OOM / Shuffle",
        },
        {
            "run_id": "978200",
            "job_name": "NYC Taxi DLT Pipeline [dev]",
            "task_key": "dlt_pipeline",
            "state": "FAILED",
            "result_state": "FAILED",
            "error_message": (
                "DLT expectation violation rate exceeded threshold. "
                "positive_fare dropped 18.3% of rows (expected < 1%). "
                "Upstream data quality issue: fare_amount contains negative values "
                "from vendor ID 2 records in 2023-02 batch."
            ),
            "started_at": (_NOW - timedelta(days=2)).isoformat(),
            "duration_seconds": 214,
            "run_url": "https://adb-example.azuredatabricks.net/jobs/11/runs/978200",
            "category": "Data Quality",
        },
    ]


# ---------------------------------------------------------------------------
# Raw pipeline code accessor (for agent context)
# ---------------------------------------------------------------------------


def get_pipeline_code(pipeline_path: str | None = None) -> str:
    """Return the raw source code of the DLT pipeline file."""
    path = pipeline_path or _DEFAULT_PIPELINE_PATH
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"# Pipeline file not found: {path}"
