"""Tool definitions for the Mistral NYC hackathon agent.

Each function represents a tool that the Mistral model can invoke during
agentic workflows. External API calls are kept here so that main_agent.py
stays clean and composable.

Copilot: follow the pattern below when adding new tools – define a Python
function plus a matching JSON schema entry in get_tools().
"""

import os
from typing import Any

# NYC Taxi bundle path – can be overridden via env var for CI / different checkouts
_NYC_TAXI_BUNDLE_PATH = os.getenv(
    "NYC_TAXI_BUNDLE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "DataBolt-Edge", "databricks"),
)


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

def wandb_log(metrics: dict[str, Any], step: int | None = None) -> str:
    """Log a dictionary of metrics to Weights & Biases.

    Args:
        metrics: Key/value pairs to log (e.g. {"loss": 0.42}).
        step: Optional global step counter.

    Returns:
        Confirmation message.
    """
    import wandb  # noqa: PLC0415

    if wandb.run is None:
        wandb.init(project=os.getenv("WANDB_PROJECT", "mistral-hackathon-nyc"))
    wandb.log(metrics, step=step)
    return f"Logged metrics: {list(metrics.keys())}"


# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------

def hf_list_models(query: str, limit: int = 5) -> list[str]:
    """Search Hugging Face Hub for models matching a query.

    Args:
        query: Search term (e.g. "mistral instruct").
        limit: Maximum number of results to return.

    Returns:
        List of model IDs.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=os.getenv("HF_TOKEN"))
    models = api.list_models(search=query, limit=limit)
    return [m.modelId for m in models]


# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------

def elevenlabs_speak(text: str, voice_id: str = "Rachel", output_path: str = "/tmp/output.mp3") -> str:
    """Convert text to speech using ElevenLabs and save to a file.

    Args:
        text: The text to synthesize.
        voice_id: ElevenLabs voice name or ID (default: "Rachel").
        output_path: Destination file path for the audio.

    Returns:
        Path to the saved audio file.
    """
    from elevenlabs import ElevenLabs, save  # noqa: PLC0415

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio = client.text_to_speech.convert(text=text, voice_id=voice_id)

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    save(audio, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Databricks Asset Bundles  (via Databricks Bundles MCP server)
# ---------------------------------------------------------------------------


def databricks_bundle_deploy(target: str = "dev", bundle_path: str | None = None) -> str:
    """Deploy the NYC Taxi Databricks Asset Bundle via the MCP server.

    Runs ``databricks bundle deploy`` through the Databricks Bundles MCP
    server process.  The bundle includes a DLT pipeline (bronze → silver →
    gold → features) and an orchestrating workflow job.

    Args:
        target:      Bundle target to deploy to (``dev`` or ``prod``).
        bundle_path: Override the bundle directory.  Defaults to
                     ``DataBolt-Edge/databricks/``.

    Returns:
        JSON-encoded result from the MCP server.
    """
    import json  # noqa: PLC0415

    from mcp_servers.databricks_bundles.mcp_client import call_mcp_tool  # noqa: PLC0415

    path = bundle_path or _NYC_TAXI_BUNDLE_PATH
    result = call_mcp_tool("bundle_deploy", {"bundle_path": path, "target": target})
    return json.dumps(result, indent=2)


def databricks_bundle_run(
    resource_key: str = "nyc_taxi_workflow",
    target: str = "dev",
    full_refresh: bool = False,
    only_tasks: list[str] | None = None,
    notebook_params: dict | None = None,
    bundle_path: str | None = None,
) -> str:
    """Trigger a Databricks bundle job or DLT pipeline via the MCP server.

    Args:
        resource_key:    Bundle resource key (e.g. ``nyc_taxi_workflow`` or
                         ``nyc_taxi_dlt``).
        target:          Bundle target (``dev`` or ``prod``).
        full_refresh:    If True, passes ``--full-refresh`` for DLT pipeline runs.
        only_tasks:      Run only these task keys, e.g. ``["download_raw_data"]``.
        notebook_params: Key-value pairs passed as notebook parameters.
        bundle_path:     Override bundle directory.

    Returns:
        JSON-encoded result including run_id from the MCP server.
    """
    import json  # noqa: PLC0415

    from mcp_servers.databricks_bundles.mcp_client import call_mcp_tool  # noqa: PLC0415

    path = bundle_path or _NYC_TAXI_BUNDLE_PATH
    payload: dict = {
        "bundle_path": path,
        "resource_key": resource_key,
        "target": target,
        "full_refresh": full_refresh,
    }
    if only_tasks:
        payload["only_tasks"] = only_tasks
    if notebook_params:
        payload["notebook_params"] = notebook_params
    result = call_mcp_tool("bundle_run", payload)
    return json.dumps(result, indent=2)


def databricks_download_taxi_data(
    start_year_month: str = "2023-01",
    end_year_month: str = "2023-03",
    target: str = "dev",
    bundle_path: str | None = None,
) -> str:
    """Download NYC yellow taxi data from the TLC CloudFront CDN into DBFS.

    Runs only the ``download_raw_data`` task of the NYC Taxi workflow.
    Data source follows the pattern from
    https://github.com/toddwschneider/nyc-taxi-data:

        https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet

    Files land in ``dbfs:/tmp/nyc-taxi-raw/yellow/`` and are skipped if
    already present (idempotent). After this finishes, run
    ``databricks_bundle_run(resource_key='nyc_taxi_dlt')`` to process them.

    Args:
        start_year_month: First month to download, e.g. ``"2023-01"``.
        end_year_month:   Last month to download (inclusive), e.g. ``"2023-03"``.
        target:           Bundle target (``dev`` or ``prod``).
        bundle_path:      Override bundle directory.

    Returns:
        JSON with run status and Databricks job run URL.
    """
    import json  # noqa: PLC0415

    from mcp_servers.databricks_bundles.mcp_client import call_mcp_tool  # noqa: PLC0415

    path = bundle_path or _NYC_TAXI_BUNDLE_PATH
    result = call_mcp_tool(
        "bundle_run",
        {
            "bundle_path": path,
            "resource_key": "nyc_taxi_workflow",
            "target": target,
            "only_tasks": ["download_raw_data"],
            "notebook_params": {
                "start_year_month": start_year_month,
                "end_year_month": end_year_month,
            },
        },
    )
    return json.dumps(result, indent=2)


def databricks_bundle_get_run_status(run_id: str) -> str:
    """Poll the status of a Databricks job run via the MCP server.

    Args:
        run_id: Databricks job run ID (integer as string).

    Returns:
        JSON with ``life_cycle_state``, ``result_state``, and ``run_page_url``.
    """
    import json  # noqa: PLC0415

    from mcp_servers.databricks_bundles.mcp_client import call_mcp_tool  # noqa: PLC0415

    result = call_mcp_tool("bundle_get_run_status", {"run_id": run_id})
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool schema registry (for Mistral function calling)
# ---------------------------------------------------------------------------

def get_tools() -> list[dict[str, Any]]:
    """Return the list of tool schemas for Mistral function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "wandb_log",
                "description": "Log metrics to Weights & Biases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "object",
                            "description": "Key/value metric pairs to log.",
                        },
                        "step": {
                            "type": "integer",
                            "description": "Optional global step counter.",
                        },
                    },
                    "required": ["metrics"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hf_list_models",
                "description": "Search Hugging Face Hub for models.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "elevenlabs_speak",
                "description": "Convert text to speech using ElevenLabs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to synthesize.",
                        },
                        "voice_id": {
                            "type": "string",
                            "description": "ElevenLabs voice name or ID.",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "databricks_download_taxi_data",
                "description": (
                    "Download NYC Yellow Taxi parquet files from the TLC CloudFront CDN "
                    "(https://github.com/toddwschneider/nyc-taxi-data) into DBFS on Databricks. "
                    "Call this BEFORE running the DLT pipeline when data hasn't been downloaded yet. "
                    "The download is idempotent — already-present files are skipped."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_year_month": {
                            "type": "string",
                            "description": "First month to download in YYYY-MM format, e.g. '2023-01'.",
                        },
                        "end_year_month": {
                            "type": "string",
                            "description": "Last month to download (inclusive) in YYYY-MM format, e.g. '2023-03'.",
                        },
                        "target": {
                            "type": "string",
                            "enum": ["dev", "prod"],
                            "description": "Bundle target environment.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "databricks_bundle_deploy",
                "description": (
                    "Deploy the NYC Taxi Databricks Asset Bundle (DLT pipeline + workflow) "
                    "to a Databricks workspace via the Databricks Bundles MCP server. "
                    "Call this before running the pipeline for the first time or after "
                    "making changes to the bundle definition."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "enum": ["dev", "prod"],
                            "description": "Bundle target environment. Use 'dev' for testing.",
                        },
                        "bundle_path": {
                            "type": "string",
                            "description": "Override bundle directory path (optional).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "databricks_bundle_run",
                "description": (
                    "Trigger a Databricks bundle job or DLT pipeline run via the MCP server. "
                    "Use resource_key='nyc_taxi_dlt' to run only the DLT pipeline, or "
                    "'nyc_taxi_workflow' to run the full orchestrated workflow including validation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resource_key": {
                            "type": "string",
                            "enum": ["nyc_taxi_workflow", "nyc_taxi_dlt"],
                            "description": "Bundle resource to run.",
                        },
                        "target": {
                            "type": "string",
                            "enum": ["dev", "prod"],
                            "description": "Bundle target environment.",
                        },
                        "full_refresh": {
                            "type": "boolean",
                            "description": "Full-refresh the DLT pipeline (reprocesses all data).",
                        },
                        "only_tasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Run only these task keys, e.g. ['download_raw_data', 'run_dlt_pipeline'].",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "databricks_bundle_get_run_status",
                "description": (
                    "Get the current status of a Databricks job run by its run ID. "
                    "Returns life_cycle_state (RUNNING, TERMINATED, etc.) and result_state "
                    "(SUCCESS, FAILED, etc.)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "Databricks job run ID returned by databricks_bundle_run.",
                        },
                    },
                    "required": ["run_id"],
                },
            },
        },
    ]
