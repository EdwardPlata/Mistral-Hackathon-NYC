"""Tool definitions for the Mistral NYC hackathon agent.

Each function represents a tool that the Mistral model can invoke during
agentic workflows. External API calls are kept here so that main_agent.py
stays clean and composable.

Copilot: follow the pattern below when adding new tools – define a Python
function plus a matching JSON schema entry in get_tools().
"""

import os
from typing import Any


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

def wandb_list_runs(project: str | None = None, limit: int = 10) -> list[dict]:
    """List recent W&B runs for a project, with key metrics.

    Args:
        project: W&B project name (defaults to WANDB_PROJECT env var or
                 "agentops-studio").
        limit: Maximum number of runs to return.

    Returns:
        List of dicts with run id, name, state, created_at, and summary metrics.
    """
    import wandb  # noqa: PLC0415

    project = project or os.getenv("WANDB_PROJECT", "agentops-studio")
    api = wandb.Api()
    runs = api.runs(project, per_page=limit)
    result = []
    for run in runs:
        result.append({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
        })
        if len(result) >= limit:
            break
    return result


def wandb_get_run_metrics(run_id: str, project: str | None = None) -> dict:
    """Fetch summary metrics and config for a specific W&B run.

    Args:
        run_id: W&B run ID (short hash shown in the UI).
        project: W&B project name (defaults to WANDB_PROJECT env var or
                 "agentops-studio").

    Returns:
        Dict with run metadata, config, and summary metrics.
    """
    import wandb  # noqa: PLC0415

    project = project or os.getenv("WANDB_PROJECT", "agentops-studio")
    api = wandb.Api()
    entity = api.default_entity
    run = api.run(f"{entity}/{project}/{run_id}")
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "config": dict(run.config),
        "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
        "tags": list(run.tags),
    }


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
# Tool schema registry (for Mistral function calling)
# ---------------------------------------------------------------------------

def get_tools() -> list[dict[str, Any]]:
    """Return the list of tool schemas for Mistral function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "wandb_list_runs",
                "description": "List recent W&B runs for a project with their metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "W&B project name (defaults to WANDB_PROJECT env var).",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max number of runs to return (default 10).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wandb_get_run_metrics",
                "description": "Get config and summary metrics for a specific W&B run.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "W&B run ID (short hash shown in the dashboard).",
                        },
                        "project": {
                            "type": "string",
                            "description": "W&B project name (defaults to WANDB_PROJECT env var).",
                        },
                    },
                    "required": ["run_id"],
                },
            },
        },
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
    ]
