# Copilot Instructions for Mistral Hackathon NYC

## Repository Purpose
This repository is a GPU-friendly hackathon workspace for building Mistral-based agentic workflows with integrations across NVIDIA tooling, HuggingFace models, ElevenLabs voice, and Weights & Biases experiment tracking.

## Tech Stack
- Python 3.11+
- UV-managed project dependencies (`pyproject.toml`, `uv.lock`)
- Core libraries: `mistralai`, `transformers`, `datasets`, `wandb`, `elevenlabs`, `huggingface_hub`
- Dev quality tools: `ruff`, `pytest`
- GitHub Actions CI on Ubuntu

## Build / Test / Quality Commands
Use UV-first workflows:

```bash
uv sync --extra langchain --extra dev
uv run ruff check .
uv run pytest -q
```

Fallback (when UV cannot resolve in CI):

```bash
pip install ".[dev]"
```

## Folder Conventions

### `agents/`
- Contains orchestration logic, tool bridges, and agent configuration.
- Keep orchestration policy separate from provider-specific implementation details.
- Prefer explicit configuration updates in `agents/config.yaml` over hidden defaults.

### `ml/`
- Reserved for model-training/evaluation and experiment tracking code.
- Standardize W&B initialization and metric/artifact logging.
- Keep scripts reproducible via config + environment variables.

### `skills/`
- Contains planning/reference docs for reusable skill workflows.
- Keep skill documentation task-oriented, concise, and versioned with code updates.

## Coding Conventions
- Prefer small, composable functions with typed interfaces.
- Avoid hard-coded secrets; load credentials from env/config.
- Keep docs and CI workflow updates in sync when changing setup commands.
- Use `uv run` for Python execution to ensure virtualenv consistency.
