# AGENTS.md (Workspace-Wide Instructions)

This file defines baseline instructions for all coding agents operating in this repository.

## Scope and Ownership

### Folder Responsibilities
- `agents/`: Agent orchestration and runtime tooling for Mistral-centric workflows.
- `ml/`: Training, evaluation, and experiment-tracking code (especially W&B usage).
- `skills/`: Skill plans and documentation for repeatable workflows.
- `prompts/`: Prompt templates and supporting prompt docs.
- `.github/`: CI/workflow automation and GitHub agent/instructions metadata.

## Security and Configuration
- Do not commit secrets, tokens, or private credentials.
- Use environment variables, `.env`-style local config, and GitHub/Codespaces secrets.
- Keep provider integrations configurable and avoid hard-coded endpoints/keys unless public + documented.

## Testing Expectations
- Run lint and tests for all non-trivial changes:
  - `uv run ruff check .`
  - `uv run pytest -q`
- If adding new scripts, prefer UV-compatible execution paths.
- Keep CI and docs aligned with the effective setup/test commands.

## Branching and PR Expectations
- Work from a feature branch and use clear, scoped commits.
- Include concise summaries of functional impact and validation steps in PR descriptions.
- Update relevant documentation alongside code/config changes.

## Agent-Specific Notes
- For Claude-specific behavior and extra guidance, see `CLAUDE.md`.
