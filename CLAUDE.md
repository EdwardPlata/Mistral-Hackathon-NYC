# CLAUDE.md

This file provides Claude-specific operating guidance for this repository.

## Scope
- Applies to Claude-family agents when they are working in this repo.
- Complements (does not replace) the shared workspace rules in `AGENTS.md`.

## Behavior
- Follow repository conventions for UV-based setup, lint, and test execution.
- Prefer minimal, high-signal patches and keep docs in sync with implementation changes.
- Preserve provider-agnostic boundaries between orchestration logic and integration wrappers.
- Escalate uncertainty by documenting assumptions in commit/PR summaries.

## Security
- Never hard-code secrets.
- Prefer environment variables and documented configuration paths.
