# Codespaces and Prebuilds

## Startup expectations with prebuilds

Codespaces prebuilds are configured for the default branch (`main`) and high-traffic branches (`develop`, `release/**`) so dependency installation happens before developers launch a Codespace. The `.devcontainer/setup.sh` script now fingerprints `devcontainer.json` and dependency files and runs `uv sync --extra langchain --extra dev` during prebuild/update when those inputs change.

In a healthy prebuild path, first open should mostly skip dependency resolution and land in a ready-to-code state.

## When a full rebuild is still required

Use **Rebuild Container** when:
- The base image or devcontainer Features change in ways that cannot be applied incrementally.
- Required secrets/environment values change and tool auth is stale.
- Dependency installation fails during prebuild or fingerprint state is corrupted.
