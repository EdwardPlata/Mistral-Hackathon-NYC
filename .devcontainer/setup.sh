#!/usr/bin/env bash
set -euo pipefail

# Suppress UV hardlink warnings in containerized environments
export UV_LINK_MODE=copy

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER_FILE="${WORKSPACE_DIR}/.devcontainer/.setup-complete"
HASH_FILE="${WORKSPACE_DIR}/.devcontainer/.dependency-fingerprint"

# Install uv if missing (Codespaces-friendly)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "${WORKSPACE_DIR}"

# Ensure .venv exists
if [[ ! -d .venv ]]; then
  uv venv --python 3.11
fi

fingerprint_inputs=(
  ".devcontainer/devcontainer.json"
  ".devcontainer/devcontainer.gpu.json"
  "pyproject.toml"
  "uv.lock"
  "requirements.txt"
  "requirements-dev.txt"
)

fingerprint_payload=""
for file in "${fingerprint_inputs[@]}"; do
  if [[ -f "${file}" ]]; then
    fingerprint_payload+="$(sha256sum "${file}")\n"
  fi
done

dependency_fingerprint="$(printf "%b" "${fingerprint_payload}" | sha256sum | awk '{print $1}')"
current_fingerprint=""
if [[ -f "${HASH_FILE}" ]]; then
  current_fingerprint="$(cat "${HASH_FILE}")"
fi

if [[ ! -f "${MARKER_FILE}" || "${current_fingerprint}" != "${dependency_fingerprint}" ]]; then
  uv sync --extra langchain --extra dev
  uv pip install --python .venv/bin/python wandb
  printf "%s" "${dependency_fingerprint}" > "${HASH_FILE}"
  touch "${MARKER_FILE}"
fi
