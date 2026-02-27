#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER_FILE="${WORKSPACE_DIR}/.devcontainer/.setup-complete"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -f "${MARKER_FILE}" ]]; then
  cd "${WORKSPACE_DIR}"
  # Fast startup path: install only core + development tooling.
  uv sync --extra dev
  touch "${MARKER_FILE}"
fi
