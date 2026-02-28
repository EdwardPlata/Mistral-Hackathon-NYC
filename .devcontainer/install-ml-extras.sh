#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_DIR}"

# Ensure .venv exists
if [[ ! -d .venv ]]; then
  uv venv --python 3.11
fi

uv sync --extra dev --extra ml-gpu --extra langchain
