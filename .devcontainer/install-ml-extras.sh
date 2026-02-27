#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${WORKSPACE_DIR}"
uv sync --extra dev --extra ml-gpu --extra langchain
