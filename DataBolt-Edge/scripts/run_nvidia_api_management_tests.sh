#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[nvidia_api_management] Running unit tests..."
uv run python -m unittest discover -s DataBolt-Edge/tests -p "test_nvidia_*.py"

echo "[nvidia_api_management] Running lint checks..."
uv run ruff check DataBolt-Edge/nvidia_api_management DataBolt-Edge/tests DataBolt-Edge/testing.py

echo "[nvidia_api_management] Done."
