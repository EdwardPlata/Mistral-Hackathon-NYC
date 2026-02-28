.PHONY: help test-nvidia lint-nvidia check-nvidia probe-nvidia-live probe-nvidia-app

help:
	@echo "Available targets:"
	@echo "  make test-nvidia      # Run nvidia_api_management unit tests"
	@echo "  make lint-nvidia      # Lint nvidia_api_management package and tests"
	@echo "  make check-nvidia     # Run tests + lint"
	@echo "  make probe-nvidia-app # Run app-level probe script"
	@echo "  make probe-nvidia-live# Run live NVIDIA API probe (requires NVIDIA_API_KEY)"

test-nvidia:
	uv run python -m unittest discover -s tests -p "test_nvidia_*.py"

lint-nvidia:
	uv run ruff check nvidia_api_management tests testing.py

check-nvidia: test-nvidia lint-nvidia

probe-nvidia-app:
	uv run python testing.py

probe-nvidia-live:
	@if [ -z "$$NVIDIA_BEARER_TOKEN" ]; then echo "NVIDIA_BEARER_TOKEN is required"; exit 1; fi
	uv run python scripts/live_nvidia_probe.py
