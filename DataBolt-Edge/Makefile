.PHONY: help test-nvidia lint-nvidia check-nvidia probe-nvidia-live probe-nvidia-app \
        test-hf check-hf probe-hf-live test-all

help:
	@echo "Available targets:"
	@echo "  make test-nvidia       # Run nvidia_api_management unit tests"
	@echo "  make lint-nvidia       # Lint nvidia_api_management package and tests"
	@echo "  make check-nvidia      # Run tests + lint (NVIDIA)"
	@echo "  make probe-nvidia-app  # Run interactive NVIDIA probe (prompts for input)"
	@echo "  make probe-nvidia-live # Run live NVIDIA API probe (requires NVIDIA_BEARER_TOKEN)"
	@echo "  make test-hf           # Run huggingface_integration unit tests"
	@echo "  make check-hf          # Run huggingface_integration tests + lint"
	@echo "  make probe-hf-live     # Run live HuggingFace probe (requires HUGGINGFACE_TOKEN or HF_TOKEN)"
	@echo "  make test-all          # Run all unit tests"

# ── NVIDIA ──────────────────────────────────────────────────────────────────

test-nvidia:
	uv run python -m unittest discover -s tests -p "test_nvidia_*.py"

lint-nvidia:
	uv run ruff check nvidia_api_management tests testing.py

check-nvidia: test-nvidia lint-nvidia

probe-nvidia-app:
	uv run python testing.py

probe-nvidia-live:
	@if [ -z "$$NVIDIA_BEARER_TOKEN" ] && [ -z "$$NVIDIA_API_KEY" ]; then \
		echo "Set NVIDIA_BEARER_TOKEN or NVIDIA_API_KEY before running this target"; exit 1; \
	fi
	uv run python scripts/live_nvidia_probe.py

# ── HuggingFace ─────────────────────────────────────────────────────────────

test-hf:
	uv run python -m unittest discover -s tests -p "test_hf_*.py"

lint-hf:
	uv run ruff check huggingface_integration tests/test_hf_*.py scripts/live_hf_probe.py

check-hf: test-hf lint-hf

probe-hf-live:
	@if [ -z "$$HUGGINGFACE_TOKEN" ] && [ -z "$$HF_TOKEN" ]; then \
		echo "Set HUGGINGFACE_TOKEN or HF_TOKEN before running this target"; exit 1; \
	fi
	uv run python scripts/live_hf_probe.py

# ── All ─────────────────────────────────────────────────────────────────────

test-all:
	uv run python -m unittest discover -s tests -p "test_*.py"
