"""Live HuggingFace probe — requires HUGGINGFACE_TOKEN or HF_TOKEN."""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from huggingface_integration import run_probe  # noqa: E402
from huggingface_integration.client import HuggingFaceClient  # noqa: E402


def main() -> int:
    # 1. Token validation
    print("── whoami ──────────────────────────────────────")
    result = run_probe()
    print(f"success       = {result.success}")
    print(f"latency_ms    = {result.latency_ms:.2f}")
    if result.username:
        print(f"username      = {result.username}")
    if result.email:
        print(f"email         = {result.email}")
    if result.plan:
        print(f"plan          = {result.plan}")
    if result.error:
        print(f"error         = {result.error}")
        return 1

    client = HuggingFaceClient()

    # 2. Model info for a known Mistral model
    print("\n── model_info: mistralai/Mistral-7B-Instruct-v0.1 ──")
    try:
        info = client.model_info("mistralai/Mistral-7B-Instruct-v0.1")
        print(f"id            = {info['id']}")
        print(f"pipeline_tag  = {info.get('pipeline_tag', '—')}")
        print(f"downloads     = {info.get('downloads', 0):,}")
        print(f"likes         = {info.get('likes', 0):,}")
        print(f"gated         = {info.get('gated', False)}")
        tags = info.get("tags", [])[:5]
        if tags:
            print(f"tags (first 5)= {tags}")
    except Exception as exc:
        print(f"model_info error: {exc}")

    # 3. Top Mistral text-generation models
    print("\n── list_models: mistralai / text-generation (top 8) ──")
    try:
        models = client.list_models(author="mistralai", pipeline_tag="text-generation", limit=8)
        for m in models:
            print(f"  {m.model_id:<55} ↓{m.downloads:>8,}  ♥{m.likes:>5,}")
    except Exception as exc:
        print(f"list_models error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
