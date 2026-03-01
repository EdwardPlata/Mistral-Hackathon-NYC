"""Download Mistral model weights from HuggingFace Hub.

Usage:
    uv run python DataBolt-Edge/scripts/download_model.py
    uv run python DataBolt-Edge/scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.1
    uv run python DataBolt-Edge/scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.1 --output-dir ./models

Requires:
    HUGGINGFACE_TOKEN env var (or HF_TOKEN fallback)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


SUPPORTED_MODELS = {
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral-7b-instruct-v03": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b-base": "mistralai/Mistral-7B-v0.1",
}

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
DEFAULT_OUTPUT_DIR = APP_ROOT / "models"


def _resolve_token() -> str:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HUGGINGFACE_TOKEN or HF_TOKEN before running this script.")
        sys.exit(1)
    return token


def download_model(model_id: str, output_dir: Path, token: str) -> Path:
    """Download model weights to output_dir using huggingface_hub snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: uv sync --extra ml-gpu")
        sys.exit(1)

    safe_name = model_id.replace("/", "--")
    local_dir = output_dir / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} → {local_dir}")
    print("This may take several minutes (~14 GB for Mistral-7B)…")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    print(f"Download complete: {local_dir}")
    return local_dir


def verify_download(local_dir: Path) -> bool:
    """Basic check that key model files are present."""
    required_patterns = ["config.json", "tokenizer.json"]
    weight_patterns = ["*.safetensors", "*.bin"]

    for pattern in required_patterns:
        if not list(local_dir.glob(pattern)):
            print(f"WARNING: {pattern} not found in {local_dir}")
            return False

    has_weights = any(list(local_dir.glob(p)) for p in weight_patterns)
    if not has_weights:
        print(f"WARNING: No weight files (*.safetensors or *.bin) found in {local_dir}")
        return False

    config = local_dir / "config.json"
    weight_files = list(local_dir.glob("*.safetensors")) or list(local_dir.glob("*.bin"))
    total_size_gb = sum(f.stat().st_size for f in weight_files) / 1e9

    print(f"Verified: config.json ✓ | {len(weight_files)} weight file(s) | {total_size_gb:.1f} GB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Mistral model weights from HuggingFace Hub")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local directory to save weights (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported shorthand model names and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Supported shorthand names:")
        for name, model_id in SUPPORTED_MODELS.items():
            print(f"  {name:<30} → {model_id}")
        return 0

    model_id = SUPPORTED_MODELS.get(args.model, args.model)
    token = _resolve_token()

    local_dir = download_model(model_id, args.output_dir, token)

    if not verify_download(local_dir):
        print("Download verification failed.")
        return 1

    print(f"\nNext step: run export_to_onnx.py --model-dir {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
