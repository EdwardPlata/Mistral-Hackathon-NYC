"""Export a downloaded Mistral model to ONNX format.

NOTE: Requires a machine with an NVIDIA GPU, CUDA 11.8+, and the ml-gpu extras:
    uv sync --extra ml-gpu

Usage:
    uv run python DataBolt-Edge/scripts/export_to_onnx.py \\
        --model-dir ./models/mistralai--Mistral-7B-Instruct-v0.1 \\
        --output-dir ./models/onnx

The exported ONNX file is used as input for build_trt_engine.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

DEFAULT_ONNX_DIR = APP_ROOT / "models" / "onnx"
OPSET_VERSION = 17
MAX_SEQ_LEN = 512


def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "optimum"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print("Install with: uv sync --extra ml-gpu")
        sys.exit(1)


def _check_gpu() -> None:
    import torch
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Export will run on CPU — this is very slow.")
        print("         Recommended: RTX 3080+ with 16+ GB VRAM.")
        response = input("Continue on CPU? [y/N]: ").strip().lower()
        if response != "y":
            sys.exit(0)


def export_to_onnx(model_dir: Path, output_dir: Path) -> Path:
    """Export model to ONNX using Optimum (preferred) or raw torch.onnx fallback."""
    import torch
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "mistral.onnx"

    print(f"Loading tokenizer from {model_dir}…")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    try:
        # Preferred path: Optimum handles dynamic axes and past-key-values properly
        from optimum.exporters.onnx import main_export
        print("Exporting via Optimum (recommended)…")
        main_export(
            model_name_or_path=str(model_dir),
            output=str(output_dir),
            task="text-generation-with-past",
            opset=OPSET_VERSION,
            device="cuda" if torch.cuda.is_available() else "cpu",
            fp16=torch.cuda.is_available(),
        )
        # Optimum writes model.onnx or model_merged.onnx
        candidates = list(output_dir.glob("*.onnx"))
        if candidates:
            onnx_path = candidates[0]
    except ImportError:
        # Fallback: raw torch.onnx.export (no past-key-value optimization)
        from transformers import AutoModelForCausalLM
        print("Optimum not available — falling back to torch.onnx.export…")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device} (may take several minutes)…")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()

        dummy_input = torch.ones((1, MAX_SEQ_LEN), dtype=torch.long).to(device)
        print(f"Exporting to {onnx_path}…")
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input,),
                str(onnx_path),
                opset_version=OPSET_VERSION,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "logits": {0: "batch", 1: "seq_len"},
                },
                do_constant_folding=True,
            )
        del model
        torch.cuda.empty_cache()

    return onnx_path


def verify_onnx(onnx_path: Path) -> bool:
    """Run ONNX model checker to confirm the graph is valid."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        size_gb = onnx_path.stat().st_size / 1e9
        print(f"ONNX check passed ✓ | {onnx_path.name} | {size_gb:.2f} GB")
        return True
    except ImportError:
        print("onnx package not installed — skipping graph verification.")
        return True
    except Exception as exc:
        print(f"ONNX check failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Mistral model to ONNX")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to downloaded model weights (output of download_model.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help=f"Directory for ONNX output (default: {DEFAULT_ONNX_DIR})",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"ERROR: model-dir not found: {args.model_dir}")
        print("Run download_model.py first.")
        return 1

    _check_deps()
    _check_gpu()

    onnx_path = export_to_onnx(args.model_dir, args.output_dir)

    if not verify_onnx(onnx_path):
        return 1

    print(f"\nNext step: run build_trt_engine.py --onnx-path {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
