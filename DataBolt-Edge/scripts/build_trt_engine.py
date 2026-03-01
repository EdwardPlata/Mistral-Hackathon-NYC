"""Build a TensorRT-LLM engine from an ONNX Mistral model.

NOTE: Requires an NVIDIA GPU with TensorRT-LLM installed:
    - CUDA 11.8+ / 12.x
    - TensorRT 9+ (part of TensorRT-LLM)
    - Install: pip install tensorrt-llm (or via NVIDIA NGC container)

Usage:
    uv run python DataBolt-Edge/scripts/build_trt_engine.py \\
        --onnx-path ./models/onnx/mistral.onnx \\
        --output-dir ./models/trt \\
        --dtype int8

The resulting .engine file is loaded by the local TRT inference backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

DEFAULT_TRT_DIR = APP_ROOT / "models" / "trt"
SUPPORTED_DTYPES = ("fp16", "int8", "fp8")


def _check_tensorrt() -> str:
    """Returns the TRT build method available: 'trtllm', 'trt', or raises."""
    try:
        import tensorrt_llm  # noqa: F401
        return "trtllm"
    except ImportError:
        pass
    try:
        import tensorrt  # noqa: F401
        return "trt"
    except ImportError:
        pass
    print("ERROR: Neither tensorrt_llm nor tensorrt is installed.")
    print("Install TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM")
    print("Or use the NGC container: nvcr.io/nvidia/tensorrt:24.01-py3")
    sys.exit(1)


def build_with_trtllm(onnx_path: Path, output_dir: Path, dtype: str) -> Path:
    """Build engine using TensorRT-LLM high-level API."""
    import tensorrt_llm
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.models import MistralForCausalLM

    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = output_dir / f"mistral_{dtype}.engine"

    builder = Builder()
    builder_config = builder.create_builder_config(
        name="mistral",
        precision=dtype,
        timing_cache=str(output_dir / "timing.cache"),
        max_batch_size=1,
        max_input_len=2048,
        max_output_len=2048,
        max_beam_width=1,
        use_inflight_batching=False,
        paged_kv_cache=False,
    )

    print(f"Building TensorRT-LLM engine ({dtype})…")
    t0 = time.perf_counter()
    network = builder.create_network()
    with builder.trt_builder.create_builder_config() as trt_config:
        engine = builder.build_engine(network, trt_config)

    elapsed = time.perf_counter() - t0
    with open(engine_path, "wb") as f:
        f.write(engine)

    print(f"Engine built in {elapsed:.1f}s → {engine_path}")
    return engine_path


def build_with_trt(onnx_path: Path, output_dir: Path, dtype: str) -> Path:
    """Build engine using base TensorRT API (fallback when TRT-LLM is unavailable)."""
    import tensorrt as trt

    output_dir.mkdir(parents=True, exist_ok=True)
    dtype_flag = {
        "fp16": trt.BuilderFlag.FP16,
        "int8": trt.BuilderFlag.INT8,
        "fp8": trt.BuilderFlag.FP8,
    }[dtype]
    engine_path = output_dir / f"mistral_{dtype}.engine"

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_flag(dtype_flag)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    print(f"Parsing ONNX: {onnx_path}…")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parser error {i}: {parser.get_error(i)}")
            sys.exit(1)

    # Dynamic shape profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 1), (1, 512), (1, 2048))
    config.add_optimization_profile(profile)

    print(f"Building TensorRT engine ({dtype}) — this can take 10-30 minutes…")
    t0 = time.perf_counter()
    engine_bytes = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - t0

    if engine_bytes is None:
        print("ERROR: Engine build failed.")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    print(f"Engine built in {elapsed:.1f}s → {engine_path}")
    return engine_path


def write_engine_metadata(engine_path: Path, dtype: str, onnx_path: Path) -> None:
    """Write a sidecar JSON with engine metadata for the inference loader."""
    meta = {
        "engine_path": str(engine_path),
        "source_onnx": str(onnx_path),
        "dtype": dtype,
        "max_input_len": 2048,
        "max_output_len": 2048,
    }
    meta_path = engine_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata written: {meta_path}")


def benchmark_engine(engine_path: Path) -> None:
    """Run a quick latency benchmark using the built engine."""
    try:
        import tensorrt as trt
        import numpy as np

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        seq_len = 128
        dummy = np.ones((1, seq_len), dtype=np.int32)
        d_input = cuda.mem_alloc(dummy.nbytes)
        cuda.memcpy_htod(d_input, dummy)

        # warmup
        context.execute_v2([int(d_input)])

        runs = 5
        t0 = time.perf_counter()
        for _ in range(runs):
            context.execute_v2([int(d_input)])
        elapsed_ms = (time.perf_counter() - t0) / runs * 1000

        print(f"Benchmark (seq_len={seq_len}): {elapsed_ms:.1f} ms/query avg over {runs} runs")
    except Exception as exc:
        print(f"Benchmark skipped: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from Mistral ONNX model")
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Path to the ONNX model (output of export_to_onnx.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRT_DIR,
        help=f"Directory for TRT engine output (default: {DEFAULT_TRT_DIR})",
    )
    parser.add_argument(
        "--dtype",
        choices=SUPPORTED_DTYPES,
        default="int8",
        help="Quantization dtype: fp16 (baseline), int8 (~2x faster), fp8 (Ampere+)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a latency benchmark after building",
    )
    args = parser.parse_args()

    if not args.onnx_path.exists():
        print(f"ERROR: onnx-path not found: {args.onnx_path}")
        print("Run export_to_onnx.py first.")
        return 1

    method = _check_tensorrt()
    print(f"TRT build method: {method}")

    if method == "trtllm":
        engine_path = build_with_trtllm(args.onnx_path, args.output_dir, args.dtype)
    else:
        engine_path = build_with_trt(args.onnx_path, args.output_dir, args.dtype)

    write_engine_metadata(engine_path, args.dtype, args.onnx_path)

    if args.benchmark:
        benchmark_engine(engine_path)

    print(f"\nEngine ready: {engine_path}")
    print("Set INFERENCE_BACKEND=local and TRT_ENGINE_PATH to use it:")
    print(f"  export INFERENCE_BACKEND=local")
    print(f"  export TRT_ENGINE_PATH={engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
