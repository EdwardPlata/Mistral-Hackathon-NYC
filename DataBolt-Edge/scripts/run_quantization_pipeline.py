#!/usr/bin/env python
"""DataBolt Edge — Model Conversion & Quantization Pipeline CLI.

Converts a float32 ONNX model (produced by export_to_onnx.py) into FP16,
dynamic INT8, and optionally static INT8 variants, then benchmarks all
produced models and writes a summary report.

Usage examples
--------------
# Full pipeline (FP16 + dynamic INT8 + benchmark)
  uv run python scripts/run_quantization_pipeline.py \\
      --source models/mistral-7b-instruct.onnx \\
      --output-dir models/quantized

# Only dynamic INT8, no benchmark
  uv run python scripts/run_quantization_pipeline.py \\
      --source models/mistral-7b-instruct.onnx \\
      --no-fp16 --no-benchmark

# Static INT8 with real tokenizer
  uv run python scripts/run_quantization_pipeline.py \\
      --source models/mistral-7b-instruct.onnx \\
      --static-int8 \\
      --tokenizer models/mistral-7b-instruct

# Dry-run mode (validate source + print config, no conversion)
  uv run python scripts/run_quantization_pipeline.py \\
      --source models/mistral-7b-instruct.onnx \\
      --dry-run

Outputs
-------
  <output-dir>/fp16/model_fp16.onnx
  <output-dir>/dynamic_int8/model_dynamic_int8.onnx
  <output-dir>/static_int8/model_static_int8.onnx   (if --static-int8)
  <output-dir>/pipeline_report.json
  <output-dir>/pipeline_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure DataBolt-Edge root is importable when run from repo root
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DataBolt Edge — Model Conversion & Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O
    p.add_argument(
        "--source", "-s",
        required=True,
        help="Path to source float32 ONNX model (export_to_onnx.py output).",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="models/quantized",
        help="Root output directory. Default: models/quantized",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer directory for calibration (static INT8 only).",
    )

    # Steps
    p.add_argument("--no-fp16", action="store_true", help="Skip FP16 conversion.")
    p.add_argument("--no-dynamic-int8", action="store_true", help="Skip dynamic INT8 quantization.")
    p.add_argument("--static-int8", action="store_true", help="Enable static INT8 quantization (needs tokenizer or uses synthetic data).")
    p.add_argument("--no-benchmark", action="store_true", help="Skip benchmark step.")
    p.add_argument("--dry-run", action="store_true", help="Validate config and source; do not convert.")

    # Benchmark options
    p.add_argument("--benchmark-runs", type=int, default=20, help="Number of timed runs per model. Default: 20")
    p.add_argument("--benchmark-seq-len", type=int, default=64, help="Token sequence length for benchmarks. Default: 64")

    # Quantization options
    p.add_argument("--weight-type", choices=["QInt8", "QUInt8"], default="QInt8", help="INT8 weight type. Default: QInt8")
    p.add_argument("--per-channel", action="store_true", help="Enable per-channel INT8 quantization (more accurate, slightly slower).")
    p.add_argument("--fail-fast", action="store_true", help="Abort pipeline on first step failure.")

    # Misc
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ── import pipeline ───────────────────────────────────────────────────────
    try:
        from conversion import PipelineConfig, QuantizationPipeline
    except ImportError as exc:
        log.error("Could not import conversion package: %s", exc)
        log.error("Make sure onnx and onnxruntime are installed: uv add onnx onnxruntime onnxconverter-common")
        return 1

    # ── build config ──────────────────────────────────────────────────────────
    cfg = PipelineConfig(
        source_onnx=args.source,
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer,

        run_fp16=not args.no_fp16,
        run_dynamic_int8=not args.no_dynamic_int8,
        run_static_int8=args.static_int8,
        run_benchmark=not args.no_benchmark and not args.dry_run,

        benchmark_n_runs=args.benchmark_runs,
        benchmark_seq_len=args.benchmark_seq_len,

        int8_weight_type=args.weight_type,
        int8_per_channel=args.per_channel,
        fail_fast=args.fail_fast,
    )

    # ── print config summary ──────────────────────────────────────────────────
    log.info("Source ONNX : %s", cfg.source_onnx)
    log.info("Output dir  : %s", cfg.output_dir)
    log.info("FP16        : %s", "yes" if cfg.run_fp16 else "no")
    log.info("Dynamic INT8: %s", "yes" if cfg.run_dynamic_int8 else "no")
    log.info("Static INT8 : %s", "yes" if cfg.run_static_int8 else "no")
    log.info("Benchmark   : %s  (%d runs, seq=%d)", "yes" if cfg.run_benchmark else "no", cfg.benchmark_n_runs, cfg.benchmark_seq_len)

    if args.dry_run:
        log.info("Dry-run mode — validating source model only")
        import onnx
        src = Path(args.source)
        if not src.exists():
            log.error("Source ONNX not found: %s", src)
            return 1
        try:
            model = onnx.load(str(src))
            onnx.checker.check_model(model)
            size_mb = src.stat().st_size / (1024 ** 2)
            log.info(
                "Source valid — %.1f MB  %d nodes  opset=%d",
                size_mb, len(model.graph.node), model.opset_import[0].version,
            )
            return 0
        except Exception as exc:
            log.error("Source model invalid: %s", exc)
            return 1

    # ── run pipeline ──────────────────────────────────────────────────────────
    pipeline = QuantizationPipeline(cfg)
    report = pipeline.run()

    # ── print report ──────────────────────────────────────────────────────────
    print("\n" + report.summary_table())

    if report.failed_variants:
        log.warning("Failed variants: %s", report.failed_variants)

    status = 0 if report.succeeded or report.successful_variants else 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
