"""End-to-end model conversion & quantization pipeline for DataBolt Edge.

Orchestrates the full workflow:

  1. Validate source ONNX model (schema + checker)
  2. FP16 conversion
  3. Dynamic INT8 quantization
  4. Static INT8 quantization (with calibration data)
  5. Benchmark all variants
  6. Generate comparison report

Each step is independently skippable via :class:`PipelineConfig`.  The
pipeline writes all outputs under a versioned sub-directory of ``output_dir``
and saves a ``pipeline_report.json`` summary alongside the models.

Example::

    from conversion import PipelineConfig, QuantizationPipeline

    cfg = PipelineConfig(
        source_onnx="models/mistral-7b-instruct.onnx",
        output_dir="models/quantized",
        run_fp16=True,
        run_dynamic_int8=True,
        run_static_int8=False,   # needs tokenizer
        run_benchmark=True,
        benchmark_n_runs=20,
    )
    pipeline = QuantizationPipeline(cfg)
    report = pipeline.run()
    print(report.summary_table())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

log = logging.getLogger(__name__)


# ── step enum ─────────────────────────────────────────────────────────────────

class PipelineStep(Enum):
    VALIDATE = auto()
    FP16 = auto()
    DYNAMIC_INT8 = auto()
    STATIC_INT8 = auto()
    BENCHMARK = auto()
    REPORT = auto()


# ── config dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for :class:`QuantizationPipeline`.

    Parameters
    ----------
    source_onnx:
        Path to the source float32 ONNX model.
    output_dir:
        Root directory for all pipeline outputs.  Sub-directories
        ``fp16/``, ``dynamic_int8/``, ``static_int8/`` are created
        automatically.
    tokenizer_dir:
        HuggingFace tokenizer directory for static INT8 calibration.
        If None, synthetic calibration data is used.
    run_fp16:
        Whether to run the FP16 conversion step.
    run_dynamic_int8:
        Whether to run the dynamic INT8 quantization step.
    run_static_int8:
        Whether to run the static INT8 quantization step.
    run_benchmark:
        Whether to benchmark all produced model variants.
    benchmark_n_runs:
        Number of timed inference calls per model.
    benchmark_seq_len:
        Token sequence length for benchmark synthetic inputs.
    fp16_keep_io_types:
        Keep model input/output in float32 for FP16 variant.
    int8_weight_type:
        "QInt8" (default) or "QUInt8".
    int8_per_channel:
        Enable per-channel quantization for more accuracy.
    calibration_seq_len:
        Token sequence length for calibration data.
    extra_calibration_prompts:
        Additional prompts for static INT8 calibration.
    fail_fast:
        Stop the pipeline on the first failed step.
    """

    source_onnx: str | Path = ""
    output_dir: str | Path = "models/quantized"

    tokenizer_dir: str | Path | None = None

    run_fp16: bool = True
    run_dynamic_int8: bool = True
    run_static_int8: bool = False   # requires tokenizer / calibration data
    run_benchmark: bool = True

    benchmark_n_runs: int = 20
    benchmark_seq_len: int = 64

    fp16_keep_io_types: bool = True
    int8_weight_type: str = "QInt8"
    int8_per_channel: bool = False
    calibration_seq_len: int = 128
    extra_calibration_prompts: list[str] = field(default_factory=list)

    fail_fast: bool = False

    def __post_init__(self) -> None:
        self.source_onnx = Path(self.source_onnx) if self.source_onnx else Path("")
        self.output_dir = Path(self.output_dir)
        if self.tokenizer_dir:
            self.tokenizer_dir = Path(self.tokenizer_dir)


# ── step result ───────────────────────────────────────────────────────────────

@dataclass
class StepOutcome:
    step: PipelineStep
    success: bool
    duration_s: float = 0.0
    message: str = ""
    data: dict = field(default_factory=dict)


# ── pipeline ──────────────────────────────────────────────────────────────────

class QuantizationPipeline:
    """Orchestrates the DataBolt model conversion & quantization pipeline.

    All step methods return a :class:`StepOutcome` regardless of success so
    the pipeline can continue or abort cleanly.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.outcomes: list[StepOutcome] = []
        self._quant_results: list = []      # QuantizationResult objects
        self._bench_results: list = []      # BenchmarkResult objects

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> "PipelineReport":
        """Execute the full pipeline and return a :class:`PipelineReport`."""
        from .report import PipelineReport

        log.info("=" * 60)
        log.info("DataBolt Edge — Quantization Pipeline")
        log.info("Source: %s", self.cfg.source_onnx)
        log.info("Output: %s", self.cfg.output_dir)
        log.info("=" * 60)

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: validate
        outcome = self._step_validate()
        self.outcomes.append(outcome)
        if not outcome.success and self.cfg.fail_fast:
            return self._build_report()

        # Step 2: FP16
        if self.cfg.run_fp16:
            outcome = self._step_fp16()
            self.outcomes.append(outcome)
            if not outcome.success and self.cfg.fail_fast:
                return self._build_report()

        # Step 3: Dynamic INT8
        if self.cfg.run_dynamic_int8:
            outcome = self._step_dynamic_int8()
            self.outcomes.append(outcome)
            if not outcome.success and self.cfg.fail_fast:
                return self._build_report()

        # Step 4: Static INT8
        if self.cfg.run_static_int8:
            outcome = self._step_static_int8()
            self.outcomes.append(outcome)
            if not outcome.success and self.cfg.fail_fast:
                return self._build_report()

        # Step 5: Benchmark
        if self.cfg.run_benchmark:
            outcome = self._step_benchmark()
            self.outcomes.append(outcome)

        # Step 6: Report
        report = self._build_report()
        self._step_save_report(report)
        return report

    # ── step implementations ──────────────────────────────────────────────────

    def _step_validate(self) -> StepOutcome:
        t0 = time.perf_counter()
        log.info("[validate] Checking source model: %s", self.cfg.source_onnx)
        try:
            import onnx
            if not self.cfg.source_onnx.exists():
                return StepOutcome(
                    step=PipelineStep.VALIDATE,
                    success=False,
                    duration_s=time.perf_counter() - t0,
                    message=f"Source ONNX not found: {self.cfg.source_onnx}",
                )
            model = onnx.load(str(self.cfg.source_onnx))
            onnx.checker.check_model(model)
            size_mb = self.cfg.source_onnx.stat().st_size / (1024 ** 2)
            n_nodes = len(model.graph.node)
            msg = f"Valid  {size_mb:.1f} MB  {n_nodes} nodes  opset={model.opset_import[0].version}"
            log.info("[validate] %s", msg)
            return StepOutcome(
                step=PipelineStep.VALIDATE,
                success=True,
                duration_s=time.perf_counter() - t0,
                message=msg,
                data={"size_mb": size_mb, "n_nodes": n_nodes},
            )
        except Exception as exc:
            msg = f"Validation failed: {exc}"
            log.warning("[validate] %s", msg)
            # Non-fatal: allow pipeline to continue (model may still quantize)
            return StepOutcome(
                step=PipelineStep.VALIDATE,
                success=False,
                duration_s=time.perf_counter() - t0,
                message=msg,
            )

    def _step_fp16(self) -> StepOutcome:
        from .quantize_onnx import convert_to_fp16
        t0 = time.perf_counter()
        log.info("[fp16] Starting FP16 conversion")
        out_path = self.cfg.output_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(
            self.cfg.source_onnx,
            out_path,
            keep_io_types=self.cfg.fp16_keep_io_types,
        )
        self._quant_results.append(result)
        log.info("[fp16] %s", result)
        return StepOutcome(
            step=PipelineStep.FP16,
            success=result.success,
            duration_s=time.perf_counter() - t0,
            message=str(result),
            data=result.__dict__ if result.success else {"error": result.error},
        )

    def _step_dynamic_int8(self) -> StepOutcome:
        from .quantize_onnx import quantize_dynamic_int8
        t0 = time.perf_counter()
        log.info("[dynamic_int8] Starting dynamic INT8 quantization")
        out_path = self.cfg.output_dir / "dynamic_int8" / "model_dynamic_int8.onnx"
        result = quantize_dynamic_int8(
            self.cfg.source_onnx,
            out_path,
            weight_type=self.cfg.int8_weight_type,
            per_channel=self.cfg.int8_per_channel,
        )
        self._quant_results.append(result)
        log.info("[dynamic_int8] %s", result)
        return StepOutcome(
            step=PipelineStep.DYNAMIC_INT8,
            success=result.success,
            duration_s=time.perf_counter() - t0,
            message=str(result),
            data=result.__dict__ if result.success else {"error": result.error},
        )

    def _step_static_int8(self) -> StepOutcome:
        from .calibrate import DataBoltCalibrationDataReader
        from .quantize_onnx import quantize_static_int8
        t0 = time.perf_counter()
        log.info("[static_int8] Building calibration dataset …")

        cal_dir = self.cfg.output_dir / "calibration_data"
        cal_dir.mkdir(parents=True, exist_ok=True)

        reader = DataBoltCalibrationDataReader(
            tokenizer_dir=self.cfg.tokenizer_dir,
            seq_len=self.cfg.calibration_seq_len,
            prompts=None,  # use built-in DataBolt prompts
        )

        out_path = self.cfg.output_dir / "static_int8" / "model_static_int8.onnx"
        log.info("[static_int8] Starting static INT8 quantization")
        result = quantize_static_int8(
            self.cfg.source_onnx,
            out_path,
            calibration_data_reader=reader,
            weight_type=self.cfg.int8_weight_type,
            per_channel=self.cfg.int8_per_channel,
        )
        self._quant_results.append(result)
        log.info("[static_int8] %s", result)
        return StepOutcome(
            step=PipelineStep.STATIC_INT8,
            success=result.success,
            duration_s=time.perf_counter() - t0,
            message=str(result),
            data=result.__dict__ if result.success else {"error": result.error},
        )

    def _step_benchmark(self) -> StepOutcome:
        from .benchmark import benchmark_model
        t0 = time.perf_counter()
        log.info("[benchmark] Running latency benchmarks")

        models_to_bench: dict[str, Path] = {}

        # Always include source (fp32)
        if self.cfg.source_onnx.exists():
            models_to_bench["fp32"] = self.cfg.source_onnx

        # Include successfully quantized variants
        for qr in self._quant_results:
            if qr.success and qr.output_path.exists():
                models_to_bench[qr.variant] = qr.output_path

        results = []
        for variant, path in models_to_bench.items():
            log.info("[benchmark] %s …", variant)
            br = benchmark_model(
                path,
                variant=variant,
                n_runs=self.cfg.benchmark_n_runs,
                seq_len=self.cfg.benchmark_seq_len,
            )
            self._bench_results.append(br)
            results.append(str(br))
            log.info("[benchmark] %s", br)

        return StepOutcome(
            step=PipelineStep.BENCHMARK,
            success=True,
            duration_s=time.perf_counter() - t0,
            message=f"Benchmarked {len(results)} model(s)",
            data={"variants": list(models_to_bench.keys())},
        )

    def _build_report(self) -> "PipelineReport":
        from .report import PipelineReport
        return PipelineReport(
            config=self.cfg,
            outcomes=self.outcomes,
            quant_results=self._quant_results,
            bench_results=self._bench_results,
        )

    def _step_save_report(self, report: "PipelineReport") -> None:
        from .report import generate_report
        try:
            generate_report(report, self.cfg.output_dir)
        except Exception as exc:
            log.warning("Could not save report: %s", exc)
