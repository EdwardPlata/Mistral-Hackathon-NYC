"""ONNX model quantization utilities for DataBolt Edge.

Supports three quantization strategies:

FP16 Conversion
---------------
Converts float32 weights to float16 using onnxconverter-common.  Reduces
model size by ~50 % with virtually no accuracy loss on modern GPUs.

Dynamic INT8
------------
Quantizes weights to INT8 at build time; activations are quantized to INT8
at runtime using computed per-operation statistics.  No calibration data is
required — works on CPU or GPU and is the recommended starting point.

Static INT8
-----------
Quantizes both weights and activations using fixed ranges derived from a
calibration dataset.  Produces the smallest and often fastest model but
requires calibration data representative of production inputs.

Hardware note
-------------
FP16 inference is only faster on GPUs with native FP16 support (Volta/Turing+).
On CPUs, dynamic INT8 is typically the fastest variant.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

if TYPE_CHECKING:
    from .calibrate import DataBoltCalibrationDataReader

log = logging.getLogger(__name__)


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QuantizationResult:
    """Outcome of a single quantization step."""
    variant: str                      # "fp16" | "dynamic_int8" | "static_int8"
    output_path: Path
    success: bool
    source_size_mb: float = 0.0
    output_size_mb: float = 0.0
    compression_ratio: float = 0.0
    duration_s: float = 0.0
    node_counts: dict[str, int] = field(default_factory=dict)
    error: str = ""

    def __str__(self) -> str:
        if not self.success:
            return f"[{self.variant}] FAILED — {self.error}"
        return (
            f"[{self.variant}] {self.source_size_mb:.1f} MB → "
            f"{self.output_size_mb:.1f} MB  "
            f"(×{self.compression_ratio:.2f} smaller)  "
            f"{self.duration_s:.1f}s"
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def _count_node_types(model: onnx.ModelProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _validate_onnx(path: Path) -> None:
    """Raise if the ONNX model at *path* fails the checker."""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


# ── FP16 conversion ───────────────────────────────────────────────────────────

def convert_to_fp16(
    source: str | Path,
    output: str | Path,
    *,
    keep_io_types: bool = True,
    validate: bool = True,
) -> QuantizationResult:
    """Convert a float32 ONNX model to float16.

    Parameters
    ----------
    source:
        Path to the source float32 ONNX file.
    output:
        Path where the FP16 model will be saved.
    keep_io_types:
        If True the model's input/output tensors remain float32 — only
        internal weights and operations are converted.  This avoids the need
        to cast inputs at inference time.
    validate:
        Run onnx.checker after conversion.
    """
    from onnxconverter_common import float16  # type: ignore[import]

    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    src_size = 0.0

    try:
        if not source.exists():
            return QuantizationResult(
                variant="fp16",
                output_path=output,
                success=False,
                duration_s=time.perf_counter() - t0,
                error=f"Source ONNX not found: {source}",
            )
        log.info("FP16 conversion: loading %s", source)
        src_size = _file_size_mb(source)
        model_f32 = onnx.load(str(source))
        src_nodes = _count_node_types(model_f32)

        log.info("Converting weights to FP16 …")
        model_f16 = float16.convert_float_to_float16(
            model_f32, keep_io_types=keep_io_types
        )

        onnx.save(model_f16, str(output))
        out_size = _file_size_mb(output)

        if validate:
            _validate_onnx(output)

        duration = time.perf_counter() - t0
        out_nodes = _count_node_types(model_f16)
        log.info("FP16 done in %.1fs  %.1f→%.1f MB", duration, src_size, out_size)

        return QuantizationResult(
            variant="fp16",
            output_path=output,
            success=True,
            source_size_mb=src_size,
            output_size_mb=out_size,
            compression_ratio=src_size / out_size if out_size else 0.0,
            duration_s=duration,
            node_counts=out_nodes,
        )
    except Exception as exc:
        log.error("FP16 conversion failed: %s", exc)
        return QuantizationResult(
            variant="fp16",
            output_path=output,
            success=False,
            source_size_mb=src_size,
            duration_s=time.perf_counter() - t0,
            error=str(exc),
        )


# ── Dynamic INT8 ──────────────────────────────────────────────────────────────

def quantize_dynamic_int8(
    source: str | Path,
    output: str | Path,
    *,
    weight_type: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
    validate: bool = True,
) -> QuantizationResult:
    """Quantize model weights to INT8 dynamically (no calibration required).

    Dynamic quantization computes activation quantization parameters at
    runtime, so no calibration data is needed.  This is the fastest path to
    a quantized model and works well for Transformer-based LLMs.

    Parameters
    ----------
    source:
        Float32 ONNX model produced by export_to_onnx.py.
    output:
        Save path for the INT8 model.
    weight_type:
        "QInt8" (default, supported everywhere) or "QUInt8".
    per_channel:
        Enable per-channel quantization for Conv/MatMul nodes.  More accurate
        but slightly more overhead.
    reduce_range:
        Use 7-bit INT8 range to avoid VPMADDUBSW overflow on some CPUs.
    validate:
        Run onnx.checker after quantization.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import]

    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    src_size = 0.0

    try:
        if not source.exists():
            return QuantizationResult(
                variant="dynamic_int8",
                output_path=output,
                success=False,
                duration_s=time.perf_counter() - t0,
                error=f"Source ONNX not found: {source}",
            )
        src_size = _file_size_mb(source)
        log.info("Dynamic INT8: quantizing %s", source)
        wtype = QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8

        quantize_dynamic(
            model_input=str(source),
            model_output=str(output),
            weight_type=wtype,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )

        out_size = _file_size_mb(output)
        if validate:
            _validate_onnx(output)

        duration = time.perf_counter() - t0
        model = onnx.load(str(output))
        out_nodes = _count_node_types(model)
        log.info("Dynamic INT8 done in %.1fs  %.1f→%.1f MB", duration, src_size, out_size)

        return QuantizationResult(
            variant="dynamic_int8",
            output_path=output,
            success=True,
            source_size_mb=src_size,
            output_size_mb=out_size,
            compression_ratio=src_size / out_size if out_size else 0.0,
            duration_s=duration,
            node_counts=out_nodes,
        )
    except Exception as exc:
        log.error("Dynamic INT8 failed: %s", exc)
        return QuantizationResult(
            variant="dynamic_int8",
            output_path=output,
            success=False,
            source_size_mb=src_size,
            duration_s=time.perf_counter() - t0,
            error=str(exc),
        )


# ── Static INT8 ───────────────────────────────────────────────────────────────

def quantize_static_int8(
    source: str | Path,
    output: str | Path,
    calibration_data_reader: "DataBoltCalibrationDataReader",
    *,
    weight_type: str = "QInt8",
    activation_type: str = "QInt8",
    calibrate_method: str = "MinMax",
    per_channel: bool = False,
    validate: bool = True,
) -> QuantizationResult:
    """Quantize model weights and activations to INT8 using calibration data.

    Static quantization produces fixed quantization ranges derived from a
    representative calibration dataset.  This is typically more accurate than
    dynamic quantization for production deployments.

    Parameters
    ----------
    source:
        Float32 ONNX model produced by export_to_onnx.py.
    output:
        Save path for the quantized model.
    calibration_data_reader:
        A :class:`DataBoltCalibrationDataReader` providing input batches.
    weight_type / activation_type:
        "QInt8" or "QUInt8".
    calibrate_method:
        "MinMax" (default, fast) or "Entropy" (more accurate, slower).
    per_channel:
        Per-channel quantization for Conv/MatMul (more accurate, slightly
        slower).
    validate:
        Run onnx.checker after quantization.
    """
    from onnxruntime.quantization import (  # type: ignore[import]
        CalibrationMethod,
        QuantType,
        quantize_static,
    )

    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    src_size = 0.0

    try:
        if not source.exists():
            return QuantizationResult(
                variant="static_int8",
                output_path=output,
                success=False,
                duration_s=time.perf_counter() - t0,
                error=f"Source ONNX not found: {source}",
            )
        src_size = _file_size_mb(source)
        log.info("Static INT8: calibrating + quantizing %s", source)
        wtype = QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8
        atype = QuantType.QInt8 if activation_type == "QInt8" else QuantType.QUInt8
        cal_method = (
            CalibrationMethod.MinMax
            if calibrate_method == "MinMax"
            else CalibrationMethod.Entropy
        )

        # Static quantization needs a pre-processed (fp32 optimised) model
        with tempfile.TemporaryDirectory() as tmp:
            pre_path = Path(tmp) / "pre.onnx"
            _preprocess_for_static_quant(source, pre_path)

            quantize_static(
                model_input=str(pre_path),
                model_output=str(output),
                calibration_data_reader=calibration_data_reader,
                weight_type=wtype,
                activation_type=atype,
                calibrate_method=cal_method,
                per_channel=per_channel,
            )

        out_size = _file_size_mb(output)
        if validate:
            _validate_onnx(output)

        duration = time.perf_counter() - t0
        model = onnx.load(str(output))
        out_nodes = _count_node_types(model)
        log.info("Static INT8 done in %.1fs  %.1f→%.1f MB", duration, src_size, out_size)

        return QuantizationResult(
            variant="static_int8",
            output_path=output,
            success=True,
            source_size_mb=src_size,
            output_size_mb=out_size,
            compression_ratio=src_size / out_size if out_size else 0.0,
            duration_s=duration,
            node_counts=out_nodes,
        )
    except Exception as exc:
        log.error("Static INT8 failed: %s", exc)
        return QuantizationResult(
            variant="static_int8",
            output_path=output,
            success=False,
            source_size_mb=src_size,
            duration_s=time.perf_counter() - t0,
            error=str(exc),
        )


def _preprocess_for_static_quant(source: Path, output: Path) -> None:
    """Run ONNX Runtime's model optimizer before static quantization."""
    try:
        from onnxruntime.quantization.preprocess import quant_pre_process  # type: ignore[import]
        quant_pre_process(str(source), str(output), skip_symbolic_shape=True)
    except Exception:
        # Fallback: just copy — the quantizer will still work
        shutil.copy2(source, output)
