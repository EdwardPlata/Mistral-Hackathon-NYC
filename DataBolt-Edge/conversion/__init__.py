"""DataBolt Edge — Model Conversion & Quantization Pipeline.

Transforms a float32 ONNX model (produced by scripts/export_to_onnx.py) into
several optimised variants for edge deployment:

  FP16      — half-precision weights, ~2× smaller, minimal quality loss
  DYN_INT8  — runtime dynamic quantization, no calibration required
  STAT_INT8 — static INT8 with calibration data, best INT8 accuracy

Usage::

    from conversion import QuantizationPipeline, PipelineConfig

    cfg = PipelineConfig(source_onnx="/path/to/model.onnx", output_dir="/path/to/out")
    pipeline = QuantizationPipeline(cfg)
    report = pipeline.run()
    print(report.summary_table())
"""

from .quantize_onnx import (
    QuantizationResult,
    convert_to_fp16,
    quantize_dynamic_int8,
    quantize_static_int8,
)
from .calibrate import DataBoltCalibrationDataReader, build_calibration_dataset
from .benchmark import BenchmarkResult, OnnxBenchmark, benchmark_model
from .pipeline import PipelineConfig, PipelineStep, QuantizationPipeline
from .report import PipelineReport, generate_report

__all__ = [
    # quantize_onnx
    "QuantizationResult",
    "convert_to_fp16",
    "quantize_dynamic_int8",
    "quantize_static_int8",
    # calibrate
    "DataBoltCalibrationDataReader",
    "build_calibration_dataset",
    # benchmark
    "BenchmarkResult",
    "OnnxBenchmark",
    "benchmark_model",
    # pipeline
    "PipelineConfig",
    "PipelineStep",
    "QuantizationPipeline",
    # report
    "PipelineReport",
    "generate_report",
]
