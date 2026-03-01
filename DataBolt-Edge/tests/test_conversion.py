"""Unit tests for DataBolt Edge model conversion & quantization pipeline.

Uses a minimal synthetic ONNX model (MatMul+Add) to test the full
pipeline without requiring GPU hardware or real model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure DataBolt-Edge root is importable
EDGE_ROOT = Path(__file__).resolve().parents[1]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest

from conversion.benchmark import BenchmarkResult, OnnxBenchmark, benchmark_model, compare_models
from conversion.calibrate import DataBoltCalibrationDataReader, build_calibration_dataset
from conversion.pipeline import PipelineConfig, QuantizationPipeline
from conversion.quantize_onnx import QuantizationResult, convert_to_fp16, quantize_dynamic_int8
from conversion.report import PipelineReport, generate_report


# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

def _make_tiny_onnx(path: Path, input_size: int = 8) -> None:
    """Create a minimal float32 ONNX model: Y = X @ W + B.

    Inputs:  X  (float32, shape [1, input_size])
    Outputs: Y  (float32, shape [1, input_size])
    """
    rng = np.random.default_rng(42)
    W = rng.standard_normal((input_size, input_size)).astype(np.float32)
    B = rng.standard_normal(input_size).astype(np.float32)

    W_init = onh.from_array(W, name="W")
    B_init = onh.from_array(B, name="B")

    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, input_size])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, input_size])

    matmul = oh.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
    add = oh.make_node("Add", inputs=["Z", "B"], outputs=["Y"])

    graph = oh.make_graph(
        nodes=[matmul, add],
        name="tiny",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, B_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def tiny_onnx(tmp_path: Path) -> Path:
    p = tmp_path / "model_fp32.onnx"
    _make_tiny_onnx(p)
    return p


# ════════════════════════════════════════════════════════════════════════════
# quantize_onnx — FP16 conversion
# ════════════════════════════════════════════════════════════════════════════

class TestFP16Conversion:
    def test_success(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out)
        assert result.success, result.error
        assert out.exists()

    def test_output_size_smaller_or_equal(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out)
        assert result.success
        # FP16 should be <= source (tiny model may be similar size due to overhead)
        assert result.output_size_mb <= result.source_size_mb * 1.2

    def test_source_size_recorded(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out)
        assert result.source_size_mb > 0

    def test_compression_ratio_positive(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out)
        assert result.success
        assert result.compression_ratio > 0

    def test_duration_recorded(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out)
        assert result.duration_s > 0

    def test_variant_label(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out)
        assert result.variant == "fp16"

    def test_onnx_valid_after_conversion(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out, validate=True)
        assert result.success

    def test_missing_source_returns_failure(self, tmp_dir):
        out = tmp_dir / "fp16.onnx"
        result = convert_to_fp16(tmp_dir / "nonexistent.onnx", out)
        assert not result.success
        assert result.error

    def test_result_str_contains_variant(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        result = convert_to_fp16(tiny_onnx, out)
        assert "fp16" in str(result)

    def test_return_type(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "fp16" / "model_fp16.onnx"
        assert isinstance(convert_to_fp16(tiny_onnx, out), QuantizationResult)


# ════════════════════════════════════════════════════════════════════════════
# quantize_onnx — Dynamic INT8
# ════════════════════════════════════════════════════════════════════════════

class TestDynamicInt8:
    def test_success(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "dyn_int8.onnx"
        result = quantize_dynamic_int8(tiny_onnx, out)
        assert result.success, result.error
        assert out.exists()

    def test_variant_label(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "dyn_int8.onnx"
        result = quantize_dynamic_int8(tiny_onnx, out)
        assert result.variant == "dynamic_int8"

    def test_source_size_recorded(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "dyn_int8.onnx"
        result = quantize_dynamic_int8(tiny_onnx, out)
        assert result.source_size_mb > 0

    def test_duration_positive(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "dyn_int8.onnx"
        result = quantize_dynamic_int8(tiny_onnx, out)
        assert result.duration_s > 0

    def test_onnx_valid_after_quantization(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "dyn_int8.onnx"
        result = quantize_dynamic_int8(tiny_onnx, out, validate=True)
        assert result.success

    def test_missing_source_failure(self, tmp_dir):
        result = quantize_dynamic_int8(tmp_dir / "missing.onnx", tmp_dir / "out.onnx")
        assert not result.success

    def test_node_counts_populated(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "dyn_int8.onnx"
        result = quantize_dynamic_int8(tiny_onnx, out)
        assert result.success
        assert isinstance(result.node_counts, dict)


# ════════════════════════════════════════════════════════════════════════════
# calibrate
# ════════════════════════════════════════════════════════════════════════════

class TestCalibrate:
    def test_default_reader_returns_data(self):
        reader = DataBoltCalibrationDataReader(tokenizer_dir=None)
        sample = reader.get_next()
        assert sample is not None
        assert "input_ids" in sample

    def test_input_ids_shape(self):
        reader = DataBoltCalibrationDataReader(seq_len=64)
        sample = reader.get_next()
        assert sample["input_ids"].shape[-1] == 64

    def test_attention_mask_present(self):
        reader = DataBoltCalibrationDataReader()
        sample = reader.get_next()
        assert "attention_mask" in sample

    def test_returns_none_when_exhausted(self):
        reader = DataBoltCalibrationDataReader(prompts=["one prompt"])
        reader.get_next()
        assert reader.get_next() is None

    def test_rewind_resets_iterator(self):
        reader = DataBoltCalibrationDataReader(prompts=["p1", "p2"])
        reader.get_next()
        reader.get_next()
        assert reader.get_next() is None
        reader.rewind()
        assert reader.get_next() is not None

    def test_custom_prompts_count(self):
        prompts = ["a", "b", "c"]
        reader = DataBoltCalibrationDataReader(prompts=prompts)
        count = 0
        while reader.get_next() is not None:
            count += 1
        assert count == 3

    def test_custom_input_names_filter(self):
        reader = DataBoltCalibrationDataReader(
            input_names=["input_ids"],
            prompts=["test"],
        )
        sample = reader.get_next()
        assert set(sample.keys()) == {"input_ids"}

    def test_dtype_int64(self):
        reader = DataBoltCalibrationDataReader(prompts=["hello"])
        sample = reader.get_next()
        assert sample["input_ids"].dtype == np.int64

    def test_build_calibration_dataset_saves_npz(self, tmp_dir):
        dataset = build_calibration_dataset(tmp_dir / "cal")
        assert len(dataset) > 0
        assert (tmp_dir / "cal" / "calibration_data.npz").exists()

    def test_built_dataset_has_correct_keys(self, tmp_dir):
        dataset = build_calibration_dataset(tmp_dir / "cal")
        for sample in dataset:
            assert "input_ids" in sample


# ════════════════════════════════════════════════════════════════════════════
# benchmark
# ════════════════════════════════════════════════════════════════════════════

class TestBenchmark:
    def test_benchmark_fp32(self, tiny_onnx):
        result = benchmark_model(tiny_onnx, variant="fp32", n_runs=5, seq_len=8)
        assert result.success, result.error
        assert result.mean_ms > 0

    def test_latency_stats_populated(self, tiny_onnx):
        result = benchmark_model(tiny_onnx, n_runs=10, seq_len=8)
        assert result.median_ms > 0
        assert result.p95_ms >= result.median_ms
        assert result.min_ms <= result.mean_ms

    def test_tokens_per_second_positive(self, tiny_onnx):
        result = benchmark_model(tiny_onnx, n_runs=5, seq_len=8)
        assert result.tokens_per_second > 0

    def test_model_size_mb_recorded(self, tiny_onnx):
        result = benchmark_model(tiny_onnx, n_runs=3, seq_len=8)
        assert result.model_size_mb > 0

    def test_missing_model_returns_failure(self, tmp_dir):
        result = benchmark_model(tmp_dir / "nonexistent.onnx", n_runs=1)
        assert not result.success
        assert "not found" in result.error.lower()

    def test_variant_inferred_from_filename(self, tiny_onnx, tmp_dir):
        fp16_path = tmp_dir / "model_fp16.onnx"
        import shutil
        shutil.copy(tiny_onnx, fp16_path)
        result = benchmark_model(fp16_path, n_runs=3, seq_len=8)
        assert result.variant == "fp16"

    def test_as_dict_keys(self, tiny_onnx):
        r = benchmark_model(tiny_onnx, n_runs=3, seq_len=8)
        d = r.as_dict()
        for key in ("variant", "mean_ms", "median_ms", "p95_ms", "tokens_per_second", "success"):
            assert key in d

    def test_compare_models_sorted(self, tiny_onnx, tmp_dir):
        import shutil
        m2 = tmp_dir / "model_b.onnx"
        shutil.copy(tiny_onnx, m2)
        results = compare_models({"fp32": tiny_onnx, "copy": m2}, n_runs=5, seq_len=8)
        # Should be a list sorted by mean_ms (fastest first)
        assert len(results) == 2
        latencies = [r.mean_ms for r in results if r.success]
        assert latencies == sorted(latencies)

    def test_return_type(self, tiny_onnx):
        assert isinstance(benchmark_model(tiny_onnx, n_runs=3, seq_len=8), BenchmarkResult)

    def test_str_output_non_empty(self, tiny_onnx):
        r = benchmark_model(tiny_onnx, n_runs=3, seq_len=8)
        assert len(str(r)) > 10


# ════════════════════════════════════════════════════════════════════════════
# pipeline (integration)
# ════════════════════════════════════════════════════════════════════════════

class TestPipeline:
    def test_full_pipeline_fp16_dyn_int8(self, tiny_onnx, tmp_dir):
        cfg = PipelineConfig(
            source_onnx=tiny_onnx,
            output_dir=tmp_dir / "out",
            run_fp16=True,
            run_dynamic_int8=True,
            run_static_int8=False,
            run_benchmark=True,
            benchmark_n_runs=5,
            benchmark_seq_len=8,
        )
        pipeline = QuantizationPipeline(cfg)
        report = pipeline.run()
        assert isinstance(report, PipelineReport)
        assert len(report.successful_variants) >= 1

    def test_output_files_created(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "out"
        cfg = PipelineConfig(
            source_onnx=tiny_onnx, output_dir=out,
            run_fp16=True, run_dynamic_int8=True,
            run_static_int8=False, run_benchmark=False,
        )
        QuantizationPipeline(cfg).run()
        assert (out / "fp16" / "model_fp16.onnx").exists()
        assert (out / "dynamic_int8" / "model_dynamic_int8.onnx").exists()

    def test_report_json_written(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "out"
        cfg = PipelineConfig(
            source_onnx=tiny_onnx, output_dir=out,
            run_fp16=True, run_dynamic_int8=False,
            run_static_int8=False, run_benchmark=False,
        )
        QuantizationPipeline(cfg).run()
        assert (out / "pipeline_report.json").exists()

    def test_report_markdown_written(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "out"
        cfg = PipelineConfig(
            source_onnx=tiny_onnx, output_dir=out,
            run_fp16=True, run_dynamic_int8=False,
            run_static_int8=False, run_benchmark=False,
        )
        QuantizationPipeline(cfg).run()
        assert (out / "pipeline_report.md").exists()

    def test_pipeline_with_missing_source(self, tmp_dir):
        cfg = PipelineConfig(
            source_onnx=tmp_dir / "missing.onnx",
            output_dir=tmp_dir / "out",
            run_fp16=True, run_dynamic_int8=True,
            run_static_int8=False, run_benchmark=False,
        )
        report = QuantizationPipeline(cfg).run()
        # validate step should fail; FP16 / INT8 steps should also fail gracefully
        assert isinstance(report, PipelineReport)

    def test_benchmark_step_runs(self, tiny_onnx, tmp_dir):
        out = tmp_dir / "out"
        cfg = PipelineConfig(
            source_onnx=tiny_onnx, output_dir=out,
            run_fp16=True, run_dynamic_int8=False,
            run_static_int8=False, run_benchmark=True,
            benchmark_n_runs=3, benchmark_seq_len=8,
        )
        report = QuantizationPipeline(cfg).run()
        assert len(report.bench_results) >= 1


# ════════════════════════════════════════════════════════════════════════════
# report
# ════════════════════════════════════════════════════════════════════════════

class TestReport:
    def _make_report(self, tiny_onnx, tmp_dir):
        cfg = PipelineConfig(
            source_onnx=tiny_onnx, output_dir=tmp_dir / "out",
            run_fp16=True, run_dynamic_int8=True,
            run_static_int8=False, run_benchmark=True,
            benchmark_n_runs=3, benchmark_seq_len=8,
        )
        pipeline = QuantizationPipeline(cfg)
        return pipeline.run()

    def test_summary_table_non_empty(self, tiny_onnx, tmp_dir):
        report = self._make_report(tiny_onnx, tmp_dir)
        table = report.summary_table()
        assert len(table) > 100
        assert "DataBolt Edge" in table

    def test_to_dict_structure(self, tiny_onnx, tmp_dir):
        report = self._make_report(tiny_onnx, tmp_dir)
        d = report.to_dict()
        assert "config" in d
        assert "steps" in d
        assert "quantization" in d
        assert "benchmarks" in d
        assert "summary" in d

    def test_to_markdown_contains_headers(self, tiny_onnx, tmp_dir):
        report = self._make_report(tiny_onnx, tmp_dir)
        md = report.to_markdown()
        assert "# DataBolt Edge" in md
        assert "## Pipeline Steps" in md

    def test_generate_report_writes_files(self, tiny_onnx, tmp_dir):
        report = self._make_report(tiny_onnx, tmp_dir)
        out = tmp_dir / "reports"
        paths = generate_report(report, out)
        assert paths["json"].exists()
        assert paths["markdown"].exists()

    def test_to_dataframe_shape(self, tiny_onnx, tmp_dir):
        report = self._make_report(tiny_onnx, tmp_dir)
        df = report.to_dataframe()
        assert len(df) >= 1
        assert "Variant" in df.columns

    def test_successful_variants_list(self, tiny_onnx, tmp_dir):
        report = self._make_report(tiny_onnx, tmp_dir)
        assert isinstance(report.successful_variants, list)
        assert len(report.successful_variants) >= 1
