"""ONNX Runtime inference benchmarking for DataBolt Edge.

Measures latency and throughput for ONNX models using ONNX Runtime's
CPUExecutionProvider (or CUDAExecutionProvider when a GPU is available).

Typical use::

    result = benchmark_model("/path/to/model.onnx", n_runs=50)
    print(result)
    # [fp32] mean=42.3 ms  p50=41.8 ms  p95=45.1 ms  tps=23.6

Models without real weights can be benchmarked on synthetic input tensors —
useful for CI validation of the pipeline without GPU hardware.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_WARMUP_RUNS = 5
_DEFAULT_RUNS = 30
_DEFAULT_SEQ_LEN = 64   # shorter for fast CPU benchmarks
_DEFAULT_BATCH = 1
_DEFAULT_VOCAB_SIZE = 32_000   # Mistral vocabulary


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Collected latency statistics for one benchmark run."""
    model_path: Path
    variant: str                          # "fp32", "fp16", "dynamic_int8", etc.
    n_runs: int = 0
    provider: str = "CPUExecutionProvider"

    # latency in milliseconds
    latencies_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    # throughput
    tokens_per_second: float = 0.0      # output tokens generated per second
    seq_len: int = _DEFAULT_SEQ_LEN

    # model metadata
    model_size_mb: float = 0.0

    # error
    success: bool = True
    error: str = ""

    def __str__(self) -> str:
        if not self.success:
            return f"[{self.variant}] BENCHMARK FAILED — {self.error}"
        return (
            f"[{self.variant}]  "
            f"mean={self.mean_ms:.1f}ms  "
            f"p50={self.median_ms:.1f}ms  "
            f"p95={self.p95_ms:.1f}ms  "
            f"tps={self.tokens_per_second:.1f}  "
            f"size={self.model_size_mb:.1f}MB"
        )

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "provider": self.provider,
            "n_runs": self.n_runs,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "model_size_mb": round(self.model_size_mb, 2),
            "success": self.success,
            "error": self.error,
        }


# ── benchmark engine ──────────────────────────────────────────────────────────

class OnnxBenchmark:
    """Benchmarks a single ONNX model file using ONNX Runtime.

    Parameters
    ----------
    model_path:
        Path to the ONNX model file.
    variant:
        Human-readable label for the model variant (e.g. "fp16", "dynamic_int8").
    seq_len:
        Token sequence length used for synthetic input tensors.
    batch_size:
        Batch size for each inference call.
    n_warmup:
        Number of warm-up runs before timing begins.
    providers:
        ONNX Runtime execution providers in priority order.  Defaults to
        ["CUDAExecutionProvider", "CPUExecutionProvider"] — ONNX Runtime
        will silently fall back to CPU if CUDA is unavailable.
    """

    def __init__(
        self,
        model_path: str | Path,
        variant: str = "unknown",
        *,
        seq_len: int = _DEFAULT_SEQ_LEN,
        batch_size: int = _DEFAULT_BATCH,
        n_warmup: int = _WARMUP_RUNS,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.variant = variant
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_warmup = n_warmup
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = None

    def _get_session(self):
        if self._session is None:
            import onnxruntime as ort  # type: ignore[import]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3  # suppress verbose ORT output
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=self.providers,
            )
        return self._session

    def _build_inputs(self, session) -> dict[str, np.ndarray]:
        """Build synthetic input arrays matching the model's input specs."""
        inputs: dict[str, np.ndarray] = {}
        for inp in session.get_inputs():
            shape = list(inp.shape)
            # Replace dynamic dims (None / symbolic) with concrete values
            for i, dim in enumerate(shape):
                if not isinstance(dim, int) or dim <= 0:
                    if i == 0:
                        shape[i] = self.batch_size
                    elif i == 1:
                        shape[i] = self.seq_len
                    else:
                        shape[i] = 1

            dtype_map = {
                "tensor(int64)": np.int64,
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
                "tensor(bool)": np.bool_,
            }
            np_dtype = dtype_map.get(inp.type, np.float32)

            if np_dtype == np.int64:
                # token IDs / attention mask
                arr = np.ones(shape, dtype=np_dtype)
            else:
                arr = np.random.randn(*shape).astype(np_dtype)
            inputs[inp.name] = arr
        return inputs

    def run(self, n_runs: int = _DEFAULT_RUNS) -> BenchmarkResult:
        """Run the benchmark and return a :class:`BenchmarkResult`."""
        result = BenchmarkResult(
            model_path=self.model_path,
            variant=self.variant,
            seq_len=self.seq_len,
        )

        if not self.model_path.exists():
            result.success = False
            result.error = f"Model file not found: {self.model_path}"
            return result

        result.model_size_mb = self.model_path.stat().st_size / (1024 ** 2)

        try:
            session = self._get_session()
            result.provider = session.get_providers()[0]
            inputs = self._build_inputs(session)
            output_names = [o.name for o in session.get_outputs()]

            # warm-up
            log.info("Benchmarking %s (%d warm-up + %d timing runs)", self.variant, self.n_warmup, n_runs)
            for _ in range(self.n_warmup):
                session.run(output_names, inputs)

            # timing
            latencies: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                session.run(output_names, inputs)
                latencies.append((time.perf_counter() - t0) * 1000)

            result.n_runs = n_runs
            result.latencies_ms = latencies
            result.mean_ms = statistics.mean(latencies)
            result.median_ms = statistics.median(latencies)

            sorted_lat = sorted(latencies)
            result.p95_ms = sorted_lat[int(0.95 * len(sorted_lat))]
            result.p99_ms = sorted_lat[int(0.99 * len(sorted_lat))]
            result.min_ms = min(latencies)
            result.max_ms = max(latencies)

            # tokens/sec: seq_len tokens generated per forward pass
            result.tokens_per_second = (self.seq_len * 1000) / result.mean_ms if result.mean_ms else 0.0

            log.info("  %s", result)
        except Exception as exc:
            result.success = False
            result.error = str(exc)
            log.error("Benchmark failed for %s: %s", self.variant, exc)

        return result


# ── convenience function ──────────────────────────────────────────────────────

def benchmark_model(
    model_path: str | Path,
    variant: str | None = None,
    *,
    n_runs: int = _DEFAULT_RUNS,
    seq_len: int = _DEFAULT_SEQ_LEN,
    batch_size: int = _DEFAULT_BATCH,
) -> BenchmarkResult:
    """Benchmark a single ONNX model file.

    Parameters
    ----------
    model_path:
        Path to the ONNX model.
    variant:
        Human-readable label (inferred from filename if None).
    n_runs:
        Number of timed inference calls.
    seq_len:
        Token sequence length for synthetic inputs.
    batch_size:
        Batch size for synthetic inputs.

    Returns
    -------
    :class:`BenchmarkResult`
    """
    path = Path(model_path)
    if variant is None:
        # Infer from filename: "model_fp16.onnx" → "fp16"
        stem = path.stem
        for suffix in ("fp16", "dynamic_int8", "static_int8", "fp32"):
            if suffix in stem:
                variant = suffix
                break
        else:
            variant = stem

    bench = OnnxBenchmark(path, variant, seq_len=seq_len, batch_size=batch_size)
    return bench.run(n_runs=n_runs)


def compare_models(
    models: dict[str, str | Path],
    *,
    n_runs: int = _DEFAULT_RUNS,
    seq_len: int = _DEFAULT_SEQ_LEN,
) -> list[BenchmarkResult]:
    """Benchmark multiple ONNX models and return results sorted by mean latency.

    Parameters
    ----------
    models:
        Mapping of variant name → model path.
    n_runs:
        Timed runs per model.

    Returns
    -------
    List of :class:`BenchmarkResult` sorted fastest to slowest.
    """
    results: list[BenchmarkResult] = []
    for variant, path in models.items():
        r = benchmark_model(path, variant=variant, n_runs=n_runs, seq_len=seq_len)
        results.append(r)

    results.sort(key=lambda r: r.mean_ms if r.success else float("inf"))
    return results
