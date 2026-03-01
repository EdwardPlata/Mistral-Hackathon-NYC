"""Inference telemetry helpers for DataBolt Edge.

Wraps inference backend calls to capture latency, token counts,
and optionally GPU metrics (VRAM / utilization) via pynvml.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetryRecord:
    backend: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    gpu_stats: dict[str, Any] = field(default_factory=dict)


def _collect_gpu_stats() -> dict[str, Any]:
    """Best-effort VRAM and GPU utilization via pynvml."""
    try:
        import pynvml  # type: ignore[import]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "vram_used_mb": mem.used // (1024 * 1024),
            "vram_total_mb": mem.total // (1024 * 1024),
            "gpu_utilization_pct": util.gpu,
        }
    except Exception:
        return {}


def run_with_telemetry(backend, request) -> tuple[Any, TelemetryRecord]:
    """Call backend.generate(request) and wrap result with a TelemetryRecord.

    Args:
        backend: InferenceBackend instance.
        request: InferenceRequest instance.

    Returns:
        (InferenceResponse, TelemetryRecord)
    """
    t0 = time.perf_counter()
    response = backend.generate(request)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    prompt_tokens = response.prompt_tokens or 0
    completion_tokens = response.completion_tokens or 0
    total_tokens = prompt_tokens + completion_tokens

    tokens_per_second = (
        (completion_tokens / (latency_ms / 1000)) if latency_ms > 0 and completion_tokens > 0 else 0.0
    )

    telemetry = TelemetryRecord(
        backend=response.backend,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tokens_per_second=round(tokens_per_second, 1),
        gpu_stats=_collect_gpu_stats(),
    )
    return response, telemetry
