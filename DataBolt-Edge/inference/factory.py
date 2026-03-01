"""Backend factory — selects the right inference backend automatically.

Selection order (controlled by INFERENCE_BACKEND env var):
    auto       — try local_trt first, fall back to nvidia_api (default)
    local_trt  — force local TensorRT engine (fails if not available)
    nvidia_api — force NVIDIA hosted API

Environment variables:
    INFERENCE_BACKEND   — 'auto' | 'local_trt' | 'nvidia_api'  (default: auto)
    TRT_ENGINE_PATH     — required for local_trt backend
    TRT_TOKENIZER_DIR   — required for local_trt backend
    NVIDIA_BEARER_TOKEN — required for nvidia_api backend
"""

from __future__ import annotations

import os

from .base import InferenceBackend
from .local_trt_backend import LocalTRTBackend
from .nvidia_api_backend import NvidiaAPIBackend


def get_backend(backend: str | None = None) -> InferenceBackend:
    """Return the appropriate InferenceBackend instance.

    Args:
        backend: 'auto', 'local_trt', or 'nvidia_api'.
                 Defaults to INFERENCE_BACKEND env var, then 'auto'.
    """
    mode = (backend or os.environ.get("INFERENCE_BACKEND", "auto")).lower()

    if mode == "local_trt":
        b = LocalTRTBackend()
        if not b.is_available():
            raise RuntimeError(
                "local_trt backend requested but not available. "
                "Check TRT_ENGINE_PATH, TRT_TOKENIZER_DIR, and TensorRT installation."
            )
        return b

    if mode == "nvidia_api":
        b = NvidiaAPIBackend()
        if not b.is_available():
            raise RuntimeError(
                "nvidia_api backend requested but NVIDIA_BEARER_TOKEN is not set."
            )
        return b

    # auto — prefer local if engine exists, otherwise fall back to API
    local = LocalTRTBackend()
    if local.is_available():
        return local

    api = NvidiaAPIBackend()
    if api.is_available():
        return api

    raise RuntimeError(
        "No inference backend is available. Either:\n"
        "  - Set NVIDIA_BEARER_TOKEN for the hosted API backend, or\n"
        "  - Set TRT_ENGINE_PATH + TRT_TOKENIZER_DIR for the local TRT backend."
    )
