"""NVIDIA API inference backend — uses nvidia_api_management under the hood.

This is the default backend for the hackathon demo and any environment
without a local TensorRT engine. Requires NVIDIA_BEARER_TOKEN to be set.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from nvidia_api_management.auth import resolve_api_key
from nvidia_api_management.client import NvidiaAPIClient
from nvidia_api_management.config import NvidiaAPIConfig
from nvidia_api_management.errors import NvidiaAPIError

from .base import InferenceBackend, InferenceRequest, InferenceResponse


class NvidiaAPIBackend(InferenceBackend):
    """Inference via NVIDIA's hosted Mistral endpoint (integrate.api.nvidia.com)."""

    def __init__(
        self,
        config: NvidiaAPIConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self._config = config or NvidiaAPIConfig.from_env()
        self._api_key = api_key
        self._client = NvidiaAPIClient(config=self._config)

    @property
    def name(self) -> str:
        return "nvidia_api"

    def is_available(self) -> bool:
        try:
            resolve_api_key(self._config, self._api_key)
            return True
        except Exception:
            return False

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        t0 = perf_counter()
        try:
            raw = self._client.chat_completion(
                content=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=False,
                api_key=self._api_key,
            )
        except NvidiaAPIError as exc:
            raise RuntimeError(f"NVIDIA API inference failed: {exc}") from exc

        latency_ms = (perf_counter() - t0) * 1000

        choices = raw.get("choices", [])
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = raw.get("usage", {})

        return InferenceResponse(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            backend=self.name,
            raw=raw,
        )

    def health_check(self) -> dict:
        from nvidia_api_management.testing import run_probe
        result = run_probe(content="ping", api_key=self._api_key)
        return {
            "backend": self.name,
            "available": result.success,
            "latency_ms": round(result.latency_ms, 1),
            "error": result.error,
        }
