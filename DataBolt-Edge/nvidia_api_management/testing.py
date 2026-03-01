from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .client import NvidiaAPIClient
from .config import NvidiaAPIConfig
from .errors import NvidiaAPIError, RequestFailedError


@dataclass(slots=True)
class ProbeResult:
    success: bool
    latency_ms: float
    status_code: int | None = None
    response: dict[str, Any] | None = None
    error: str | None = None


def run_probe(
    content: str = "What is the capital of France?",
    model: str | None = None,
    stream: bool = False,
    api_key: str | None = None,
    config: NvidiaAPIConfig | None = None,
) -> ProbeResult:
    start = perf_counter()
    client = NvidiaAPIClient(config=config)

    try:
        response = client.chat_completion(
            content=content,
            model=model,
            stream=stream,
            api_key=api_key,
        )
        latency_ms = (perf_counter() - start) * 1000
        return ProbeResult(success=True, latency_ms=latency_ms, response=response)
    except RequestFailedError as exc:
        latency_ms = (perf_counter() - start) * 1000
        return ProbeResult(
            success=False,
            latency_ms=latency_ms,
            status_code=exc.status_code,
            error=str(exc),
        )
    except NvidiaAPIError as exc:
        latency_ms = (perf_counter() - start) * 1000
        return ProbeResult(success=False, latency_ms=latency_ms, error=str(exc))
