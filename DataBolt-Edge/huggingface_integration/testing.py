from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .client import HuggingFaceClient
from .config import HuggingFaceConfig
from .errors import HuggingFaceError, HFRequestError


@dataclass(slots=True)
class HFProbeResult:
    success: bool
    latency_ms: float
    username: str | None = None
    email: str | None = None
    plan: str | None = None
    status_code: int | None = None
    error: str | None = None


def run_probe(
    token: str | None = None,
    config: HuggingFaceConfig | None = None,
) -> HFProbeResult:
    """Verify the HF token by calling whoami and extracting user details."""
    start = perf_counter()
    client = HuggingFaceClient(config=config)

    try:
        info = client.whoami(token=token)
        latency_ms = (perf_counter() - start) * 1000
        return HFProbeResult(
            success=True,
            latency_ms=latency_ms,
            username=info.get("name") or info.get("fullname"),
            email=info.get("email"),
            plan=info.get("plan", {}).get("name") if isinstance(info.get("plan"), dict) else None,
        )
    except HFRequestError as exc:
        latency_ms = (perf_counter() - start) * 1000
        return HFProbeResult(
            success=False,
            latency_ms=latency_ms,
            status_code=exc.status_code,
            error=str(exc),
        )
    except HuggingFaceError as exc:
        latency_ms = (perf_counter() - start) * 1000
        return HFProbeResult(success=False, latency_ms=latency_ms, error=str(exc))
