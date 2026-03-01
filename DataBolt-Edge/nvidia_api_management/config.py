from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_status_codes(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default

    status_codes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            status_codes.append(int(item))
        except ValueError:
            continue

    return tuple(status_codes) if status_codes else default


def _parse_headers(value: str | None) -> dict[str, str]:
    if not value:
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    headers: dict[str, str] = {}
    for key, header_value in parsed.items():
        if isinstance(key, str) and isinstance(header_value, str):
            headers[key] = header_value
    return headers


@dataclass(slots=True)
class NvidiaAPIConfig:
    base_url: str = "https://integrate.api.nvidia.com"
    chat_completions_path: str = "/v1/chat/completions"
    api_key_env_var: str = "NVIDIA_BEARER_TOKEN"
    default_model: str = "mistralai/mistral-large-3-675b-instruct-2512"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    accept_header: str = "application/json"
    user_agent: str = "nvidia-api-management/0.1.0"
    default_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "NvidiaAPIConfig":
        return cls(
            base_url=os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com"),
            chat_completions_path=os.getenv("NVIDIA_API_CHAT_PATH", "/v1/chat/completions"),
            api_key_env_var=os.getenv("NVIDIA_API_KEY_ENV_VAR", "NVIDIA_BEARER_TOKEN"),
            default_model=os.getenv(
                "NVIDIA_API_MODEL", "mistralai/mistral-large-3-675b-instruct-2512"
            ),
            timeout_seconds=_as_float(os.getenv("NVIDIA_API_TIMEOUT_SECONDS"), 30.0),
            max_retries=_as_int(os.getenv("NVIDIA_API_MAX_RETRIES"), 3),
            backoff_factor=_as_float(os.getenv("NVIDIA_API_BACKOFF_FACTOR"), 0.5),
            retry_status_codes=_parse_status_codes(
                os.getenv("NVIDIA_API_RETRY_STATUS_CODES"),
                (429, 500, 502, 503, 504),
            ),
            accept_header=os.getenv("NVIDIA_API_ACCEPT", "application/json"),
            user_agent=os.getenv("NVIDIA_API_USER_AGENT", "nvidia-api-management/0.1.0"),
            default_headers=_parse_headers(os.getenv("NVIDIA_API_EXTRA_HEADERS")),
        )
