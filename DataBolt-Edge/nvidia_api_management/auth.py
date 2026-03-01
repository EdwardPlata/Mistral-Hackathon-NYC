from __future__ import annotations

import os

from .config import NvidiaAPIConfig
from .errors import MissingCredentialError


def resolve_api_key(config: NvidiaAPIConfig, api_key: str | None = None) -> str:
    token = api_key or os.getenv(config.api_key_env_var)
    if not token:
        raise MissingCredentialError(
            f"Missing NVIDIA API token. Set {config.api_key_env_var} in your environment."
        )
    return token


def build_headers(config: NvidiaAPIConfig, api_key: str | None = None) -> dict[str, str]:
    token = resolve_api_key(config, api_key)

    headers: dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Accept": config.accept_header,
        "Content-Type": "application/json",
        "User-Agent": config.user_agent,
    }
    headers.update(config.default_headers)
    return headers
