from __future__ import annotations

import os

from .config import HuggingFaceConfig
from .errors import MissingTokenError


def resolve_token(config: HuggingFaceConfig, token: str | None = None) -> str:
    """Return an HF token, checking the caller arg then each env var in order.

    huggingface_hub also reads HF_TOKEN automatically, but we resolve it
    explicitly so every call is auditable and testable.
    """
    if token:
        return token
    for var in config.token_env_vars:
        value = os.getenv(var)
        if value:
            return value
    raise MissingTokenError(
        f"No HuggingFace token found. Set one of: {', '.join(config.token_env_vars)}"
    )
