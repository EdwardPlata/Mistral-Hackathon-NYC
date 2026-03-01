from __future__ import annotations

import logging
from typing import Mapping


SENSITIVE_HEADER_NAMES = {"authorization", "x-api-key", "api-key"}


def get_logger(name: str = "nvidia_api_management", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def redact_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}

    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADER_NAMES:
            sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = value
    return sanitized
