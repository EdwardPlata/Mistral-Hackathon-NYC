"""Retry logic for Mistral API calls with exponential backoff.

This module provides a decorator that wraps Mistral API calls with automatic
retry logic for transient errors (rate limits, network issues, server errors).
Non-retryable errors (auth, validation) fail immediately.

Configuration via environment variables:
    MISTRAL_RETRY_MAX_ATTEMPTS (default: 3)
    MISTRAL_RETRY_MIN_WAIT (default: 1 second)
    MISTRAL_RETRY_MAX_WAIT (default: 60 seconds)
"""

import logging
import os
import sys
from typing import Any, Callable

from tenacity import (
    after_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Configure logging for retry attempts
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


# Retry configuration from environment variables
MAX_ATTEMPTS = int(os.getenv("MISTRAL_RETRY_MAX_ATTEMPTS", "3"))
MIN_WAIT = int(os.getenv("MISTRAL_RETRY_MIN_WAIT", "1"))
MAX_WAIT = int(os.getenv("MISTRAL_RETRY_MAX_WAIT", "60"))


def is_retryable_error(exception: Exception) -> bool:
    """Determine if an exception should trigger a retry.

    Retryable errors:
        - Rate limits (HTTP 429)
        - Server errors (HTTP 5xx)
        - Network/connection errors
        - Timeout errors

    Non-retryable errors (fail immediately):
        - Auth errors (HTTP 401, 403)
        - Validation errors (HTTP 400, 422)
        - Other client errors (HTTP 4xx except 429)

    Args:
        exception: The exception to check.

    Returns:
        True if the error is retryable, False otherwise.
    """
    # Check if it's a Mistral API exception with status code
    if hasattr(exception, "status_code"):
        status_code = exception.status_code
        # Retry on rate limits and server errors
        if status_code == 429:  # Rate limit
            logger.info(f"Rate limit hit (429), will retry with backoff")
            return True
        if 500 <= status_code < 600:  # Server errors
            logger.info(f"Server error ({status_code}), will retry")
            return True
        # Don't retry on client errors (auth, validation, etc.)
        if 400 <= status_code < 500:
            logger.warning(f"Client error ({status_code}), will not retry")
            return False

    # Retry on connection and timeout errors
    if isinstance(exception, (ConnectionError, TimeoutError, OSError)):
        logger.info(f"Network error ({type(exception).__name__}), will retry")
        return True

    # Don't retry on other errors (programming errors, etc.)
    logger.warning(f"Non-retryable error ({type(exception).__name__}), failing immediately")
    return False


def retry_mistral_call(func: Callable) -> Callable:
    """Decorator that adds retry logic with exponential backoff to Mistral API calls.

    Configuration is controlled via environment variables:
        - MISTRAL_RETRY_MAX_ATTEMPTS: Maximum number of retry attempts (default: 3)
        - MISTRAL_RETRY_MIN_WAIT: Minimum wait time between retries in seconds (default: 1)
        - MISTRAL_RETRY_MAX_WAIT: Maximum wait time between retries in seconds (default: 60)

    Retry strategy:
        - Exponential backoff with jitter
        - Retries: rate limits (429), server errors (5xx), network errors
        - Fails immediately: auth errors (401, 403), validation errors (400, 422)

    Example usage:
        @retry_mistral_call
        def call_mistral():
            return client.chat.complete(...)

    Args:
        func: The function to wrap with retry logic.

    Returns:
        The wrapped function with retry behavior.
    """
    return retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=MIN_WAIT, max=MAX_WAIT),
        retry=retry_if_exception(is_retryable_error),
        after=after_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def get_retry_config() -> dict[str, Any]:
    """Return the current retry configuration.

    Useful for debugging and displaying in UI/logs.

    Returns:
        Dictionary with retry configuration parameters.
    """
    return {
        "max_attempts": MAX_ATTEMPTS,
        "min_wait_seconds": MIN_WAIT,
        "max_wait_seconds": MAX_WAIT,
    }
