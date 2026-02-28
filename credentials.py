"""
Centralized credential management for all APIs and services.

Supports loading credentials from:
1. Codespace secrets (environment variables set by GitHub Codespaces)
2. .env file (for local development)
3. Explicit parameter passing (for testing/overrides)

Usage:
    from credentials import load_credentials, validate_credentials

    # Load and validate all credentials
    creds = load_credentials()
    validate_credentials(creds, required=["MISTRAL_API_KEY", "NVIDIA_BEARER_TOKEN"])

    mistral_key = creds.get("MISTRAL_API_KEY")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class Credentials:
    """Container for all API credentials."""

    mistral_api_key: Optional[str] = None
    nvidia_bearer_token: Optional[str] = None
    databricks_pat: Optional[str] = None
    wandb_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    def get(self, key: str) -> Optional[str]:
        """Get credential by env variable name."""
        mapping = {
            "MISTRAL_API_KEY": self.mistral_api_key,
            "NVIDIA_BEARER_TOKEN": self.nvidia_bearer_token,
            "DATABRICKS_PAT": self.databricks_pat,
            "WANDB_API_KEY": self.wandb_api_key,
            "ELEVENLABS_API_KEY": self.elevenlabs_api_key,
            "HF_TOKEN": self.hf_token,
        }
        return mapping.get(key)

    def to_dict(self) -> dict[str, Optional[str]]:
        """Convert to dictionary for display/logging."""
        return {
            "MISTRAL_API_KEY": self.mistral_api_key,
            "NVIDIA_BEARER_TOKEN": self.nvidia_bearer_token,
            "DATABRICKS_PAT": self.databricks_pat,
            "WANDB_API_KEY": self.wandb_api_key,
            "ELEVENLABS_API_KEY": self.elevenlabs_api_key,
            "HF_TOKEN": self.hf_token,
        }


def _load_dotenv_safe(env_file: Path = None) -> None:
    """Load .env file if it exists (safe operation)."""
    if env_file is None:
        env_file = Path(".env")

    if env_file.exists():
        load_dotenv(env_file)


def load_credentials(
    env_file: Optional[Path] = None,
    load_dotenv: bool = True,
) -> Credentials:
    """
    Load credentials from environment (Codespace secrets or .env file).

    Priority order:
    1. Environment variables (set by Codespaces or manually)
    2. .env file (for local development)
    3. None (if not found)

    Args:
        env_file: Path to .env file. Defaults to .env in current directory.
        load_dotenv: If True, attempt to load .env file first.

    Returns:
        Credentials dataclass with all available credentials.
    """
    if load_dotenv:
        _load_dotenv_safe(env_file)

    return Credentials(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        nvidia_bearer_token=os.getenv("NVIDIA_BEARER_TOKEN"),
        databricks_pat=os.getenv("DATABRICKS_PAT"),
        wandb_api_key=os.getenv("WANDB_API_KEY"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        hf_token=os.getenv("HF_TOKEN"),
    )


def validate_credentials(
    credentials: Credentials,
    required: list[str] | None = None,
) -> None:
    """
    Validate that required credentials are present.

    Args:
        credentials: Credentials object to validate.
        required: List of required credential names (e.g., ["MISTRAL_API_KEY"]).
                 If None, only checks essential credentials.

    Raises:
        ValueError: If any required credential is missing.
    """
    if required is None:
        required = []

    missing = []
    for key in required:
        if not credentials.get(key):
            missing.append(key)

    if missing:
        available = ", ".join(
            [k for k in credentials.to_dict().keys() if credentials.get(k)]
        )
        raise ValueError(
            f"Missing required credentials: {', '.join(missing)}\n"
            f"Available credentials: {available or 'None'}\n"
            f"Please set these environment variables or add them to .env file.\n"
            f"See docs/CREDENTIALS.md for setup instructions."
        )


def inject_credentials(
    credentials: Credentials,
    persist: bool = False,
) -> None:
    """
    Inject credentials into os.environ for backward compatibility.

    Args:
        credentials: Credentials object to inject.
        persist: If True, credentials persist in os.environ. If False, only
                used for this session (useful for testing).
    """
    creds_dict = credentials.to_dict()
    for key, value in creds_dict.items():
        if value:
            os.environ[key] = value
