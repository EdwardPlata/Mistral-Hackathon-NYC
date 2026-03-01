from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class HuggingFaceConfig:
    # Primary env var is HUGGINGFACE_TOKEN (Codespace secret name).
    # HF_TOKEN is the fallback (matches huggingface_hub defaults + .env.example).
    token_env_vars: tuple[str, ...] = ("HUGGINGFACE_TOKEN", "HF_TOKEN")
    endpoint: str = "https://huggingface.co"

    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        return cls(
            endpoint=os.getenv("HF_ENDPOINT", "https://huggingface.co"),
        )
