"""Standardised request/response types and abstract backend for DataBolt inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceRequest:
    prompt: str
    system_prompt: str = (
        "You are DataBolt Edge, an expert AI assistant that helps data engineers "
        "debug Spark, Airflow, and SQL issues. Be concise and actionable."
    )
    max_tokens: int = 1024
    temperature: float = 0.15
    top_p: float = 1.0
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    backend: str = "unknown"
    raw: dict[str, Any] | None = None


class InferenceBackend(ABC):
    """Abstract base class all inference backends must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. 'nvidia_api', 'local_trt')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can serve requests right now."""

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference and return a response."""

    def health_check(self) -> dict[str, Any]:
        """Return a dict describing backend health. Override for richer checks."""
        return {"backend": self.name, "available": self.is_available()}
