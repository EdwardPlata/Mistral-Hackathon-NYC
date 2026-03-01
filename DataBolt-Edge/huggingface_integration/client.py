from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from .auth import resolve_token
from .config import HuggingFaceConfig
from .errors import HFRequestError, MissingTokenError


@dataclass
class ModelSummary:
    """Lightweight summary of a Hub model."""
    model_id: str
    pipeline_tag: str | None
    downloads: int
    likes: int
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_model_info(cls, info: Any) -> "ModelSummary":
        return cls(
            model_id=info.id,
            pipeline_tag=getattr(info, "pipeline_tag", None),
            downloads=getattr(info, "downloads", 0) or 0,
            likes=getattr(info, "likes", 0) or 0,
            tags=list(getattr(info, "tags", []) or []),
        )


class HuggingFaceClient:
    """Wrapper around huggingface_hub.HfApi for DataBolt-Edge."""

    def __init__(
        self,
        config: HuggingFaceConfig | None = None,
        _api: HfApi | None = None,  # injectable for tests
    ) -> None:
        self.config = config or HuggingFaceConfig.from_env()
        self._api = _api  # None until first use — resolved with token at call time

    def _get_api(self, token: str | None) -> HfApi:
        """Return an HfApi instance authenticated with the resolved token."""
        if self._api is not None:
            return self._api
        resolved = resolve_token(self.config, token)
        return HfApi(token=resolved, endpoint=self.config.endpoint)

    # ── public methods ───────────────────────────────────────────────────────

    def whoami(self, token: str | None = None) -> dict[str, Any]:
        """Return authenticated user info dict from /api/whoami-v2."""
        try:
            return self._get_api(token).whoami()
        except HfHubHTTPError as exc:
            raise HFRequestError(
                str(exc),
                status_code=exc.response.status_code if exc.response is not None else None,
            ) from exc
        except MissingTokenError:
            raise

    def model_info(self, model_id: str, token: str | None = None) -> dict[str, Any]:
        """Return metadata dict for a specific model on the Hub."""
        try:
            info = self._get_api(token).model_info(model_id)
            # Convert ModelInfo to a plain dict for consistent downstream use
            return {
                "id": info.id,
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "tags": list(getattr(info, "tags", []) or []),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "created_at": str(getattr(info, "created_at", "")),
                "last_modified": str(getattr(info, "last_modified", "")),
                "private": getattr(info, "private", False),
                "gated": getattr(info, "gated", False),
            }
        except HfHubHTTPError as exc:
            raise HFRequestError(
                str(exc),
                status_code=exc.response.status_code if exc.response is not None else None,
            ) from exc

    def list_models(
        self,
        author: str = "mistralai",
        pipeline_tag: str = "text-generation",
        limit: int = 10,
        token: str | None = None,
    ) -> list[ModelSummary]:
        """List models on the Hub filtered by author and pipeline tag."""
        try:
            api = self._get_api(token)
            results = api.list_models(
                author=author,
                pipeline_tag=pipeline_tag,
                limit=limit,
                sort="downloads",
                direction=-1,
            )
            return [ModelSummary.from_model_info(m) for m in results]
        except HfHubHTTPError as exc:
            raise HFRequestError(
                str(exc),
                status_code=exc.response.status_code if exc.response is not None else None,
            ) from exc

    def dataset_info(self, dataset_id: str, token: str | None = None) -> dict[str, Any]:
        """Return metadata dict for a specific dataset on the Hub."""
        try:
            info = self._get_api(token).dataset_info(dataset_id)
            return {
                "id": info.id,
                "tags": list(getattr(info, "tags", []) or []),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "private": getattr(info, "private", False),
            }
        except HfHubHTTPError as exc:
            raise HFRequestError(
                str(exc),
                status_code=exc.response.status_code if exc.response is not None else None,
            ) from exc
