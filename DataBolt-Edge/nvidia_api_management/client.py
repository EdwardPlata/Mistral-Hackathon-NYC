from __future__ import annotations

import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .auth import build_headers
from .config import NvidiaAPIConfig
from .errors import RequestFailedError
from .logging_utils import get_logger, redact_headers


class NvidiaAPIClient:
    def __init__(
        self,
        config: NvidiaAPIConfig | None = None,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or NvidiaAPIConfig.from_env()
        self.logger = logger or get_logger()
        self.session = session or requests.Session()
        self._configure_retry_adapter(self.session)

    def _configure_retry_adapter(self, session: requests.Session) -> None:
        retry = Retry(
            total=self.config.max_retries,
            connect=self.config.max_retries,
            read=self.config.max_retries,
            status=self.config.max_retries,
            allowed_methods=frozenset(["POST"]),
            status_forcelist=self.config.retry_status_codes,
            backoff_factor=self.config.backoff_factor,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

    def _build_chat_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        chat_path = self.config.chat_completions_path.lstrip("/")
        return f"{base_url}/{chat_path}"

    def chat_completion(
        self,
        content: str,
        model: str | None = None,
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.15,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        url = self._build_chat_url()
        headers = build_headers(self.config, api_key=api_key)
        payload = {
            "model": model or self.config.default_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream,
        }

        self.logger.info(
            "nvidia_request method=POST url=%s timeout=%s headers=%s",
            url,
            self.config.timeout_seconds,
            redact_headers(headers),
        )

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise RequestFailedError(f"Failed to reach NVIDIA API: {exc}") from exc

        self.logger.info(
            "nvidia_response status_code=%s content_type=%s",
            response.status_code,
            response.headers.get("Content-Type", "unknown"),
        )

        if not response.ok:
            response_text = response.text[:500]
            raise RequestFailedError(
                f"NVIDIA API returned HTTP {response.status_code}",
                status_code=response.status_code,
                response_text=response_text,
            )

        if stream:
            return {"stream": response.iter_lines()}

        try:
            return response.json()
        except ValueError as exc:
            raise RequestFailedError(
                "NVIDIA API returned invalid JSON response",
                status_code=response.status_code,
                response_text=response.text[:500],
            ) from exc
