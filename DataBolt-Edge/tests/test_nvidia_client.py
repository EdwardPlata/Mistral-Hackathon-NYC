from __future__ import annotations

from pathlib import Path
import sys
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from nvidia_api_management.client import NvidiaAPIClient  # noqa: E402
from nvidia_api_management.config import NvidiaAPIConfig  # noqa: E402
from nvidia_api_management.errors import RequestFailedError  # noqa: E402


class MockResponse:
    def __init__(self, ok: bool, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.ok = ok
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = {"Content-Type": "application/json"}

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("invalid json")
        return self._payload

    def iter_lines(self):
        yield b"line"


class MockSession:
    def __init__(self, response: MockResponse) -> None:
        self.response = response
        self.mount_calls: list[tuple[str, object]] = []

    def mount(self, prefix: str, adapter: object) -> None:
        self.mount_calls.append((prefix, adapter))

    def post(self, *args, **kwargs) -> MockResponse:
        return self.response


def _config() -> NvidiaAPIConfig:
    return NvidiaAPIConfig(api_key_env_var="NO_KEY_NEEDED_FOR_TEST")


class TestNvidiaClient(unittest.TestCase):
    def test_chat_completion_success(self) -> None:
        session = MockSession(MockResponse(ok=True, status_code=200, payload={"id": "abc"}))
        client = NvidiaAPIClient(config=_config(), session=session)

        result = client.chat_completion(content="hello", api_key="token")

        self.assertEqual(result["id"], "abc")

    def test_chat_completion_http_error_raises(self) -> None:
        session = MockSession(
            MockResponse(ok=False, status_code=503, payload=None, text="unavailable")
        )
        client = NvidiaAPIClient(config=_config(), session=session)

        with self.assertRaises(RequestFailedError) as exc:
            client.chat_completion(content="hello", api_key="token")

        self.assertEqual(exc.exception.status_code, 503)

    def test_chat_completion_invalid_json_raises(self) -> None:
        session = MockSession(MockResponse(ok=True, status_code=200, payload=None, text="not-json"))
        client = NvidiaAPIClient(config=_config(), session=session)

        with self.assertRaises(RequestFailedError) as exc:
            client.chat_completion(content="hello", api_key="token")

        self.assertIn("invalid JSON", str(exc.exception))

    def test_chat_completion_stream_returns_iterator(self) -> None:
        session = MockSession(MockResponse(ok=True, status_code=200, payload={"ignored": True}))
        client = NvidiaAPIClient(config=_config(), session=session)

        result = client.chat_completion(content="hello", api_key="token", stream=True)

        self.assertIn("stream", result)
        chunks = list(result["stream"])
        self.assertEqual(chunks, [b"line"])


if __name__ == "__main__":
    unittest.main()
