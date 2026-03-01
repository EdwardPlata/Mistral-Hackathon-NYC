from __future__ import annotations

from pathlib import Path
import sys
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from nvidia_api_management.client import NvidiaAPIClient  # noqa: E402
from nvidia_api_management.config import NvidiaAPIConfig  # noqa: E402


class CapturingSession:
    def __init__(self) -> None:
        self.mounts: list[tuple[str, object]] = []

    def mount(self, prefix: str, adapter: object) -> None:
        self.mounts.append((prefix, adapter))


class TestNvidiaRetry(unittest.TestCase):
    def test_retry_adapter_is_mounted_for_http_and_https(self) -> None:
        session = CapturingSession()
        config = NvidiaAPIConfig(max_retries=4)

        NvidiaAPIClient(config=config, session=session)  # type: ignore[arg-type]

        prefixes = [prefix for prefix, _ in session.mounts]
        self.assertIn("https://", prefixes)
        self.assertIn("http://", prefixes)

    def test_chat_url_building_respects_base_and_path(self) -> None:
        config = NvidiaAPIConfig(
            base_url="https://integrate.api.nvidia.com/",
            chat_completions_path="v1/chat/completions",
        )
        client = NvidiaAPIClient(config=config)

        self.assertEqual(
            client._build_chat_url(),
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
