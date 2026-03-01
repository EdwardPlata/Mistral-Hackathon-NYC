from pathlib import Path
import sys
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from nvidia_api_management.auth import build_headers  # noqa: E402
from nvidia_api_management.config import NvidiaAPIConfig  # noqa: E402
from nvidia_api_management.errors import MissingCredentialError  # noqa: E402


class TestNvidiaAuth(unittest.TestCase):
    def test_build_headers_with_explicit_api_key(self) -> None:
        config = NvidiaAPIConfig(default_headers={"X-Test": "1"})
        headers = build_headers(config, api_key="secret-token")

        self.assertEqual(headers["Authorization"], "Bearer secret-token")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["X-Test"], "1")

    def test_build_headers_missing_api_key_raises(self) -> None:
        config = NvidiaAPIConfig(api_key_env_var="MISSING_KEY")
        with self.assertRaises(MissingCredentialError):
            build_headers(config)


if __name__ == "__main__":
    unittest.main()
