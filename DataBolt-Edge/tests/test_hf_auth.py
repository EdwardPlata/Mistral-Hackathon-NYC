from __future__ import annotations

import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from huggingface_integration.auth import resolve_token  # noqa: E402
from huggingface_integration.config import HuggingFaceConfig  # noqa: E402
from huggingface_integration.errors import MissingTokenError  # noqa: E402


class TestHFAuth(unittest.TestCase):
    def _config_no_env(self) -> HuggingFaceConfig:
        # Point to env vars that aren't set so env-fallback always fails cleanly
        return HuggingFaceConfig(token_env_vars=("MISSING_HF_VAR_A", "MISSING_HF_VAR_B"))

    def test_resolve_explicit_token(self) -> None:
        config = self._config_no_env()
        self.assertEqual(resolve_token(config, token="hf-explicit-token"), "hf-explicit-token")

    def test_resolve_missing_raises(self) -> None:
        config = self._config_no_env()
        with self.assertRaises(MissingTokenError):
            resolve_token(config)

    def test_resolve_env_fallback(self) -> None:
        import os
        config = HuggingFaceConfig(token_env_vars=("_TEST_HF_PRIMARY", "_TEST_HF_FALLBACK"))
        os.environ["_TEST_HF_FALLBACK"] = "hf-from-env"
        try:
            token = resolve_token(config)
            self.assertEqual(token, "hf-from-env")
        finally:
            del os.environ["_TEST_HF_FALLBACK"]

    def test_explicit_beats_env(self) -> None:
        import os
        config = HuggingFaceConfig(token_env_vars=("HUGGINGFACE_TOKEN", "HF_TOKEN"))
        os.environ["_TEST_HF_SHOULD_NOT_USE"] = "env-token"
        try:
            token = resolve_token(config, token="explicit-wins")
            self.assertEqual(token, "explicit-wins")
        finally:
            os.environ.pop("_TEST_HF_SHOULD_NOT_USE", None)

    def test_first_env_var_wins(self) -> None:
        import os
        config = HuggingFaceConfig(token_env_vars=("_TEST_PRIMARY", "_TEST_SECONDARY"))
        os.environ["_TEST_PRIMARY"] = "primary-token"
        os.environ["_TEST_SECONDARY"] = "secondary-token"
        try:
            token = resolve_token(config)
            self.assertEqual(token, "primary-token")
        finally:
            del os.environ["_TEST_PRIMARY"]
            del os.environ["_TEST_SECONDARY"]


if __name__ == "__main__":
    unittest.main()
