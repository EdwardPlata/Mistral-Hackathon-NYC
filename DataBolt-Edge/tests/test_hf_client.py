from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from huggingface_integration.client import HuggingFaceClient, ModelSummary  # noqa: E402
from huggingface_integration.config import HuggingFaceConfig  # noqa: E402
from huggingface_integration.errors import HFRequestError  # noqa: E402


def _config() -> HuggingFaceConfig:
    return HuggingFaceConfig(token_env_vars=("MISSING_TOKEN_FOR_TEST",))


def _fake_model(model_id: str, downloads: int = 100, likes: int = 10) -> MagicMock:
    m = MagicMock()
    m.id = model_id
    m.pipeline_tag = "text-generation"
    m.downloads = downloads
    m.likes = likes
    m.tags = ["text-generation", "mistral"]
    return m


def _fake_model_info_obj(model_id: str) -> MagicMock:
    m = MagicMock()
    m.id = model_id
    m.pipeline_tag = "text-generation"
    m.tags = ["text-generation", "mistral"]
    m.downloads = 500
    m.likes = 42
    m.created_at = "2024-01-01"
    m.last_modified = "2024-06-01"
    m.private = False
    m.gated = False
    return m


class TestHFClientWhoami(unittest.TestCase):
    def test_whoami_success(self) -> None:
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "test-user", "email": "t@example.com"}
        client = HuggingFaceClient(config=_config(), _api=mock_api)

        result = client.whoami(token="hf-test")
        self.assertEqual(result["name"], "test-user")
        mock_api.whoami.assert_called_once()

    def test_whoami_http_error_raises_hf_request_error(self) -> None:
        from huggingface_hub.utils import HfHubHTTPError
        mock_api = MagicMock()
        fake_resp = MagicMock()
        fake_resp.status_code = 401
        mock_api.whoami.side_effect = HfHubHTTPError("Unauthorized", response=fake_resp)
        client = HuggingFaceClient(config=_config(), _api=mock_api)

        with self.assertRaises(HFRequestError) as ctx:
            client.whoami(token="bad-token")
        self.assertEqual(ctx.exception.status_code, 401)


class TestHFClientModelInfo(unittest.TestCase):
    def test_model_info_success(self) -> None:
        mock_api = MagicMock()
        mock_api.model_info.return_value = _fake_model_info_obj("mistralai/Mistral-7B-Instruct-v0.1")
        client = HuggingFaceClient(config=_config(), _api=mock_api)

        result = client.model_info("mistralai/Mistral-7B-Instruct-v0.1", token="hf-test")
        self.assertEqual(result["id"], "mistralai/Mistral-7B-Instruct-v0.1")
        self.assertEqual(result["downloads"], 500)
        self.assertFalse(result["private"])

    def test_model_info_not_found_raises(self) -> None:
        from huggingface_hub.utils import HfHubHTTPError
        mock_api = MagicMock()
        fake_resp = MagicMock()
        fake_resp.status_code = 404
        mock_api.model_info.side_effect = HfHubHTTPError("Not Found", response=fake_resp)
        client = HuggingFaceClient(config=_config(), _api=mock_api)

        with self.assertRaises(HFRequestError) as ctx:
            client.model_info("mistralai/doesnotexist", token="hf-test")
        self.assertEqual(ctx.exception.status_code, 404)


class TestHFClientListModels(unittest.TestCase):
    def test_list_models_returns_summaries(self) -> None:
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([
            _fake_model("mistralai/Mistral-7B-v0.1", downloads=1000, likes=50),
            _fake_model("mistralai/Mistral-7B-Instruct-v0.1", downloads=800, likes=40),
        ])
        client = HuggingFaceClient(config=_config(), _api=mock_api)

        results = client.list_models(token="hf-test")
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ModelSummary)
        self.assertEqual(results[0].model_id, "mistralai/Mistral-7B-v0.1")
        self.assertEqual(results[0].downloads, 1000)

    def test_list_models_empty(self) -> None:
        mock_api = MagicMock()
        mock_api.list_models.return_value = iter([])
        client = HuggingFaceClient(config=_config(), _api=mock_api)

        results = client.list_models(token="hf-test")
        self.assertEqual(results, [])


class TestModelSummary(unittest.TestCase):
    def test_from_model_info(self) -> None:
        fake = _fake_model("mistralai/Mistral-7B-v0.1", downloads=999, likes=88)
        summary = ModelSummary.from_model_info(fake)
        self.assertEqual(summary.model_id, "mistralai/Mistral-7B-v0.1")
        self.assertEqual(summary.downloads, 999)
        self.assertEqual(summary.likes, 88)
        self.assertIn("text-generation", summary.tags)


if __name__ == "__main__":
    unittest.main()
