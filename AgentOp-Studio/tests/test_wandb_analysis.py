"""Tests for backend.wandb_analysis — WandbTrainingRun and TrainingConfig.

All tests mock wandb so they run without a real WANDB_API_KEY or internet
connection.
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure AgentOp-Studio/ is on sys.path
_root = os.path.join(os.path.dirname(__file__), "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.wandb_analysis import (  # noqa: E402
    TrainingConfig,
    WandbTrainingRun,
    _simulate_loss,
    _simulate_lr,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_wandb_mock() -> MagicMock:
    """Return a fully mocked wandb module."""
    mock_wandb = MagicMock()
    # run.url
    mock_run = MagicMock()
    mock_run.url = "https://wandb.ai/test/test/runs/abc123"
    mock_wandb.init.return_value = mock_run
    mock_wandb.run = mock_run
    # Table
    mock_table = MagicMock()
    mock_wandb.Table.return_value = mock_table
    # Artifact
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact
    return mock_wandb


def _patch_env(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-testing")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")


# ---------------------------------------------------------------------------
# TrainingConfig tests
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.model_name == "mistral-7b-instruct-v0.2"
        assert cfg.dataset_name == "tatsu-lab/alpaca"
        assert cfg.num_steps == 100
        assert cfg.eval_every == 10
        assert 0 < cfg.learning_rate < 1
        assert cfg.lora_rank > 0

    def test_custom_values(self):
        cfg = TrainingConfig(num_steps=50, lora_rank=8, learning_rate=1e-3)
        assert cfg.num_steps == 50
        assert cfg.lora_rank == 8
        assert cfg.learning_rate == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# Internal simulation helpers
# ---------------------------------------------------------------------------


class TestSimulationHelpers:
    def test_loss_decreases_overall(self):
        """Loss at step 100 should be lower than at step 1 (on average)."""
        early = [_simulate_loss(1, 100) for _ in range(20)]
        late = [_simulate_loss(100, 100) for _ in range(20)]
        assert sum(early) / len(early) > sum(late) / len(late)

    def test_loss_positive(self):
        for step in [1, 10, 50, 100]:
            assert _simulate_loss(step, 100) > 0

    def test_lr_warmup(self):
        """LR should rise during warmup then fall."""
        peak = 2e-4
        warmup = 10
        mid_warmup = _simulate_lr(5, 100, peak, warmup)
        at_peak = _simulate_lr(10, 100, peak, warmup)
        after_peak = _simulate_lr(90, 100, peak, warmup)
        assert mid_warmup < at_peak
        assert at_peak > after_peak

    def test_lr_reaches_near_zero_at_end(self):
        lr_end = _simulate_lr(100, 100, 1e-4, 10)
        assert lr_end < 1e-6


# ---------------------------------------------------------------------------
# WandbTrainingRun tests
# ---------------------------------------------------------------------------


class TestWandbTrainingRun:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        run = WandbTrainingRun()
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            run._start()

    def test_context_manager_calls_finish(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=5)):
                pass

        mock_wandb.finish.assert_called_once()

    def test_run_url_set_after_start(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=5)) as run:
                assert run.run_url == "https://wandb.ai/test/test/runs/abc123"

    def test_training_loop_returns_metrics(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=10, eval_every=5)) as run:
                results = run.run_training_loop()

        assert "final_train_loss" in results
        assert "final_perplexity" in results
        assert "final_token_accuracy" in results
        assert results["total_steps"] == 10
        assert results["final_train_loss"] > 0
        assert results["final_perplexity"] > 1
        assert 0 < results["final_token_accuracy"] <= 1

    def test_training_loop_logs_correct_step_count(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=15, eval_every=5)) as run:
                run.run_training_loop()

        # wandb.log called once per step
        assert mock_wandb.log.call_count >= 15

    def test_log_dataset_creates_table_and_artifact(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()
        samples = [
            {"instruction": f"Q{i}", "input": "", "output": f"A{i}"}
            for i in range(5)
        ]

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=5)) as run:
                run.log_dataset(samples)

        mock_wandb.Table.assert_called_once()
        mock_wandb.Artifact.assert_called_once()
        mock_wandb.log_artifact.assert_called_once()

    def test_log_dataset_skips_empty(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=5)) as run:
                run.log_dataset([])  # should be a no-op

        mock_wandb.Table.assert_not_called()

    def test_log_agent_analysis_skips_empty(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=5)) as run:
                before = mock_wandb.log.call_count
                run.log_agent_analysis([])
                after = mock_wandb.log.call_count

        assert after == before  # no extra log calls

    def test_log_agent_analysis_computes_success_rate(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()
        records = [
            {"run_id": "a", "status": "success", "total_tokens": 100, "total_cost": 0.001},
            {"run_id": "b", "status": "error", "total_tokens": 50, "total_cost": 0.0005},
        ]

        logged_metrics = {}

        def capture_log(metrics, **kwargs):
            logged_metrics.update(metrics)

        mock_wandb.log.side_effect = capture_log

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=5)) as run:
                run.log_agent_analysis(records)

        assert logged_metrics.get("agent/total_runs") == 2
        assert logged_metrics.get("agent/success_rate") == pytest.approx(0.5)
        assert logged_metrics.get("agent/avg_tokens_per_run") == pytest.approx(75.0)

    def test_auto_run_name_generated(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(config=TrainingConfig(num_steps=5)) as run:
                assert run.run_name.startswith("mistral-ft-")

    def test_custom_run_name(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(
                config=TrainingConfig(num_steps=5), run_name="my-custom-run"
            ) as run:
                assert run.run_name == "my-custom-run"

    def test_eval_metrics_logged_at_correct_steps(self, monkeypatch):
        _patch_env(monkeypatch)
        mock_wandb = _make_wandb_mock()
        logged = []

        def capture(metrics, **kwargs):
            logged.append(dict(metrics))

        mock_wandb.log.side_effect = capture

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with WandbTrainingRun(
                config=TrainingConfig(num_steps=20, eval_every=10)
            ) as run:
                run.run_training_loop()

        eval_logs = [m for m in logged if "eval/loss" in m]
        # eval logged at steps 10 and 20 → 2 eval entries
        assert len(eval_logs) == 2
