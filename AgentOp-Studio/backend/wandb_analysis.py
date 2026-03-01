"""W&B analysis module for Mistral agent training runs.

Provides WandbTrainingRun — a context-managed class that:
  - Creates a W&B run for a Mistral fine-tuning experiment
  - Streams a HuggingFace dataset sample and logs statistics + artifact
  - Simulates a realistic training loop (loss, perplexity, lr, accuracy)
  - Logs per-step metrics and eval checkpoints
  - Logs agent-behavior metrics from AgentOps run records
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import uuid
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyper-parameters for a simulated Mistral fine-tuning run."""

    model_name: str = "mistral-7b-instruct-v0.2"
    dataset_name: str = "tatsu-lab/alpaca"
    num_steps: int = 100
    eval_every: int = 10
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_seq_len: int = 512
    lora_rank: int = 16
    warmup_steps: int = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_loss(step: int, total: int, noise: float = 0.05) -> float:
    """Realistic training loss: exponential decay from ~2.5 → ~0.7."""
    base = 2.5 * math.exp(-3.5 * step / max(1, total)) + 0.7
    return max(0.3, base + random.gauss(0, noise))


def _simulate_lr(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class WandbTrainingRun:
    """Context-managed W&B run for Mistral agent training analysis.

    Usage::

        config = TrainingConfig(num_steps=50)
        with WandbTrainingRun(config=config) as run:
            run.log_dataset(samples)
            metrics = run.run_training_loop()
        print(run.run_url)
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        run_name: str | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.run_name = run_name or f"mistral-ft-{uuid.uuid4().hex[:8]}"
        self._wandb = None
        self.run_url: str | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WandbTrainingRun":
        self._start()
        return self

    def __exit__(self, *_) -> None:
        self._finish()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        if not os.getenv("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY environment variable is not set")
        import wandb  # noqa: PLC0415

        self._wandb = wandb
        project = os.getenv("WANDB_PROJECT", "agentops-studio")
        run = wandb.init(
            project=project,
            name=self.run_name,
            config={
                "model": self.config.model_name,
                "dataset": self.config.dataset_name,
                "num_steps": self.config.num_steps,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "max_seq_len": self.config.max_seq_len,
                "lora_rank": self.config.lora_rank,
                "warmup_steps": self.config.warmup_steps,
            },
            tags=["mistral", "fine-tuning", "agentops"],
            reinit=True,
        )
        self.run_url = run.url if run else None

    def _finish(self) -> None:
        if self._wandb and self._wandb.run:
            self._wandb.finish()

    # ------------------------------------------------------------------
    # Dataset logging
    # ------------------------------------------------------------------

    def log_dataset(self, samples: list[dict]) -> None:
        """Upload HuggingFace dataset sample as a W&B artifact + table.

        Logs summary statistics (num_samples, avg_instruction_words,
        avg_output_words) alongside a browsable W&B Table.
        """
        if not samples:
            return
        w = self._wandb

        # Build W&B Table from sample columns
        columns = list(samples[0].keys())
        # Ensure the most useful columns appear first
        for preferred in ("output", "input", "instruction"):
            if preferred in columns:
                columns.remove(preferred)
                columns.insert(0, preferred)

        table = w.Table(columns=columns)
        for s in samples:
            table.add_data(*[str(s.get(c, ""))[:512] for c in columns])

        w.log({"dataset/sample_table": table})

        # Upload JSON artifact
        artifact = w.Artifact(
            name=f"dataset-{self.run_name}",
            type="dataset",
            description=f"HuggingFace dataset sample: {self.config.dataset_name}",
            metadata={"source": self.config.dataset_name, "num_samples": len(samples)},
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
            tmp_path = f.name
        artifact.add_file(tmp_path, name="samples.json")
        w.log_artifact(artifact)

        # Summary statistics
        instr_lens = [len(str(s.get("instruction", s.get("prompt", ""))).split()) for s in samples]
        out_lens = [len(str(s.get("output", s.get("response", ""))).split()) for s in samples]
        w.log({
            "dataset/num_samples": len(samples),
            "dataset/avg_instruction_words": sum(instr_lens) / max(1, len(instr_lens)),
            "dataset/avg_output_words": sum(out_lens) / max(1, len(out_lens)),
            "dataset/max_instruction_words": max(instr_lens, default=0),
            "dataset/max_output_words": max(out_lens, default=0),
        })

    # ------------------------------------------------------------------
    # Training loop simulation
    # ------------------------------------------------------------------

    def run_training_loop(self) -> dict:
        """Simulate Mistral fine-tuning and log metrics step-by-step.

        Logs:
          - train/loss, train/perplexity, train/token_accuracy
          - train/learning_rate, train/grad_norm
          - eval/loss, eval/token_accuracy, eval/perplexity (every eval_every steps)

        Returns:
            dict with final_train_loss, final_perplexity, final_token_accuracy,
            total_steps.
        """
        w = self._wandb
        cfg = self.config
        last_metrics: dict = {}

        for step in range(1, cfg.num_steps + 1):
            train_loss = _simulate_loss(step, cfg.num_steps)
            lr = _simulate_lr(step, cfg.num_steps, cfg.learning_rate, cfg.warmup_steps)
            perplexity = math.exp(min(train_loss, 10.0))
            token_acc = min(0.95, 0.35 + 0.60 * (1.0 - math.exp(-4.0 * step / cfg.num_steps)))
            grad_norm = abs(random.gauss(1.5, 0.6))

            log_dict: dict = {
                "train/loss": train_loss,
                "train/perplexity": perplexity,
                "train/token_accuracy": token_acc,
                "train/learning_rate": lr,
                "train/grad_norm": grad_norm,
            }

            if step % cfg.eval_every == 0:
                eval_loss = max(0.3, train_loss + random.gauss(0.05, 0.03))
                eval_acc = max(0.1, token_acc - random.uniform(0.01, 0.05))
                log_dict["eval/loss"] = eval_loss
                log_dict["eval/token_accuracy"] = eval_acc
                log_dict["eval/perplexity"] = math.exp(min(eval_loss, 10.0))

            w.log(log_dict, step=step)
            last_metrics = log_dict

        return {
            "final_train_loss": last_metrics.get("train/loss", 0.0),
            "final_perplexity": last_metrics.get("train/perplexity", 0.0),
            "final_token_accuracy": last_metrics.get("train/token_accuracy", 0.0),
            "total_steps": cfg.num_steps,
        }

    # ------------------------------------------------------------------
    # Agent analysis
    # ------------------------------------------------------------------

    def log_agent_analysis(self, run_records: list[dict]) -> None:
        """Log AgentOps run records as agent-analysis metrics in W&B.

        Args:
            run_records: List of dicts with keys run_id, status,
                         total_tokens, total_cost.
        """
        if not run_records:
            return
        w = self._wandb

        total_tokens = sum(r.get("total_tokens") or 0 for r in run_records)
        total_cost = sum(r.get("total_cost") or 0.0 for r in run_records)
        success_count = sum(1 for r in run_records if r.get("status") == "success")
        n = len(run_records)

        w.log({
            "agent/total_runs": n,
            "agent/success_rate": success_count / max(1, n),
            "agent/avg_tokens_per_run": total_tokens / max(1, n),
            "agent/avg_cost_usd": total_cost / max(1, n),
            "agent/total_cost_usd": total_cost,
        })

        table = w.Table(columns=["run_id", "status", "tokens", "cost_usd"])
        for r in run_records:
            table.add_data(
                str(r.get("run_id", ""))[:12],
                r.get("status", "unknown"),
                r.get("total_tokens") or 0,
                round(r.get("total_cost") or 0.0, 6),
            )
        w.log({"agent/run_summary": table})
