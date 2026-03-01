#!/usr/bin/env python3
"""Demo: Create a Mistral agent training run in W&B with HuggingFace data.

Fetches a small sample from a HuggingFace dataset, then creates a W&B
training run that logs dataset statistics, a simulated fine-tuning loop
(loss / perplexity / accuracy curves), and any existing AgentOps runs
as agent-analysis metrics.

Usage:
    PYTHONPATH=AgentOp-Studio python AgentOp-Studio/scripts/mistral_training_demo.py

Environment variables:
    WANDB_API_KEY    (required) Weights & Biases API key
    HF_TOKEN         (optional) HuggingFace token for private datasets
    WANDB_PROJECT    (optional) W&B project name  [default: agentops-studio]
    HF_DATASET       (optional) HuggingFace dataset id  [default: tatsu-lab/alpaca]
    HF_SAMPLE_SIZE   (optional) Number of HF samples to pull  [default: 30]
    TRAINING_STEPS   (optional) Simulated training steps  [default: 100]
"""

from __future__ import annotations

import os
import sys

# Allow running from repo root without installing the package
_repo_root = os.path.join(os.path.dirname(__file__), "..")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dotenv import load_dotenv

load_dotenv()

from backend.wandb_analysis import TrainingConfig, WandbTrainingRun  # noqa: E402


# ---------------------------------------------------------------------------
# HuggingFace data fetch
# ---------------------------------------------------------------------------


def fetch_hf_samples(dataset_name: str, sample_size: int) -> list[dict]:
    """Stream a small sample from a HuggingFace dataset.

    Falls back to synthetic placeholder data if the dataset is unavailable
    (e.g. HF_TOKEN missing or network issues) so the demo always completes.
    """
    print(f"  Fetching {sample_size} samples from '{dataset_name}' on HuggingFace …")
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        samples = []
        for row in ds:
            samples.append(dict(row))
            if len(samples) >= sample_size:
                break
        print(f"  Retrieved {len(samples)} samples.")
        return samples

    except Exception as exc:
        print(f"  Warning: could not load dataset ({exc}). Using synthetic data.")
        return [
            {
                "instruction": f"Explain concept #{i} in machine learning.",
                "input": "",
                "output": (
                    f"Concept #{i} is a foundational principle in ML. "
                    "It relates to how models learn from data through "
                    "iterative optimization of a loss function."
                ),
            }
            for i in range(1, sample_size + 1)
        ]


# ---------------------------------------------------------------------------
# AgentOps run records (best-effort — works even if backend is not running)
# ---------------------------------------------------------------------------


def _load_agentops_runs() -> list[dict]:
    """Load recent AgentOps runs from DuckDB for agent analysis logging."""
    try:
        from backend.db import get_conn  # noqa: PLC0415

        conn = get_conn()
        rows = conn.execute(
            "SELECT run_id, status, total_tokens, total_cost FROM runs LIMIT 20"
        ).fetchall()
        conn.close()
        return [
            {"run_id": r[0], "status": r[1], "total_tokens": r[2], "total_cost": r[3]}
            for r in rows
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not os.getenv("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY environment variable is not set.")
        print("       Set it in your .env file or Codespace secrets.")
        sys.exit(1)

    dataset_name = os.getenv("HF_DATASET", "tatsu-lab/alpaca")
    sample_size = int(os.getenv("HF_SAMPLE_SIZE", "30"))
    training_steps = int(os.getenv("TRAINING_STEPS", "100"))

    print("=" * 60)
    print("  Mistral Agent Training Demo — AgentOp-Studio")
    print("=" * 60)

    # 1. Fetch HuggingFace data
    print("\n[1/4] Fetching HuggingFace dataset …")
    samples = fetch_hf_samples(dataset_name, sample_size)

    # 2. Load existing AgentOps runs for analysis
    print("\n[2/4] Loading AgentOps run records …")
    agent_records = _load_agentops_runs()
    print(f"  Found {len(agent_records)} AgentOps run(s).")

    # 3. Configure and start training run
    config = TrainingConfig(
        model_name="mistral-7b-instruct-v0.2",
        dataset_name=dataset_name,
        num_steps=training_steps,
        eval_every=max(1, training_steps // 10),
        learning_rate=2e-4,
        batch_size=4,
        max_seq_len=512,
        lora_rank=16,
        warmup_steps=max(1, training_steps // 10),
    )

    print(f"\n[3/4] Starting W&B training run …")
    print(f"  Model:    {config.model_name}")
    print(f"  Dataset:  {config.dataset_name}  ({len(samples)} samples)")
    print(f"  Steps:    {config.num_steps}  |  LR: {config.learning_rate}  |  LoRA rank: {config.lora_rank}")

    with WandbTrainingRun(config=config) as run:
        print(f"\n  W&B run URL: {run.run_url}\n")

        # Log dataset
        print("  Logging dataset artifact …")
        run.log_dataset(samples)

        # Run training simulation
        print(f"  Simulating {config.num_steps} training steps …")
        results = run.run_training_loop()

        # Log agent analysis
        if agent_records:
            print("  Logging AgentOps agent analysis …")
            run.log_agent_analysis(agent_records)

    # 4. Summary
    print("\n[4/4] Training complete.")
    print("=" * 60)
    print(f"  Final train loss:       {results['final_train_loss']:.4f}")
    print(f"  Final perplexity:       {results['final_perplexity']:.2f}")
    print(f"  Final token accuracy:   {results['final_token_accuracy']:.2%}")
    print(f"  Total steps:            {results['total_steps']}")
    print(f"\n  View run: {run.run_url}")
    print("=" * 60)


if __name__ == "__main__":
    main()
