"""Calibration dataset for DataBolt Edge static INT8 quantization.

Provides a :class:`DataBoltCalibrationDataReader` that feeds representative
Spark / Airflow / SQL log prompts through a tokenizer to generate the input
tensors needed by ONNX Runtime's static quantization calibrator.

If no tokenizer directory is available (e.g., in a Codespace without model
weights), a :func:`build_fake_calibration_dataset` helper generates synthetic
token-ID arrays of the correct shape, allowing the full pipeline to run end-
to-end without real model weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


# ── representative DataBolt prompts ───────────────────────────────────────────

_CALIBRATION_PROMPTS: list[str] = [
    # Spark
    "Log output:\njava.lang.OutOfMemoryError: GC overhead limit exceeded\n\nWhat is the root cause?",
    "Log output:\nException in task 3.0 in stage 5.0: FetchFailed\n\nHow should I fix this shuffle error?",
    "Log output:\nLost executor 2 on 10.0.0.5: Remote RPC client disassociated\n\nWhat caused the executor loss?",
    # Airflow
    "Log output:\nFileNotFoundError: /data/input/sales_2024.csv not found\n\nHow do I debug this DAG failure?",
    "Log output:\nAirflowSensorTimeout: Sensor timed out after 3600s\n\nWhat retry strategy should I use?",
    "Log output:\nImportError: cannot import name 'read_parquet' from 'pandas'\n\nWhich version should I install?",
    # SQL
    "Log output:\nFull table scan on orders: 5M rows, cost=521432\n\nHow do I add an index here?",
    "Log output:\nCross join detected: products × categories = 50M rows\n\nHow do I rewrite this query?",
    "Log output:\nUsing filesort on ORDER BY created_at DESC\n\nWhat index would eliminate this filesort?",
    # General DataBolt
    "Describe the most common causes of Spark OOM errors on large shuffle operations.",
    "What are the best practices for partitioning data in Apache Spark?",
    "How do I tune executor memory and GC settings for a Spark job processing 1TB?",
    "Explain how Airflow backfills work and common failure modes.",
    "What does a high cost estimate in a SQL EXPLAIN plan indicate?",
    "How do I detect and fix data skew in a distributed SQL join?",
    "What is the difference between FP16 and INT8 inference for LLMs?",
]

# Default sequence length for calibration inputs
_DEFAULT_SEQ_LEN = 128
_DEFAULT_BATCH_SIZE = 1


# ── ONNX Runtime CalibrationDataReader ───────────────────────────────────────

class DataBoltCalibrationDataReader:
    """Feeds tokenized DataBolt prompts to the ONNX Runtime static quantizer.

    If a *tokenizer_dir* is provided and the tokenizer loads successfully, real
    token IDs are generated.  Otherwise falls back to synthetic random INT64
    token IDs (good enough for calibration range estimation on toy models).

    Parameters
    ----------
    tokenizer_dir:
        Path to a directory containing ``tokenizer.json`` (HF tokenizer
        format).  Pass None to use synthetic token IDs.
    input_names:
        ONNX model input names.  Typical Mistral ONNX export names are
        ``["input_ids", "attention_mask"]``.
    seq_len:
        Token sequence length for calibration inputs.
    batch_size:
        Batch size for each calibration step.
    prompts:
        Optional list of prompt strings.  Defaults to built-in DataBolt
        calibration prompts.
    """

    def __init__(
        self,
        tokenizer_dir: str | Path | None = None,
        input_names: list[str] | None = None,
        seq_len: int = _DEFAULT_SEQ_LEN,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        prompts: list[str] | None = None,
    ) -> None:
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self.input_names = input_names or ["input_ids", "attention_mask"]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.prompts = prompts or _CALIBRATION_PROMPTS
        self._tokenizer = self._load_tokenizer()
        self._iter: Iterator[dict[str, np.ndarray]] | None = None

    # --- tokenizer loading ---

    def _load_tokenizer(self):
        """Attempt to load a HuggingFace tokenizer; return None on failure."""
        if not self.tokenizer_dir or not self.tokenizer_dir.exists():
            log.info("No tokenizer dir — using synthetic calibration data")
            return None
        try:
            from transformers import AutoTokenizer  # type: ignore[import]
            tok = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            log.info("Loaded tokenizer from %s", self.tokenizer_dir)
            return tok
        except Exception as exc:
            log.warning("Could not load tokenizer (%s) — using synthetic data", exc)
            return None

    # --- data generation ---

    def _tokenize(self, prompt: str) -> dict[str, np.ndarray]:
        """Tokenize a single prompt → model input arrays."""
        if self._tokenizer is not None:
            enc = self._tokenizer(
                prompt,
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            return {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
        # Synthetic fallback: deterministic hash-based token IDs
        rng = np.random.default_rng(seed=hash(prompt) & 0xFFFFFFFF)
        ids = rng.integers(0, 32_000, size=(self.batch_size, self.seq_len), dtype=np.int64)
        mask = np.ones((self.batch_size, self.seq_len), dtype=np.int64)
        return {"input_ids": ids, "attention_mask": mask}

    def _build_inputs(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return only the subset of arrays that match ``input_names``."""
        return {k: v for k, v in data.items() if k in self.input_names}

    # --- CalibrationDataReader interface (called by onnxruntime) ---

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self._generate())
        try:
            return next(self._iter)
        except StopIteration:
            return None

    def _generate(self) -> Iterator[dict[str, np.ndarray]]:
        for prompt in self.prompts:
            data = self._tokenize(prompt)
            yield self._build_inputs(data)

    # Allow re-use
    def rewind(self) -> None:
        self._iter = None


# ── calibration dataset builder ───────────────────────────────────────────────

def build_calibration_dataset(
    output_dir: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
    seq_len: int = _DEFAULT_SEQ_LEN,
    extra_prompts: list[str] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Build and optionally save a list of calibration input dicts.

    Returns the dataset as a list so it can be inspected before being passed
    to :func:`~conversion.quantize_onnx.quantize_static_int8`.

    Parameters
    ----------
    output_dir:
        Directory to save ``calibration_data.npz`` (one file per prompt).
    tokenizer_dir:
        HuggingFace tokenizer directory.  None → synthetic data.
    seq_len:
        Sequence length for tokenization.
    extra_prompts:
        Additional prompts to include beyond the built-in set.

    Returns
    -------
    list of dicts mapping input name → numpy array.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = _CALIBRATION_PROMPTS[:]
    if extra_prompts:
        prompts.extend(extra_prompts)

    reader = DataBoltCalibrationDataReader(
        tokenizer_dir=tokenizer_dir,
        seq_len=seq_len,
        prompts=prompts,
    )

    dataset: list[dict[str, np.ndarray]] = []
    while True:
        sample = reader.get_next()
        if sample is None:
            break
        dataset.append(sample)

    # Save as .npz for reproducibility
    npz_path = output_dir / "calibration_data.npz"
    flat: dict[str, np.ndarray] = {}
    for i, sample in enumerate(dataset):
        for k, v in sample.items():
            flat[f"sample{i:03d}_{k}"] = v
    np.savez(npz_path, **flat)
    log.info("Saved %d calibration samples to %s", len(dataset), npz_path)

    return dataset
