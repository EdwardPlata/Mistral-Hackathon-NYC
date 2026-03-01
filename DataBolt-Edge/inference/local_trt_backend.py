"""Local TensorRT inference backend — runs a .engine file on the local GPU.

This backend is used when deploying DataBolt Edge on actual NVIDIA hardware
(RTX 3080+, Jetson Orin, etc.) after running build_trt_engine.py.

When TensorRT is not installed this module still imports cleanly;
is_available() returns False and the factory falls back to NvidiaAPIBackend.

Required env vars:
    TRT_ENGINE_PATH   — path to the .engine file
    TRT_TOKENIZER_DIR — path to the tokenizer directory (model weights folder)

Optional:
    TRT_MAX_INPUT_LEN  (default: 2048)
    TRT_MAX_OUTPUT_LEN (default: 1024)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

from .base import InferenceBackend, InferenceRequest, InferenceResponse

_TRT_AVAILABLE = False
try:
    import tensorrt as trt  # noqa: F401
    _TRT_AVAILABLE = True
except ImportError:
    pass


class LocalTRTBackend(InferenceBackend):
    """Inference via a locally built TensorRT-LLM .engine file."""

    def __init__(
        self,
        engine_path: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
    ) -> None:
        self._engine_path = Path(engine_path or os.environ.get("TRT_ENGINE_PATH", ""))
        self._tokenizer_dir = Path(tokenizer_dir or os.environ.get("TRT_TOKENIZER_DIR", ""))
        self._max_input_len = int(os.environ.get("TRT_MAX_INPUT_LEN", "2048"))
        self._max_output_len = int(os.environ.get("TRT_MAX_OUTPUT_LEN", "1024"))
        self._engine = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "local_trt"

    def is_available(self) -> bool:
        if not _TRT_AVAILABLE:
            return False
        if not self._engine_path.exists():
            return False
        if not self._tokenizer_dir.exists():
            return False
        return True

    def _load(self) -> None:
        """Lazy-load the TRT engine and tokenizer on first use."""
        if self._engine is not None:
            return

        import tensorrt as trt
        from transformers import AutoTokenizer

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self._engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._tokenizer_dir))

        # Load sidecar metadata if present
        meta_path = self._engine_path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._max_input_len = meta.get("max_input_len", self._max_input_len)
            self._max_output_len = meta.get("max_output_len", self._max_output_len)

    def _greedy_decode(self, input_ids: list[int]) -> list[int]:
        """Simple greedy decoding loop over the TRT engine."""
        import numpy as np
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("pycuda is required for local TRT inference") from exc

        generated = list(input_ids)
        for _ in range(self._max_output_len):
            seq = np.array([generated[-self._max_input_len:]], dtype=np.int32)
            d_input = cuda.mem_alloc(seq.nbytes)
            cuda.memcpy_htod(d_input, seq)
            # Allocate output buffer for logits [1, seq_len, vocab_size]
            vocab_size = self._engine.get_tensor_shape(
                self._engine.get_tensor_name(1)
            )[-1]
            logits = np.empty((1, seq.shape[1], vocab_size), dtype=np.float32)
            d_output = cuda.mem_alloc(logits.nbytes)
            self._context.execute_v2([int(d_input), int(d_output)])
            cuda.memcpy_dtoh(logits, d_output)
            next_token = int(np.argmax(logits[0, -1]))
            generated.append(next_token)
            eos_id = self._tokenizer.eos_token_id
            if eos_id is not None and next_token == eos_id:
                break

        return generated[len(input_ids):]

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        if not self.is_available():
            raise RuntimeError(
                "LocalTRTBackend is not available. "
                "Set TRT_ENGINE_PATH and TRT_TOKENIZER_DIR and ensure TensorRT is installed."
            )
        self._load()

        full_prompt = f"[INST] {request.prompt} [/INST]"
        if request.system_prompt:
            full_prompt = f"<<SYS>>\n{request.system_prompt}\n<</SYS>>\n\n{full_prompt}"

        input_ids = self._tokenizer.encode(full_prompt, return_tensors=None)
        t0 = perf_counter()
        output_ids = self._greedy_decode(input_ids)
        latency_ms = (perf_counter() - t0) * 1000

        text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return InferenceResponse(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=len(input_ids),
            completion_tokens=len(output_ids),
            backend=self.name,
        )
