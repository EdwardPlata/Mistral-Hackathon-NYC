"""DataBolt Edge — All-API Test Dashboard.

Tabs:
  Overview       — live status badges for every credential
  DataBolt       — core inference layer (Spark/Airflow/SQL log debugging)
  Log Parsers    — standalone Spark / Airflow / SQL plan parser (no inference)
  NVIDIA         — raw chat completion via nvidia_api_management
  Mistral AI     — chat completion via mistralai SDK
  ElevenLabs     — list voices + optional TTS preview
  W&B            — verify token & show viewer info
  Databricks     — validate PAT against a workspace URL
  HuggingFace    — whoami, model browser, dataset info via huggingface_hub
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DataBolt-Edge"))

load_dotenv(ROOT / ".env")

# ── imports (all optional — failures shown in UI) ───────────────────────────
try:
    from inference import get_backend, InferenceRequest
    INFERENCE_OK = True
except Exception as _e:
    INFERENCE_OK = False
    _INFERENCE_ERR = str(_e)

try:
    from nvidia_api_management import run_probe as nvidia_probe
    NVIDIA_OK = True
except Exception as _e:
    NVIDIA_OK = False
    _NVIDIA_ERR = str(_e)

try:
    from mistralai import Mistral
    MISTRAL_OK = True
except Exception as _e:
    MISTRAL_OK = False
    _MISTRAL_ERR = str(_e)

try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_OK = True
except Exception as _e:
    ELEVENLABS_OK = False
    _ELEVENLABS_ERR = str(_e)

try:
    import wandb  # noqa: F401
    WANDB_OK = True
except Exception as _e:
    WANDB_OK = False
    _WANDB_ERR = str(_e)

try:
    import requests as _requests  # noqa: F401
    REQUESTS_OK = True
except Exception as _e:
    REQUESTS_OK = False

try:
    from huggingface_integration import run_probe as hf_probe
    from huggingface_integration.client import HuggingFaceClient
    HF_OK = True
except Exception as _e:
    HF_OK = False
    _HF_ERR = str(_e)

try:
    from parsers import parse_spark, parse_airflow, parse_sql_plan
    PARSERS_OK = True
except Exception as _e:
    PARSERS_OK = False
    _PARSERS_ERR = str(_e)

try:
    from conversion import PipelineConfig, QuantizationPipeline
    from conversion.quantize_onnx import convert_to_fp16, quantize_dynamic_int8
    from conversion.benchmark import benchmark_model, compare_models
    CONVERSION_OK = True
except Exception as _e:
    CONVERSION_OK = False
    _CONVERSION_ERR = str(_e)

# ── sample logs ──────────────────────────────────────────────────────────────

_SPARK_SAMPLE = """\
24/02/28 14:33:01 ERROR Executor: Exception in task 3.0 in stage 5.0 (TID 47)
java.lang.OutOfMemoryError: GC overhead limit exceeded
    at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage2.processNext(Unknown Source)
    at org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)
    at org.apache.spark.sql.execution.WholeStageCodegenEvaluatorFactory...
24/02/28 14:33:01 WARN  TaskSetManager: Lost task 3.0 in stage 5.0 (TID 47, executor 2): TaskKilled (another attempt succeeded)
24/02/28 14:33:01 ERROR TaskSchedulerImpl: Lost executor 2 on 10.0.0.5: Remote RPC client disassociated.
"""

_AIRFLOW_SAMPLE = """\
[2024-02-28 09:15:03,412] {taskinstance.py:1482} ERROR - Failed to execute task
Traceback (most recent call last):
  File "/opt/airflow/dags/etl_pipeline.py", line 47, in extract_sales_data
    df = pd.read_csv('/data/input/sales_2024.csv')
  File "/usr/local/lib/python3.10/site-packages/pandas/io/parsers/readers.py", line 912, in read_csv
    return _read(filepath_or_buffer, kwds)
FileNotFoundError: [Errno 2] No such file or directory: '/data/input/sales_2024.csv'
[2024-02-28 09:15:03,415] {taskinstance.py:1492} ERROR - Task failed with exception
[2024-02-28 09:15:03,416] {local_task_job.py:156} ERROR - Task exited with return code 1
"""

_SQL_SAMPLE = """\
EXPLAIN SELECT o.order_id, c.name, SUM(oi.price) as total
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.created_at > '2024-01-01'
GROUP BY o.order_id, c.name;

-> Table scan on orders  (cost=521432.50 rows=4983201)
-> Hash join (cost=1043215.00 rows=4983201)
    -> Table scan on customers  (cost=12450.00 rows=124500)
    -> Table scan on order_items  (cost=2987451.00 rows=28932847)
Filter: (o.created_at > '2024-01-01')  -- no index on created_at
"""

_SAMPLE_LOGS = {
    "Spark — OOM / executor failure": _SPARK_SAMPLE,
    "Airflow — FileNotFoundError": _AIRFLOW_SAMPLE,
    "SQL — slow query / full table scan": _SQL_SAMPLE,
    "Custom": "",
}

_DEFAULT_QUESTIONS = {
    "Spark — OOM / executor failure": "What caused this failure and what are the top 3 fixes?",
    "Airflow — FileNotFoundError": "Why did this task fail and how do I fix it?",
    "SQL — slow query / full table scan": "Why is this query slow and how do I optimize it?",
    "Custom": "",
}

# ── credential helpers ───────────────────────────────────────────────────────

def _env(key: str) -> str | None:
    return os.environ.get(key) or None


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataBolt Edge — API Tester",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ DataBolt Edge — API Test Dashboard")
st.caption("Verify and test every API credential in one place.")

tabs = st.tabs([
    "🏠 Overview",
    "⚡ DataBolt",
    "📝 Log Parsers",
    "🟩 NVIDIA",
    "✨ Mistral AI",
    "🎙 ElevenLabs",
    "📊 W&B",
    "🧱 Databricks",
    "🤗 HuggingFace",
    "🔧 Model Pipeline",
])

# ════════════════════════════════════════════════════════════════════════════
# Tab 0 — Overview
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.subheader("Credential Status")
    st.caption("Green = env var is set.  Click a tab to run a live test.")

    cred_rows = [
        ("NVIDIA", "NVIDIA_BEARER_TOKEN"),
        ("Mistral AI", "MISTRAL_API_KEY"),
        ("ElevenLabs", "ELEVENLABS_API_KEY"),
        ("W&B", "WANDB_API_KEY"),
        ("Databricks", "DATABRICKS_PAT"),
        ("HuggingFace", "HUGGINGFACE_TOKEN"),
    ]

    cols = st.columns(len(cred_rows))
    for col, (label, var) in zip(cols, cred_rows):
        val = _env(var)
        icon = "🟢" if val else "🔴"
        col.metric(label=label, value=icon + (" Set" if val else " Missing"))
        if val:
            col.caption(f"`{var[:20]}…`")

    st.divider()
    st.subheader("SDK Availability")
    sdk_cols = st.columns(7)
    sdk_cols[0].metric("inference layer", "✅" if INFERENCE_OK else "❌")
    sdk_cols[1].metric("log parsers", "✅" if PARSERS_OK else "❌")
    sdk_cols[2].metric("conversion pipeline", "✅" if CONVERSION_OK else "❌")
    sdk_cols[3].metric("nvidia_api_management", "✅" if NVIDIA_OK else "❌")
    sdk_cols[4].metric("mistralai", "✅" if MISTRAL_OK else "❌")
    sdk_cols[5].metric("elevenlabs", "✅" if ELEVENLABS_OK else "❌")
    sdk_cols[6].metric("huggingface_hub", "✅" if HF_OK else "❌")

    st.divider()
    st.subheader("Port Map")
    st.markdown("""
| Service | Port | Command |
|---|---|---|
| **This UI** (DataBolt API Tester) | **8501** | `streamlit run UI/streamlit_test_ui.py` |
| **AgentOp-Studio Frontend** | **8502** | `streamlit run AgentOp-Studio/frontend/app.py --server.port 8502` |
| **AgentOp-Studio Backend** | **8000** | `uvicorn backend.main:app --port 8000` |
""")

# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — DataBolt (core inference + parser enrichment)
# ════════════════════════════════════════════════════════════════════════════

# Map log type label → parser function (None = no parser for this type)
_LOG_TYPE_PARSERS = {
    "Spark — OOM / executor failure": "spark",
    "Airflow — FileNotFoundError": "airflow",
    "SQL — slow query / full table scan": "sql",
    "Custom": None,
}

with tabs[1]:
    st.subheader("DataBolt Edge — Log Debugging")
    st.caption(
        "Paste a Spark, Airflow, or SQL log and ask a question. "
        "The log is first **parsed** for structured context, then forwarded to the inference layer."
    )

    if not INFERENCE_OK:
        st.error(f"Inference layer not loaded: {_INFERENCE_ERR}")
        st.stop()

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        log_type = st.selectbox(
            "Log type / scenario",
            list(_SAMPLE_LOGS.keys()),
            key="db_log_type",
        )
        log_input = st.text_area(
            "Log / query (paste or edit sample below)",
            value=_SAMPLE_LOGS[log_type],
            height=220,
            key="db_log_input",
        )
        question = st.text_input(
            "Your question",
            value=_DEFAULT_QUESTIONS[log_type],
            key="db_question",
        )
        with st.expander("Advanced options"):
            db_max_tokens = st.slider("Max tokens", 128, 1024, 512, 64, key="db_max_tokens")
            db_temp = st.slider("Temperature", 0.0, 1.0, 0.15, 0.05, key="db_temp")
            db_backend = st.selectbox(
                "Backend override",
                ["auto", "nvidia_api", "local_trt"],
                key="db_backend",
            )
            db_use_parser = st.toggle(
                "Enrich prompt with parser context",
                value=True,
                key="db_use_parser",
                help="Runs the log parser first and injects structured error info into the prompt.",
            )

        run_db = st.button("🔍 Analyse Log", type="primary", use_container_width=True, key="btn_db_infer")

    with col_result:
        if run_db:
            if not log_input.strip():
                st.warning("Paste a log or select a sample scenario first.")
            elif not question.strip():
                st.warning("Enter a question about the log.")
            else:
                # ── Step 1: parse the log (if parsers available and type known) ──
                parser_key = _LOG_TYPE_PARSERS.get(log_type)
                parse_ctx = ""
                parse_result = None

                if db_use_parser and PARSERS_OK and parser_key:
                    try:
                        if parser_key == "spark":
                            parse_result = parse_spark(log_input)
                            parse_ctx = parse_result.to_prompt_context()
                        elif parser_key == "airflow":
                            parse_result = parse_airflow(log_input)
                            parse_ctx = parse_result.to_prompt_context()
                        elif parser_key == "sql":
                            parse_result = parse_sql_plan(log_input)
                            parse_ctx = parse_result.to_prompt_context()
                    except Exception as _pe:
                        st.warning(f"Parser error (continuing without context): {_pe}")

                # ── show parsed summary ────────────────────────────────────────
                if parse_result is not None:
                    with st.expander("🔬 Parser output", expanded=True):
                        if parser_key == "spark" and parse_result is not None:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("ERRORs", parse_result.error_count)
                            c2.metric("WARNs", parse_result.warn_count)
                            c3.metric("OOM?", "Yes" if parse_result.has_oom else "No")
                            if parse_result.categories:
                                st.write("**Categories:**", ", ".join(parse_result.categories))
                            if parse_result.exceptions:
                                st.write("**Exceptions:**", ", ".join(parse_result.exceptions[:4]))
                            if parse_result.stages:
                                st.write("**Stages:**", ", ".join(sorted(set(parse_result.stages))[:5]))

                        elif parser_key == "airflow" and parse_result is not None:
                            c1, c2 = st.columns(2)
                            c1.metric("ERRORs", parse_result.error_count)
                            c2.metric("Tracebacks", len(parse_result.tracebacks))
                            if parse_result.dag_id:
                                st.write(f"**DAG:** `{parse_result.dag_id}` / **Task:** `{parse_result.task_id}`")
                            if parse_result.categories:
                                st.write("**Categories:**", ", ".join(parse_result.categories))
                            if parse_result.exception_classes:
                                st.write("**Exception types:**", ", ".join(parse_result.exception_classes[:4]))

                        elif parser_key == "sql" and parse_result is not None:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Full scans", sum(1 for s in parse_result.scans if s.is_full_scan))
                            c2.metric("Max rows", f"{parse_result.max_rows:,}" if parse_result.max_rows else "—")
                            c3.metric("Est. cost", f"{parse_result.total_cost:,.0f}" if parse_result.total_cost else "—")
                            if parse_result.warnings:
                                for w in parse_result.warnings:
                                    st.warning(w)

                # ── Step 2: build enriched prompt ─────────────────────────────
                if parse_ctx:
                    prompt = (
                        f"Structured analysis from log parser:\n{parse_ctx}\n\n"
                        f"Raw log output:\n```\n{log_input.strip()}\n```\n\n"
                        f"{question.strip()}"
                    )
                else:
                    prompt = f"Log output:\n```\n{log_input.strip()}\n```\n\n{question.strip()}"

                # ── Step 3: run inference ──────────────────────────────────────
                with st.spinner("Running inference…"):
                    try:
                        backend = get_backend(db_backend)
                        req = InferenceRequest(
                            prompt=prompt,
                            max_tokens=db_max_tokens,
                            temperature=db_temp,
                        )
                        resp = backend.generate(req)

                        st.success(f"✅  {backend.name}  |  {resp.latency_ms:.0f} ms")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Backend", resp.backend)
                        m2.metric("Prompt tokens", resp.prompt_tokens or "—")
                        m3.metric("Completion tokens", resp.completion_tokens or "—")

                        st.divider()
                        st.markdown("**Analysis:**")
                        st.markdown(resp.text)

                        with st.expander("Request details"):
                            st.code(prompt, language="text")

                    except RuntimeError as exc:
                        st.error(f"❌  {exc}")
                    except Exception as exc:
                        st.error(f"❌  Unexpected error: {exc}")
                        st.exception(exc)
        else:
            st.info("Select a scenario, review the log, enter your question, then click **Analyse Log**.")
            with st.expander("How it works"):
                st.markdown("""
The **DataBolt** tab routes requests through two layers:

```
Log text
   ↓
parsers/  ← extracts categories, exceptions, stack frames, cost estimates
   ↓
InferenceRequest (enriched prompt + system prompt)
   ↓
  get_backend('auto')
   ↓
  NvidiaAPIBackend  ← default when NVIDIA_BEARER_TOKEN is set
  LocalTRTBackend   ← used when TRT_ENGINE_PATH + GPU are available
   ↓
  InferenceResponse (text, latency_ms, tokens, backend)
```

The **parser context** is injected before the raw log so the model receives
structured signals (error category, exception type, affected stages) alongside
the full log text. Toggle **Enrich prompt with parser context** in *Advanced
options* to disable this.
                """)

# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Log Parsers (standalone, no inference)
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("Log Parsers — Standalone")
    st.caption(
        "Parse a Spark, Airflow, or SQL EXPLAIN log without running inference. "
        "Useful for verifying the parser output before using it in the DataBolt tab."
    )

    if not PARSERS_OK:
        st.error(f"Log parser package not loaded: {_PARSERS_ERR}")
        st.stop()

    p_col_cfg, p_col_result = st.columns([1, 2])

    with p_col_cfg:
        p_log_type = st.selectbox(
            "Parser",
            ["Spark", "Airflow", "SQL Plan"],
            key="p_log_type",
        )

        _parser_samples = {
            "Spark": _SPARK_SAMPLE,
            "Airflow": _AIRFLOW_SAMPLE,
            "SQL Plan": _SQL_SAMPLE,
        }
        p_log_input = st.text_area(
            "Log text",
            value=_parser_samples[p_log_type],
            height=300,
            key="p_log_input",
        )
        run_parser = st.button("🔬 Parse Log", type="primary", use_container_width=True, key="btn_parser")

    with p_col_result:
        if run_parser:
            if not p_log_input.strip():
                st.warning("Paste a log first.")
            else:
                try:
                    if p_log_type == "Spark":
                        r = parse_spark(p_log_input)
                        st.success("✅  Spark log parsed")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("ERROR lines", r.error_count)
                        c2.metric("WARN lines", r.warn_count)
                        c3.metric("OOM detected", "Yes" if r.has_oom else "No")
                        if r.categories:
                            st.write("**Error categories:**", ", ".join(r.categories))
                        if r.exceptions:
                            st.write("**Exceptions:**", ", ".join(r.exceptions[:6]))
                        if r.stages:
                            st.write("**Stages affected:**", ", ".join(sorted(set(r.stages))[:8]))
                        if r.tasks:
                            st.write("**Tasks affected:**", ", ".join(sorted(set(r.tasks))[:8]))
                        if r.executors:
                            st.write("**Executors:**", ", ".join(sorted(set(r.executors))))
                        if r.first_exception_block:
                            with st.expander("First exception block"):
                                st.code(r.first_exception_block, language="java")
                        with st.expander("Prompt context (injected into inference)"):
                            st.code(r.to_prompt_context(), language="text")

                    elif p_log_type == "Airflow":
                        r = parse_airflow(p_log_input)
                        st.success("✅  Airflow log parsed")
                        c1, c2 = st.columns(2)
                        c1.metric("ERROR lines", r.error_count)
                        c2.metric("Tracebacks", len(r.tracebacks))
                        if r.dag_id:
                            st.write(f"**DAG:** `{r.dag_id}`  **Task:** `{r.task_id}`  **Exec:** `{r.execution_date}`")
                        if r.categories:
                            st.write("**Error categories:**", ", ".join(r.categories))
                        if r.exception_classes:
                            st.write("**Exception types:**", ", ".join(r.exception_classes))
                        for i, tb in enumerate(r.tracebacks):
                            with st.expander(f"Traceback {i + 1}: {tb.exception_class}"):
                                st.code(tb.raw, language="python")
                        with st.expander("Prompt context"):
                            st.code(r.to_prompt_context(), language="text")

                    elif p_log_type == "SQL Plan":
                        r = parse_sql_plan(p_log_input)
                        st.success("✅  SQL plan parsed")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Full scans", sum(1 for s in r.scans if s.is_full_scan))
                        c2.metric("Max rows", f"{r.max_rows:,}" if r.max_rows else "—")
                        c3.metric("Est. cost", f"{r.total_cost:,.0f}" if r.total_cost else "—")
                        flags = []
                        if r.has_full_table_scan: flags.append("Full table scan")
                        if r.has_cartesian_join: flags.append("Cartesian join")
                        if r.has_filesort: flags.append("Filesort")
                        if r.has_temp_table: flags.append("Temp table")
                        if r.has_no_index_filter: flags.append("Filter without index")
                        if flags:
                            st.write("**Flags:**", ", ".join(flags))
                        if r.warnings:
                            st.write("**Warnings:**")
                            for w in r.warnings:
                                st.warning(w)
                        if r.scans:
                            import pandas as pd
                            scan_rows = [
                                {"Table": s.table, "Full scan": s.is_full_scan,
                                 "Rows": f"{s.rows:,}", "Cost": f"{s.cost:,.0f}"}
                                for s in r.scans
                            ]
                            st.dataframe(pd.DataFrame(scan_rows), use_container_width=True)
                        with st.expander("Prompt context"):
                            st.code(r.to_prompt_context(), language="text")

                except Exception as exc:
                    st.error(f"❌  {exc}")
                    st.exception(exc)
        else:
            st.info("Select a parser, review the sample log, then click **Parse Log**.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — NVIDIA
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.subheader("NVIDIA Inference API")

    if not NVIDIA_OK:
        st.error(f"nvidia_api_management module not loaded: {_NVIDIA_ERR}")
        st.stop()

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        nvidia_key = st.text_input(
            "Bearer Token (optional override)",
            type="password",
            placeholder="Leave empty → uses NVIDIA_BEARER_TOKEN env var",
            key="nvidia_key",
        )
        nvidia_model = st.text_input(
            "Model",
            value="mistralai/mistral-large-3-675b-instruct-2512",
            key="nvidia_model",
        )
        nvidia_prompt_choice = st.selectbox(
            "Preset prompt",
            ["Custom", "What is the capital of France?",
             "Describe DataBolt Edge in one sentence.",
             "List three NVIDIA GPU families."],
            key="nvidia_preset",
        )
        if nvidia_prompt_choice == "Custom":
            nvidia_prompt = st.text_area("Prompt", value="Hello!", height=80, key="nvidia_custom_prompt")
        else:
            nvidia_prompt = nvidia_prompt_choice
        run_nvidia = st.button("🚀 Run NVIDIA Test", type="primary", key="btn_nvidia")

    with col_result:
        if run_nvidia:
            with st.spinner("Calling NVIDIA API…"):
                result = nvidia_probe(
                    content=nvidia_prompt,
                    model=nvidia_model or None,
                    api_key=nvidia_key or None,
                )

            if result.success:
                st.success(f"✅  Success  |  {result.latency_ms:.0f} ms")
                choices = (result.response or {}).get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    st.info(content)
                with st.expander("Full JSON response"):
                    st.json(result.response)
            else:
                st.error(f"❌  Failed  |  HTTP {result.status_code}")
                st.code(result.error)
        else:
            st.info("Configure and click **Run NVIDIA Test**.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 4 — Mistral AI
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("Mistral AI API")

    if not MISTRAL_OK:
        st.error(f"mistralai SDK not loaded: {_MISTRAL_ERR}")
        st.stop()

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        mistral_key = st.text_input(
            "API Key (optional override)",
            type="password",
            placeholder="Leave empty → uses MISTRAL_API_KEY env var",
            key="mistral_key",
        )
        mistral_model = st.selectbox(
            "Model",
            ["mistral-large-latest", "mistral-medium-latest",
             "mistral-small-latest", "open-mistral-7b"],
            key="mistral_model",
        )
        mistral_temp = st.slider("Temperature", 0.0, 1.5, 0.6, 0.05, key="mistral_temp")
        mistral_prompt_choice = st.selectbox(
            "Preset prompt",
            ["Custom", "What is the capital of France?",
             "Explain neural networks in 2 sentences.",
             "What are the main benefits of edge computing?"],
            key="mistral_preset",
        )
        if mistral_prompt_choice == "Custom":
            mistral_prompt = st.text_area("Prompt", value="Hello!", height=80, key="mistral_custom")
        else:
            mistral_prompt = mistral_prompt_choice

        run_mistral = st.button("🚀 Run Mistral Test", type="primary", key="btn_mistral")

    with col_result:
        if run_mistral:
            api_key = mistral_key or _env("MISTRAL_API_KEY")
            if not api_key:
                st.error("MISTRAL_API_KEY not set.")
            else:
                with st.spinner("Calling Mistral AI…"):
                    t0 = time.perf_counter()
                    try:
                        from mistralai import models as _mistral_models
                        client = Mistral(api_key=api_key)
                        resp = client.chat.complete(
                            model=mistral_model,
                            messages=[{"role": "user", "content": mistral_prompt}],
                            temperature=mistral_temp,
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000
                        answer = resp.choices[0].message.content
                        usage = resp.usage
                        st.success(f"✅  Success  |  {latency_ms:.0f} ms")
                        st.info(answer)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Prompt tokens", usage.prompt_tokens)
                        m2.metric("Completion tokens", usage.completion_tokens)
                        m3.metric("Total tokens", usage.total_tokens)
                    except _mistral_models.SDKError as exc:
                        latency_ms = (time.perf_counter() - t0) * 1000
                        if exc.status_code == 401:
                            st.error("❌  **401 Unauthorized** — your `MISTRAL_API_KEY` is expired or invalid.")
                            st.info("Regenerate your key at [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys/) and update your Codespace secret, then restart the Codespace.")
                        elif exc.status_code == 429:
                            st.error("❌  **429 Rate Limited** — too many requests. Wait a moment and retry.")
                        else:
                            st.error(f"❌  HTTP {exc.status_code} — {exc.message}")
                    except Exception as exc:
                        st.error(f"❌  {exc}")
        else:
            st.info("Configure and click **Run Mistral Test**.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 5 — ElevenLabs
# ════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader("ElevenLabs API")

    if not ELEVENLABS_OK:
        st.error(f"elevenlabs SDK not loaded: {_ELEVENLABS_ERR}")
        st.stop()

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        el_key = st.text_input(
            "API Key (optional override)",
            type="password",
            placeholder="Leave empty → uses ELEVENLABS_API_KEY env var",
            key="el_key",
        )
        el_action = st.radio(
            "Action",
            ["List voices", "Text-to-Speech preview"],
            key="el_action",
        )
        if el_action == "Text-to-Speech preview":
            el_text = st.text_area(
                "Text to speak",
                value="Hello! DataBolt Edge is running.",
                height=80,
                key="el_text",
            )
            el_voice = st.text_input(
                "Voice ID or name",
                value="Rachel",
                key="el_voice",
            )
        run_el = st.button("🚀 Run ElevenLabs Test", type="primary", key="btn_el")

    with col_result:
        if run_el:
            api_key = el_key or _env("ELEVENLABS_API_KEY")
            if not api_key:
                st.error("ELEVENLABS_API_KEY not set.")
            else:
                with st.spinner("Calling ElevenLabs…"):
                    t0 = time.perf_counter()
                    try:
                        client = ElevenLabs(api_key=api_key)
                        if el_action == "List voices":
                            voices_resp = client.voices.get_all()
                            latency_ms = (time.perf_counter() - t0) * 1000
                            st.success(f"✅  Success  |  {latency_ms:.0f} ms")
                            voices = voices_resp.voices
                            st.metric("Voices available", len(voices))
                            import pandas as pd
                            rows = [
                                {"Name": v.name, "Voice ID": v.voice_id,
                                 "Category": getattr(v, "category", "—")}
                                for v in voices[:15]
                            ]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        else:
                            audio_gen = client.text_to_speech.convert(
                                voice_id=el_voice,
                                text=el_text,
                                model_id="eleven_multilingual_v2",
                            )
                            audio_bytes = b"".join(audio_gen)
                            latency_ms = (time.perf_counter() - t0) * 1000
                            st.success(f"✅  Success  |  {latency_ms:.0f} ms")
                            st.audio(audio_bytes, format="audio/mp3")
                    except Exception as exc:
                        st.error(f"❌  {exc}")
        else:
            st.info("Configure and click **Run ElevenLabs Test**.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 6 — W&B
# ════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.subheader("Weights & Biases")

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        wb_key = st.text_input(
            "API Key (optional override)",
            type="password",
            placeholder="Leave empty → uses WANDB_API_KEY env var",
            key="wb_key",
        )
        run_wb = st.button("🚀 Verify W&B Token", type="primary", key="btn_wb")

    with col_result:
        if run_wb:
            api_key = wb_key or _env("WANDB_API_KEY")
            if not api_key:
                st.error("WANDB_API_KEY not set.")
            else:
                with st.spinner("Calling W&B API…"):
                    t0 = time.perf_counter()
                    try:
                        import requests as req
                        query = '{"query": "{ viewer { username email entity } }"}'
                        r = req.post(
                            "https://api.wandb.ai/graphql",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            data=query,
                            timeout=15,
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000
                        if r.ok:
                            data = r.json().get("data", {}).get("viewer", {})
                            st.success(f"✅  Success  |  {latency_ms:.0f} ms")
                            m1, m2 = st.columns(2)
                            m1.metric("Username", data.get("username", "—"))
                            m2.metric("Entity", data.get("entity", "—"))
                            st.caption(f"Email: {data.get('email', '—')}")
                        else:
                            st.error(f"❌  HTTP {r.status_code}: {r.text[:200]}")
                    except Exception as exc:
                        st.error(f"❌  {exc}")
        else:
            st.info("Click **Verify W&B Token** to validate your key.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 7 — Databricks
# ════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    st.subheader("Databricks PAT")

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        db_pat = st.text_input(
            "Personal Access Token (optional override)",
            type="password",
            placeholder="Leave empty → uses DATABRICKS_PAT env var",
            key="db_pat",
        )
        db_host = st.text_input(
            "Workspace URL",
            placeholder="https://your-workspace.azuredatabricks.net",
            key="db_host",
        )
        run_dbricks = st.button("🚀 Verify Databricks PAT", type="primary", key="btn_dbricks")

    with col_result:
        if run_dbricks:
            pat = db_pat or _env("DATABRICKS_PAT")
            host = db_host.rstrip("/") if db_host else None
            if not pat:
                st.error("DATABRICKS_PAT not set.")
            elif not host:
                st.warning("Enter your Databricks workspace URL to run a live test.")
                st.info(f"Token is **set** ({len(pat)} chars). Provide workspace URL to verify against the API.")
            else:
                with st.spinner("Calling Databricks API…"):
                    t0 = time.perf_counter()
                    try:
                        import requests as req
                        r = req.get(
                            f"{host}/api/2.0/token/list",
                            headers={"Authorization": f"Bearer {pat}"},
                            timeout=15,
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000
                        if r.ok:
                            tokens = r.json().get("token_infos", [])
                            st.success(f"✅  Auth OK  |  {latency_ms:.0f} ms")
                            st.metric("Active tokens on workspace", len(tokens))
                        elif r.status_code == 403:
                            st.success(f"✅  Token accepted (HTTP 403 = valid auth, insufficient scope)  |  {latency_ms:.0f} ms")
                        elif r.status_code == 401:
                            st.error("❌  HTTP 401 — token rejected by workspace.")
                        else:
                            st.warning(f"HTTP {r.status_code}: {r.text[:300]}")
                    except Exception as exc:
                        st.error(f"❌  {exc}")
        else:
            st.info("Enter your workspace URL and click **Verify Databricks PAT**.")
            st.caption("Token scope note: `/api/2.0/token/list` requires admin or token management permission. A 403 still confirms the token is valid.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 8 — HuggingFace
# ════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    st.subheader("HuggingFace Hub")

    if not HF_OK:
        st.error(f"huggingface_integration not loaded: {_HF_ERR}")
        st.stop()

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        hf_token = st.text_input(
            "Token (optional override)",
            type="password",
            placeholder="Leave empty → uses HUGGINGFACE_TOKEN env var",
            key="hf_token",
        )
        hf_action = st.radio(
            "Action",
            ["Verify token (whoami)", "Browse Mistral models", "Model details", "Dataset details"],
            key="hf_action",
        )
        if hf_action == "Model details":
            hf_model_id = st.text_input(
                "Model ID",
                value="mistralai/Mistral-7B-Instruct-v0.1",
                key="hf_model_id",
            )
        if hf_action == "Dataset details":
            hf_dataset_id = st.text_input(
                "Dataset ID",
                value="mistralai/FineTome-100k",
                key="hf_dataset_id",
            )
        if hf_action == "Browse Mistral models":
            hf_limit = st.slider("Max results", 5, 30, 10, key="hf_limit")
            hf_author = st.text_input("Author filter", value="mistralai", key="hf_author")
        run_hf = st.button("🚀 Run HuggingFace Test", type="primary", key="btn_hf")

    with col_result:
        if run_hf:
            token = hf_token or _env("HUGGINGFACE_TOKEN") or _env("HF_TOKEN")
            if not token:
                st.error("No HuggingFace token found. Set HUGGINGFACE_TOKEN or HF_TOKEN.")
            else:
                with st.spinner("Calling HuggingFace Hub…"):
                    t0 = time.perf_counter()
                    try:
                        import pandas as pd
                        client = HuggingFaceClient()

                        if hf_action == "Verify token (whoami)":
                            result = hf_probe(token=token)
                            if result.success:
                                st.success(f"✅  Authenticated  |  {result.latency_ms:.0f} ms")
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Username", result.username or "—")
                                m2.metric("Email", result.email or "—")
                                m3.metric("Plan", result.plan or "—")
                            else:
                                st.error(f"❌  {result.error}  (HTTP {result.status_code})")

                        elif hf_action == "Browse Mistral models":
                            models = client.list_models(
                                author=hf_author,
                                pipeline_tag="text-generation",
                                limit=hf_limit,
                                token=token,
                            )
                            latency_ms = (time.perf_counter() - t0) * 1000
                            st.success(f"✅  {len(models)} models  |  {latency_ms:.0f} ms")
                            rows = [
                                {
                                    "Model": m.model_id,
                                    "Pipeline": m.pipeline_tag or "—",
                                    "Downloads": f"{m.downloads:,}",
                                    "Likes": m.likes,
                                    "Tags": ", ".join(m.tags[:4]),
                                }
                                for m in models
                            ]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True)

                        elif hf_action == "Model details":
                            info = client.model_info(hf_model_id, token=token)
                            latency_ms = (time.perf_counter() - t0) * 1000
                            st.success(f"✅  {info['id']}  |  {latency_ms:.0f} ms")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Downloads", f"{info.get('downloads', 0):,}")
                            m2.metric("Likes", info.get("likes", 0))
                            m3.metric("Gated", "Yes" if info.get("gated") else "No")
                            st.caption(f"Pipeline: {info.get('pipeline_tag', '—')}  |  Private: {info.get('private', False)}")
                            tags = info.get("tags", [])
                            if tags:
                                st.write("**Tags:**", ", ".join(tags[:10]))
                            with st.expander("Full metadata"):
                                st.json(info)

                        elif hf_action == "Dataset details":
                            info = client.dataset_info(hf_dataset_id, token=token)
                            latency_ms = (time.perf_counter() - t0) * 1000
                            st.success(f"✅  {info['id']}  |  {latency_ms:.0f} ms")
                            m1, m2 = st.columns(2)
                            m1.metric("Downloads", f"{info.get('downloads', 0):,}")
                            m2.metric("Likes", info.get("likes", 0))
                            with st.expander("Full metadata"):
                                st.json(info)

                    except Exception as exc:
                        st.error(f"❌  {exc}")
        else:
            st.info("Select an action and click **Run HuggingFace Test**.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 9 — Model Pipeline (conversion & quantization)
# ════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    st.subheader("Model Conversion & Quantization Pipeline")
    st.caption(
        "Convert a float32 ONNX model to FP16 and INT8 variants, benchmark all "
        "produced models, and view a side-by-side comparison report."
    )

    if not CONVERSION_OK:
        st.error(f"Conversion package not loaded: {_CONVERSION_ERR}")
        st.stop()

    mp_col_cfg, mp_col_result = st.columns([1, 2])

    with mp_col_cfg:
        st.markdown("#### Source model")
        mp_source = st.text_input(
            "ONNX model path",
            value="DataBolt-Edge/models/mistral-7b-instruct/model.onnx",
            key="mp_source",
            help="Path to a float32 ONNX file produced by scripts/export_to_onnx.py",
        )
        mp_output_dir = st.text_input(
            "Output directory",
            value="DataBolt-Edge/models/quantized",
            key="mp_output_dir",
        )
        mp_tokenizer = st.text_input(
            "Tokenizer directory (optional)",
            value="",
            placeholder="Leave empty → synthetic calibration data",
            key="mp_tokenizer",
        )

        st.markdown("#### Steps to run")
        mp_fp16 = st.checkbox("FP16 conversion", value=True, key="mp_fp16")
        mp_dyn_int8 = st.checkbox("Dynamic INT8", value=True, key="mp_dyn_int8")
        mp_static_int8 = st.checkbox("Static INT8 (needs calibration)", value=False, key="mp_static_int8")
        mp_benchmark = st.checkbox("Benchmark all variants", value=True, key="mp_benchmark")

        with st.expander("Advanced options"):
            mp_bench_runs = st.slider("Benchmark runs", 5, 100, 20, 5, key="mp_bench_runs")
            mp_bench_seq = st.slider("Benchmark seq len", 16, 512, 64, 16, key="mp_bench_seq")
            mp_weight_type = st.radio("INT8 weight type", ["QInt8", "QUInt8"], key="mp_weight_type")
            mp_per_channel = st.checkbox("Per-channel quantization", value=False, key="mp_per_channel")

        # ── single-step tools ──────────────────────────────────────────────
        st.markdown("#### Quick tools (no full pipeline)")
        st.caption("Run individual steps on an existing ONNX file:")
        mp_single_source = st.text_input(
            "ONNX path for quick tool",
            value=mp_source,
            key="mp_single_source",
        )
        mp_single_out = st.text_input(
            "Output path",
            value="DataBolt-Edge/models/quick_output.onnx",
            key="mp_single_out",
        )
        mp_tool = st.selectbox(
            "Tool",
            ["FP16 convert", "Dynamic INT8", "Benchmark (10 runs)"],
            key="mp_tool",
        )
        run_single = st.button("▶ Run", key="mp_run_single")

        st.divider()
        run_pipeline = st.button(
            "🚀 Run Full Pipeline",
            type="primary",
            use_container_width=True,
            key="mp_run_pipeline",
        )

    with mp_col_result:

        # ── single-step tool ───────────────────────────────────────────────
        if run_single:
            src = mp_single_source.strip()
            out = mp_single_out.strip()
            if not src:
                st.warning("Enter an ONNX path for the quick tool.")
            else:
                from pathlib import Path as _Path
                with st.spinner(f"Running {mp_tool} …"):
                    try:
                        if mp_tool == "FP16 convert":
                            r = convert_to_fp16(_Path(src), _Path(out))
                            if r.success:
                                st.success(f"✅ FP16 — {r.source_size_mb:.1f} MB → {r.output_size_mb:.1f} MB  [{r.duration_s:.1f}s]")
                                st.caption(f"Saved: {r.output_path}")
                            else:
                                st.error(f"❌ {r.error}")

                        elif mp_tool == "Dynamic INT8":
                            r = quantize_dynamic_int8(_Path(src), _Path(out))
                            if r.success:
                                st.success(f"✅ Dynamic INT8 — {r.source_size_mb:.1f} MB → {r.output_size_mb:.1f} MB  [{r.duration_s:.1f}s]")
                                st.caption(f"Saved: {r.output_path}")
                            else:
                                st.error(f"❌ {r.error}")

                        elif mp_tool == "Benchmark (10 runs)":
                            r = benchmark_model(_Path(src), n_runs=10)
                            if r.success:
                                st.success(f"✅ Benchmark — mean {r.mean_ms:.1f} ms  TPS {r.tokens_per_second:.1f}")
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Mean (ms)", f"{r.mean_ms:.1f}")
                                c2.metric("P50 (ms)", f"{r.median_ms:.1f}")
                                c3.metric("P95 (ms)", f"{r.p95_ms:.1f}")
                                c4.metric("TPS", f"{r.tokens_per_second:.1f}")
                            else:
                                st.error(f"❌ {r.error}")
                    except Exception as exc:
                        st.error(f"❌ {exc}")
                        st.exception(exc)

        # ── full pipeline ──────────────────────────────────────────────────
        elif run_pipeline:
            if not mp_source.strip():
                st.warning("Enter a source ONNX path.")
            else:
                from pathlib import Path as _Path
                cfg = PipelineConfig(
                    source_onnx=mp_source.strip(),
                    output_dir=mp_output_dir.strip(),
                    tokenizer_dir=mp_tokenizer.strip() or None,
                    run_fp16=mp_fp16,
                    run_dynamic_int8=mp_dyn_int8,
                    run_static_int8=mp_static_int8,
                    run_benchmark=mp_benchmark,
                    benchmark_n_runs=mp_bench_runs,
                    benchmark_seq_len=mp_bench_seq,
                    int8_weight_type=mp_weight_type,
                    int8_per_channel=mp_per_channel,
                )

                progress_placeholder = st.empty()
                steps_done = []

                with st.spinner("Running quantization pipeline…"):
                    try:
                        pipeline = QuantizationPipeline(cfg)
                        report = pipeline.run()
                    except Exception as exc:
                        st.error(f"❌ Pipeline error: {exc}")
                        st.exception(exc)
                        report = None

                if report is not None:
                    # Step summary
                    st.markdown("**Pipeline steps:**")
                    for o in report.outcomes:
                        icon = "✅" if o.success else "❌"
                        st.write(f"{icon} `{o.step.name}` — {o.message[:80]}  `{o.duration_s:.1f}s`")

                    # Quantization results
                    if report.quant_results:
                        st.markdown("**Quantization results:**")
                        import pandas as pd
                        qrows = []
                        for r in report.quant_results:
                            if r.success:
                                qrows.append({
                                    "Variant": r.variant,
                                    "Source MB": f"{r.source_size_mb:.1f}",
                                    "Output MB": f"{r.output_size_mb:.1f}",
                                    "Ratio": f"{r.compression_ratio:.2f}x",
                                    "Time (s)": f"{r.duration_s:.1f}",
                                    "Status": "✅",
                                })
                            else:
                                qrows.append({
                                    "Variant": r.variant,
                                    "Source MB": "—", "Output MB": "—",
                                    "Ratio": "—", "Time (s)": f"{r.duration_s:.1f}",
                                    "Status": f"❌ {r.error[:40]}",
                                })
                        st.dataframe(pd.DataFrame(qrows), use_container_width=True)

                    # Benchmark results
                    if report.bench_results:
                        st.markdown("**Benchmark results:**")
                        bench_df = report.to_dataframe()
                        if not bench_df.empty:
                            st.dataframe(bench_df, use_container_width=True)

                            # Speedup chart
                            fp32_row = bench_df[bench_df["Variant"] == "fp32"]
                            if not fp32_row.empty:
                                fp32_ms = fp32_row["Mean (ms)"].iloc[0]
                                bench_df["Speedup"] = (fp32_ms / bench_df["Mean (ms)"]).round(2)
                                import plotly.express as px
                                fig = px.bar(
                                    bench_df[bench_df["OK"]],
                                    x="Variant",
                                    y="Mean (ms)",
                                    color="Variant",
                                    title="Inference Latency by Variant (lower is better)",
                                    text="Mean (ms)",
                                )
                                fig.update_traces(texttemplate="%{text:.1f}ms", textposition="outside")
                                st.plotly_chart(fig, use_container_width=True)

                    # Report files
                    out_dir = _Path(mp_output_dir.strip())
                    json_path = out_dir / "pipeline_report.json"
                    if json_path.exists():
                        with open(json_path) as fh:
                            import json as _json
                            with st.expander("Full report (JSON)"):
                                st.json(_json.load(fh))
        else:
            st.info("Configure pipeline options and click **Run Full Pipeline**, or use a quick tool above.")
            with st.expander("Architecture overview"):
                st.markdown("""
### Quantization Pipeline Flow

```
float32 ONNX model (export_to_onnx.py output)
           │
     ┌─────┼─────────────────────┐
     ▼     ▼                     ▼
  FP16  Dynamic INT8        Static INT8
   │       │                    │
   │   onnxruntime          onnxruntime
   │   quantize_dynamic()   quantize_static()
   │   (no calibration)     (calibration data)
   │                            │
   └──────────┬─────────────────┘
              ▼
         Benchmark all variants
         (OnnxBenchmark, n_runs)
              ▼
        pipeline_report.json
        pipeline_report.md
```

### When to use each variant

| Variant | Size | Speed | Quality | Use case |
|---------|------|-------|---------|----------|
| fp32 | 100% | baseline | best | development |
| fp16 | ~50% | ~1.5x GPU | near-lossless | GPU deployment |
| dynamic_int8 | ~25% | ~2x CPU | good | CPU edge devices |
| static_int8 | ~25% | ~2x CPU | best INT8 accuracy | production CPU |

### Key files
- `conversion/quantize_onnx.py` — FP16, dynamic INT8, static INT8
- `conversion/calibrate.py` — calibration dataset reader
- `conversion/benchmark.py` — latency / TPS benchmarking
- `conversion/pipeline.py` — end-to-end orchestrator
- `conversion/report.py` — JSON + Markdown report generator
- `scripts/run_quantization_pipeline.py` — CLI interface
                """)
