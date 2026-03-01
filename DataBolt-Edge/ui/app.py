"""DataBolt Edge — Streamlit developer UI.

Single-page app with 3 tabs:
    Tab 1 — Analyze    : submit log/SQL + question → parser + LLM analysis
    Tab 2 — History    : browse all analyses from DuckDB
    Tab 3 — Pipeline   : quantization benchmark results

Run:
    PYTHONPATH=DataBolt-Edge uv run --extra agentops \\
        streamlit run DataBolt-Edge/ui/app.py \\
        --server.port 8504 --server.headless true --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import sys

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Make backend importable when running directly (PYTHONPATH=DataBolt-Edge).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_BASE = os.getenv("DATABOLT_API_URL", "http://localhost:8001")

st.set_page_config(page_title="DataBolt Edge", layout="wide")
st.title("DataBolt Edge — On-Device AI Debugger")
st.caption("Spark · Airflow · SQL — local inference, no cloud calls")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, timeout: int = 10) -> dict | list:
    r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Demo data loader
# ---------------------------------------------------------------------------

_DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "demo_data")


def _load_demo(filename: str) -> str:
    path = os.path.join(_DEMO_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


DEMO_FILES = {
    "spark": ("spark_oom.log", "Why did this Spark job fail and how do I fix it?"),
    "airflow": ("airflow_failure.log", "What caused this Airflow task to fail and how do I prevent it?"),
    "sql": ("sql_bad_plan.txt", "Rewrite this SQL to eliminate the full table scan."),
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Analyze", "History", "Quantization Pipeline"])

# ===========================================================================
# Tab 1 — Analyze
# ===========================================================================

with tab1:
    st.header("Analyze Log / SQL")

    col_type, col_demo = st.columns([1, 2])
    with col_type:
        log_type = st.radio(
            "Log type", ["spark", "airflow", "sql"], horizontal=True, key="log_type"
        )
    with col_demo:
        if st.button("Load demo sample", key="load_demo"):
            fname, demo_q = DEMO_FILES[log_type]
            demo_content = _load_demo(fname)
            st.session_state["content"] = demo_content
            st.session_state["question"] = demo_q

    content = st.text_area(
        "Paste log / SQL EXPLAIN output",
        value=st.session_state.get("content", ""),
        height=220,
        key="content_area",
        placeholder="Paste Spark driver log, Airflow task log, or SQL EXPLAIN output here…",
    )
    question = st.text_input(
        "Question",
        value=st.session_state.get("question", "What is wrong and how do I fix it?"),
        key="question_input",
    )

    if st.button("Analyze", type="primary", key="analyze"):
        if not content.strip():
            st.warning("Paste some log content first.")
        else:
            with st.spinner("Parsing and running local inference…"):
                try:
                    result = _post(
                        "/analyze",
                        {"log_type": log_type, "content": content, "question": question},
                        timeout=180,
                    )

                    # Telemetry bar
                    t = result.get("telemetry", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Backend", t.get("backend", "—"))
                    c2.metric("Latency (ms)", t.get("latency_ms", "—"))
                    c3.metric("Tokens", t.get("total_tokens", "—"))
                    c4.metric("Tok/s", t.get("tokens_per_second", "—"))

                    st.success("Analysis complete — model ran locally")

                    # Model response
                    st.subheader("Diagnosis & Recommendations")
                    st.markdown(result.get("model_response", "(no response)"))

                    # SQL rewrite diff
                    if result.get("optimized_sql"):
                        st.subheader("Optimized SQL")
                        st.code(result["optimized_sql"], language="sql")

                    # Parsed summary
                    with st.expander("Parsed log summary (structured)"):
                        st.json(result.get("parsed_summary", {}))

                    # Store analysis_id for follow-up
                    st.caption(f"analysis_id: `{result.get('analysis_id')}`")

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.info(
                        "Make sure the backend is running:\n"
                        "```\nPYTHONPATH=DataBolt-Edge uv run --extra agentops "
                        "uvicorn api.server:app --port 8001\n```"
                    )

# ===========================================================================
# Tab 2 — History
# ===========================================================================

with tab2:
    st.header("Analysis History")

    col_refresh, col_detail = st.columns([1, 2])

    with col_refresh:
        if st.button("Load history", key="load_history"):
            try:
                records = _get("/analyses")
                if records:
                    import pandas as pd

                    df = pd.DataFrame(records)
                    st.dataframe(df, use_container_width=True)
                    st.session_state["history"] = records
                else:
                    st.info("No analyses yet.")
            except Exception as exc:
                st.error(f"Could not reach backend: {exc}")

    with col_detail:
        analysis_id_input = st.text_input(
            "Load detail by analysis_id", key="detail_id"
        )
        if st.button("Load Detail", key="load_detail"):
            if not analysis_id_input.strip():
                st.warning("Enter an analysis_id")
            else:
                try:
                    rec = _get(f"/results/{analysis_id_input.strip()}")
                    st.markdown(f"**Log type:** {rec.get('log_type')} | "
                                f"**Backend:** {rec.get('backend')} | "
                                f"**Latency:** {rec.get('latency_ms')} ms")
                    st.markdown("**Question:**")
                    st.info(rec.get("question"))
                    st.markdown("**Response:**")
                    st.markdown(rec.get("model_response", ""))
                    if rec.get("optimized_sql"):
                        st.code(rec["optimized_sql"], language="sql")
                except Exception as exc:
                    st.error(f"Error: {exc}")

# ===========================================================================
# Tab 3 — Quantization Pipeline
# ===========================================================================

with tab3:
    st.header("Quantization Benchmark")
    st.caption(
        "Shows FP16 vs dynamic-INT8 vs static-INT8 benchmark results "
        "from the ONNX quantization pipeline."
    )

    report_path = st.text_input(
        "Report JSON path",
        value=os.path.join(os.path.dirname(__file__), "..", "models", "quantized", "report.json"),
        key="report_path",
    )

    if st.button("Load Report", key="load_report"):
        if os.path.exists(report_path):
            import json, pandas as pd

            with open(report_path) as f:
                report = json.load(f)

            benchmarks = report.get("benchmarks", {})
            if benchmarks:
                rows = []
                for variant, bm in benchmarks.items():
                    rows.append({
                        "Variant": variant,
                        "Median (ms)": bm.get("median_ms"),
                        "P95 (ms)": bm.get("p95_ms"),
                        "P99 (ms)": bm.get("p99_ms"),
                        "Throughput (seq/s)": bm.get("throughput_seq_per_sec"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.subheader("Raw Report")
            st.json(report)
        else:
            st.warning(
                f"Report not found at `{report_path}`. "
                "Run the quantization pipeline first:\n"
                "```\nPYTHONPATH=DataBolt-Edge uv run python "
                "DataBolt-Edge/scripts/run_quantization_pipeline.py "
                "--model-path models/mistral-7b --output-dir models/quantized\n```"
            )

    st.divider()
    st.subheader("Backend Health")
    if st.button("Check backend", key="health_check"):
        try:
            h = _get("/health")
            inf = h.get("inference", {})
            st.metric("Inference backend", inf.get("backend", inf.get("reason", "—")))
            st.metric("Available", str(inf.get("available", "—")))
            env = h.get("env", {})
            st.json(env)
        except Exception as exc:
            st.error(f"Backend unreachable: {exc}")
