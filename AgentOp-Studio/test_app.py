"""AgentOp-Studio simple developer test UI.

Single-page Streamlit app with 5 tabs for quickly exercising all backend
components and verifying end-to-end functionality.

Run:
    PYTHONPATH=AgentOp-Studio streamlit run AgentOp-Studio/test_app.py \
        --server.port 8503 --server.headless true --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import sys
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

# Make backend importable for Tab 5 cost functions.
sys.path.insert(0, os.path.dirname(__file__))
from backend.costs import (  # noqa: E402
    ELEVENLABS_PRICING,
    MISTRAL_PRICING,
    estimate_cost,
    estimate_elevenlabs_cost,
)

load_dotenv()

API_BASE = os.getenv("AGENTOPS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AgentOp-Studio · Test UI", layout="wide")
st.title("AgentOp-Studio — Developer Test UI")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(path: str, timeout: int = 10) -> dict | list:
    resp = httpx.get(f"{API_BASE}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    resp = httpx.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _mask(value: str | None, show: int = 4) -> str:
    if not value:
        return "(not set)"
    return value[:show] + "..." + value[-show:] if len(value) > show * 2 else "***"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Health & Status", "Run Agent", "Inspect Runs", "Evaluations", "Cost Calculator"]
)

# ===========================================================================
# Tab 1 — Health & Status
# ===========================================================================

with tab1:
    st.header("Health & Status")

    col_env, col_ping = st.columns([1, 1])

    with col_env:
        st.subheader("Environment")
        st.markdown(f"**AGENTOPS_API_URL:** `{API_BASE}`")
        st.markdown(
            f"**MISTRAL_API_KEY:** `{_mask(os.getenv('MISTRAL_API_KEY'))}`"
        )
        db_path = os.getenv(
            "AGENTOPS_DB_PATH",
            os.path.join(os.path.dirname(__file__), "data", "agentops.duckdb"),
        )
        st.markdown(f"**AGENTOPS_DB_PATH:** `{db_path}`")

    with col_ping:
        st.subheader("Backend Ping")
        if st.button("Ping Backend", key="ping"):
            try:
                t0 = time.perf_counter()
                _get("/runs")
                latency_ms = (time.perf_counter() - t0) * 1000
                st.success(f"Backend reachable — {latency_ms:.1f} ms")
            except Exception as exc:
                st.error(f"Backend unreachable: {exc}")

    st.divider()
    st.subheader("Database Table Counts")

    if st.button("Refresh DB Counts", key="db_counts"):
        try:
            import duckdb

            conn = duckdb.connect(db_path)
            tables = ["runs", "messages", "tool_calls", "memory_snapshots", "diffs", "evaluations"]
            counts: dict[str, int] = {}
            for t in tables:
                try:
                    row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                    counts[t] = row[0] if row else 0
                except Exception:
                    counts[t] = -1  # table may not exist yet
            conn.close()

            cols = st.columns(len(tables))
            for col, (name, count) in zip(cols, counts.items()):
                col.metric(name, count if count >= 0 else "N/A")

        except Exception as exc:
            st.error(f"Could not query DB: {exc}")

# ===========================================================================
# Tab 2 — Run Agent
# ===========================================================================

with tab2:
    st.header("Run Agent")

    prompt = st.text_area(
        "Prompt",
        placeholder="Enter a prompt for the agent…",
        height=120,
        key="run_prompt",
    )
    user_id = st.text_input("User ID (optional)", value="test-user", key="run_user_id")

    if st.button("Submit", key="run_submit"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Running agent…"):
                try:
                    result = _post("/run", {"prompt": prompt, "user_id": user_id}, timeout=180)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Run ID", result.get("run_id", "—")[:8] + "…")
                    c2.metric("Tokens", result.get("total_tokens", 0))
                    c3.metric("Cost ($)", f"{result.get('total_cost', 0):.6f}")

                    st.markdown("**Response:**")
                    st.info(result.get("response", "(no response)"))

                    with st.expander("Full JSON response"):
                        st.json(result)

                except Exception as exc:
                    st.error(f"Run failed: {exc}")

# ===========================================================================
# Tab 3 — Inspect Runs
# ===========================================================================

with tab3:
    st.header("Inspect Runs")

    if st.button("Load All Runs", key="load_runs"):
        try:
            runs = _get("/runs")
            if runs:
                import pandas as pd

                st.dataframe(pd.DataFrame(runs), use_container_width=True)
            else:
                st.info("No runs recorded yet.")
        except Exception as exc:
            st.error(f"Failed to fetch runs: {exc}")

    st.divider()
    st.subheader("Run Detail")

    run_id_input = st.text_input("Run ID", placeholder="Paste a run_id here…", key="detail_run_id")

    col_detail, col_mem = st.columns(2)

    with col_detail:
        if st.button("Load Run Detail", key="load_detail"):
            if not run_id_input.strip():
                st.warning("Enter a run_id.")
            else:
                try:
                    detail = _get(f"/runs/{run_id_input.strip()}")
                    st.markdown("**Messages:**")
                    st.json(detail.get("messages", []))
                    st.markdown("**Tool Calls:**")
                    st.json(detail.get("tool_calls", []))
                except Exception as exc:
                    st.error(f"Error: {exc}")

    with col_mem:
        if st.button("Load Memory Snapshots", key="load_memory"):
            if not run_id_input.strip():
                st.warning("Enter a run_id.")
            else:
                try:
                    snapshots = _get(f"/runs/{run_id_input.strip()}/memory")
                    if snapshots:
                        st.json(snapshots)
                    else:
                        st.info("No memory snapshots for this run.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

# ===========================================================================
# Tab 4 — Evaluations
# ===========================================================================

with tab4:
    st.header("Evaluations")

    if st.button("Load All Evals", key="load_evals"):
        try:
            evals = _get("/evals")
            if evals:
                import pandas as pd

                st.dataframe(pd.DataFrame(evals), use_container_width=True)
            else:
                st.info("No evaluation records yet.")
        except Exception as exc:
            st.error(f"Failed to fetch evals: {exc}")

    st.divider()
    st.subheader("Log a Metric")

    with st.form("eval_form"):
        eval_run_id = st.text_input("Run ID", key="eval_run_id")
        metric_name = st.text_input("Metric Name", placeholder="e.g. accuracy", key="eval_metric_name")
        metric_value = st.number_input("Metric Value", value=1.0, step=0.01, key="eval_metric_value")
        submitted = st.form_submit_button("Post Eval")

    if submitted:
        if not eval_run_id.strip() or not metric_name.strip():
            st.warning("Run ID and Metric Name are required.")
        else:
            try:
                result = _post(
                    "/eval",
                    {"run_id": eval_run_id.strip(), "metrics": {metric_name.strip(): float(metric_value)}},
                )
                st.success(f"Logged — eval_id: {result.get('eval_ids', ['—'])[0]}")
            except Exception as exc:
                st.error(f"Failed to log eval: {exc}")

    st.divider()
    st.subheader("Evals for a Specific Run")

    filter_run_id = st.text_input("Run ID", placeholder="Paste a run_id…", key="eval_filter_run_id")
    if st.button("Filter Evals", key="filter_evals"):
        if not filter_run_id.strip():
            st.warning("Enter a run_id.")
        else:
            try:
                evals = _get(f"/evals/{filter_run_id.strip()}")
                if evals:
                    import pandas as pd

                    st.dataframe(pd.DataFrame(evals), use_container_width=True)
                else:
                    st.info("No evals for this run.")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ===========================================================================
# Tab 5 — Cost Calculator (works without a running backend)
# ===========================================================================

with tab5:
    st.header("Cost Calculator")
    st.caption("This tab works offline — no backend required.")

    col_mistral, col_el = st.columns(2)

    with col_mistral:
        st.subheader("Mistral")
        m_tokens = st.number_input(
            "Total tokens", min_value=1, value=1000, step=100, key="m_tokens"
        )
        m_model = st.selectbox(
            "Model", options=list(MISTRAL_PRICING.keys()), key="m_model"
        )
        if st.button("Estimate Mistral Cost", key="calc_mistral"):
            cost = estimate_cost(int(m_tokens), m_model)
            rates = MISTRAL_PRICING[m_model]
            st.metric("Estimated Cost (USD)", f"${cost:.6f}")
            st.caption(
                f"Rate: ${rates['input']:.4f}/1k input · ${rates['output']:.4f}/1k output "
                f"(averaged for single-rate estimate)"
            )

    with col_el:
        st.subheader("ElevenLabs TTS")
        el_chars = st.number_input(
            "Characters to synthesize", min_value=1, value=1000, step=100, key="el_chars"
        )
        el_tier = st.selectbox(
            "Subscription tier", options=list(ELEVENLABS_PRICING.keys()), key="el_tier"
        )
        if st.button("Estimate ElevenLabs Cost", key="calc_el"):
            cost = estimate_elevenlabs_cost(int(el_chars), el_tier)
            rate = ELEVENLABS_PRICING[el_tier]
            st.metric("Estimated Cost (USD)", f"${cost:.6f}")
            st.caption(f"Rate: ${rate:.4f} / 1k chars (tier: {el_tier})")
