"""AgentOps-Studio Streamlit dashboard.

Pages (via sidebar radio):
    Dashboard       — KPI metrics + charts + run table
    Run Agent       — Submit a prompt and view the result
    Run Detail      — Browse messages and tool calls for a run
    Replay & Diff   — Replay a run with param overrides, view diff
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Load environment variables early (.env or Codespace secrets)
load_dotenv()

API_BASE = os.getenv("AGENTOPS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AgentOps-Studio", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(path: str) -> dict | list:
    resp = httpx.get(f"{API_BASE}{path}", timeout=60)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: dict) -> dict:
    resp = httpx.post(f"{API_BASE}{path}", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _load_runs() -> pd.DataFrame:
    try:
        data = _get("/runs")
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as exc:
        st.error(f"Could not reach backend: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Run Agent", "Run Detail", "Replay & Diff", "Evaluations"],
)

# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

if page == "Dashboard":
    st.title("AgentOps-Studio — Dashboard")

    df = _load_runs()

    if df.empty:
        st.info("No runs yet. Use **Run Agent** to submit a prompt.")
    else:
        total_runs = len(df)
        total_tokens = int(df["total_tokens"].fillna(0).sum())
        total_cost = float(df["total_cost"].fillna(0).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Runs", total_runs)
        c2.metric("Total Tokens", f"{total_tokens:,}")
        c3.metric("Total Cost (USD)", f"${total_cost:.4f}")

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            fig_tokens = px.bar(
                df,
                x="run_id",
                y="total_tokens",
                title="Tokens per Run",
                labels={"run_id": "Run ID", "total_tokens": "Tokens"},
            )
            fig_tokens.update_xaxes(tickangle=45)
            st.plotly_chart(fig_tokens, use_container_width=True)

        with col_right:
            fig_cost = px.bar(
                df,
                x="run_id",
                y="total_cost",
                title="Cost per Run (USD)",
                labels={"run_id": "Run ID", "total_cost": "Cost ($)"},
            )
            fig_cost.update_xaxes(tickangle=45)
            st.plotly_chart(fig_cost, use_container_width=True)

        st.subheader("All Runs")
        st.dataframe(df, use_container_width=True)

        # Allow clicking a run ID to jump to detail
        selected = st.text_input("Enter a Run ID to view detail →")
        if selected:
            st.session_state["detail_run_id"] = selected
            st.info("Switch to the **Run Detail** page in the sidebar.")

# ---------------------------------------------------------------------------
# Page: Run Agent
# ---------------------------------------------------------------------------

elif page == "Run Agent":
    st.title("Run Agent")

    prompt = st.text_area("Prompt", height=160, placeholder="Ask the agent anything…")
    user_id = st.text_input("User ID (optional)", value="default")

    if st.button("Submit", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Running agent…"):
                try:
                    result = _post("/run", {"prompt": prompt, "user_id": user_id})
                    st.success(f"Run ID: `{result['run_id']}`")
                    st.markdown("**Response:**")
                    st.write(result["response"])
                    col1, col2 = st.columns(2)
                    col1.metric("Total Tokens", result["total_tokens"])
                    col2.metric("Estimated Cost (USD)", f"${result['total_cost']:.4f}")
                    st.session_state["detail_run_id"] = result["run_id"]
                except Exception as exc:
                    st.error(f"Error: {exc}")

# ---------------------------------------------------------------------------
# Page: Run Detail
# ---------------------------------------------------------------------------

elif page == "Run Detail":
    st.title("Run Detail")

    run_id = st.text_input(
        "Run ID",
        value=st.session_state.get("detail_run_id", ""),
    )

    if st.button("Load", type="primary") or st.session_state.get("detail_run_id"):
        if not run_id.strip():
            st.warning("Enter a Run ID.")
        else:
            try:
                run = _get(f"/runs/{run_id}")
            except Exception as exc:
                st.error(f"Could not load run: {exc}")
                run = None

            if run:
                st.subheader("Summary")
                cols = st.columns(4)
                cols[0].metric("Status", run.get("status", "—"))
                cols[1].metric("Tokens", run.get("total_tokens") or 0)
                cols[2].metric("Cost (USD)", f"${(run.get('total_cost') or 0):.4f}")
                import json as _json
                _cfg = _json.loads(run["config"]) if run.get("config") else {}
                cols[3].metric("Model", _cfg.get("model", "—"))

                st.divider()
                st.subheader("Message Thread")
                role_colors = {
                    "user": "🧑",
                    "assistant": "🤖",
                    "tool": "🔧",
                }
                for msg in run.get("messages", []):
                    icon = role_colors.get(msg["role"], "❓")
                    with st.expander(f"{icon} {msg['role'].capitalize()}  —  {msg.get('timestamp', '')}"):
                        st.write(msg["content"])

                st.divider()
                st.subheader("Tool Calls")
                tool_calls = run.get("tool_calls", [])
                if tool_calls:
                    tc_df = pd.DataFrame(tool_calls)[
                        ["tool_name", "args", "return_value", "latency_ms"]
                    ]
                    st.dataframe(tc_df, use_container_width=True)
                else:
                    st.info("No tool calls for this run.")

                st.divider()
                st.subheader("Memory Snapshots")
                try:
                    snapshots = _get(f"/runs/{run_id}/memory")
                    if snapshots:
                        for i, snap in enumerate(snapshots, 1):
                            with st.expander(
                                f"Snapshot {i} — {snap.get('timestamp', '')}",
                                expanded=False,
                            ):
                                import json as _snap_json
                                try:
                                    mem = _snap_json.loads(snap["memory_json"])
                                    st.json(mem)
                                except Exception:
                                    st.text(snap["memory_json"])
                    else:
                        st.info("No memory snapshots for this run (snapshots are captured per tool-call round).")
                except Exception as exc:
                    st.error(f"Could not load memory snapshots: {exc}")

                st.divider()
                st.subheader("Evaluations")
                try:
                    evals = _get(f"/evals/{run_id}")
                    if evals:
                        eval_df = pd.DataFrame(evals)[["metric_name", "metric_value"]]
                        st.dataframe(eval_df, use_container_width=True)
                    else:
                        st.info("No evaluations logged for this run.")
                except Exception as exc:
                    st.error(f"Could not load evaluations: {exc}")

                st.divider()
                st.subheader("Log Evaluation")
                with st.form("eval_form"):
                    metric_name = st.text_input("Metric name", placeholder="e.g. success, accuracy, latency_ok")
                    metric_value = st.number_input("Metric value", value=1.0, step=0.01)
                    submitted = st.form_submit_button("Log")
                    if submitted:
                        if not metric_name.strip():
                            st.warning("Enter a metric name.")
                        else:
                            try:
                                result = _post(
                                    "/eval",
                                    {"run_id": run_id, "metrics": {metric_name: metric_value}},
                                )
                                st.success(f"Logged: {metric_name} = {metric_value}")
                            except Exception as exc:
                                st.error(f"Failed to log: {exc}")

# ---------------------------------------------------------------------------
# Page: Replay & Diff
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Page: Evaluations
# ---------------------------------------------------------------------------

elif page == "Evaluations":
    st.title("Evaluations")

    try:
        evals = _get("/evals")
    except Exception as exc:
        st.error(f"Could not reach backend: {exc}")
        evals = []

    if not evals:
        st.info("No evaluations logged yet. Use **Run Detail** to log metrics for a run.")
    else:
        eval_df = pd.DataFrame(evals)

        # Summary KPIs
        unique_metrics = eval_df["metric_name"].nunique()
        unique_runs = eval_df["run_id"].nunique()
        c1, c2 = st.columns(2)
        c1.metric("Metric Types", unique_metrics)
        c2.metric("Runs Evaluated", unique_runs)

        st.divider()

        # Per-metric average bar chart
        avg_df = eval_df.groupby("metric_name")["metric_value"].mean().reset_index()
        avg_df.columns = ["metric_name", "avg_value"]
        fig_avg = px.bar(
            avg_df,
            x="metric_name",
            y="avg_value",
            title="Average Value per Metric",
            labels={"metric_name": "Metric", "avg_value": "Average Value"},
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        # Time series: metric values over runs (if start_time available)
        if "start_time" in eval_df.columns and eval_df["start_time"].notna().any():
            eval_df["start_time"] = pd.to_datetime(eval_df["start_time"])
            fig_ts = px.scatter(
                eval_df,
                x="start_time",
                y="metric_value",
                color="metric_name",
                title="Metric Values Over Time",
                labels={
                    "start_time": "Run Time",
                    "metric_value": "Value",
                    "metric_name": "Metric",
                },
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("All Evaluation Records")
        st.dataframe(
            eval_df[["run_id", "metric_name", "metric_value", "start_time"]].rename(
                columns={
                    "run_id": "Run ID",
                    "metric_name": "Metric",
                    "metric_value": "Value",
                    "start_time": "Time",
                }
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Page: Replay & Diff
# ---------------------------------------------------------------------------

elif page == "Replay & Diff":
    st.title("Replay & Diff")

    df = _load_runs()
    if df.empty:
        st.info("No runs available to replay.")
    else:
        run_ids = df["run_id"].tolist()
        selected_run = st.selectbox("Select a Run to Replay", run_ids)

        st.markdown("**Optional Overrides**")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.5, 0.6, 0.05)
        with col2:
            model_options = [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
                "open-mistral-7b",
            ]
            model = st.selectbox("Model", model_options)

        override_params: dict[str, str] = {
            "TEMPERATURE": str(temperature),
            "MISTRAL_MODEL": model,
        }

        if st.button("Replay", type="primary"):
            with st.spinner("Replaying run…"):
                try:
                    result = _post(
                        "/replay",
                        {
                            "run_id": selected_run,
                            "override_params": override_params,
                        },
                    )

                    st.success(f"New Run ID: `{result['new_run_id']}`")

                    st.subheader("Responses")
                    c_orig, c_new = st.columns(2)
                    with c_orig:
                        st.markdown("**Original**")
                        st.write(result["original_response"])
                    with c_new:
                        st.markdown("**Replayed**")
                        st.write(result["new_response"])

                    st.subheader("Unified Diff")
                    diff_text = result.get("diff", "")
                    if diff_text:
                        st.code(diff_text, language="diff")
                    else:
                        st.info("Responses are identical — no diff.")

                    # Metrics comparison
                    orig_run = _get(f"/runs/{selected_run}")
                    new_run = _get(f"/runs/{result['new_run_id']}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Orig Tokens", orig_run.get("total_tokens") or 0)
                    m2.metric("New Tokens", new_run.get("total_tokens") or 0)
                    m3.metric("Orig Cost", f"${(orig_run.get('total_cost') or 0):.4f}")
                    m4.metric("New Cost", f"${(new_run.get('total_cost') or 0):.4f}")

                except Exception as exc:
                    st.error(f"Replay failed: {exc}")
