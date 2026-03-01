"""AgentOps-Studio Streamlit dashboard.

Pages (via sidebar radio):
    Dashboard       — KPI metrics + charts + run table
    Run Agent       — Submit a prompt and view the result
    Run Detail      — Browse messages and tool calls for a run
    Replay & Diff   — Replay a run with param overrides, view diff
    Evaluations     — Browse and chart logged eval metrics
    W&B Training    — Launch Mistral training runs in Weights & Biases
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
    ["Dashboard", "Run Agent", "Run Detail", "Replay & Diff", "Evaluations", "W&B Training", "Databricks"],
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

# ---------------------------------------------------------------------------
# Page: W&B Training
# ---------------------------------------------------------------------------

elif page == "W&B Training":
    st.title("W&B Training — Mistral Agent Analysis")
    st.markdown(
        "Launch a **Mistral fine-tuning simulation** in [Weights & Biases](https://wandb.ai). "
        "The run fetches a HuggingFace dataset sample, logs training metrics (loss, "
        "perplexity, token accuracy), uploads a dataset artifact, and includes "
        "agent-analysis from existing AgentOps runs."
    )

    # -- W&B connection status banner --
    st.subheader("Connection Status")
    try:
        status = _get("/wandb/status")
        if status.get("connected"):
            st.success(
                f"Connected — project **{status['project']}** "
                f"/ entity **{status.get('entity', '—')}** "
                f"(key: `{status['key_prefix']}`)"
            )
        else:
            st.error(f"Not connected: {status.get('reason', 'unknown')}")
            st.info(
                "Set `WANDB_API_KEY` in your `.env` file or Codespace secrets, "
                "then restart the backend."
            )
    except Exception as exc:
        st.warning(f"Could not reach backend to check W&B status: {exc}")

    st.divider()

    # -- Training run configuration form --
    st.subheader("Launch Training Run")
    with st.form("wandb_training_form"):
        col_l, col_r = st.columns(2)

        with col_l:
            model_options = [
                "mistral-7b-instruct-v0.2",
                "mistral-7b-v0.3",
                "mixtral-8x7b-instruct-v0.1",
                "mistral-small-latest",
            ]
            model_name = st.selectbox("Model", model_options)

            dataset_options = [
                "tatsu-lab/alpaca",
                "HuggingFaceH4/ultrachat_200k",
                "databricks/databricks-dolly-15k",
                "openai/gsm8k",
            ]
            dataset_name = st.selectbox("HuggingFace Dataset", dataset_options)

        with col_r:
            num_steps = st.slider("Training Steps", min_value=10, max_value=200, value=50, step=10)
            sample_size = st.slider("HF Sample Size", min_value=5, max_value=50, value=20, step=5)

        run_name = st.text_input(
            "Run Name (optional)",
            placeholder="Leave blank for auto-generated name",
        )

        launch = st.form_submit_button("Launch Training Run", type="primary")

    if launch:
        payload = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_steps": num_steps,
            "sample_size": sample_size,
        }
        if run_name.strip():
            payload["run_name"] = run_name.strip()

        with st.spinner(f"Running {num_steps} training steps and logging to W&B …"):
            try:
                result = _post("/wandb/training-run", payload)

                st.success("Training run complete!")

                # Metrics grid
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Final Loss", f"{result.get('final_train_loss', 0):.4f}")
                m2.metric("Perplexity", f"{result.get('final_perplexity', 0):.2f}")
                m3.metric("Token Accuracy", f"{result.get('final_token_accuracy', 0):.1%}")
                m4.metric("HF Samples", result.get("dataset_samples", 0))

                # W&B run link
                run_url = result.get("run_url")
                if run_url:
                    st.markdown(f"### View in W&B\n[{run_url}]({run_url})")
                else:
                    st.info("Run complete (URL not available — check W&B dashboard).")

            except Exception as exc:
                st.error(f"Training run failed: {exc}")

    # -- Instructions for standalone script --
    st.divider()
    st.subheader("Run via CLI")
    st.markdown(
        "You can also run the training demo directly from the terminal "
        "without the backend:"
    )
    st.code(
        "PYTHONPATH=AgentOp-Studio python AgentOp-Studio/scripts/mistral_training_demo.py",
        language="bash",
    )
    st.markdown(
        "Optional env vars: `HF_DATASET`, `HF_SAMPLE_SIZE`, `TRAINING_STEPS`, "
        "`WANDB_PROJECT`"
    )

# ---------------------------------------------------------------------------
# Page: Databricks
# ---------------------------------------------------------------------------

elif page == "Databricks":
    st.title("Databricks — Job Monitor & Debug Agent")

    # Load workflow data once
    try:
        wf_data = _get("/databricks/workflow")
    except Exception as exc:
        st.error(f"Could not reach backend: {exc}")
        wf_data = {}

    failed_jobs = wf_data.get("failed_jobs", [])
    predefined = wf_data.get("predefined_errors", [])
    pipeline_data = wf_data.get("pipeline", {})

    # ── SECTION 1: Failed Jobs ─────────────────────────────────────────────
    st.subheader("Failed Jobs")

    if not failed_jobs:
        st.info("No failed jobs found.")
    else:
        # One expander per failed job
        for job in failed_jobs:
            severity_icon = "🔴" if "OOM" in job.get("category", "") else "🟠"
            header = (
                f"{severity_icon} **{job['job_name']}** › `{job['task_key']}`  "
                f"— {job.get('category', 'Error')}  "
                f"| {job.get('started_at', '')[:16].replace('T', ' ')} UTC"
            )
            with st.expander(header, expanded=False):
                st.error(job.get("error_message", ""))
                col_a, col_b, col_c = st.columns([2, 2, 1])
                col_a.markdown(f"**Run ID:** `{job['run_id']}`")
                col_b.markdown(f"**Duration:** {job.get('duration_seconds', 0)}s")
                run_url = job.get("run_url", "")
                col_c.markdown(f"[View in Databricks ↗]({run_url})" if run_url else "")

                # Pre-fill investigation from this job
                if st.button(
                    "Investigate this error",
                    key=f"inv_{job['run_id']}",
                    type="secondary",
                ):
                    st.session_state["db_error_msg"] = job["error_message"]
                    st.session_state["db_task_key"] = job["task_key"]
                    st.session_state["db_job_name"] = job["job_name"]
                    st.session_state["db_category"] = job.get("category", "")
                    st.rerun()

    st.divider()

    # ── SECTION 2: DLT Pipeline DAG (visual) ──────────────────────────────
    with st.expander("Pipeline DAG — Medallion Architecture", expanded=False):
        tables = pipeline_data.get("tables", [])
        edges = pipeline_data.get("dag_edges", [])

        if tables and edges:
            # Build Sankey diagram
            all_names = list({t["name"] for t in tables})
            label_to_idx = {n: i for i, n in enumerate(all_names)}

            layer_colors = {
                "bronze": "#cd7f32",
                "silver": "#a8a9ad",
                "gold": "#ffd700",
                "features": "#6495ed",
            }
            node_colors = [
                layer_colors.get(
                    next((t["layer"] for t in tables if t["name"] == n), ""), "#888"
                )
                for n in all_names
            ]

            sources = [label_to_idx[e["source"]] for e in edges if e["source"] in label_to_idx]
            targets_idx = [label_to_idx[e["target"]] for e in edges if e["target"] in label_to_idx]

            fig_dag = px.scatter(title="")  # placeholder — build sankey manually
            import plotly.graph_objects as go  # noqa: PLC0415

            fig_sankey = go.Figure(
                go.Sankey(
                    node=dict(
                        label=all_names,
                        color=node_colors,
                        pad=20,
                        thickness=20,
                    ),
                    link=dict(
                        source=sources,
                        target=targets_idx,
                        value=[1] * len(sources),
                    ),
                )
            )
            fig_sankey.update_layout(
                title_text="DLT Table Lineage (Bronze → Silver → Gold → Features)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

            # Table summary
            layer_counts = pipeline_data.get("layers", {})
            cols = st.columns(len(layer_counts) or 1)
            for i, (layer, names) in enumerate(sorted(layer_counts.items())):
                cols[i].metric(layer.title(), len(names))
            st.caption(
                f"Total tables: {pipeline_data.get('total_tables', 0)}  |  "
                f"Quality rules: {pipeline_data.get('quality_rule_count', 0)}"
            )
        else:
            st.info("Pipeline data unavailable.")

    st.divider()

    # ── SECTION 3: Investigate Error ──────────────────────────────────────
    st.subheader("Investigate Error with Mistral")

    # Predefined error quick-select
    st.markdown("**Quick select a predefined error category:**")
    pred_cols = st.columns(len(predefined) or 1)
    for i, err in enumerate(predefined):
        if pred_cols[i].button(
            f"{err['icon']} {err['label']}",
            key=f"pred_{err['id']}",
            use_container_width=True,
        ):
            st.session_state["db_error_msg"] = err["prompt"]
            st.session_state["db_task_key"] = ""
            st.session_state["db_job_name"] = ""
            st.session_state["db_category"] = err["label"]
            st.rerun()

    st.markdown("**— or describe the error —**")

    with st.form("db_investigate_form"):
        error_msg = st.text_area(
            "Error message / description",
            value=st.session_state.get("db_error_msg", ""),
            height=100,
            placeholder="Paste the full error message here, or click a quick-select above…",
        )
        col_l, col_r = st.columns(2)
        task_key = col_l.text_input(
            "Failed task key",
            value=st.session_state.get("db_task_key", ""),
            placeholder="e.g. validate_output",
        )
        job_name_input = col_r.text_input(
            "Job name",
            value=st.session_state.get("db_job_name", ""),
            placeholder="e.g. NYC Taxi Workflow [dev]",
        )
        extra_ctx = st.text_area(
            "Additional context (optional)",
            height=60,
            placeholder="Any extra info: cluster config, data volume, recent changes…",
        )
        include_code = st.checkbox("Include DLT pipeline code as context", value=True)

        submitted = st.form_submit_button("Investigate with Mistral", type="primary")

    if submitted and error_msg.strip():
        with st.spinner("Mistral is analyzing the error…"):
            try:
                result = _post(
                    "/databricks/investigate",
                    {
                        "error_message": error_msg,
                        "task_key": task_key,
                        "job_name": job_name_input,
                        "extra_context": extra_ctx,
                        "include_pipeline_code": include_code,
                    },
                )
                st.session_state["db_last_analysis"] = result
                st.session_state["db_last_error"] = error_msg
                st.session_state["db_last_job"] = job_name_input
                st.session_state["db_last_task"] = task_key
                st.session_state["db_last_category"] = st.session_state.get("db_category", "other")
            except Exception as exc:
                st.error(f"Investigation failed: {exc}")
                result = None

    # ── SECTION 4: Analysis Output ─────────────────────────────────────────
    analysis = st.session_state.get("db_last_analysis")
    if analysis:
        st.divider()
        st.subheader("Mistral Analysis")

        if analysis.get("error"):
            st.error(f"Agent error: {analysis['error']}")
        else:
            sev = analysis.get("severity", "unknown")
            sev_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")

            c1, c2, c3 = st.columns(3)
            c1.metric("Severity", f"{sev_color} {sev.title()}")
            c2.metric("Category", analysis.get("category", "—").replace("_", " ").title())
            c3.metric("Confidence", f"{analysis.get('confidence', 0):.0%}")

            st.markdown("**Root Cause**")
            st.info(analysis.get("root_cause", "—"))

            st.markdown("**Recommended Fix**")
            st.success(analysis.get("recommended_fix", "—"))

            fix_code = analysis.get("fix_code", "").strip()
            if fix_code:
                st.markdown("**Code Change**")
                lang = "sql" if fix_code.strip().upper().startswith("SELECT") else "python"
                st.code(fix_code, language=lang)
                fix_loc = analysis.get("fix_location", "")
                if fix_loc:
                    st.caption(f"Apply at: `{fix_loc}`")

            prevention = analysis.get("prevention", "").strip()
            if prevention:
                with st.expander("Prevention Strategy"):
                    st.write(prevention)

            st.caption(
                f"Model: `{analysis.get('model', '—')}` | "
                f"Tokens: {analysis.get('tokens_used', 0):,}"
            )

            # Save to Knowledge Base
            st.divider()
            col_save, col_wandb = st.columns(2)
            save_wandb = col_wandb.checkbox("Also log to W&B knowledge base", value=True)
            if col_save.button("Save to Knowledge Base", type="primary"):
                try:
                    kb_result = _post(
                        "/databricks/knowledge",
                        {
                            "error_category": st.session_state.get("db_last_category", "other"),
                            "error_description": st.session_state.get("db_last_error", ""),
                            "job_name": st.session_state.get("db_last_job", ""),
                            "task_key": st.session_state.get("db_last_task", ""),
                            "analysis": analysis,
                            "log_to_wandb": save_wandb,
                        },
                    )
                    st.success(f"Saved — KB ID: `{kb_result['kb_id']}`")
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

    # ── SECTION 5: Knowledge Base ──────────────────────────────────────────
    st.divider()
    st.subheader("Knowledge Base — Error/Fix Library")
    st.caption(
        "Saved investigations are stored in DuckDB and uploaded to W&B as dataset artifacts "
        "for future Mistral fine-tuning."
    )

    try:
        kb_entries = _get("/databricks/knowledge")
    except Exception:
        kb_entries = []

    if not kb_entries:
        st.info("No entries yet. Investigate an error and click **Save to Knowledge Base**.")
    else:
        kb_df = pd.DataFrame(kb_entries)
        display_cols = [c for c in ["created_at", "error_category", "job_name", "task_key",
                                    "root_cause", "confidence", "logged_to_wandb"] if c in kb_df.columns]
        st.dataframe(
            kb_df[display_cols].rename(columns={
                "created_at": "Time", "error_category": "Category",
                "job_name": "Job", "task_key": "Task",
                "root_cause": "Root Cause", "confidence": "Confidence",
                "logged_to_wandb": "W&B",
            }),
            use_container_width=True,
        )

        # Expand a single entry to see full details
        if len(kb_entries) > 0:
            kb_ids = [e["kb_id"][:8] + "…" for e in kb_entries]
            selected_idx = st.selectbox("View full entry", range(len(kb_ids)), format_func=lambda i: kb_ids[i])
            entry = kb_entries[selected_idx]
            with st.expander("Full Entry", expanded=True):
                st.markdown(f"**Root cause:** {entry.get('root_cause', '')}")
                st.markdown(f"**Recommended fix:** {entry.get('recommended_fix', '')}")
                if entry.get("fix_code"):
                    st.code(entry["fix_code"], language="python")
