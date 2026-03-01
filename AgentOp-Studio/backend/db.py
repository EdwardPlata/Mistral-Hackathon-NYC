"""DuckDB database layer for AgentOps-Studio.

Opens a fresh connection per call (DuckDB is not thread-safe with shared
connections when used across threads).
"""

import os

import duckdb

DB_PATH = os.getenv(
    "AGENTOPS_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "agentops.duckdb"),
)


def get_conn() -> duckdb.DuckDBPyConnection:
    """Return a new DuckDB connection to the configured database file."""
    return duckdb.connect(DB_PATH)


def init_schema() -> None:
    """Create all AgentOps tables if they do not already exist."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id          VARCHAR PRIMARY KEY,
            agent_id        VARCHAR,
            user_id         VARCHAR,
            start_time      TIMESTAMP,
            end_time        TIMESTAMP,
            status          VARCHAR,
            total_tokens    INTEGER,
            total_cost      DOUBLE,
            config          JSON,
            initial_prompt  TEXT,
            final_response  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id   VARCHAR PRIMARY KEY,
            run_id       VARCHAR,
            role         VARCHAR,
            content      TEXT,
            timestamp    TIMESTAMP,
            token_count  INTEGER,
            finish_reason VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_calls (
            call_id      VARCHAR PRIMARY KEY,
            run_id       VARCHAR,
            message_id   VARCHAR,
            tool_name    VARCHAR,
            args         JSON,
            return_value TEXT,
            latency_ms   INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_snapshots (
            memory_id   VARCHAR PRIMARY KEY,
            run_id      VARCHAR,
            timestamp   TIMESTAMP,
            memory_json JSON
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diffs (
            diff_id           VARCHAR PRIMARY KEY,
            run_id            VARCHAR,
            parameter_changed VARCHAR,
            before_output     TEXT,
            after_output      TEXT,
            diff_json         JSON
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id      VARCHAR PRIMARY KEY,
            run_id       VARCHAR,
            metric_name  VARCHAR,
            metric_value DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            kb_id             VARCHAR PRIMARY KEY,
            created_at        TIMESTAMP,
            error_category    VARCHAR,
            error_description TEXT,
            job_name          VARCHAR,
            task_key          VARCHAR,
            root_cause        TEXT,
            recommended_fix   TEXT,
            fix_code          TEXT,
            fix_location      VARCHAR,
            prevention        TEXT,
            confidence        DOUBLE,
            mistral_model     VARCHAR,
            tokens_used       INTEGER,
            logged_to_wandb   BOOLEAN
        )
    """)
    conn.close()
