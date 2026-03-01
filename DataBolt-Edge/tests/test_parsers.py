"""Unit tests for DataBolt Edge log parsers."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure DataBolt-Edge is importable from the test runner
EDGE_ROOT = Path(__file__).resolve().parents[1]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

import pytest

from parsers.spark import parse_spark, SparkParseResult
from parsers.airflow import parse_airflow, AirflowParseResult
from parsers.sql_plan import parse_sql_plan, SqlPlanResult


# ════════════════════════════════════════════════════════════════════════════
# Spark parser tests
# ════════════════════════════════════════════════════════════════════════════

SPARK_OOM_LOG = """\
24/02/28 14:33:01 ERROR Executor: Exception in task 3.0 in stage 5.0 (TID 47)
java.lang.OutOfMemoryError: GC overhead limit exceeded
    at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage2.processNext(Unknown Source)
    at org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)
24/02/28 14:33:01 WARN  TaskSetManager: Lost task 3.0 in stage 5.0 (TID 47, executor 2): TaskKilled (another attempt succeeded)
24/02/28 14:33:01 ERROR TaskSchedulerImpl: Lost executor 2 on 10.0.0.5: Remote RPC client disassociated.
"""

SPARK_SHUFFLE_LOG = """\
24/02/28 15:00:01 ERROR MapOutputTrackerMasterEndpoint: Error communicating with MapOutputTracker
org.apache.spark.shuffle.FetchFailed: Failed to connect to host1/10.0.0.2:7337
    at org.apache.spark.shuffle.BlockStoreShuffleReader.read(BlockStoreShuffleReader.scala:73)
24/02/28 15:00:02 WARN  DAGScheduler: Resubmitting stage 3 (some-rdd) due to fetch failure
24/02/28 15:00:03 ERROR TaskSchedulerImpl: Lost executor 1 on host1: Remote RPC disassociated.
"""

SPARK_EMPTY_LOG = """
24/02/28 12:00:00 INFO SparkContext: Running Spark version 3.5.0
24/02/28 12:00:05 INFO SparkUI: Bound SparkUI to 0.0.0.0, port 4040
"""


class TestSparkParser:
    def test_oom_detected(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert r.has_oom is True

    def test_error_count(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert r.error_count == 2

    def test_warn_count(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert r.warn_count == 1

    def test_exception_extracted(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert any("OutOfMemoryError" in e for e in r.exceptions)

    def test_stage_and_task(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert "5.0" in r.stages
        assert "3.0" in r.tasks

    def test_tid_extracted(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert "47" in r.tids

    def test_executor_and_host(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert "2" in r.executors
        assert any("10.0.0.5" in h for h in r.hosts)

    def test_categories_oom(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert any("OOM" in c or "Memory" in c for c in r.categories)

    def test_categories_executor_lost(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert any("Executor" in c for c in r.categories)

    def test_first_exception_block_non_empty(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert "OutOfMemoryError" in r.first_exception_block

    def test_error_lines_capped(self):
        r = parse_spark(SPARK_OOM_LOG)
        assert len(r.error_lines) <= 10

    def test_shuffle_category(self):
        r = parse_spark(SPARK_SHUFFLE_LOG)
        assert any("Shuffle" in c for c in r.categories)

    def test_empty_log_no_errors(self):
        r = parse_spark(SPARK_EMPTY_LOG)
        assert r.error_count == 0
        assert r.has_oom is False

    def test_prompt_context_non_empty(self):
        r = parse_spark(SPARK_OOM_LOG)
        ctx = r.to_prompt_context()
        assert len(ctx) > 20
        assert "ERROR" in ctx

    def test_return_type(self):
        assert isinstance(parse_spark(SPARK_OOM_LOG), SparkParseResult)


# ════════════════════════════════════════════════════════════════════════════
# Airflow parser tests
# ════════════════════════════════════════════════════════════════════════════

AIRFLOW_FILENOTFOUND_LOG = """\
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

AIRFLOW_WITH_METADATA = """\
dag_id=etl_pipeline, task_id=extract_sales_data, execution_date=2024-02-28T09:00:00+00:00
[2024-02-28 09:15:03,412] {taskinstance.py:1482} ERROR - Failed to execute task
ImportError: cannot import name 'read_parquet' from 'pandas'
[2024-02-28 09:15:03,415] {taskinstance.py:1492} ERROR - Task failed
"""

AIRFLOW_SENSOR_LOG = """\
[2024-02-28 10:30:00,000] {sensor.py:121} ERROR - AirflowSensorTimeout: Sensor timed out; failing task.
[2024-02-28 10:30:00,001] {taskinstance.py:1500} WARNING - Sensor poking timeout exceeded after 3600s
"""


class TestAirflowParser:
    def test_error_count(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        assert r.error_count >= 2

    def test_traceback_extracted(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        assert len(r.tracebacks) >= 1

    def test_exception_class_in_traceback(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        assert any("FileNotFoundError" in tb.exception_class for tb in r.tracebacks)

    def test_traceback_has_frames(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        tb = r.tracebacks[0]
        assert len(tb.frames) >= 1

    def test_traceback_frame_file(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        tb = r.tracebacks[0]
        assert any("/opt/airflow" in f["file"] for f in tb.frames)

    def test_exception_classes_list(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        assert "FileNotFoundError" in r.exception_classes

    def test_category_file_not_found(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        assert any("File not found" in c or "File" in c for c in r.categories)

    def test_dag_task_extraction(self):
        r = parse_airflow(AIRFLOW_WITH_METADATA)
        assert r.dag_id == "etl_pipeline"
        assert r.task_id == "extract_sales_data"

    def test_execution_date_extraction(self):
        r = parse_airflow(AIRFLOW_WITH_METADATA)
        assert "2024-02-28" in r.execution_date

    def test_import_error_category(self):
        r = parse_airflow(AIRFLOW_WITH_METADATA)
        assert any("Import" in c for c in r.categories)

    def test_sensor_timeout_category(self):
        r = parse_airflow(AIRFLOW_SENSOR_LOG)
        assert any("Sensor" in c for c in r.categories)

    def test_traceback_summary_non_empty(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        assert r.tracebacks[0].summary()

    def test_prompt_context_non_empty(self):
        r = parse_airflow(AIRFLOW_FILENOTFOUND_LOG)
        ctx = r.to_prompt_context()
        assert len(ctx) > 10

    def test_return_type(self):
        assert isinstance(parse_airflow(AIRFLOW_FILENOTFOUND_LOG), AirflowParseResult)

    def test_empty_log(self):
        r = parse_airflow("No log content here.")
        assert r.error_count == 0
        assert r.tracebacks == []


# ════════════════════════════════════════════════════════════════════════════
# SQL plan parser tests
# ════════════════════════════════════════════════════════════════════════════

SQL_TEXT_EXPLAIN = """\
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

SQL_FILESORT = """\
-> Sort: o.created_at  (cost=100000.00 rows=500000)
    -> Table scan on orders  (cost=90000.00 rows=500000)
    Using filesort
"""

SQL_JSON_PLAN = """\
{
  "query_block": {
    "table": {
      "table_name": "orders",
      "access_type": "ALL",
      "rows_examined_per_scan": 5000000
    }
  }
}
"""

SQL_CARTESIAN = """\
-> Nested Loop(cost=9999999.00 rows=50000000)
    -> Table scan on products (cost=1500.00 rows=5000)
    CROSS JOIN
    -> Table scan on categories (cost=100.00 rows=10000)
"""

SQL_CLEAN = """\
-> Index lookup on orders using idx_created_at (created_at > '2024-01-01')
   (cost=1200.00 rows=12000)
-> Ref access on customers using PRIMARY (id=o.customer_id)
   (cost=450.00 rows=4500)
"""


class TestSqlPlanParser:
    def test_full_table_scan_detected(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        assert r.has_full_table_scan is True

    def test_scan_tables_found(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        table_names = [s.table for s in r.scans]
        assert "orders" in table_names

    def test_multiple_scans(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        assert len(r.scans) >= 2

    def test_hash_join_detected(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        join_types = [j.join_type for j in r.joins]
        assert any("Hash" in jt for jt in join_types)

    def test_max_rows(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        assert r.max_rows >= 4_983_201

    def test_total_cost(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        assert r.total_cost >= 500_000

    def test_no_index_filter_detected(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        assert r.has_no_index_filter is True

    def test_warnings_generated(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        assert len(r.warnings) > 0

    def test_large_table_warning_present(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        assert any("Full table scan" in w or "full scan" in w.lower() or "rows" in w.lower() for w in r.warnings)

    def test_filesort_detected(self):
        r = parse_sql_plan(SQL_FILESORT)
        assert r.has_filesort is True

    def test_filesort_warning(self):
        r = parse_sql_plan(SQL_FILESORT)
        assert any("filesort" in w.lower() or "Filesort" in w for w in r.warnings)

    def test_json_plan_full_scan(self):
        r = parse_sql_plan(SQL_JSON_PLAN)
        assert r.has_full_table_scan is True

    def test_json_plan_table_name(self):
        r = parse_sql_plan(SQL_JSON_PLAN)
        assert any(s.table == "orders" for s in r.scans)

    def test_cartesian_detected(self):
        r = parse_sql_plan(SQL_CARTESIAN)
        assert r.has_cartesian_join is True

    def test_cartesian_warning(self):
        r = parse_sql_plan(SQL_CARTESIAN)
        assert any("artesian" in w or "cross" in w.lower() for w in r.warnings)

    def test_prompt_context_non_empty(self):
        r = parse_sql_plan(SQL_TEXT_EXPLAIN)
        ctx = r.to_prompt_context()
        assert len(ctx) > 10

    def test_return_type(self):
        assert isinstance(parse_sql_plan(SQL_TEXT_EXPLAIN), SqlPlanResult)

    def test_empty_plan(self):
        r = parse_sql_plan("")
        assert isinstance(r, SqlPlanResult)
        assert not r.warnings  # should not crash, just empty
