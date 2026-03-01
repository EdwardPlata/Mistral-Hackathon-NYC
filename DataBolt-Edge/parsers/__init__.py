"""DataBolt Edge — log parsers.

Each parser takes raw log text and returns a structured result that can be
injected into an InferenceRequest prompt for richer, more targeted analysis.

Usage::

    from parsers import parse_spark, parse_airflow, parse_sql_plan

    result = parse_spark(log_text)
    result = parse_airflow(log_text)
    result = parse_sql_plan(log_text)
"""

from .spark import SparkParseResult, parse_spark
from .airflow import AirflowParseResult, parse_airflow
from .sql_plan import SqlPlanResult, parse_sql_plan

__all__ = [
    "SparkParseResult",
    "parse_spark",
    "AirflowParseResult",
    "parse_airflow",
    "SqlPlanResult",
    "parse_sql_plan",
]
