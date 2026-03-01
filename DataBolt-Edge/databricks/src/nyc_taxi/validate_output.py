# Databricks notebook source
# MAGIC %md
# MAGIC ## NYC Taxi Pipeline Output Validation
# MAGIC
# MAGIC Spot-checks row counts and column nullability on the gold and feature tables.
# MAGIC Fails the workflow task if any assertion fails.

# COMMAND ----------

import sys

dbutils.widgets.text("catalog", "main")   # noqa: F821
dbutils.widgets.text("schema", "nyc_taxi")

catalog = dbutils.widgets.get("catalog")   # noqa: F821
schema  = dbutils.widgets.get("schema")    # noqa: F821

print(f"Validating tables in {catalog}.{schema}")

# COMMAND ----------

def assert_min_rows(table: str, min_rows: int) -> None:
    count = spark.table(f"{catalog}.{schema}.{table}").count()  # noqa: F821
    assert count >= min_rows, (
        f"FAIL: {table} has {count} rows, expected >= {min_rows}"
    )
    print(f"  OK  {table}: {count:,} rows")


def assert_no_nulls(table: str, columns: list) -> None:
    df = spark.table(f"{catalog}.{schema}.{table}")  # noqa: F821
    for col in columns:
        null_count = df.filter(df[col].isNull()).count()
        assert null_count == 0, (
            f"FAIL: {table}.{col} has {null_count} null rows"
        )
        print(f"  OK  {table}.{col}: no nulls")


# COMMAND ----------

print("── Row counts ─────────────────────────────────────────────────────────")
assert_min_rows("bronze_trips_raw",        1_000)
assert_min_rows("silver_trips_clean",      1_000)
assert_min_rows("gold_trips_by_hour",         24)
assert_min_rows("gold_trips_by_pickup_zone",   1)
assert_min_rows("gold_trips_by_vendor",        1)
assert_min_rows("features_fare_prediction", 1_000)

# COMMAND ----------

print("── Null checks ────────────────────────────────────────────────────────")
assert_no_nulls("silver_trips_clean", [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "fare_amount",
    "trip_distance",
    "passenger_count",
    "trip_seconds",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_date",
])

assert_no_nulls("features_fare_prediction", [
    "label_fare_amount",
    "trip_distance",
    "trip_seconds",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
])

# COMMAND ----------

print()
print("All validation checks passed.")
