# Databricks notebook source
"""NYC Taxi Delta Live Tables pipeline.

Four-layer medallion architecture:
  bronze  → raw ingestion from NYC TLC parquet files
  silver  → cleaned and validated trips
  gold    → aggregated summaries (by zone, hour, vendor)
  features → ML feature table for fare prediction

This notebook is referenced by the DLT pipeline defined in databricks.yml.
Runtime configuration keys (set in databricks.yml configuration block):
  nyc_taxi_source_path  – DBFS / cloud storage path to raw parquet files
  catalog               – Unity Catalog name
  schema                – target schema / database name

DLT manages table lineage; all tables are written to the catalog.schema
specified in the pipeline configuration.
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Runtime config (injected by DLT from pipeline configuration block)
# ---------------------------------------------------------------------------
_SOURCE_PATH = spark.conf.get(  # noqa: F821 – spark available in DLT context
    "nyc_taxi_source_path",
    "dbfs:/databricks-datasets/nyctaxi/tables/nyctaxi_yellow",
)


# ============================================================================
# BRONZE – raw ingestion
# ============================================================================


@dlt.table(
    name="bronze_trips_raw",
    comment="Raw NYC Yellow Taxi trip records – append-only, schema as-is.",
    table_properties={"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_trips_raw():
    """Ingest raw parquet files from NYC TLC source into a Delta table.

    Adds a load timestamp and the source file path for lineage.
    """
    return (
        spark.read.format("parquet")  # noqa: F821
        .option("mergeSchema", "true")
        .load(_SOURCE_PATH)
        .withColumn("_load_ts", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )


# ============================================================================
# SILVER – cleaned and validated trips
# ============================================================================

# Expectation thresholds
_MIN_FARE = 0.01
_MAX_FARE = 5_000.0
_MIN_TRIP_SECONDS = 30
_MAX_TRIP_SECONDS = 10 * 3600  # 10 hours


@dlt.table(
    name="silver_trips_clean",
    comment="Cleaned, type-cast, and validated NYC Taxi trips.",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("positive_fare", "fare_amount > 0")
@dlt.expect_or_drop("valid_trip_duration", "trip_seconds BETWEEN 30 AND 36000")
@dlt.expect_or_drop("valid_passenger_count", "passenger_count BETWEEN 1 AND 9")
@dlt.expect_or_drop("pickup_before_dropoff", "tpep_dropoff_datetime > tpep_pickup_datetime")
def silver_trips_clean():
    """Apply quality rules, derive computed columns, and cast to clean types."""
    return (
        dlt.read("bronze_trips_raw")
        # ── cast columns ─────────────────────────────────────────────────────
        .withColumn("tpep_pickup_datetime", F.col("tpep_pickup_datetime").cast(TimestampType()))
        .withColumn("tpep_dropoff_datetime", F.col("tpep_dropoff_datetime").cast(TimestampType()))
        .withColumn("fare_amount", F.col("fare_amount").cast(DoubleType()))
        .withColumn("tip_amount", F.col("tip_amount").cast(DoubleType()))
        .withColumn("total_amount", F.col("total_amount").cast(DoubleType()))
        .withColumn("trip_distance", F.col("trip_distance").cast(DoubleType()))
        .withColumn("passenger_count", F.col("passenger_count").cast(IntegerType()))
        .withColumn("PULocationID", F.col("PULocationID").cast(LongType()))
        .withColumn("DOLocationID", F.col("DOLocationID").cast(LongType()))
        .withColumn("VendorID", F.col("VendorID").cast(IntegerType()))
        # ── derived columns ───────────────────────────────────────────────────
        .withColumn(
            "trip_seconds",
            (
                F.unix_timestamp("tpep_dropoff_datetime")
                - F.unix_timestamp("tpep_pickup_datetime")
            ).cast(IntegerType()),
        )
        .withColumn(
            "cost_per_mile",
            F.when(F.col("trip_distance") > 0, F.col("fare_amount") / F.col("trip_distance")).otherwise(None),
        )
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        .withColumn(
            "vendor_name",
            F.when(F.col("VendorID") == 1, F.lit("Creative Mobile Technologies"))
            .when(F.col("VendorID") == 2, F.lit("VeriFone"))
            .otherwise(F.lit("Unknown")),
        )
        .withColumn(
            "payment_type_label",
            F.when(F.col("payment_type") == 1, F.lit("Credit Card"))
            .when(F.col("payment_type") == 2, F.lit("Cash"))
            .when(F.col("payment_type") == 3, F.lit("No Charge"))
            .when(F.col("payment_type") == 4, F.lit("Dispute"))
            .otherwise(F.lit("Unknown")),
        )
        # ── drop internal load columns ────────────────────────────────────────
        .drop("_load_ts", "_source_file")
    )


# ============================================================================
# GOLD – aggregated summaries
# ============================================================================


@dlt.table(
    name="gold_trips_by_hour",
    comment="Hourly trip volume and revenue aggregated across all pickup hours.",
    table_properties={"quality": "gold"},
)
def gold_trips_by_hour():
    """Aggregate trips by pickup hour of day (0–23)."""
    return (
        dlt.read("silver_trips_clean")
        .groupBy("pickup_hour")
        .agg(
            F.count("*").alias("trip_count"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.sum("fare_amount"), 2).alias("total_fare_revenue"),
            F.round(F.avg("trip_distance"), 2).alias("avg_distance_miles"),
            F.round(F.avg("trip_seconds") / 60.0, 1).alias("avg_duration_mins"),
            F.round(F.avg("tip_amount"), 2).alias("avg_tip"),
        )
        .orderBy("pickup_hour")
    )


@dlt.table(
    name="gold_trips_by_pickup_zone",
    comment="Per-zone trip counts, average fares, and distances.",
    table_properties={"quality": "gold"},
)
def gold_trips_by_pickup_zone():
    """Aggregate trips by NYC TLC pickup location ID."""
    return (
        dlt.read("silver_trips_clean")
        .groupBy("PULocationID")
        .agg(
            F.count("*").alias("trip_count"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.avg("trip_distance"), 2).alias("avg_distance_miles"),
            F.round(F.avg("tip_amount"), 2).alias("avg_tip"),
            F.round(F.sum("total_amount"), 2).alias("total_revenue"),
        )
        .orderBy(F.desc("trip_count"))
    )


@dlt.table(
    name="gold_trips_by_vendor",
    comment="Per-vendor daily trip volumes and revenue.",
    table_properties={"quality": "gold"},
)
def gold_trips_by_vendor():
    """Aggregate trips by vendor and pickup date."""
    return (
        dlt.read("silver_trips_clean")
        .groupBy("pickup_date", "vendor_name")
        .agg(
            F.count("*").alias("trip_count"),
            F.round(F.sum("fare_amount"), 2).alias("total_fare"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.avg("trip_distance"), 2).alias("avg_distance_miles"),
        )
        .orderBy("pickup_date", "vendor_name")
    )


# ============================================================================
# FEATURES – ML feature table for fare prediction
# ============================================================================


@dlt.table(
    name="features_fare_prediction",
    comment=(
        "Feature table for fare prediction models. "
        "One row per trip; label is fare_amount."
    ),
    table_properties={
        "quality": "gold",
        "delta.feature.allowColumnDefaults": "supported",
    },
)
def features_fare_prediction():
    """Produce a flat feature vector suitable for fare regression models.

    Features selected cover trip context, temporal patterns, and spatial info.
    All nulls in feature columns are filled with sensible defaults.
    """
    return (
        dlt.read("silver_trips_clean")
        .select(
            # ── label ────────────────────────────────────────────────────────
            F.col("fare_amount").alias("label_fare_amount"),
            # ── trip features ─────────────────────────────────────────────────
            F.col("trip_distance"),
            F.col("trip_seconds"),
            F.col("passenger_count"),
            F.col("cost_per_mile"),
            # ── temporal features ─────────────────────────────────────────────
            F.col("pickup_hour"),
            F.col("pickup_day_of_week"),
            F.month("tpep_pickup_datetime").alias("pickup_month"),
            F.year("tpep_pickup_datetime").alias("pickup_year"),
            (F.col("pickup_day_of_week").isin(1, 7)).cast(IntegerType()).alias("is_weekend"),
            # ── spatial features ──────────────────────────────────────────────
            F.col("PULocationID").alias("pickup_location_id"),
            F.col("DOLocationID").alias("dropoff_location_id"),
            # ── vendor / payment ──────────────────────────────────────────────
            F.col("VendorID").alias("vendor_id"),
            F.col("payment_type"),
            F.col("tip_amount"),
            F.col("tolls_amount"),
            F.col("extra"),
            F.col("mta_tax"),
        )
        # Fill nulls so downstream ML code receives a clean tensor
        .na.fill(
            {
                "trip_distance": 0.0,
                "trip_seconds": 0,
                "passenger_count": 1,
                "cost_per_mile": 0.0,
                "tip_amount": 0.0,
                "tolls_amount": 0.0,
                "extra": 0.0,
                "mta_tax": 0.0,
            }
        )
    )
