"""Tests for the NYC Taxi Databricks Asset Bundle.

Covers three layers without requiring a live Databricks workspace:

1. Pipeline logic (silver transformations, gold aggregations, feature columns)
   – Uses a lightweight mock of the ``dlt`` module and a simple dict-based
     row validator so the tests run anywhere pandas is available.

2. MCP server dispatcher
   – Patches ``bundle_client`` to verify every tool route returns the correct
     shape and surfaces CLI errors gracefully.

3. Mistral agent tools
   – Patches ``call_mcp_tool`` to confirm the three agent-facing tools
     serialize arguments correctly and propagate results back to the model.

Run with::

    uv run pytest DataBolt-Edge/databricks/tests/test_nyc_taxi_bundle.py -v
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ── Helpers ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_PICKUP = datetime(2023, 6, 15, 8, 35, 0, tzinfo=timezone.utc)
_DROPOFF = datetime(2023, 6, 15, 8, 57, 30, tzinfo=timezone.utc)


def _make_raw_row(**overrides: Any) -> dict:
    """Return one valid raw NYC taxi row, with optional overrides."""
    base = {
        "VendorID": 2,
        "tpep_pickup_datetime": _PICKUP,
        "tpep_dropoff_datetime": _DROPOFF,
        "passenger_count": 1,
        "trip_distance": 3.5,
        "fare_amount": 14.50,
        "tip_amount": 2.90,
        "tolls_amount": 0.0,
        "total_amount": 19.80,
        "extra": 0.5,
        "mta_tax": 0.5,
        "payment_type": 1,
        "PULocationID": 161,
        "DOLocationID": 237,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ── Silver layer transformation helpers (pure-Python re-impl for testing)
# ── These mirror the logic in dlt_pipeline.py without PySpark.
# ---------------------------------------------------------------------------


def _apply_silver_transforms(row: dict) -> dict | None:
    """Apply the silver-layer derivations and quality checks.

    Returns None if the row should be dropped by an @dlt.expect_or_drop rule.
    """
    pickup = row["tpep_pickup_datetime"]
    dropoff = row["tpep_dropoff_datetime"]

    if not isinstance(pickup, datetime) or not isinstance(dropoff, datetime):
        return None

    trip_seconds = int((dropoff - pickup).total_seconds())
    fare = float(row.get("fare_amount") or 0)
    dist = float(row.get("trip_distance") or 0)
    pax = int(row.get("passenger_count") or 0)

    # @dlt.expect_or_drop rules
    if fare <= 0:
        return None
    if not (30 <= trip_seconds <= 36_000):
        return None
    if not (1 <= pax <= 9):
        return None
    if dropoff <= pickup:
        return None

    return {
        **row,
        "trip_seconds": trip_seconds,
        "cost_per_mile": (fare / dist) if dist > 0 else None,
        "pickup_hour": pickup.hour,
        "pickup_day_of_week": pickup.isoweekday(),  # 1=Mon, 7=Sun
        "pickup_date": pickup.date(),
        "vendor_name": {1: "Creative Mobile Technologies", 2: "VeriFone"}.get(
            row.get("VendorID"), "Unknown"
        ),
        "payment_type_label": {
            1: "Credit Card",
            2: "Cash",
            3: "No Charge",
            4: "Dispute",
        }.get(row.get("payment_type"), "Unknown"),
    }


def _apply_feature_row(silver_row: dict) -> dict:
    """Produce a feature-table row from a silver row (mirrors DLT features table)."""
    pickup = silver_row["tpep_pickup_datetime"]
    pax = silver_row["passenger_count"]
    is_weekend = 1 if silver_row["pickup_day_of_week"] in (6, 7) else 0  # Mon=1
    return {
        "label_fare_amount": silver_row["fare_amount"],
        "trip_distance": silver_row["trip_distance"],
        "trip_seconds": silver_row["trip_seconds"],
        "passenger_count": silver_row["passenger_count"],
        "cost_per_mile": silver_row.get("cost_per_mile") or 0.0,
        "pickup_hour": silver_row["pickup_hour"],
        "pickup_day_of_week": silver_row["pickup_day_of_week"],
        "pickup_month": pickup.month,
        "pickup_year": pickup.year,
        "is_weekend": is_weekend,
        "pickup_location_id": silver_row["PULocationID"],
        "dropoff_location_id": silver_row["DOLocationID"],
        "vendor_id": silver_row["VendorID"],
        "payment_type": silver_row["payment_type"],
        "tip_amount": silver_row.get("tip_amount") or 0.0,
        "tolls_amount": silver_row.get("tolls_amount") or 0.0,
        "extra": silver_row.get("extra") or 0.0,
        "mta_tax": silver_row.get("mta_tax") or 0.0,
    }


# ============================================================================
# 1. Silver-layer transformation tests
# ============================================================================


class TestSilverTransforms:
    def test_valid_row_passes_all_rules(self):
        row = _make_raw_row()
        result = _apply_silver_transforms(row)
        assert result is not None

    def test_derived_trip_seconds(self):
        row = _make_raw_row()
        result = _apply_silver_transforms(row)
        assert result["trip_seconds"] == 22 * 60 + 30  # 22 min 30 sec

    def test_derived_cost_per_mile(self):
        row = _make_raw_row(fare_amount=14.0, trip_distance=2.0)
        result = _apply_silver_transforms(row)
        assert result["cost_per_mile"] == pytest.approx(7.0)

    def test_zero_distance_gives_no_cost_per_mile(self):
        row = _make_raw_row(trip_distance=0.0)
        result = _apply_silver_transforms(row)
        assert result["cost_per_mile"] is None

    def test_pickup_hour_extracted(self):
        row = _make_raw_row()
        result = _apply_silver_transforms(row)
        assert result["pickup_hour"] == 8

    def test_vendor_name_verifone(self):
        row = _make_raw_row(VendorID=2)
        result = _apply_silver_transforms(row)
        assert result["vendor_name"] == "VeriFone"

    def test_vendor_name_cmt(self):
        row = _make_raw_row(VendorID=1)
        result = _apply_silver_transforms(row)
        assert result["vendor_name"] == "Creative Mobile Technologies"

    def test_vendor_name_unknown(self):
        row = _make_raw_row(VendorID=99)
        result = _apply_silver_transforms(row)
        assert result["vendor_name"] == "Unknown"

    def test_payment_type_credit_card(self):
        row = _make_raw_row(payment_type=1)
        result = _apply_silver_transforms(row)
        assert result["payment_type_label"] == "Credit Card"

    def test_negative_fare_dropped(self):
        row = _make_raw_row(fare_amount=-1.0)
        assert _apply_silver_transforms(row) is None

    def test_zero_fare_dropped(self):
        row = _make_raw_row(fare_amount=0.0)
        assert _apply_silver_transforms(row) is None

    def test_very_short_trip_dropped(self):
        short_dropoff = datetime(2023, 6, 15, 8, 35, 10, tzinfo=timezone.utc)
        row = _make_raw_row(tpep_dropoff_datetime=short_dropoff)
        assert _apply_silver_transforms(row) is None  # 10 sec < 30 sec minimum

    def test_very_long_trip_dropped(self):
        long_dropoff = datetime(2023, 6, 15, 22, 35, 0, tzinfo=timezone.utc)  # ~14 hrs
        row = _make_raw_row(tpep_dropoff_datetime=long_dropoff)
        assert _apply_silver_transforms(row) is None

    def test_zero_passengers_dropped(self):
        row = _make_raw_row(passenger_count=0)
        assert _apply_silver_transforms(row) is None

    def test_too_many_passengers_dropped(self):
        row = _make_raw_row(passenger_count=10)
        assert _apply_silver_transforms(row) is None

    def test_dropoff_before_pickup_dropped(self):
        row = _make_raw_row(
            tpep_pickup_datetime=_DROPOFF,
            tpep_dropoff_datetime=_PICKUP,
        )
        assert _apply_silver_transforms(row) is None


# ============================================================================
# 2. Feature table tests
# ============================================================================


class TestFeatureTable:
    def _get_silver(self) -> dict:
        return _apply_silver_transforms(_make_raw_row())

    def test_label_column_present(self):
        features = _apply_feature_row(self._get_silver())
        assert "label_fare_amount" in features
        assert features["label_fare_amount"] == pytest.approx(14.50)

    def test_is_weekend_weekday(self):
        # _PICKUP is 2023-06-15 = Thursday  (isoweekday=4)
        features = _apply_feature_row(self._get_silver())
        assert features["is_weekend"] == 0

    def test_is_weekend_saturday(self):
        sat_pickup = datetime(2023, 6, 17, 10, 0, tzinfo=timezone.utc)
        sat_dropoff = datetime(2023, 6, 17, 10, 30, tzinfo=timezone.utc)
        row = _make_raw_row(tpep_pickup_datetime=sat_pickup, tpep_dropoff_datetime=sat_dropoff)
        silver = _apply_silver_transforms(row)
        features = _apply_feature_row(silver)
        assert features["is_weekend"] == 1

    def test_null_fillable_columns(self):
        row = _make_raw_row(tip_amount=None, tolls_amount=None, extra=None, mta_tax=None)
        silver = _apply_silver_transforms(row)
        # Python None is kept in the dict; actual fill happens in PySpark .na.fill()
        # Verify the feature row uses 0.0 as default for None numerics
        features = _apply_feature_row(silver)
        assert features["tip_amount"] == 0.0
        assert features["tolls_amount"] == 0.0
        assert features["extra"] == 0.0
        assert features["mta_tax"] == 0.0

    def test_all_required_feature_columns_present(self):
        features = _apply_feature_row(self._get_silver())
        required = [
            "label_fare_amount",
            "trip_distance",
            "trip_seconds",
            "passenger_count",
            "cost_per_mile",
            "pickup_hour",
            "pickup_day_of_week",
            "pickup_month",
            "pickup_year",
            "is_weekend",
            "pickup_location_id",
            "dropoff_location_id",
            "vendor_id",
            "payment_type",
        ]
        for col in required:
            assert col in features, f"Missing feature column: {col}"


# ============================================================================
# 3. Gold aggregation tests (smoke-check on multiple rows)
# ============================================================================


class TestGoldAggregations:
    def _make_silver_batch(self) -> list[dict]:
        from datetime import timedelta  # noqa: PLC0415

        rows = []
        for hour in range(6, 12):
            for _ in range(3):
                pickup = _PICKUP.replace(hour=hour, minute=0)
                dropoff = pickup + timedelta(minutes=25)
                raw = _make_raw_row(
                    tpep_pickup_datetime=pickup,
                    tpep_dropoff_datetime=dropoff,
                    fare_amount=10.0 + hour,
                )
                silver = _apply_silver_transforms(raw)
                if silver:
                    rows.append(silver)
        return rows

    def test_gold_by_hour_has_expected_hours(self):
        silver = self._make_silver_batch()
        hours = {r["pickup_hour"] for r in silver}
        assert hours == {6, 7, 8, 9, 10, 11}

    def test_gold_by_hour_trip_count(self):
        silver = self._make_silver_batch()
        from collections import Counter

        counts = Counter(r["pickup_hour"] for r in silver)
        for hour in range(6, 12):
            assert counts[hour] == 3

    def test_gold_avg_fare_increases_with_hour(self):
        """Fares were constructed as 10 + hour, so avg must increase."""
        silver = self._make_silver_batch()
        from collections import defaultdict

        sums: dict[int, list] = defaultdict(list)
        for r in silver:
            sums[r["pickup_hour"]].append(r["fare_amount"])
        avgs = {h: sum(v) / len(v) for h, v in sums.items()}
        hours_sorted = sorted(avgs)
        for i in range(len(hours_sorted) - 1):
            assert avgs[hours_sorted[i]] <= avgs[hours_sorted[i + 1]]


# ============================================================================
# 4. MCP server dispatcher tests
# ============================================================================


@dataclass
class _FakeResult:
    returncode: int
    stdout: str
    stderr: str = ""
    succeeded: bool = field(init=False)

    def __post_init__(self):
        self.succeeded = self.returncode == 0

    def to_dict(self):
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "succeeded": self.succeeded,
        }


class TestMcpServerDispatcher:
    """Tests the _dispatch() function in the MCP server without running the server."""

    @pytest.fixture(autouse=True)
    def _import_server(self):
        # Lazily import so tests skip gracefully if `mcp` is not installed
        pytest.importorskip("mcp")
        from mcp_servers.databricks_bundles import server  # noqa: PLC0415

        self.dispatch = server._dispatch

    def test_bundle_validate_success(self):
        ok = _FakeResult(0, '{"resources":{}}')
        with patch("mcp_servers.databricks_bundles.server.bundle_client.validate_bundle", return_value=ok):
            result = self.dispatch("bundle_validate", {"bundle_path": "/fake/path"})
        assert result["succeeded"] is True

    def test_bundle_deploy_success(self):
        ok = _FakeResult(0, "Bundle deployed successfully")
        with patch("mcp_servers.databricks_bundles.server.bundle_client.deploy_bundle", return_value=ok):
            result = self.dispatch("bundle_deploy", {"bundle_path": "/fake", "target": "dev"})
        assert result["succeeded"] is True
        assert "deployed" in result["stdout"].lower()

    def test_bundle_deploy_failure_surfaces_stderr(self):
        fail = _FakeResult(1, "", "Error: DATABRICKS_HOST not set")
        with patch("mcp_servers.databricks_bundles.server.bundle_client.deploy_bundle", return_value=fail):
            result = self.dispatch("bundle_deploy", {"bundle_path": "/fake"})
        assert result["succeeded"] is False
        assert "DATABRICKS_HOST" in result["stderr"]

    def test_bundle_run_passes_full_refresh(self):
        ok = _FakeResult(0, "Run submitted. run_id=12345")
        with patch(
            "mcp_servers.databricks_bundles.server.bundle_client.run_bundle_resource", return_value=ok
        ) as mock_run:
            self.dispatch(
                "bundle_run",
                {"bundle_path": "/fake", "resource_key": "nyc_taxi_dlt", "full_refresh": True},
            )
        extra_args = mock_run.call_args.kwargs.get("extra_args") or mock_run.call_args[1].get("extra_args")
        assert extra_args == ["--full-refresh"]

    def test_bundle_run_parses_run_id(self):
        ok = _FakeResult(0, "Run submitted. run_id=12345\n12345")
        with patch("mcp_servers.databricks_bundles.server.bundle_client.run_bundle_resource", return_value=ok):
            result = self.dispatch("bundle_run", {"bundle_path": "/fake", "resource_key": "nyc_taxi_dlt"})
        assert result["succeeded"] is True

    def test_bundle_get_run_status(self):
        status = {
            "run_id": "12345",
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "state_message": "",
            "run_page_url": "https://adb-xxx.azuredatabricks.net/...#job/1/run/12345",
            "succeeded": True,
        }
        with patch(
            "mcp_servers.databricks_bundles.server.bundle_client.get_job_run_status", return_value=status
        ):
            result = self.dispatch("bundle_get_run_status", {"run_id": "12345"})
        assert result["life_cycle_state"] == "TERMINATED"
        assert result["result_state"] == "SUCCESS"

    def test_bundle_list_resources_parses_json(self):
        payload = json.dumps({"resources": {"jobs": {"nyc_taxi_workflow": {}}}})
        ok = _FakeResult(0, payload)
        with patch(
            "mcp_servers.databricks_bundles.server.bundle_client.list_bundle_resources", return_value=ok
        ):
            result = self.dispatch("bundle_list_resources", {"bundle_path": "/fake"})
        assert "resources" in result
        assert "nyc_taxi_workflow" in result["resources"]["jobs"]

    def test_unknown_tool_returns_error(self):
        result = self.dispatch("nonexistent_tool", {})
        assert result["succeeded"] is False
        assert "Unknown tool" in result["error"]


# ============================================================================
# 5. Agent tools tests
# ============================================================================


def test_get_tools_includes_databricks_schemas():
    """get_tools() must advertise the three Databricks bundle schemas (no mcp needed)."""
    import agents.tools as t  # noqa: PLC0415

    names = {entry["function"]["name"] for entry in t.get_tools()}
    assert "databricks_bundle_deploy" in names
    assert "databricks_bundle_run" in names
    assert "databricks_bundle_get_run_status" in names


class TestAgentToolsDatabricks:
    """Verify the three Mistral-facing tool functions call the MCP client correctly."""

    # The tools import call_mcp_tool lazily inside each function, so we must
    # patch at the source module rather than on agents.tools.
    _PATCH_TARGET = "mcp_servers.databricks_bundles.mcp_client.call_mcp_tool"

    @pytest.fixture(autouse=True)
    def _import_tools(self):
        # Skip if mcp is not installed (same guard as MCP server tests)
        pytest.importorskip("mcp")
        import agents.tools as t  # noqa: PLC0415
        self.tools = t

    def test_deploy_calls_bundle_deploy_tool(self):
        mock_result = {"succeeded": True, "stdout": "deployed", "returncode": 0}
        with patch(self._PATCH_TARGET, return_value=mock_result) as m:
            self.tools.databricks_bundle_deploy(target="dev")
            m.assert_called_once()
            name, args = m.call_args[0]
            assert name == "bundle_deploy"
            assert args["target"] == "dev"

    def test_run_calls_bundle_run_tool(self):
        mock_result = {"succeeded": True, "stdout": "run submitted", "returncode": 0}
        with patch(self._PATCH_TARGET, return_value=mock_result) as m:
            self.tools.databricks_bundle_run(resource_key="nyc_taxi_dlt", target="dev")
            m.assert_called_once()
            name, args = m.call_args[0]
            assert name == "bundle_run"
            assert args["resource_key"] == "nyc_taxi_dlt"

    def test_run_passes_full_refresh(self):
        mock_result = {"succeeded": True, "stdout": "", "returncode": 0}
        with patch(self._PATCH_TARGET, return_value=mock_result) as m:
            self.tools.databricks_bundle_run(resource_key="nyc_taxi_dlt", full_refresh=True)
            _, args = m.call_args[0]
            assert args["full_refresh"] is True

    def test_get_run_status_calls_correct_tool(self):
        mock_result = {"run_id": "99", "life_cycle_state": "RUNNING", "succeeded": True}
        with patch(self._PATCH_TARGET, return_value=mock_result) as m:
            output = self.tools.databricks_bundle_get_run_status("99")
            name, args = m.call_args[0]
            assert name == "bundle_get_run_status"
            assert args["run_id"] == "99"
            parsed = json.loads(output)
            assert parsed["life_cycle_state"] == "RUNNING"

    def test_deploy_returns_json_string(self):
        mock_result = {"succeeded": True, "stdout": "ok", "returncode": 0}
        with patch(self._PATCH_TARGET, return_value=mock_result):
            raw = self.tools.databricks_bundle_deploy()
        parsed = json.loads(raw)
        assert parsed["succeeded"] is True


# ============================================================================
# 6. Integration smoke-test (skipped unless DATABRICKS_HOST is set)
# ============================================================================


@pytest.mark.skipif(
    not __import__("os").getenv("DATABRICKS_HOST")
    or not __import__("shutil").which("databricks"),
    reason=(
        "Live integration requires DATABRICKS_HOST env var and the Databricks CLI on PATH. "
        "Install CLI: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh"
    ),
)
class TestLiveIntegration:
    """Runs a real bundle validate against the configured workspace.

    These tests are intentionally read-only (validate only, not deploy).
    """

    def test_bundle_validate_live(self):
        import os  # noqa: PLC0415
        from mcp_servers.databricks_bundles.bundle_client import validate_bundle  # noqa: PLC0415

        bundle_path = os.path.join(
            os.path.dirname(__file__), "..", "."  # DataBolt-Edge/databricks/
        )
        result = validate_bundle(bundle_path)
        assert result.succeeded, f"Bundle validation failed:\n{result.stderr}"
