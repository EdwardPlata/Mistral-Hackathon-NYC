# Databricks notebook source
"""Download NYC Yellow Taxi trip data from the TLC CloudFront CDN into DBFS.

Data source: https://github.com/toddwschneider/nyc-taxi-data
CDN URL pattern:
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet

Writes parquet files to dbfs:/tmp/nyc-taxi-raw/yellow/ using the DBFS REST API
(chunked streaming upload) — works on classic and serverless compute without
requiring any local filesystem mount.

Parameters (workflow base_parameters):
  start_year_month   – first month to download, e.g. "2023-01"
  end_year_month     – last month to download (inclusive), e.g. "2023-03"
  target_dbfs_path   – DBFS destination directory (default: dbfs:/tmp/nyc-taxi-raw/yellow)
"""

# COMMAND ----------

import base64
import json
import urllib.request

dbutils.widgets.text("start_year_month", "2023-01")  # noqa: F821
dbutils.widgets.text("end_year_month", "2023-03")  # noqa: F821
dbutils.widgets.text("target_dbfs_path", "dbfs:/tmp/nyc-taxi-raw/yellow")  # noqa: F821

start_ym = dbutils.widgets.get("start_year_month")  # noqa: F821
end_ym = dbutils.widgets.get("end_year_month")  # noqa: F821
target_dir = dbutils.widgets.get("target_dbfs_path")  # noqa: F821

_TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

print(f"Downloading yellow taxi data: {start_ym} → {end_ym}")
print(f"Target: {target_dir}")

# COMMAND ----------


def _ym_range(start: str, end: str):
    """Yield (year, month) tuples inclusive between start and end (YYYY-MM strings)."""
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# COMMAND ----------

# Resolve workspace host and bearer token from the notebook execution context.
# These are available on any Databricks compute type (classic, serverless, DLT).
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
_WORKSPACE_URL = ctx.apiUrl().get()
_TOKEN = ctx.apiToken().get()
_DBFS_API = f"{_WORKSPACE_URL}/api/2.0/dbfs"
_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MB per chunk

print(f"Workspace: {_WORKSPACE_URL}")


def _dbfs_request(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_DBFS_API}/{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _dbfs_exists(dbfs_path: str) -> bool:
    """Return True if *dbfs_path* already exists in DBFS."""
    try:
        dbutils.fs.ls(dbfs_path)  # noqa: F821
        return True
    except Exception:  # noqa: BLE001
        return False


def _upload_to_dbfs(url: str, dbfs_path: str) -> int:
    """Stream *url* into *dbfs_path* using the DBFS streaming API.

    Returns the number of bytes written.
    """
    # Open the upload handle (always overwrite)
    handle = _dbfs_request("create", {"path": dbfs_path, "overwrite": True})["handle"]

    total = 0
    with urllib.request.urlopen(url) as response:
        while True:
            chunk = response.read(_CHUNK_BYTES)
            if not chunk:
                break
            _dbfs_request(
                "add-block",
                {"handle": handle, "data": base64.b64encode(chunk).decode()},
            )
            total += len(chunk)

    _dbfs_request("close", {"handle": handle})
    return total


# COMMAND ----------

# Ensure target directory exists (no-op if already present)
dbutils.fs.mkdirs(target_dir)  # noqa: F821

downloaded, skipped = [], []

for year, month in _ym_range(start_ym, end_ym):
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    url = f"{_TLC_BASE}/{filename}"
    dbfs_dest = f"{target_dir}/{filename}"

    if _dbfs_exists(dbfs_dest):
        print(f"  skip  {filename}  (already present in DBFS)")
        skipped.append(filename)
        continue

    print(f"  fetch {filename}  ← {url}")
    try:
        nbytes = _upload_to_dbfs(url, dbfs_dest)
        print(f"         done   {nbytes / 1_048_576:.1f} MB  →  {dbfs_dest}")
        downloaded.append(filename)
    except Exception as exc:  # noqa: BLE001
        print(f"         WARN   {filename}: {exc}")

# COMMAND ----------

print(f"\nSummary: {len(downloaded)} downloaded, {len(skipped)} skipped")
files = dbutils.fs.ls(target_dir)  # noqa: F821
total_bytes = sum(f.size for f in files)
print(f"Files in {target_dir}: {len(files)} ({total_bytes / 1_048_576:.0f} MB total)")
for f in files:
    print(f"  {f.name}  {f.size / 1_048_576:.1f} MB")
