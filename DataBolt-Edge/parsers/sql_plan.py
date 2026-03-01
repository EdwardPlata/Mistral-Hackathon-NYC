"""SQL query plan parser for DataBolt Edge.

Supports both:
  1. MySQL / PostgreSQL / Spark text EXPLAIN output (indented operators)
  2. JSON EXPLAIN output (MySQL EXPLAIN FORMAT=JSON, Spark logical plans)

Identifies costly operations:
  - Full table scans (no index access)
  - Large row estimates
  - Cartesian products (CROSS JOIN / nested loop with no join condition)
  - Expensive aggregations / sorts with high cost
  - Missing indexes

Example text EXPLAIN input::

    -> Table scan on orders  (cost=521432.50 rows=4983201)
    -> Hash join (cost=1043215.00 rows=4983201)
        -> Table scan on customers  (cost=12450.00 rows=124500)
        -> Table scan on order_items  (cost=2987451.00 rows=28932847)
    Filter: (o.created_at > '2024-01-01')  -- no index on created_at
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


# ── compiled regexes ──────────────────────────────────────────────────────────

# Full / index scan line  (MySQL EXPLAIN text / pg EXPLAIN)
_RE_TABLE_SCAN = re.compile(
    r"(?:^|\s)(?:Table scan|ALL|Seq Scan|Full scan)\s+(?:on\s+)?(?P<table>\w+)",
    re.IGNORECASE | re.MULTILINE,
)

# Cost / rows estimates:   (cost=521432.50 rows=4983201)  or  rows: 29384
_RE_COST = re.compile(
    r"cost\s*[=:]\s*(?P<cost>[\d.]+)",
    re.IGNORECASE,
)
_RE_ROWS = re.compile(
    r"rows\s*[=:]\s*(?P<rows>[\d,]+)",
    re.IGNORECASE,
)

# Hash join / nested loop / merge join
_RE_JOIN = re.compile(
    r"(?P<join_type>Hash Join|Nested Loop|Merge Join|CROSS JOIN|BNL|Block Nested Loop|Cartesian)",
    re.IGNORECASE,
)

# Filter without index (comment or note about missing index)
_RE_NO_INDEX = re.compile(
    r"no\s+index|without\s+index|full.?table|filter.*--.*(?:no|missing|without)\s+index"
    r"|possible_keys\s*:\s*NULL",
    re.IGNORECASE,
)

# Sort / filesort
_RE_SORT = re.compile(r"Using filesort|Sort\s+\(|ORDER BY.*(?:sort|type=ALL)", re.IGNORECASE)

# Temporary table
_RE_TEMP_TABLE = re.compile(r"Using temporary|TempTableScan|temp_table", re.IGNORECASE)

# Aggregate operators
_RE_AGGREGATE = re.compile(r"Aggregate|GROUP BY|ROLLUP|groupingExpressions", re.IGNORECASE)

# Individual EXPLAIN row in MySQL tabular format:
#   id | select_type | table | ... | type | possible_keys | key | ... | rows | Extra
_RE_MYSQL_TABULAR_ROW = re.compile(
    r"^\s*\d+\s*\|.*\|\s*(?P<table>\w+)\s*\|.*\|\s*(?P<type>ALL|index|range|ref|eq_ref|const|system|NULL)"
    r"\s*\|\s*(?P<possible_keys>[^|]*)\|\s*(?P<key>[^|]*)\|[^|]*\|\s*(?P<rows>[\d,]+)\s*\|",
    re.MULTILINE,
)

# Spark logical plan: "LocalTableScan", "FileScan", "Relation"
_RE_SPARK_SCAN = re.compile(
    r"(?:LocalTableScan|FileScan|Relation|ExternalRDDScan|InMemoryTableScan)"
    r"(?:\s+\[)?(?P<cols>[^\]]*)\]?",
    re.IGNORECASE,
)

_THRESHOLDS = {
    "large_table_rows": 1_000_000,   # warn if row estimate > 1M
    "high_cost": 500_000,            # warn if cost > 500K
    "cartesian_rows": 10_000,        # cartesian join is bad even at 10K
}


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ScanInfo:
    """A single table scan found in the plan."""
    table: str
    is_full_scan: bool = True
    rows: int = 0
    cost: float = 0.0


@dataclass
class JoinInfo:
    """A join node found in the plan."""
    join_type: str
    is_cartesian: bool = False


@dataclass
class SqlPlanResult:
    """Structured representation of a parsed SQL EXPLAIN plan."""

    # Detected table scans
    scans: list[ScanInfo] = field(default_factory=list)

    # Detected join types
    joins: list[JoinInfo] = field(default_factory=list)

    # Overall cost / row estimates
    total_cost: float = 0.0
    max_rows: int = 0

    # Flags for costly patterns
    has_full_table_scan: bool = False
    has_cartesian_join: bool = False
    has_filesort: bool = False
    has_temp_table: bool = False
    has_no_index_filter: bool = False

    # Performance warnings (human-readable)
    warnings: list[str] = field(default_factory=list)

    # Detected operations summary
    operations: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Return compact structured string for LLM prompt injection."""
        parts: list[str] = []
        if self.warnings:
            parts.append("Performance issues detected:")
            for w in self.warnings:
                parts.append(f"  - {w}")
        if self.scans:
            scan_info = [
                f"{s.table} ({'full scan' if s.is_full_scan else 'index'}, "
                f"{s.rows:,} rows, cost {s.cost:.0f})"
                for s in self.scans[:6]
            ]
            parts.append(f"Table access: {'; '.join(scan_info)}")
        if self.joins:
            join_strs = [j.join_type + (" (CARTESIAN!)" if j.is_cartesian else "") for j in self.joins]
            parts.append(f"Join types: {', '.join(join_strs)}")
        if self.total_cost:
            parts.append(f"Total cost estimate: {self.total_cost:,.0f}")
        if self.max_rows:
            parts.append(f"Max row estimate: {self.max_rows:,}")
        if self.operations:
            parts.append(f"Operations: {', '.join(self.operations)}")
        return "\n".join(parts)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_rows(text: str) -> int:
    m = _RE_ROWS.search(text)
    if m:
        return int(m.group("rows").replace(",", ""))
    return 0


def _parse_cost(text: str) -> float:
    m = _RE_COST.search(text)
    if m:
        return float(m.group("cost"))
    return 0.0


def _try_json_parse(text: str) -> dict | list | None:
    """Attempt to parse *text* as JSON. Returns None if it isn't JSON."""
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


# ── JSON plan parsing ─────────────────────────────────────────────────────────

def _walk_json_plan(node: dict | list, result: SqlPlanResult, depth: int = 0) -> None:
    """Recursively walk a MySQL/Spark JSON explain plan."""
    if isinstance(node, list):
        for item in node:
            _walk_json_plan(item, result, depth)
        return
    if not isinstance(node, dict):
        return

    # MySQL JSON EXPLAIN format
    table_info = node.get("table", {})
    if isinstance(table_info, dict):
        table_name = table_info.get("table_name", "")
        access_type = table_info.get("access_type", "")
        rows = int(table_info.get("rows_examined_per_scan", 0))
        key = table_info.get("key")
        if table_name:
            is_full = access_type in ("ALL", "FULL") or key is None
            scan = ScanInfo(
                table=table_name, is_full_scan=is_full, rows=rows,
            )
            result.scans.append(scan)
            if is_full:
                result.has_full_table_scan = True

    # Nested plans
    for key in ("nested_loop", "ordering_operation", "grouping_operation",
                "duplicates_removal", "query_block", "table", "attached_condition"):
        child = node.get(key)
        if child:
            _walk_json_plan(child, result, depth + 1)


# ── text plan parsing ─────────────────────────────────────────────────────────

def _parse_text_plan(text: str, result: SqlPlanResult) -> None:
    """Parse a human-readable EXPLAIN text output."""
    all_costs: list[float] = []
    all_rows: list[int] = []

    # Full / table scans
    for m in _RE_TABLE_SCAN.finditer(text):
        table = m.group("table")
        # look for cost / rows on the same line
        line_end = text.find("\n", m.start())
        line = text[m.start():line_end] if line_end != -1 else text[m.start():]
        rows = _parse_rows(line)
        cost = _parse_cost(line)
        all_costs.append(cost)
        all_rows.append(rows)
        result.scans.append(ScanInfo(table=table, is_full_scan=True, rows=rows, cost=cost))
        result.has_full_table_scan = True

    # Joins
    for m in _RE_JOIN.finditer(text):
        jtype = m.group("join_type")
        is_cart = bool(re.search(r"CROSS JOIN|Cartesian|BNL|Block Nested Loop", jtype, re.IGNORECASE))
        result.joins.append(JoinInfo(join_type=jtype, is_cartesian=is_cart))
        if is_cart:
            result.has_cartesian_join = True

    # Global cost / rows from all lines
    for m in _RE_COST.finditer(text):
        all_costs.append(float(m.group("cost")))
    for m in _RE_ROWS.finditer(text):
        all_rows.append(int(m.group("rows").replace(",", "")))

    # Flags
    if _RE_SORT.search(text):
        result.has_filesort = True
    if _RE_TEMP_TABLE.search(text):
        result.has_temp_table = True
    if _RE_NO_INDEX.search(text):
        result.has_no_index_filter = True
    if _RE_AGGREGATE.search(text):
        if "Aggregate" not in result.operations and "GROUP BY" not in result.operations:
            result.operations.append("Aggregate / GROUP BY")

    if all_costs:
        result.total_cost = max(all_costs)
    if all_rows:
        result.max_rows = max(all_rows)


def _mysql_tabular(text: str, result: SqlPlanResult) -> bool:
    """Try to parse MySQL tabular EXPLAIN format. Returns True if rows found."""
    found = False
    for m in _RE_MYSQL_TABULAR_ROW.finditer(text):
        found = True
        table = m.group("table")
        access_type = m.group("type")
        rows = int(m.group("rows").replace(",", ""))
        key = m.group("key").strip()
        is_full = access_type == "ALL" or not key
        result.scans.append(ScanInfo(table=table, is_full_scan=is_full, rows=rows))
        if is_full:
            result.has_full_table_scan = True
    return found


# ── warnings builder ──────────────────────────────────────────────────────────

def _build_warnings(result: SqlPlanResult) -> None:
    """Populate result.warnings based on detected flags and thresholds."""
    if result.has_full_table_scan:
        large = [s for s in result.scans if s.is_full_scan and s.rows > _THRESHOLDS["large_table_rows"]]
        if large:
            for s in large:
                result.warnings.append(
                    f"Full table scan on `{s.table}` with ~{s.rows:,} rows — consider adding an index"
                )
        else:
            tables = [s.table for s in result.scans if s.is_full_scan]
            result.warnings.append(f"Full table scan on: {', '.join(tables)}")

    if result.has_cartesian_join:
        result.warnings.append("Cartesian / cross join detected — this can produce an explosive result set")

    if result.has_filesort:
        result.warnings.append("Filesort (Using filesort) — ORDER BY or GROUP BY cannot use an index")

    if result.has_temp_table:
        result.warnings.append("Temporary table created — GROUP BY / DISTINCT / UNION may need index tuning")

    if result.has_no_index_filter:
        result.warnings.append("Filter applied without index — adding an index on the filter column may help")

    if result.total_cost > _THRESHOLDS["high_cost"]:
        result.warnings.append(
            f"Very high estimated cost ({result.total_cost:,.0f}) — query may scan large amounts of data"
        )

    if result.max_rows > _THRESHOLDS["large_table_rows"]:
        result.warnings.append(
            f"Large row estimate ({result.max_rows:,}) — pagination, partitioning, or caching may help"
        )

    if not result.joins and result.has_full_table_scan and result.max_rows > 100_000:
        result.warnings.append("No index-accelerated join found — verify foreign key indexes exist")


# ── public API ────────────────────────────────────────────────────────────────

def parse_sql_plan(plan_text: str) -> SqlPlanResult:
    """Parse a SQL EXPLAIN / query plan and return a :class:`SqlPlanResult`.

    Accepts:
      - Text EXPLAIN output (MySQL, PostgreSQL, Spark)
      - JSON EXPLAIN output (MySQL EXPLAIN FORMAT=JSON)
      - Mixed text+explain (e.g. the query followed by the plan)
    """
    result = SqlPlanResult()

    # Try JSON first
    json_obj = _try_json_parse(plan_text)
    if json_obj is not None:
        _walk_json_plan(json_obj, result)
    else:
        # Try MySQL tabular first, fall back to text
        if not _mysql_tabular(plan_text, result):
            _parse_text_plan(plan_text, result)

    # Spark-specific scan patterns (may appear alongside text)
    for m in _RE_SPARK_SCAN.finditer(plan_text):
        tbl = m.group("cols").strip().split(",")[0].strip()
        if tbl and not any(s.table == tbl for s in result.scans):
            result.scans.append(ScanInfo(table=tbl or "unknown", is_full_scan=True))

    # Operations summary
    if result.has_filesort and "Filesort" not in result.operations:
        result.operations.append("Filesort")
    if result.has_temp_table and "Temp table" not in result.operations:
        result.operations.append("Temp table")
    if any(j.is_cartesian for j in result.joins) and "Cartesian join" not in result.operations:
        result.operations.append("Cartesian join")

    _build_warnings(result)
    return result
