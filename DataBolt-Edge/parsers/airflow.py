"""Airflow log parser for DataBolt Edge.

Extracts structured error information from Airflow task-instance logs including:
- DAG id, task id, execution date (from log path or log content)
- ERROR / WARNING lines
- Python tracebacks (exception type + message + stack frames)
- Airflow-specific error patterns (XCom, connection, sensor timeout, etc.)

Typical Airflow log header::

    [2024-02-28 09:15:03,412] {taskinstance.py:1482} ERROR - Failed to execute task
    Traceback (most recent call last):
      File "/opt/airflow/dags/etl_pipeline.py", line 47, in extract_sales_data
        df = pd.read_csv('/data/input/sales_2024.csv')
    FileNotFoundError: [Errno 2] No such file or directory: '/data/input/sales_2024.csv'
    [2024-02-28 09:15:03,415] {taskinstance.py:1492} ERROR - Task failed with exception
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── compiled regexes ──────────────────────────────────────────────────────────

# Standard Airflow log line: [timestamp] {module:line} LEVEL - message
_RE_AF_LINE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+\{(?P<module>[^}]+)\}\s+"
    r"(?P<level>ERROR|WARNING|WARN|INFO|DEBUG|CRITICAL)\s+-\s+(?P<message>.+)$",
    re.MULTILINE,
)

# Alternative: simpler timestamp + level
_RE_SIMPLE_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[,.\d+Z-]+)?)"
    r"\s+(?P<level>ERROR|WARNING|WARN|INFO|DEBUG|CRITICAL)\s+-?\s*(?P<message>.+)$",
    re.MULTILINE,
)

# DAG id and task id from log header lines like:
#   dag_id=my_dag, task_id=my_task, execution_date=2024-02-28T09:00:00+00:00
_RE_DAG_TASK = re.compile(
    r"dag_id\s*[=:]\s*(?P<dag_id>[^\s,;]+).*?task_id\s*[=:]\s*(?P<task_id>[^\s,;]+)",
    re.IGNORECASE | re.DOTALL,
)
_RE_EXEC_DATE = re.compile(
    r"execution_date\s*[=:]\s*(?P<exec_date>[^\s,;]+)",
    re.IGNORECASE,
)

# Python traceback start
_RE_TRACEBACK = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)

# Python traceback frame: "  File "path", line N, in func_name"
_RE_TB_FRAME = re.compile(
    r'^\s+File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>\S+)$',
    re.MULTILINE,
)

# Final exception line: ExceptionClass: message (at end of traceback)
_RE_FINAL_EXC = re.compile(
    r"^(?P<exc_class>[A-Za-z][A-Za-z0-9_.]*(?:Error|Exception|Warning|Interrupt|Exit|Failure|Fault)):\s*(?P<exc_msg>.*)$",
    re.MULTILINE,
)

# Airflow-specific patterns
_AF_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("File not found", re.compile(r"FileNotFoundError|No such file or directory|cannot(?:\s+be)?\s+found", re.IGNORECASE)),
    ("Connection error", re.compile(r"ConnectionError|ConnectionRefusedError|airflow\.exceptions\.AirflowNotFoundException|AirflowBadRequest", re.IGNORECASE)),
    ("Sensor timeout", re.compile(r"AirflowSensorTimeout|Sensor timed out|poking.*timeout", re.IGNORECASE)),
    ("XCom error", re.compile(r"XCom|xcom_pull|xcom_push", re.IGNORECASE)),
    ("Import error", re.compile(r"ImportError|ModuleNotFoundError|cannot import name", re.IGNORECASE)),
    ("Permission error", re.compile(r"PermissionError|Permission denied|Access denied", re.IGNORECASE)),
    ("Database error", re.compile(r"OperationalError|DatabaseError|psycopg2|sqlalchemy", re.IGNORECASE)),
    ("Dependency failed", re.compile(r"upstream task.*failed|dependency.*failed|TriggerRule", re.IGNORECASE)),
    ("Retry exceeded", re.compile(r"max_retries|retries\s+exceeded|Task exited with return code", re.IGNORECASE)),
    ("Kubernetes / pod", re.compile(r"KubernetesPodOperator|pod.*failed|container.*error", re.IGNORECASE)),
]


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TracebackInfo:
    """Single Python traceback extracted from an Airflow log."""
    exception_class: str
    exception_message: str
    frames: list[dict[str, str]] = field(default_factory=list)  # {file, line, func, code}
    raw: str = ""

    def summary(self, max_frames: int = 3) -> str:
        lines = [f"{self.exception_class}: {self.exception_message}"]
        for fr in self.frames[-max_frames:]:  # innermost frames (most relevant)
            lines.append(f"  File \"{fr['file']}\", line {fr['line']}, in {fr['func']}")
            if fr.get("code"):
                lines.append(f"    {fr['code']}")
        return "\n".join(lines)


@dataclass
class AirflowParseResult:
    """Structured representation of a parsed Airflow task log."""

    dag_id: str = ""
    task_id: str = ""
    execution_date: str = ""

    error_count: int = 0
    warning_count: int = 0

    # Extracted error/warning messages (first 10 ERROR lines, capped)
    error_lines: list[str] = field(default_factory=list)

    # Python tracebacks found
    tracebacks: list[TracebackInfo] = field(default_factory=list)

    # Error categories detected
    categories: list[str] = field(default_factory=list)

    # All unique exception class names
    exception_classes: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Return compact structured string for LLM prompt injection."""
        parts: list[str] = []
        if self.dag_id:
            parts.append(f"DAG: {self.dag_id}  Task: {self.task_id}  Execution: {self.execution_date}")
        if self.categories:
            parts.append(f"Error categories: {', '.join(self.categories)}")
        if self.exception_classes:
            parts.append(f"Exception types: {', '.join(self.exception_classes[:5])}")
        parts.append(f"Log counts — ERROR: {self.error_count}  WARN: {self.warning_count}")
        if self.tracebacks:
            parts.append(f"\nTraceback (most recent):\n{self.tracebacks[-1].summary()}")
        return "\n".join(parts)


# ── traceback extraction ──────────────────────────────────────────────────────

def _extract_tracebacks(text: str) -> list[TracebackInfo]:
    """Find all Python tracebacks in *text* and return structured info."""
    results: list[TracebackInfo] = []
    for tb_start in _RE_TRACEBACK.finditer(text):
        start = tb_start.start()
        # scan forward until we find the exception line
        snippet = text[start:]
        frames: list[dict[str, str]] = []
        exc_class = ""
        exc_msg = ""
        lines = snippet.splitlines()
        in_frames = False
        raw_lines: list[str] = [lines[0]]
        i = 1
        while i < len(lines):
            ln = lines[i]
            raw_lines.append(ln)
            frame_m = _RE_TB_FRAME.match(ln)
            if frame_m:
                in_frames = True
                frame: dict[str, str] = {
                    "file": frame_m.group("file"),
                    "line": frame_m.group("line"),
                    "func": frame_m.group("func"),
                    "code": "",
                }
                # next line may be the source code snippet
                if i + 1 < len(lines) and lines[i + 1].startswith("    ") and not _RE_TB_FRAME.match(lines[i + 1]):
                    i += 1
                    frame["code"] = lines[i].strip()
                    raw_lines.append(lines[i])
                frames.append(frame)
            elif in_frames:
                # should be the final exception line
                exc_m = _RE_FINAL_EXC.match(ln)
                if exc_m:
                    exc_class = exc_m.group("exc_class")
                    exc_msg = exc_m.group("exc_msg")
                    raw_lines.append(ln)
                break
            i += 1

        if exc_class:
            results.append(TracebackInfo(
                exception_class=exc_class,
                exception_message=exc_msg,
                frames=frames,
                raw="\n".join(raw_lines),
            ))
    return results


# ── public API ────────────────────────────────────────────────────────────────

def parse_airflow(log_text: str) -> AirflowParseResult:
    """Parse an Airflow task-instance log and return an :class:`AirflowParseResult`."""
    result = AirflowParseResult()

    # --- DAG / task metadata ---
    dag_task_m = _RE_DAG_TASK.search(log_text)
    if dag_task_m:
        result.dag_id = dag_task_m.group("dag_id")
        result.task_id = dag_task_m.group("task_id")
    exec_m = _RE_EXEC_DATE.search(log_text)
    if exec_m:
        result.execution_date = exec_m.group("exec_date")

    # --- count log levels and collect error lines ---
    for pattern in (_RE_AF_LINE, _RE_SIMPLE_LINE):
        for m in pattern.finditer(log_text):
            level = m.group("level").upper()
            if level in ("ERROR", "CRITICAL"):
                result.error_count += 1
                if len(result.error_lines) < 10:
                    result.error_lines.append(m.group("message").strip())
            elif level in ("WARNING", "WARN"):
                result.warning_count += 1

    # --- extract tracebacks ---
    result.tracebacks = _extract_tracebacks(log_text)

    # --- collect unique exception classes ---
    seen: set[str] = set()
    for tb in result.tracebacks:
        if tb.exception_class and tb.exception_class not in seen:
            seen.add(tb.exception_class)
            result.exception_classes.append(tb.exception_class)
    # also scan for bare exception lines outside tracebacks
    for m in _RE_FINAL_EXC.finditer(log_text):
        cls = m.group("exc_class")
        if cls not in seen:
            seen.add(cls)
            result.exception_classes.append(cls)

    # --- categorise errors ---
    seen_cats: set[str] = set()
    for cat_name, pattern in _AF_PATTERNS:
        if pattern.search(log_text) and cat_name not in seen_cats:
            result.categories.append(cat_name)
            seen_cats.add(cat_name)

    return result
