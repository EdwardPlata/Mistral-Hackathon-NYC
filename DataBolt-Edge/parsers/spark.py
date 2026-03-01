"""Spark log parser for DataBolt Edge.

Extracts structured error information from Spark application logs including:
- Exception types and messages
- Stage / task identifiers
- Executor / host info
- Stack trace snippets
- Memory-related errors
- Lost executor / RPC disconnects

Typical Spark log lines look like::

    24/02/28 14:33:01 ERROR Executor: Exception in task 3.0 in stage 5.0 (TID 47)
    java.lang.OutOfMemoryError: GC overhead limit exceeded
        at ...
    24/02/28 14:33:01 WARN  TaskSetManager: Lost task 3.0 in stage 5.0 ...
    24/02/28 14:33:01 ERROR TaskSchedulerImpl: Lost executor 2 on 10.0.0.5: Remote RPC ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── compiled regexes ──────────────────────────────────────────────────────────

_RE_LOG_LINE = re.compile(
    r"^(?P<ts>\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+(?P<level>ERROR|WARN|INFO|DEBUG|TRACE)"
    r"\s+(?P<logger>\S+):\s+(?P<message>.+)$",
    re.MULTILINE,
)

# task N.M in stage S.T (TID T)
_RE_TASK = re.compile(
    r"task\s+(?P<task_id>[\d.]+)\s+in\s+stage\s+(?P<stage_id>[\d.]+)"
    r"(?:\s+\(TID\s+(?P<tid>\d+)\))?",
    re.IGNORECASE,
)

# Java/Scala exception line:  some.package.ExceptionName: message
_RE_EXCEPTION = re.compile(
    r"^(?P<exc_class>(?:[a-zA-Z_$][a-zA-Z0-9_$]*\.)+[A-Z][a-zA-Z0-9_$]*):\s*(?P<exc_msg>.*)$",
    re.MULTILINE,
)

# Python exception: ExceptionType: message (no dots before UpperCase)
_RE_PY_EXCEPTION = re.compile(
    r"^(?P<exc_class>[A-Z][a-zA-Z0-9_]*(?:Error|Exception|Warning|Fault)):\s*(?P<exc_msg>.*)$",
    re.MULTILINE,
)

# stack frame:  "    at Class.method(File.java:123)"
_RE_STACK_FRAME = re.compile(r"^\s+at\s+[\w$.<>]+\([\w$.]+(?::\d+)?\)$", re.MULTILINE)

# executor reference: executor N on host
_RE_EXECUTOR = re.compile(
    r"executor\s+(?P<executor_id>\d+)\s+on\s+(?P<host>[\w.:/\-]+)",
    re.IGNORECASE,
)

# memory / OOM keywords
_MEMORY_KEYWORDS = frozenset(
    [
        "OutOfMemoryError",
        "GC overhead limit exceeded",
        "Java heap space",
        "Cannot allocate memory",
        "Not enough memory",
        "spillToDisk",
        "SpillWriter",
        "exceeded memory limit",
        "ExecutorLostFailure",
    ]
)

# categories of common Spark errors
_ERROR_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("OOM / Memory", re.compile(r"OutOfMemory|GC overhead|heap space|Cannot allocate memory", re.IGNORECASE)),
    ("Executor Lost", re.compile(r"Lost executor|ExecutorLostFailure|Remote RPC client disassociated", re.IGNORECASE)),
    ("Task Failed", re.compile(r"Lost task|Task failed|Exception in task", re.IGNORECASE)),
    ("Shuffle Error", re.compile(r"shuffle|FetchFailed|org\.apache\.spark\.shuffle", re.IGNORECASE)),
    ("Disk / IO", re.compile(r"FileNotFoundException|DiskBlockObjectWriter|IOException", re.IGNORECASE)),
    ("Network", re.compile(r"Connection refused|timed out|SocketException|ConnectException", re.IGNORECASE)),
    ("Data Skew", re.compile(r"Stage took|very long|skew", re.IGNORECASE)),
]


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SparkParseResult:
    """Structured representation of a parsed Spark log."""

    # Raw counts
    error_count: int = 0
    warn_count: int = 0

    # Unique exception class names found
    exceptions: list[str] = field(default_factory=list)

    # Stage / task identifiers mentioned
    stages: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    tids: list[str] = field(default_factory=list)

    # Executors and hosts
    executors: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)

    # Error categorisation (may contain multiple)
    categories: list[str] = field(default_factory=list)

    # Whether any OOM / memory indicator was detected
    has_oom: bool = False

    # Condensed error lines (first 10 ERROR lines)
    error_lines: list[str] = field(default_factory=list)

    # First exception block (class + message + up to 5 stack frames)
    first_exception_block: str = ""

    def to_prompt_context(self) -> str:
        """Return a compact string for injection into an LLM prompt."""
        parts: list[str] = []
        if self.categories:
            parts.append(f"Error categories: {', '.join(self.categories)}")
        if self.exceptions:
            parts.append(f"Exceptions: {', '.join(self.exceptions[:5])}")
        if self.stages:
            parts.append(f"Affected stages: {', '.join(sorted(set(self.stages))[:5])}")
        if self.tasks:
            parts.append(f"Affected tasks: {', '.join(sorted(set(self.tasks))[:5])}")
        if self.executors:
            parts.append(f"Affected executors: {', '.join(sorted(set(self.executors))[:5])}")
        if self.has_oom:
            parts.append("OOM / memory pressure detected")
        parts.append(f"Log counts — ERROR: {self.error_count}  WARN: {self.warn_count}")
        if self.first_exception_block:
            parts.append(f"\nFirst exception:\n{self.first_exception_block}")
        return "\n".join(parts)


# ── public API ────────────────────────────────────────────────────────────────

def parse_spark(log_text: str) -> SparkParseResult:
    """Parse a Spark application log and return a :class:`SparkParseResult`."""
    result = SparkParseResult()

    # --- count log levels and collect error lines ---
    for m in _RE_LOG_LINE.finditer(log_text):
        level = m.group("level")
        if level == "ERROR":
            result.error_count += 1
            if len(result.error_lines) < 10:
                result.error_lines.append(m.group(0).strip())
        elif level == "WARN":
            result.warn_count += 1

    # --- extract task / stage identifiers ---
    for m in _RE_TASK.finditer(log_text):
        result.stages.append(m.group("stage_id"))
        result.tasks.append(m.group("task_id"))
        if m.group("tid"):
            result.tids.append(m.group("tid"))

    # --- extract executor info ---
    for m in _RE_EXECUTOR.finditer(log_text):
        result.executors.append(m.group("executor_id"))
        result.hosts.append(m.group("host"))

    # --- extract exception class names ---
    seen_exceptions: set[str] = set()
    for m in _RE_EXCEPTION.finditer(log_text):
        cls = m.group("exc_class")
        if cls not in seen_exceptions:
            seen_exceptions.add(cls)
            result.exceptions.append(cls)
    for m in _RE_PY_EXCEPTION.finditer(log_text):
        cls = m.group("exc_class")
        if cls not in seen_exceptions:
            seen_exceptions.add(cls)
            result.exceptions.append(cls)

    # --- OOM check ---
    result.has_oom = any(kw in log_text for kw in _MEMORY_KEYWORDS)

    # --- categorise errors ---
    seen_cats: set[str] = set()
    for cat_name, pattern in _ERROR_CATEGORIES:
        if pattern.search(log_text) and cat_name not in seen_cats:
            result.categories.append(cat_name)
            seen_cats.add(cat_name)

    # --- first exception block ---
    exc_match = _RE_EXCEPTION.search(log_text) or _RE_PY_EXCEPTION.search(log_text)
    if exc_match:
        start = exc_match.start()
        # grab the exception line + up to 5 subsequent stack frame lines
        block_lines = log_text[start:].splitlines()
        kept: list[str] = [block_lines[0]]
        frame_count = 0
        for ln in block_lines[1:]:
            if _RE_STACK_FRAME.match(ln):
                kept.append(ln)
                frame_count += 1
                if frame_count >= 5:
                    break
            elif ln.strip() == "":
                break
            else:
                kept.append(ln)
                break  # non-frame continuation
        result.first_exception_block = "\n".join(kept)

    return result
