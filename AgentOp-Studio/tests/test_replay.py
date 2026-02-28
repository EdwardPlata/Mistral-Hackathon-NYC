"""Tests for diff generation logic (pure stdlib, no API calls)."""

import difflib


def _compute_diff(original: str, replayed: str) -> str:
    """Replicates the diff logic used in backend.replay."""
    lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            replayed.splitlines(keepends=True),
            fromfile="original",
            tofile="replayed",
        )
    )
    return "".join(lines)


def test_identical_strings_produce_empty_diff():
    diff = _compute_diff("hello world", "hello world")
    assert diff == ""


def test_changed_string_produces_diff():
    diff = _compute_diff("foo bar", "foo baz")
    assert "-foo bar" in diff
    assert "+foo baz" in diff


def test_diff_contains_unified_header():
    diff = _compute_diff("line1\nline2\n", "line1\nchanged\n")
    assert "---" in diff
    assert "+++" in diff


def test_multiline_diff():
    original = "apple\nbanana\ncherry\n"
    replayed = "apple\nblueberry\ncherry\n"
    diff = _compute_diff(original, replayed)
    assert "-banana" in diff
    assert "+blueberry" in diff


def test_added_lines():
    original = "line1\n"
    replayed = "line1\nline2\n"
    diff = _compute_diff(original, replayed)
    assert "+line2" in diff


def test_removed_lines():
    original = "line1\nline2\n"
    replayed = "line1\n"
    diff = _compute_diff(original, replayed)
    assert "-line2" in diff
