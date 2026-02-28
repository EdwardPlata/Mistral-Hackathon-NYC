---
name: run-tests
description: "Fast, targeted test execution with actionable output"
allowed-tools: ["run_in_terminal", "read_file", "grep_search", "get_errors"]
---

# Run Tests

Execute targeted tests efficiently and report actionable results.

## Objective
Run the smallest effective test suite for the current work, report failures clearly, and suggest fixes.

## Smart Test Selection
Choose the appropriate scope based on context:

### Full Suite (use sparingly)
```bash
uv run pytest -q
```

### Single File (most common)
```bash
uv run pytest -q tests/test_agents.py -v
```

### Single Test/Class
```bash
uv run pytest -q tests/test_agents.py::TestMainAgent -v
uv run pytest -q tests/test_agents.py::test_tool_execution -v
```

### Failed Tests Only (after a failure)
```bash
uv run pytest -q --lf -v
```

### With Coverage (when requested)
```bash
uv run pytest -q --cov=agents --cov-report=term-missing
```

## Output Format

### On Success ✅
```
✅ All tests passed (X passed in Y.Ys)

Summary:
- Test file: tests/test_agents.py
- Tests run: 12
- Duration: 1.2s
```

### On Failure ❌
```
❌ N tests failed

Failed Tests:
1. test_tool_execution (tests/test_agents.py:45)
   AssertionError: Expected 'success' but got 'error'
   
   Likely cause: Tool validation logic may be incorrect
   Suggested fix: Check agents/tools.py:validate_tool_input()

2. test_config_loading (tests/test_agents.py:67)
   FileNotFoundError: config.yaml not found
   
   Likely cause: Test fixture not setting up config file
   Suggested fix: Add config.yaml fixture in conftest.py

Next steps:
- Fix the issues above and re-run with: uv run pytest -q --lf -v
- Or investigate specific failures: /deep-research <failed_module>
```

## Integration
- Uses `uv run pytest` for consistency with repo setup
- Respects pytest.ini or pyproject.toml test configuration
- Can read test output and suggest fixes based on error patterns

## Usage
Invoke with: `/run-tests [optional_path]`

Examples:
```
/run-tests
/run-tests tests/test_agents.py
/run-tests tests/test_agents.py::test_config_loading
```

## Tips
- Use `-v` for verbose output when debugging
- Use `--lf` (last failed) to re-run only failed tests
- Use `-x` to stop on first failure for rapid iteration
- Check `get_errors` for IDE-reported issues before running tests
