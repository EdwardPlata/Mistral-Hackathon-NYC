---
name: refactor-with-tests
description: "Test-first refactoring workflow with validation and rollback option"
disable-model-invocation: true
allowed-tools: ["read_file", "grep_search", "semantic_search", "file_search", "list_dir", "create_file", "replace_string_in_file", "multi_replace_string_in_file", "run_in_terminal", "get_errors"]
---

# Refactor With Tests

A disciplined refactoring workflow that ensures quality through testing.

## Workflow
This is a **manual-only skill** (user must invoke it explicitly via `/refactor-with-tests`).

### Phase 1: Understand & Scope
1. Read the target file(s) and understand current implementation
2. Identify the refactoring goal (clarify with user if ambiguous)
3. Check for existing tests covering the target code
4. Document the refactoring plan (what changes, why, expected impact)

### Phase 2: Write/Extend Tests
1. If tests are missing: Create comprehensive tests for current behavior
2. If tests exist but incomplete: Extend to cover edge cases
3. Run tests to establish baseline: `uv run pytest -q <test_file> -v`
4. Confirm all tests pass before refactoring

### Phase 3: Refactor
1. Make the planned code changes
2. Keep changes focused and incremental
3. Maintain backward compatibility unless explicitly agreed otherwise
4. Update docstrings and type hints

### Phase 4: Validate
1. Run tests: `uv run pytest -q <test_file> -v`
2. Run linting: `uv run ruff check <changed_files>`
3. Check for type errors (if applicable)
4. Report results clearly

### Phase 5: Review & Options
If tests pass:
- ✅ Summarize what changed and why
- 📋 Suggest next steps (documentation, related refactors, etc.)

If tests fail:
- ❌ Report failures with clear diagnostics
- 🔄 Offer to fix the issue or rollback changes
- 💡 Explain why tests failed

## Safety Features
- **Test-first**: Never refactor without tests
- **Incremental**: Make changes in small, verifiable steps
- **Rollback-ready**: Git integration allows easy reset
- **Quality gates**: Lint + test before declaring success

## Usage
Invoke with: `/refactor-with-tests <target_file_or_module>`

Example:
```
/refactor-with-tests agents/main_agent.py
```

## Integration with This Repo
- Uses `uv run pytest` for testing
- Uses `uv run ruff check` for linting
- Respects pyproject.toml configuration
- Works with the existing `.venv` managed by UV
