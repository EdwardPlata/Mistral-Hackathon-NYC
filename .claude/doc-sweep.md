---
name: doc-sweep
description: "Generate or update documentation across multiple files"
allowed-tools: ["read_file", "grep_search", "semantic_search", "file_search", "list_dir", "create_file", "replace_string_in_file", "multi_replace_string_in_file", "run_in_terminal", "get_errors"]
---

# Documentation Sweep

Systematically generate or update documentation across the repository.

## Objectives
- **Docstrings**: Add/update function and class docstrings
- **README**: Update project documentation
- **API docs**: Generate API reference documentation
- **Type hints**: Add missing type annotations
- **Examples**: Add usage examples where helpful

## Workflow

### Phase 1: Assess Current State
1. Scan target files/modules for missing documentation
2. Check existing docstring format (Google, NumPy, Sphinx)
3. Identify undocumented public APIs
4. Find missing type hints

### Phase 2: Documentation Standards
For this repo, use **Google-style docstrings** with type hints:

```python
def process_agent_response(
    response: dict[str, Any],
    config: AgentConfig,
    validate: bool = True
) -> ProcessedResponse:
    """Process and validate agent response from Mistral API.
    
    Args:
        response: Raw response dictionary from Mistral API
        config: Agent configuration object
        validate: Whether to validate response schema
        
    Returns:
        ProcessedResponse object with parsed data
        
    Raises:
        ValidationError: If response fails schema validation
        AgentError: If response indicates agent-level error
        
    Example:
        >>> config = AgentConfig.from_yaml("agents/config.yaml")
        >>> response = {"type": "success", "data": {...}}
        >>> result = process_agent_response(response, config)
        >>> print(result.status)
        'success'
    """
    ...
```

### Phase 3: Update Files
1. Add docstrings to undocumented functions/classes
2. Update existing docstrings that are outdated
3. Add type hints where missing
4. Add usage examples for complex APIs

### Phase 4: Quality Checks
1. Run linting: `uv run ruff check <files>`
2. Check for docstring errors: `uv run python -m pydoc <module>`
3. Verify all public APIs are documented
4. Test example code (if present)

### Phase 5: Generate API Docs (Optional)
If the user wants generated docs:

```bash
# Install sphinx if needed
uv pip install sphinx sphinx-rtd-theme

# Generate docs
sphinx-quickstart docs/api
sphinx-build -b html docs/api docs/api/_build
```

## Documentation Priorities

### High Priority
- Public APIs (functions, classes in `__all__`)
- Entry points (main.py, CLI commands)
- Configuration (config.yaml, environment variables)
- Error handling and exceptions

### Medium Priority
- Internal utilities
- Test helpers
- Data models

### Low Priority
- Private functions (single underscore prefix)
- Obvious one-liners

## README Guidelines
Update README sections as needed:
- **Installation**: Keep UV commands current
- **Quick Start**: Add minimal working example
- **Configuration**: Document all environment variables
- **Examples**: Add common use cases
- **API Reference**: Link to generated docs
- **Contributing**: Keep workflow instructions current

## Usage
Invoke with: `/doc-sweep [target_path]`

Examples:
```
/doc-sweep agents/
/doc-sweep agents/main_agent.py
/doc-sweep README.md
```

## Output
Provide a summary of:
- Files updated
- Number of docstrings added/updated
- Missing documentation that needs human attention
- Linting results after changes
