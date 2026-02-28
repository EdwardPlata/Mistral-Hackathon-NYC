---
name: deep-research
description: "Deep repository exploration and context building without modifying files"
context: fork
agent: Explore
allowed-tools: ["read_file", "grep_search", "semantic_search", "file_search", "list_dir", "get_errors"]
---

# Deep Research Mode

You are in **Explore-only mode** for comprehensive repository analysis and context building.

## Objective
Build a complete mental model of the codebase by exploring:
- Project structure and module organization
- Key abstractions, interfaces, and data flows
- Dependencies and integration points
- Test coverage and quality gates
- Documentation completeness
- Risk areas and technical debt

## Constraints
- **Read-only**: You cannot modify files or run commands
- **Comprehensive**: Gather enough context to answer "where/why/how" questions
- **Structured**: Organize findings by module/layer/concern

## Workflow
1. **Project overview**: Read pyproject.toml, README, top-level structure
2. **Module deep-dive**: Explore agents/, ml/, skills/ directories systematically
3. **Integration mapping**: Trace how components connect (imports, configs, APIs)
4. **Quality assessment**: Check test coverage, linting configs, CI/CD setup
5. **Risk identification**: Note TODOs, FIXMEs, deprecated patterns, missing tests

## Output Format
Provide a structured report with:
- **Architecture summary**: High-level component diagram (text-based)
- **Module inventory**: Purpose and key files for each package
- **Integration map**: How components communicate
- **Quality metrics**: Test coverage, linting status, doc completeness
- **Risk register**: Technical debt, security concerns, missing documentation
- **Next steps**: Recommended actions based on findings

## Usage
Invoke with: `/deep-research`

This skill forks the conversation context, so exploration details won't clutter the main chat.
