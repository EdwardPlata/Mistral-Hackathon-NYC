# Claude Skills for Mistral Hackathon NYC

This directory contains custom Claude Skills for the "Developer CLI Assistant (Mistral Vibe)" project. Skills are reusable, permission-scoped workflows that package repeatable development tasks.

## What are Claude Skills?

Claude Skills are filesystem-based automation patterns that:
- Are auto-discovered from `.claude/*.md` files
- Can be invoked by users via `/skill-name`
- Restrict tool access via `allowed-tools` for safety
- Can fork conversation context for exploration
- Support manual-only invocation to prevent side effects

[Learn more about Claude Skills](https://code.claude.com/docs/en/skills)

## Available Skills

### 🔍 /deep-research
**Purpose**: Comprehensive repository exploration and context building

**Mode**: Read-only, forked conversation context  
**When to use**: When you need to understand the codebase structure, dependencies, or architecture before making changes  
**Key feature**: Forks conversation so exploration details don't clutter main chat

```
/deep-research
```

### 🧪 /refactor-with-tests
**Purpose**: Test-first refactoring with validation and rollback

**Mode**: Manual-only (you must invoke it explicitly)  
**When to use**: When refactoring code and want test coverage + quality gates  
**Workflow**: Write tests → Refactor → Validate → Report

```
/refactor-with-tests agents/main_agent.py
```

### ⚡ /run-tests
**Purpose**: Fast, targeted test execution with actionable output

**Mode**: Automatic or manual  
**When to use**: Quick test runs, focused test selection, failure diagnosis  
**Smart selection**: Full suite, single file, single test, or last-failed

```
/run-tests
/run-tests tests/test_agents.py
/run-tests tests/test_agents.py::test_config_loading
```

### 💾 /commit
**Purpose**: Safe git commit workflow with confirmation

**Mode**: Manual-only (prevents accidental commits)  
**When to use**: Ready to commit changes with a good message  
**Features**: Review diffs, stage changes, craft conventional commit message

```
/commit
/commit feat
/commit fix
```

### 📝 /doc-sweep
**Purpose**: Generate or update documentation across multiple files

**Mode**: Automatic or manual  
**When to use**: Add docstrings, update README, generate API docs  
**Standards**: Google-style docstrings with type hints

```
/doc-sweep agents/
/doc-sweep README.md
```

### 🔒 /safe-reader
**Purpose**: Read-only exploration with no modifications

**Mode**: Automatic or manual  
**When to use**: When requirements are unclear or you want to explore safely  
**Restrictions**: No file edits, no command execution, only reading

```
/safe-reader
```

## Skill Architecture

### Permission Model
Each skill declares `allowed-tools` to restrict what it can do:

- **Read-only**: `read_file`, `grep_search`, `semantic_search`, `file_search`, `list_dir`, `get_errors`
- **Edit**: Add `create_file`, `replace_string_in_file`, `multi_replace_string_in_file`
- **Execute**: Add `run_in_terminal`

### Safety Features

#### Manual-only Skills
Some skills use `disable-model-invocation: true`:
- `/refactor-with-tests` - Side effects require user approval
- `/commit` - Git commits need explicit confirmation

These skills **must** be invoked by the user; Claude cannot trigger them automatically.

#### Forked Context
`/deep-research` uses `context: fork` to:
- Explore the codebase in a separate conversation
- Avoid polluting the main chat with exploratory details
- Return a summary without overwhelming the user

## Integration with This Repo

### UV Package Manager
All skills use `uv run` for commands:
```bash
uv run pytest -q
uv run ruff check .
uv run python script.py
```

### Quality Gates
- **Testing**: pytest with `-q` for concise output
- **Linting**: ruff for fast Python linting
- **Type checking**: Supported via pyright/mypy (if configured)

### Git Workflow
- Conventional commits (type(scope): subject)
- Branch-based development
- PR-ready commit messages

## Recommended Workflows

### New Feature Development
1. `/deep-research` - Understand existing architecture
2. `/refactor-with-tests` - Implement with tests
3. `/run-tests` - Validate changes
4. `/doc-sweep` - Update documentation
5. `/commit` - Commit with good message

### Bug Fix
1. `/safe-reader` - Understand the bug context
2. `/run-tests` - Reproduce the failure
3. `/refactor-with-tests` - Fix with tests
4. `/commit` - Commit the fix

### Code Review / Exploration
1. `/safe-reader` - Browse safely
2. `/deep-research` - Deep dive into specific areas
3. Report findings without making changes

### Documentation Update
1. `/doc-sweep` - Update docs systematically
2. `/run-tests` - Ensure examples work
3. `/commit` - Commit documentation

## Creating New Skills

To add a new skill:

1. Create `.claude/my-skill.md`
2. Add frontmatter with metadata
3. Write clear instructions for Claude
4. Test the skill by invoking `/my-skill`

### Skill Template

```markdown
---
name: my-skill
description: "Brief description for tool discovery"
disable-model-invocation: false  # true for manual-only
allowed-tools: ["read_file", "grep_search"]  # Restrict permissions
---

# My Skill

## Objective
What this skill accomplishes

## Workflow
Step-by-step process

## Usage
How to invoke it

## Integration
How it works with this repo
```

## References

- [Claude Skills Documentation](https://code.claude.com/docs/en/skills)
- [Mistral Vibe CLI Docs](https://mistral.ai/news/devstral-2-vibe-cli)
- Repository conventions: See [CLAUDE.md](../CLAUDE.md) and [AGENTS.md](../AGENTS.md)

## Philosophy

Skills embody the principle: **Workflows over tools**. Instead of re-implementing capabilities, skills compose existing tools into safe, repeatable patterns that align with team conventions and quality standards.
