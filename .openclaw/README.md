# OpenClaw Workspace

This directory contains OpenClaw configurations and planning workflows for multiple projects in the Mistral Hackathon NYC repository.

## Purpose

OpenClaw is configured here to provide structured **Research → Plan → Implement** workflows with file-based tracking across AI agent sessions. This ensures context preservation, systematic progress, and effective collaboration between human developers and AI agents.

## Projects Configured

### 1. DataBolt-Edge
**Path:** [`./databolt-edge/`](./databolt-edge/)  
**Project:** NVIDIA API Management & GPU-Accelerated Data Processing

**Focus:**
- NVIDIA API client and authentication
- GPU-accelerated processing workflows
- API management patterns and retry logic

[→ See DataBolt-Edge README](./databolt-edge/README.md)

---

### 2. AgentOp-Studio
**Path:** [`./agentop-studio/`](./agentop-studio/)  
**Project:** Agent Observability, Cost Tracking & Replay System

**Focus:**
- Agent orchestration and instrumentation
- Cost tracking and token usage monitoring
- Replay engine for debugging
- Streamlit UI for observability

[→ See AgentOp-Studio README](./agentop-studio/README.md)

---

## Workflow Overview

Each project follows the same structured workflow pattern:

### Research → Plan → Implement → Validate

```
┌──────────────┐
│   Research   │  Systematic codebase exploration
│              │  Document findings as you discover
└──────┬───────┘
       ↓
┌──────────────┐
│   Planning   │  Break down task into phases
│              │  Log architectural decisions
└──────┬───────┘
       ↓
┌──────────────┐
│  Implement   │  Execute against the plan
│              │  Track progress and failed attempts
└──────┬───────┘
       ↓
┌──────────────┐
│   Validate   │  Test, verify, document
│              │  Update metrics and learnings
└──────────────┘
```

## Core Workflow Rules

### 1. 2-Action Save Rule ⚡
**Save findings after every 2 file reads/searches**

Prevents context loss and ensures discoveries are documented.

```bash
Read file 1 → Read file 2 → SAVE to findings.md
```

### 2. Read-Before-Decide 📖
**Re-read planning files before major implementation decisions**

Ensures alignment with overall plan and past architectural decisions.

```bash
Before implementing → Re-read task_plan.md → Make decision → Log rationale
```

### 3. 3-Strike Protocol 🔄
**Track failed attempts and pivot after 3 failures**

Prevents infinite loops on failing approaches.

```bash
Attempt 1 → Failed (log)
Attempt 2 → Failed (log)
Attempt 3 → Failed (log)
→ PIVOT to new strategy
```

## File Structure

Each project workspace contains:

```
project-name/
├── config.yaml      # OpenClaw configuration
├── task_plan.md     # Phase tracking, decisions, blockers
├── findings.md      # Research documentation
├── progress.md      # Session log, tests, metrics
└── README.md        # Project-specific workflow guide
```

### File Purposes

| File | Purpose | When to Use |
|------|---------|-------------|
| `config.yaml` | Project configuration, focus areas, rules | Initial setup, context |
| `task_plan.md` | Phase breakdown, decision log, success criteria | Before planning, before implementing |
| `findings.md` | Research discoveries, architecture notes | During research (every 2 actions) |
| `progress.md` | Session activities, tests, errors, learnings | Throughout all phases |
| `README.md` | Workflow guide, commands, tips | Start of every session |

## Quick Start

### Starting Work on DataBolt-Edge

```bash
# 1. Read the workflow guide
cat .openclaw/databolt-edge/README.md

# 2. Review or create a task plan
vim .openclaw/databolt-edge/task_plan.md

# 3. Begin research phase
# - Explore DataBolt-Edge/ directory
# - Document findings every 2 file reads
# - Update .openclaw/databolt-edge/findings.md

# 4. Run tests
cd DataBolt-Edge
uv run pytest tests/ -v
```

### Starting Work on AgentOp-Studio

```bash
# 1. Read the workflow guide
cat .openclaw/agentop-studio/README.md

# 2. Review or create a task plan
vim .openclaw/agentop-studio/task_plan.md

# 3. Begin research phase
# - Explore AgentOp-Studio/ directory
# - Document findings every 2 file reads
# - Update .openclaw/agentop-studio/findings.md

# 4. Run tests
cd AgentOp-Studio
uv run pytest tests/ -v
streamlit run frontend/app.py  # If working on frontend
```

## Example Prompts for AI Agents

### Research Phase
```
Research the [DataBolt-Edge/AgentOp-Studio] codebase.
Focus on understanding [component/feature].
Document findings in .openclaw/[project]/findings.md as you discover them.
Follow the 2-action-save rule: save after every 2 file reads.
```

### Planning Phase
```
Based on findings in .openclaw/[project]/findings.md, 
create a detailed implementation plan for [task].
Break it into clear phases with success criteria.
Update .openclaw/[project]/task_plan.md with the plan.
Log key architectural decisions with rationale.
```

### Implementation Phase
```
Implement Phase 3 subtasks from .openclaw/[project]/task_plan.md.
Log all activities in .openclaw/[project]/progress.md.
Run tests after each significant change.
Apply 3-strike protocol for failed attempts.
```

### Validation Phase
```
Validate implementation against success criteria in task_plan.md.
Run full test suite and document results in progress.md.
Update metrics and log key learnings.
```

## Best Practices

### ✅ Do This

- **Read workflow README at session start** - Restores context quickly
- **Save findings frequently** - Prevent context loss (2-action rule)
- **Re-read plan before big changes** - Ensure alignment
- **Log failed attempts** - Track what doesn't work (3-strike protocol)
- **Update progress in real-time** - Maintain accurate session log
- **Test incrementally** - Catch issues early

### ❌ Avoid This

- Starting implementation without research phase complete
- Forgetting to save findings after file reads
- Making architectural decisions without logging rationale
- Repeating failed approaches without pivoting
- Skipping test validation after changes
- Working on multiple phases simultaneously

## Integration with Repository

### Repository-Level Files
- `/AGENTS.md` - Workspace-wide agent instructions
- `/CLAUDE.md` - Claude-specific guidance
- `/pyproject.toml` - Project dependencies
- `/.github/copilot-instructions.md` - Copilot guidance

### Project-Level Files
- `DataBolt-Edge/README.md` - Project documentation
- `DataBolt-Edge/Makefile` - Build commands
- `AgentOp-Studio/README.md` - Project documentation
- `AgentOp-Studio/PROGRESS.md` - Project milestones

### OpenClaw Planning Files
- `.openclaw/databolt-edge/` - DataBolt-Edge workflow
- `.openclaw/agentop-studio/` - AgentOp-Studio workflow

**Relationship:** OpenClaw planning files complement (not replace) project documentation. Use them for active task tracking and session continuity.

## Debugging & Troubleshooting

### Lost Context?
1. Read `task_plan.md` - Restore task context
2. Review `findings.md` - Recall research discoveries
3. Check `progress.md` - See what was last done

### Stuck on Implementation?
1. Check `progress.md` failed attempts log
2. Review if 3-strike protocol triggered (pivot needed)
3. Re-read `task_plan.md` decision log
4. Consult `findings.md` for architectural patterns

### Tests Failing?
1. Check `progress.md` for previous test results
2. Review recent commits in session log
3. Verify against success criteria in `task_plan.md`
4. Check if changes affected multiple components

### Need to Switch Projects?
1. Save current state in `progress.md`
2. Switch to other project directory
3. Read that project's `README.md`
4. Load context from `task_plan.md`

## Metrics & Tracking

Each project's `progress.md` tracks:
- Sessions completed
- Files modified
- Tests passing/added
- Commits made
- Features added
- Failed attempts (3-strike)

Use this data to:
- Measure progress
- Identify blockers
- Optimize workflow
- Learn from patterns

## Contributing to OpenClaw Configs

When improving these configurations:

1. **Test the workflow** - Use it for a real task first
2. **Document changes** - Update relevant README
3. **Keep consistency** - Both projects follow same pattern
4. **Preserve examples** - Templates help future sessions
5. **Update this README** - Keep master doc in sync

## Questions?

- **For DataBolt-Edge workflow:** See [`./databolt-edge/README.md`](./databolt-edge/README.md)
- **For AgentOp-Studio workflow:** See [`./agentop-studio/README.md`](./agentop-studio/README.md)
- **For repository guidelines:** See `/AGENTS.md` and `/CLAUDE.md`
- **For Copilot context:** See `/.github/copilot-instructions.md`

---

**Last Updated:** February 28, 2026  
**Pattern Version:** Research-Plan-Implement v1.0  
**Projects Configured:** 2 (DataBolt-Edge, AgentOp-Studio)
