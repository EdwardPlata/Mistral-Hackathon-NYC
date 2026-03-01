# OpenClaw Workflow: DataBolt-Edge

This directory contains OpenClaw configuration and planning files for the DataBolt-Edge project (NVIDIA API Management system).

## Purpose

This workflow implements a **Research → Plan → Implement** pattern with file-based tracking to maintain context and progress across AI agent sessions.

## Files in This Directory

### Configuration
- **`config.yaml`** - OpenClaw project configuration, focus areas, and workflow rules

### Planning & Tracking Files
- **`task_plan.md`** - Phase tracking, decision log, and task breakdown
- **`findings.md`** - Research documentation and codebase discoveries
- **progress.md`** - Session log, activities, tests, and metrics

## Workflow Pattern

### Phase 1: Research
Before implementing any task, research the codebase to understand:
- Architecture and component relationships
- Existing patterns and conventions
- Integration points and dependencies
- Test infrastructure

**Actions:**
1. Start by reading `task_plan.md` to set context
2. Systematically explore directories in `DataBolt-Edge/`
3. Document discoveries in `findings.md`
4. **Save every 2 search/view operations** (2-action-save rule)

**Prompt Template:**
```
Research the DataBolt-Edge codebase in [directory/component].
Focus on understanding [specific aspect].
Document findings in .openclaw/databolt-edge/findings.md as you discover them.
Follow the 2-action-save rule.
```

### Phase 2: Planning
Convert research into actionable plan:
- Break down task into phases
- Identify dependencies and success criteria
- Make architectural decisions (and log them)

**Actions:**
1. Re-read `findings.md` to refresh context
2. Update `task_plan.md` with phases and subtasks
3. Add decision rationale to decision log
4. Define validation strategy

**Prompt Template:**
```
Based on findings in .openclaw/databolt-edge/findings.md, create a detailed plan for [task].
Break it into phases with clear success criteria.
Update .openclaw/databolt-edge/task_plan.md with the plan.
Log key decisions with rationale.
```

### Phase 3: Implementation
Execute against the plan:
- Reference plan before major code changes
- Update progress log after each activity
- Track failed attempts (3-strike protocol)

**Actions:**
1. **Before implementing:** Re-read `task_plan.md` Phase 3
2. Make changes incrementally
3. Log each activity in `progress.md`
4. Track errors and debugging steps
5. If approach fails 3x, pivot strategy

**Prompt Template:**
```
Implement Phase 3 subtasks from .openclaw/databolt-edge/task_plan.md.
Log all activities and changes in .openclaw/databolt-edge/progress.md.
Run tests after each significant change.
Apply 3-strike protocol for failed attempts.
```

### Phase 4: Validation
Test and verify:
- Run test suite
- Verify success criteria from plan
- Update documentation

**Actions:**
1. Execute test commands (see `progress.md` Command Reference)
2. Check against Phase 4 success criteria
3. Update metrics in `progress.md`
4. Document learnings

## Key Rules

### 1. 2-Action Save Rule
After every 2 file reads or searches, immediately save findings to prevent context loss:

```
✅ Good:
1. Read nvidia_api_management/client.py
2. Read nvidia_api_management/auth.py
3. → SAVE to findings.md
4. Read tests/test_nvidia_client.py
5. Read tests/test_nvidia_auth.py
6. → SAVE to findings.md

❌ Bad:
1. Read 10 files
2. Forget findings before saving
```

### 2. Read-Before-Decide
Before major implementation decisions, re-read planning files to ensure alignment:

```
✅ Good:
1. Re-read task_plan.md Phase 3
2. Review decision log
3. Make implementation decision
4. Log decision rationale

❌ Bad:
1. Start coding without checking plan
2. Implement approach that conflicts with logged decisions
```

### 3. 3-Strike Protocol
Track failed attempts and mutate approach after 3 failures:

```
✅ Good:
Attempt 1: Try approach A → Failed (log error)
Attempt 2: Tweak approach A → Failed (log error)
Attempt 3: Modify approach A → Failed (log error)
→ PIVOT to completely different approach B

❌ Bad:
Attempt 1-10: Keep trying same approach with minor tweaks
```

## Project-Specific Context

### DataBolt-Edge Structure
```
DataBolt-Edge/
├── nvidia_api_management/  # Core API client and auth
├── scripts/                 # Testing and probing utilities
├── tests/                   # Pytest test suite
├── Makefile                 # Build automation
└── README.md                # Project documentation
```

### Priority Files to Understand
1. `nvidia_api_management/client.py` - Main API client
2. `nvidia_api_management/auth.py` - Authentication flow
3. `nvidia_api_management/config.py` - Configuration management
4. `tests/test_nvidia_client.py` - Core test patterns

### Common Commands
```bash
# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_nvidia_client.py -v

# Lint
uv run ruff check .

# Run probe script
uv run python scripts/live_nvidia_probe.py

# Check dependencies
uv sync --extra dev
```

## Tips for Effective Use

1. **Start Every Session:** Read `task_plan.md` to restore context
2. **After Research:** Always have findings documented before planning
3. **Before Coding:** Verify plan is complete and decisions are logged
4. **During Implementation:** Keep `progress.md` updated in real-time
5. **Hit a Wall?:** Check failed attempts log - might need to pivot

## Directory Reference

This OpenClaw workspace: `/workspaces/Mistral-Hackathon-NYC/.openclaw/databolt-edge/`  
Target project: `/workspaces/Mistral-Hackathon-NYC/DataBolt-Edge/`

## Questions or Issues?

- Review `findings.md` for unanswered questions section
- Check `progress.md` blockers log
- Re-read this README for workflow guidance
