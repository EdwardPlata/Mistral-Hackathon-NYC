# OpenClaw Workflow: AgentOp-Studio

This directory contains OpenClaw configuration and planning files for the AgentOp-Studio project (Agent Observability & Cost Tracking Platform).

## Purpose

This workflow implements a **Research → Plan → Implement** pattern with file-based tracking to maintain context and progress across AI agent sessions.

## Files in This Directory

### Configuration
- **`config.yaml`** - OpenClaw project configuration, focus areas, and workflow rules

### Planning & Tracking Files
- **`task_plan.md`** - Phase tracking, decision log, and task breakdown
- **`findings.md`** - Research documentation and codebase discoveries
- **`progress.md`** - Session log, activities, tests, and metrics

## Workflow Pattern

### Phase 1: Research
Before implementing any task, research the codebase to understand:
- Multi-layer architecture (agents, backend, frontend)
- Agent instrumentation and wrapping mechanism
- Cost tracking implementation
- Database schema and persistence
- Replay engine architecture
- UI components and data visualization

**Actions:**
1. Start by reading `task_plan.md` to set context
2. Systematically explore directories in `AgentOp-Studio/`
3. Document discoveries in `findings.md`
4. **Save every 2 search/view operations** (2-action-save rule)

**Prompt Template:**
```
Research the AgentOp-Studio codebase focusing on [component: agents/backend/frontend].
Understand [specific aspect: cost tracking/replay/instrumentation].
Document findings in .openclaw/agentop-studio/findings.md as you discover them.
Follow the 2-action-save rule.
```

### Phase 2: Planning
Convert research into actionable plan:
- Break down task into phases
- Consider both backend AND frontend impact
- Identify component dependencies
- Make architectural decisions (and log them)

**Actions:**
1. Re-read `findings.md` to refresh context
2. Update `task_plan.md` with phases and subtasks
3. Add decision rationale to decision log
4. Define validation strategy (backend tests + frontend manual testing)

**Prompt Template:**
```
Based on findings in .openclaw/agentop-studio/findings.md, create a detailed plan for [task].
Consider impact on both backend and frontend components.
Break it into phases with clear success criteria.
Update .openclaw/agentop-studio/task_plan.md with the plan.
Log key decisions with rationale.
```

### Phase 3: Implementation
Execute against the plan:
- Reference plan before major code changes
- Update progress log after each activity
- Test backend changes with pytest
- Verify frontend changes with Streamlit
- Track failed attempts (3-strike protocol)

**Actions:**
1. **Before implementing:** Re-read `task_plan.md` Phase 3
2. Make changes incrementally
3. Log each activity in `progress.md`
4. Track errors and debugging steps
5. If approach fails 3x, pivot strategy

**Prompt Template:**
```
Implement Phase 3 subtasks from .openclaw/agentop-studio/task_plan.md.
Log all activities and changes in .openclaw/agentop-studio/progress.md.
Run backend tests after changes to backend/.
Test frontend manually if modifying frontend/.
Apply 3-strike protocol for failed attempts.
```

### Phase 4: Validation
Test and verify:
- Run backend test suite
- Test frontend UI manually
- Verify end-to-end data flow
- Check against success criteria from plan

**Actions:**
1. Execute test commands (see `progress.md` Command Reference)
2. Manual UI testing (run Streamlit app)
3. Check against Phase 4 success criteria
4. Update metrics in `progress.md`
5. Document learnings

## Key Rules

### 1. 2-Action Save Rule
After every 2 file reads or searches, immediately save findings to prevent context loss:

```
✅ Good:
1. Read backend/costs.py
2. Read backend/db.py
3. → SAVE to findings.md
4. Read frontend/app.py
5. Read agents/main_agent.py
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

### AgentOp-Studio Architecture
```
AgentOp-Studio/
├── agents/              # Agent orchestration layer
│   ├── main_agent.py    # Main agent logic
│   ├── tools.py         # Tool definitions
│   └── config.yaml      # Agent configuration
├── backend/             # Backend services
│   ├── instrumented_agent.py  # Agent wrapper for observability
│   ├── costs.py         # Cost tracking logic
│   ├── db.py            # Database layer
│   ├── replay.py        # Replay engine
│   └── main.py          # Backend entry point
├── frontend/            # Streamlit UI
│   └── app.py           # Main UI application
├── tests/               # Pytest test suite
│   ├── test_costs.py
│   ├── test_db.py
│   └── test_replay.py
├── infra/               # Infrastructure
│   ├── Dockerfile
│   └── docker-compose.yml
└── data/                # Data storage
```

### Component Interaction Flow
```
┌─────────────┐
│   Frontend  │ (Streamlit)
│   app.py    │
└──────┬──────┘
       │
       ↓
┌──────────────────────────┐
│      Backend             │
├──────────────────────────┤
│ instrumented_agent.py    │ ← Wraps agent execution
│ costs.py                 │ ← Tracks token usage/costs
│ db.py                    │ ← Persists execution data
│ replay.py                │ ← Enables replay/debugging
└──────┬───────────────────┘
       │
       ↓
┌─────────────┐
│   Agents    │
│ main_agent  │ ← Core agent logic
│ tools       │ ← Tool definitions
└─────────────┘
```

### Priority Files to Understand
1. `agents/main_agent.py` - Core agent orchestration
2. `backend/instrumented_agent.py` - Execution wrapping
3. `backend/costs.py` - Cost calculation logic
4. `backend/db.py` - Data persistence
5. `backend/replay.py` - Replay mechanism
6. `frontend/app.py` - UI and visualization

### Common Commands

#### Backend Development
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_costs.py -v

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

#### Frontend Development
```bash
# Run Streamlit app
streamlit run frontend/app.py

# Run on specific port
streamlit run frontend/app.py --server.port 8501
```

#### Infrastructure
```bash
# Build Docker image
docker-compose -f infra/docker-compose.yml build

# Run with Docker
docker-compose -f infra/docker-compose.yml up

# Stop and cleanup
docker-compose -f infra/docker-compose.yml down
```

## Tips for Effective Use

1. **Start Every Session:** Read `task_plan.md` to restore context
2. **After Research:** Always have findings documented before planning
3. **Before Coding:** Verify plan is complete and decisions are logged
4. **Consider Both Layers:** Most changes affect both backend AND frontend
5. **During Implementation:** Keep `progress.md` updated in real-time
6. **Frontend Changes:** Always test manually with `streamlit run frontend/app.py`
7. **Backend Changes:** Always run `pytest` before committing
8. **Hit a Wall?:** Check failed attempts log - might need to pivot

## Multi-Layer Testing Strategy

When making changes, consider this testing approach:

1. **Backend Unit Tests** - Run `pytest tests/`
2. **Backend Integration** - Test component interactions
3. **Frontend Manual Test** - Run Streamlit app and verify UI
4. **End-to-End Flow** - Test complete user journey
5. **Cost Tracking Validation** - Verify costs are calculated correctly
6. **Replay Testing** - If touching replay logic, test replay functionality

## Common Pitfalls

❌ **Changing backend without testing frontend** - Always verify UI still works  
❌ **Forgetting to update database schema** - Changes to stored data need migration  
❌ **Not testing cost calculations** - Cost tracking is critical, always validate  
❌ **Skipping replay tests** - Replay bugs are hard to catch later  
❌ **Not documenting architectural decisions** - Future changes need context

## Directory Reference

This OpenClaw workspace: `/workspaces/Mistral-Hackathon-NYC/.openclaw/agentop-studio/`  
Target project: `/workspaces/Mistral-Hackathon-NYC/AgentOp-Studio/`

## Related Documentation

- Project-level progress: `AgentOp-Studio/PROGRESS.md`
- Project README: `AgentOp-Studio/README.md`

## Questions or Issues?

- Review `findings.md` for unanswered questions section
- Check `progress.md` blockers log
- Re-read this README for workflow guidance
- Consult `task_plan.md` decision log for past architectural decisions
