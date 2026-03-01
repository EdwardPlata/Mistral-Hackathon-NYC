# Task Plan: AgentOp-Studio

## Current Task
**Task:** Add Retry Logic for Mistral API Calls  
**Status:** Planning Phase 🔵  
**Created:** February 28, 2026  
**Last Updated:** February 28, 2026

**Context:** Phase 1 MVP is complete but lacks error recovery. Mistral API calls in `agents/main_agent.py` and `backend/instrumented_agent.py` have no retry logic, making the system fragile under rate limits or network errors. Will implement exponential backoff with the `tenacity` library.

---

## Project Status Overview

### Phase 1: Core Observability MVP ✅ COMPLETE
- All 19 tests passing
- Full agent instrumentation implemented
- Cost tracking for 7 Mistral models
- DuckDB persistence (6 tables)
- Replay engine with unified diffs
- 4-page Streamlit dashboard
- 4-endpoint FastAPI backend
- Docker compose infrastructure
- Optional W&B logging

### Phase 2: Integrations & Evaluation ⏳ PENDING
- Memory snapshot capture
- Memory viewer UI
- Evaluation metrics logging
- Evaluation dashboard
- HuggingFace leaderboard export
- ElevenLabs cost tracking
- W&B artifact upload
- Multi-agent run grouping

### Phase 3: Polish, Auth & CI/CD ⏳ PENDING
- API key auth middleware
- GitHub Actions CI
- Cost spike alerts
- Fine-tune loop automation
- Kubernetes manifests

---

## Potential Next Tasks (Prioritized)

### Option 1: Add Retry Logic for Mistral API (HIGH PRIORITY)
**Why:** Current system fragile under rate limits, no error recovery
**Effort:** Medium (2-3 hours)
**Impact:** High (reliability improvement)

### Option 2: Implement Memory Snapshot Capture (PHASE 2)
**Why:** Table exists but never populated, blocking memory viewer feature
**Effort:** Medium (3-4 hours)
**Impact:** Medium (enables Phase 2 memory features)

### Option 3: Set Up GitHub Actions CI (PHASE 3)
**Why:** No automated testing, manual test runs required
**Effort:** Low (1-2 hours)
**Impact:** High (dev workflow improvement)

### Option 4: Add Evaluation Logging Endpoint (PHASE 2)
**Why:** Evaluations table unused, no way to track agent performance metrics
**Effort:** Medium (2-3 hours)
**Impact:** Medium (enables evaluation dashboard)

### Option 5: Add API Authentication (PHASE 3)
**Why:** FastAPI endpoints currently unprotected
**Effort:** Medium (2-3 hours)
**Impact:** High (security requirement for deployment)

---

## Sample Task Plan: Add Retry Logic for Mistral API

### Phase 1: Research ✅ COMPLETE
**Goal:** Understand current Mistral API call patterns and failure modes

**Activities:**
- [x] Identify all Mistral API call sites (main_agent.py, instrumented_agent.py)
- [x] Review current error handling (none exists)
- [x] Document failure scenarios (rate limits, network errors, timeouts)
- [x] Research retry patterns (exponential backoff, max attempts)

**Findings Reference:** See `findings.md` Section 2 & 3

### Phase 2: Planning ✅ COMPLETE
**Status:** Complete  
**Goal:** Design retry wrapper with exponential backoff

**Activities:**
- [x] Define retry configuration (max_attempts=3, base_delay=1s, max_delay=60s)
- [x] Choose retry library: **tenacity** (widely used, well-maintained, flexible)
- [x] Plan integration points: 4 call sites identified
  - `agents/main_agent.py` lines 78, 109
  - `backend/instrumented_agent.py` lines 131, 240
- [x] Define logging strategy: Log retry attempts to stderr + observability
- [x] Determine error classification: 
  - **Retryable:** Rate limits (429), network errors, timeouts
  - **Non-retryable:** Auth errors (401), validation errors (400)

**Dependencies:**
- Phase 1 complete ✅

**Design Decisions Made:**

#### Decision 1: Use tenacity library
**Rationale:** 
- Well-established library with 3K+ stars
- Supports exponential backoff out-of-box
- Easy to configure retryable exceptions
- Already used in similar projects
- Better than custom implementation (less code, less bugs)

#### Decision 2: Create decorator for reusability
**Approach:** Single `@retry_mistral_call` decorator applied to all 4 call sites
**Rationale:**
- DRY principle: one implementation, four uses
- Easy to test decorator in isolation
- Consistent retry behavior across codebase
- Easy to tune configuration in one place

#### Decision 3: Retry at call level, not client level
**Rationale:**
- More granular control per operation
- Easier to add operation-specific retry config later
- Clearer in tracing/logs which operation failed

#### Decision 4: Log retry attempts to observability
**Implementation:**
- Add `retry_attempt` column to `messages` table (optional migration)
- OR: Log to stderr with structured format for external monitoring
- **Choice:** Start with stderr logging, add DB later if needed
**Rationale:** Simpler initial implementation, no schema migration required

#### Decision 5: Configuration via environment variables
**New env vars:**
- `MISTRAL_RETRY_MAX_ATTEMPTS` (default: 3)
- `MISTRAL_RETRY_MIN_WAIT` (default: 1)
- `MISTRAL_RETRY_MAX_WAIT` (default: 60)
**Rationale:** Allows tuning without code changes, follows existing pattern

### Phase 3: Implementation ✅ COMPLETE
**Status:** Complete  
**Goal:** Add retry decorator/wrapper to Mistral API calls

**Subtasks:**
- [x] Add `tenacity` to pyproject.toml [agentops] dependencies
- [x] ~~Install tenacity: `uv sync --extra agentops`~~ (user must run manually)
- [x] Create retry decorator in new file `AgentOp-Studio/backend/retry.py`
- [x] Configure retry behavior (3 attempts, exponential backoff 1-60s)
- [x] Define retryable exceptions (rate limit, network, timeout)
- [x] Apply decorator to 4 Mistral call sites:
  - [x] `agents/main_agent.py` line 78 (tool round)
  - [x] `agents/main_agent.py` line 109 (final response)
  - [x] `backend/instrumented_agent.py` line 131 (tool round)
  - [x] `backend/instrumented_agent.py` line 240 (final response)
- [x] Add environment variable support for retry config
- [x] Add logging for retry attempts (stderr with structured format)
- [x] Create test file: `tests/test_retry.py` (13 test cases)
- [ ] Test manually with rate limit simulation (pending tenacity install)

**Implementation Summary:**
Created `backend/retry.py` (136 lines) with:
- `retry_mistral_call` decorator using tenacity
- `is_retryable_error()` for error classification
- `get_retry_config()` for debugging
- Environment variable configuration
- Structured logging to stderr

Applied decorator to all 4 Mistral API call sites by wrapping them in local functions.

**Dependencies:**
- Phase 2 complete ✅

### Phase 4: Validation
**Status:** Not Started
**Goal:** Test retry logic under failure conditions

**Activities:**
- [ ] Add unit tests for retry logic
- [ ] Simulate rate limit errors (mock Mistral client)
- [ ] Verify exponential backoff timing
- [ ] Test max attempts respected
- [ ] Verify non-retryable errors fail fast
- [ ] Check retry attempts logged correctly
- [ ] Run full test suite (ensure no regressions)
- [ ] Manual testing with actual Mistral API

**Success Criteria:**
- [ ] Retry logic handles rate limit errors gracefully
- [ ] Max 3 retry attempts with exponential backoff
- [ ] Non-retryable errors (auth, validation) fail immediately
- [ ] Retry attempts visible in logs/observability
- [ ] All existing tests still passing
- [ ] No performance degradation for successful calls

---

## Phases (Generic Template - Use for New Tasks)

### Phase 1: Research
**Status:** ⏳ Not Started  
**Goal:** Understand the AgentOp-Studio architecture, components, and data flow

**Activities:**
- [ ] Review relevant files in `findings.md`
- [ ] Identify affected components
- [ ] Document current behavior
- [ ] Research best practices/libraries

**Findings RIn Progress  
**Goal:** Test retry logic under various failure conditions

**Activities:**
- [ ] Install tenacity: `uv sync --extra agentops`
- [ ] Run new retry unit tests: `pytest tests/test_retry.py -v` (13 tests expected)
- [ ] Run full existing test suite: `pytest tests/ -v` (should be 32 tests: 19 + 13)
- [ ] Verify exponential backoff timing (check logs)
- [ ] Test max attempts respected (check logs)
- [ ] Verify non-retryable errors fail fast
- [ ] Check retry attempts logged correctly
- [ ] Manual testing with actual Mistral API (if API key available)
- [ ] Consider: Simulate rate limit with mock/patch for integration test

**Success Criteria:**
- [x] Retry logic implemented in all 4 call sites
- [x] Non-retryable errors classified correctly
- [x] Retry attempts will be visible in logs (stderr)
- [x] No syntax errors in modified files
- [ ] All 13 new retry tests passing
- [ ] All 19 existing tests still passing (no regressions)
- [ ] Retry behavior verified via logging output

**Testing Commands:**
```bash
# Install dependency
cd /workspaces/Mistral-Hackathon-NYC
uv sync --extra agentops

# Run retry tests only
cd AgentOp-Studio
uv run pytest tests/test_retry.py -v

# Run all tests
uv run pytest tests/ -v

# Check for lint errors
uv run ruff check .

# Manual test (if MISTRAL_API_KEY available)
uv run python -c "from agents.main_agent import run_agent; print(run_agent('Hello'))"
```

**Expected Test Count:** 32 total (19 existing + 13 new retry tests)

### Phase 3: Implementation
**Status:** ⏳ Not Started  
**Goal:** Execute the planned changes

**Subtasks:**
<!-- Add specific subtasks based on your plan -->
- [ ] Subtask 1
- [ ] Subtask 2
- [ ] Subtask 3

**Dependencies:**
- Phase 2 must be complete

---

### Phase 4: Validation
**Status:** ⏳ Not Started  
**Goal:** Test and verify implementation

**Activities:**
- [ ] Run backend test suite
- [ ] Test frontend UI functionality
- [ ] Verify cost tracking accuracy
- [ ] Test replay functionality
- [ ] Integration testing (end-to-end)
- [ ] Check database migrations (if applicable)
- [ ] Update documentation
- [ ] Code quality check (lint/format)

**Success Criteria:**
- [ ] All tests passing
- [ ] UI functional and responsive  
- [ ] No regressions in cost tracking
- [ ] Replay works correctly
- [ ] Documentation updated

---

## Decision Log

### Decision 1: February 28, 2026 - Research Complete
**Context:** Completed initial research of AgentOp-Studio. Phase 1 MVP complete, need to prioritize next tasks.  
**Options Considered:** 
1. Proceed with Phase 2 features (memory, evaluations)
2. Focus on reliability improvements (retry logic, CI)
3. Add security features (auth, rate limiting)
4. Performance optimization (Postgres migration)

**Decision:** Recommend reliability improvements first (retry logic, CI) before adding new features  
**Rationale:** Phase 1 is functional but fragile (no retry logic, no automated testing). Stabilize before expanding. Can revisit Phase 2 features after establishing solid CI/CD pipeline.

---

## Blockers & Risks

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| No Mistral API retry logic | High | Implement exponential backoff wrapper | Open |
| No automated CI/CD | Medium | Set up GitHub Actions | Open |
| DuckDB not production-ready | Medium | Document Postgres migration path for production | Open |
| memory_snapshots/evaluations tables unused | Low | Phase 2 implementation | Acceptable |
| No API authentication | High | Add FastAPI middleware with API keys | Open |
| ElevenLabs cost tracking missing | Low | Extend costs.py similar to Mistral pricing | Open |

---

## Component Dependencies

```
┌─────────────┐
│   Frontend  │ (Streamlit UI)
│   app.py    │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────┐
│         Backend                 │
├─────────────────────────────────┤
│ instrumented_agent.py           │
│ costs.py (tracking)             │
│ db.py (persistence)             │
│ replay.py (replay engine)       │
└──────┬──────────────────────────┘
       │
       ↓
┌─────────────┐
│   Agents    │ (Orchestration)
│ main_agent  │
│ tools       │
└─────────────┘
```

---

## Notes
- Remember to save findings after every 2 search/view operations
- Re-read this plan before major implementation decisions
- Apply 3-strike protocol: if approach fails 3 times, pivot strategy
- Consider impact on both frontend AND backend when making changes
