# Research Findings: AgentOp-Studio

**Last Updated:** February 28, 2026  
**Research Sessions:** 1

---

## Section 1: Architecture Overview

### Directory Structure
```
AgentOp-Studio/
├── agents/          # Agent orchestration and tool definitions
├── backend/         # Instrumented agent, cost tracking, DB, replay
├── frontend/        # Streamlit UI dashboard
├── tests/           # Pytest test suite (19 tests, all passing)
├── infra/           # Docker setup (Dockerfile + compose)
├── data/            # DuckDB storage location
├── .plan            # Planning file (discovered)
├── PROGRESS.md      # Project-level progress tracker
└── README.md        # Architecture documentation
```

**Key Observations:**
- **Phase 1 is COMPLETE**: Full observability MVP with instrumented agent, cost tracking, DB persistence, replay/diff, and Streamlit UI
- **19 tests all passing** (test_costs, test_replay, test_db)
- **DuckDB-based persistence** with 6 tables: runs, messages, tool_calls, memory_snapshots, diffs, evaluations
- **W&B optional logging** built in (logs if WANDB_API_KEY present)
- Uses **Mistral Agents API** for LLM, supports 7 models with cost tracking
- **Docker-ready** with backend (port 8000) + frontend (port 8501)

---

## Section 2: Agent Orchestration

### Main Agent
**File:** `agents/main_agent.py`
- **Purpose:** Uninstrumented agentic loop for Mistral model interactions
- **Key Functions:**
  - `run_agent(prompt)` - Main entry point for agent execution
  - `_dispatch_tool(name, arguments)` - Tool invocation by name lookup
- **Configuration:** Uses env vars: `MISTRAL_API_KEY`, `MISTRAL_MODEL`, `MAX_TOKENS`, `TEMPERATURE`
- **Tool Integration:** Max 5 tool-call rounds before stopping
- **Model:** Defaults to `mistral-large-latest`

**File:** `agents/tools.py`
- **Purpose:** Tool definitions with JSON schemas for Mistral function calling
- **Available Tools:**
  1. `wandb_log(metrics, step)` - Log metrics to W&B
  2. `hf_list_models(query, limit)` - Search HuggingFace Hub
  3. `elevenlabs_speak(text, voice_id, output_path)` - Text-to-speech via ElevenLabs
- **Tool Pattern:** Python function + matching JSON schema in `get_tools()`
- **External APIs:** W&B, HuggingFace Hub, ElevenLabs

**File:** `agents/config.yaml`
- **Purpose:** Model/tool settings with env-var interpolation
- **Status:** Referenced in PROGRESS.md but configuration handled via environment variables

### Agent Workflow
1. **User prompt** → messages list initialized
2. **Model call** → Mistral API with tools parameter
3. **Tool requests** → Dispatched via `_dispatch_tool()`
4. **Tool results** → Fed back as tool messages
5. **Loop continues** → Up to 5 rounds or until text response
6. **Final response** → Returned to caller

---

## Section 3: Backend Components

### Instrumentation Layer
**File:** `backend/instrumented_agent.py`
- **Purpose:** Wraps agent execution with full observability logging to DuckDB
- **Key Function:** `run_instrumented(prompt, user_id="default", agent_id="main")`
- **How it Works:**
  1. Generates unique `run_id` (UUID)
  2. Inserts run record with status='running'
  3. Executes agent loop (same as main_agent.py)
  4. Logs every message, tool call, token count
  5. Updates run status to 'success' or 'error'
  6. Optionally logs to W&B if `WANDB_API_KEY` set
- **Data Captured:**
  - Run metadata (id, start/end time, status, config)
  - All messages (user, assistant, tool)
  - Tool calls (name, args, return value, latency_ms)
  - Token counts and cost per run
- **Integration Points:** Called by FastAPI backend, writes to DuckDB

### Cost Tracking
**File:** `backend/costs.py`
- **Purpose:** Estimate USD cost for Mistral model token usage
- **Metrics Tracked:**
  - Token usage (total count)
  - Model-specific pricing (input/output rates per 1K tokens)
  - Total cost in USD
- **Cost Model:** 7 Mistral models supported with different pricing tiers
  - `mistral-large-latest`: $0.003 input / $0.009 output per 1K tokens
  - `mistral-small-latest`: $0.001 input / $0.003 output per 1K tokens
  - `open-mistral-7b`: $0.00025 per 1K tokens (both)
  - etc.
- **Function:** `estimate_cost(tokens, model)` - Uses average of input+output rates
- **Storage:** Cost written to `runs.total_cost` column

### Database Layer
**File:** `backend/db.py`
- **Purpose:** DuckDB persistence layer for all observability data
- **Database Type:** DuckDB (file-based, located in `data/agentops.duckdb`)
- **Connection Pattern:** Fresh connection per call (not thread-safe with shared connections)
- **Schema Overview:**
  1. **runs** - Run metadata (id, agent_id, user_id, times, status, tokens, cost, config, prompt, response)
  2. **messages** - Message log (id, run_id, role, content, timestamp, tokens, finish_reason)
  3. **tool_calls** - Tool invocations (id, run_id, message_id, tool_name, args JSON, return_value, latency_ms)
  4. **memory_snapshots** - Memory state (id, run_id, timestamp, memory_json) [NOT YET USED]
  5. **diffs** - Replay comparisons (id, run_id, parameter_changed, before/after output, diff_json)
  6. **evaluations** - Metrics (id, run_id, metric_name, metric_value) [NOT YET USED]
- **Key Functions:**
  - `get_conn()` - Returns new DuckDB connection
  - `init_schema()` - Idempotent table creation

### Replay Engine
**File:** `backend/replay.py`
- **Purpose:** Re-run previous agent executions with parameter overrides and generate diffs
- **Key Function:** `replay_run(run_id, override_params=None)`
- **Replay Mechanism:**
  1. Fetch original run from DB (prompt, config, response)
  2. Temporarily override env vars (e.g., MISTRAL_MODEL, TEMPERATURE)
  3. Call `run_instrumented()` with original prompt
  4. Restore original env vars
  5. Generate unified diff between original and new response
  6. Store diff in `diffs` table
- **Data Requirements:** Original run must exist in `runs` table
- **Use Cases:**
  - Test model sensitivity to temperature changes
  - Compare different models on same prompt
  - Debug regression issues
- **Diff Format:** Python `difflib.unified_diff` format

---

## Section 4: Frontend UI

### Streamlit Application
**File:** `frontend/app.py`
- **Purpose:** Multi-page Streamlit dashboard for agent observability
- **API Integration:** Communicates with FastAPI backend via httpx
- **Base URL:** `AGENTOPS_API_URL` env var (default: http://localhost:8000)

### UI Components (4 Pages)

#### 1. Dashboard Page
- **KPI Metrics:** Total Runs, Total Tokens, Total Cost (USD)
- **Charts:** 
  - Bar chart: Tokens per Run
  - Bar chart: Cost per Run (USD)
- **Data Table:** All runs with full details
- **Navigation:** Input field to jump to Run Detail by ID

#### 2. Run Agent Page
- **Input:** Prompt text area + user ID field
- **Action:** Submit button → POST /run
- **Output:** Displays response, run_id, tokens, cost

#### 3. Run Detail Page
- **Input:** Run ID selector/input
- **Display:**
  - Message thread with role icons (user/assistant/tool)
  - Tool calls table (name, args, return value, latency)
- **Data:** Fetched from GET /runs/{run_id}

#### 4. Replay & Diff Page
- **Input:** 
  - Run selector (dropdown)
  - Override parameters (MISTRAL_MODEL, TEMPERATURE)
- **Action:** Replay button → POST /replay
- **Output:** 
  - Side-by-side comparison (original vs new response)
  - Unified diff view
  - Metrics comparison

### UI Flow
1. **Dashboard** → Browse all runs, see metrics
2. **Run Agent** → Submit new prompt, see result
3. **Run Detail** → Deep dive into specific run
4. **Replay & Diff** → Experiment with parameter changes

---

## Section 5: Testing Infrastructure

### Test Files
- `tests/test_costs.py` - **6 tests** for cost estimation logic
  - Known model costs
  - Zero tokens edge case
  - Unknown model fallback to default pricing
  - Proportional scaling validation
  - Price comparison between models
- `tests/test_db.py` - **7 tests** for DuckDB schema and CRUD operations
  - Schema creation (idempotent)
  - Runs, messages, tool_calls, diffs, evaluations table writes
  - Uses `tmp_path` fixtures for isolated DB per test
- `tests/test_replay.py` - **6 tests** for diff generation
  - Identical strings (no diff)
  - Changed strings with unified format
  - Multiline diffs
  - Add/remove line detection
- `tests/conftest.py` - Test configuration
  - Adds `AgentOp-Studio/` to sys.path for imports
  - Shared fixtures and setup

### Test Patterns
- **Isolated DB tests:** Each test uses `tmp_path` fixture for clean DuckDB instance
- **Edge case coverage:** Zero tokens, unknown models, empty strings
- **Proportional validation:** Tests that costs scale linearly with tokens
- **Deterministic diffs:** Uses difflib.unified_diff for reproducible output

### Coverage Analysis
- **Well-tested (19 tests total, all passing):**
  - Cost estimation (backend/costs.py)
  - Database schema and CRUD (backend/db.py)
  - Replay diff logic (backend/replay.py)
- **Coverage gaps:**
  - No tests for `instrumented_agent.py` (integration complexity)
  - No tests for `frontend/app.py` (Streamlit UI)
  - No tests for `agents/main_agent.py` (requires Mistral API)
  - No tests for `agents/tools.py` (requires external API keys)
  - `memory_snapshots` and `evaluations` tables never written (Phase 2 feature)

---

## Section 6: Infrastructure

### Docker Setup
**File:** `infra/Dockerfile`
- **Base Image:** Python 3.11-slim
- **Package Manager:** UV (fast Python package installer)
- **Dependencies:** Installs `agentops` extras from pyproject.toml
- **Working Directory:** /app
- **Build Context:** Builds from repository root

**File:** `infra/docker-compose.yml`
- **Services:**
  1. **backend** (FastAPI)
     - Port: 8000 (exposed)
     - Command: `uvicorn backend.main:app --host 0.0.0.0`
     - PYTHONPATH: `/app/AgentOp-Studio`
     - Volume: `data/` mounted for DuckDB persistence
     - Environment: Loaded from `../../.env`
  2. **frontend** (Streamlit)
     - Port: 8501 (exposed)
     - Command: `streamlit run AgentOp-Studio/frontend/app.py`
     - Environment: `AGENTOPS_API_URL=http://backend:8000`
     - Depends on: backend service
- **Networking:** Frontend connects to backend via Docker service name (`backend:8000`)
- **Data Persistence:** DuckDB file mounted as volume (survives container restarts)

---

## Section 7: Data Flow & FastAPI Backend

### FastAPI Backend
**File:** `backend/main.py`
- **Framework:** FastAPI with Pydantic models
- **Title:** AgentOps-Studio v0.1.0
- **Startup:** Calls `init_schema()` to ensure tables exist

### API Endpoints

#### POST /run
- **Purpose:** Execute agent with a prompt
- **Request:** `{prompt: str, user_id: str = "default"}`
- **Process:**
  1. Calls `run_instrumented(prompt, user_id)`
  2. Agent executes with full observability logging
  3. All data persisted to DuckDB
- **Response:** `{run_id, response, total_tokens, total_cost}`

#### POST /replay
- **Purpose:** Replay a run with parameter overrides
- **Request:** `{run_id: str, override_params: dict | None}`
- **Process:**
  1. Calls `replay_run(run_id, override_params)`
  2. Fetches original run
  3. Temporarily overrides env vars (e.g., model, temperature)
  4. Re-runs agent with original prompt
  5. Generates unified diff
  6. Stores diff in database
- **Response:** `{diff_id, run_id, new_run_id, original_response, new_response, diff}`

#### GET /runs
- **Purpose:** List all runs
- **Query:** Orders by start_time DESC
- **Response:** `[{run_id, agent_id, user_id, start_time, end_time, status, total_tokens, total_cost, ...}]`

#### GET /runs/{run_id}
- **Purpose:** Get detailed run information
- **Query:** Joins runs + messages + tool_calls
- **Response:** 
  ```json
  {
    "summary": {...run metadata...},
    "messages": [{message_id, role, content, timestamp, tokens}],
    "tool_calls": [{call_id, tool_name, args, return_value, latency_ms}]
  }
  ```

### Execution Flow
```
User (Streamlit UI)
  ↓
  POST /run {prompt: "..."}
  ↓
[FastAPI Backend]
  ↓
run_instrumented()
  ├─→ INSERT INTO runs (status='running')
  ├─→ Mistral API call (with tools)
  ├─→ Tool dispatch (_dispatch_tool)
  ├─→ INSERT INTO messages (user, assistant, tool)
  ├─→ INSERT INTO tool_calls (name, args, result, latency)
  ├─→ estimate_cost() → UPDATE runs.total_cost
  ├─→ Optional: _log_to_wandb()
  └─→ UPDATE runs (status='success', end_time)
  ↓
Response: {run_id, response, total_tokens, total_cost}
  ↓
[Streamlit UI displays result]
```

### Data Persistence
- **What gets stored:**
  - Run metadata (config, timing, status, tokens, cost)
  - Complete message history (user, assistant, tool messages)
  - Tool call records (name, arguments, return values, latency)
  - Replay diffs (parameter changes, before/after outputs)
  - (Future: memory snapshots, evaluation metrics)
- **When it's stored:** 
  - Run record: Created at start (status='running'), updated at end
  - Messages: Inserted after each Mistral API response
  - Tool calls: Inserted immediately after tool execution
  - Diffs: Created during replay operations
- **How to query:**
  - Via FastAPI endpoints (GET /runs, GET /runs/{id})
  - Direct DuckDB queries via `get_conn()`
  - Streamlit UI uses API for all data access

### W&B Optional Logging
- **Trigger:** If `WANDB_API_KEY` env var is set
- **Data Logged:**
  - total_tokens (int)
  - total_cost_usd (float)
  - status (1 for success, 0 for error)
  - model (string)
- **When:** After run completes in `run_instrumented()`
- **Project:** Configurable via `WANDB_PROJECT` env var (default: agentops-studio)
- **Run ID:** Uses same UUID as AgentOps run_id for correlation
- **Error Handling:** Silent failure (best-effort logging)

---

## Section 8: Configuration & Dependencies

### Configuration Files
**File:** `agents/config.yaml`
- **Model Settings:**
  - Provider: mistral
  - Name: mistral-large-latest
  - Endpoint: `${MISTRAL_ENDPOINT}` (env var interpolation)
  - API Key: `${MISTRAL_API_KEY}`
- **Tool Enablement:**
  - nvidia_nemo: enabled (requires `NVIDIA_API_KEY`)
  - huggingface: enabled (requires `HF_TOKEN`)
  - elevenlabs: enabled (requires `ELEVENLABS_API_KEY`)
  - wandb: enabled, project: `mistral-hackathon-nyc`
- **Settings:**
  - max_tokens: 2048
  - temperature: 0.6

**Environment Variables (from .env or Codespaces secrets):**
- `MISTRAL_API_KEY` - **Required** for agent execution
- `WANDB_API_KEY` - Optional for W&B logging
- `HF_TOKEN` - For HuggingFace Hub tool
- `ELEVENLABS_API_KEY` - For TTS tool
- `NVIDIA_API_KEY` - For NVIDIA NeMo toolkit
- `AGENTOPS_DB_PATH` - DuckDB file path (default: `data/agentops.duckdb`)
- `AGENTOPS_API_URL` - Frontend → Backend URL (default: http://localhost:8000)
- `MISTRAL_MODEL` - Model override (default: mistral-large-latest)
- `MAX_TOKENS` - Token limit override (default: 2048)
- `TEMPERATURE` - Sampling temperature (default: 0.6)
- `WANDB_PROJECT` - W&B project name (default: agentops-studio)

### External Dependencies
```
mistralai - Mistral API client for LLM and function calling
streamlit - Frontend UI framework
fastapi - Backend REST API framework
uvicorn - ASGI server for FastAPI
pydantic - Request/response validation
duckdb - Embedded SQL database for observability data
httpx - HTTP client for frontend ↔ backend communication
pandas - Data manipulation for UI tables
plotly - Interactive charts in Streamlit
wandb - Experiment tracking (optional)
huggingface_hub - HuggingFace API client
elevenlabs - Text-to-speech API client
python-dotenv - Environment variable loading
```

### Integration Points
1. **Integration with: Mistral API**
   - **How:** Python SDK (`mistralai` package)
   - **Files involved:** `agents/main_agent.py`, `backend/instrumented_agent.py`
   - **Purpose:** LLM reasoning, function calling, multi-turn conversations

2. **Integration with: Weights & Biases**
   - **How:** W&B Python SDK (optional, best-effort logging)
   - **Files involved:** `backend/instrumented_agent.py`, `agents/tools.py`
   - **Purpose:** Experiment tracking, metrics logging

3. **Integration with: HuggingFace Hub**
   - **How:** `huggingface_hub` API client
   - **Files involved:** `agents/tools.py` (`hf_list_models` tool)
   - **Purpose:** Model search and discovery

4. **Integration with: ElevenLabs**
   - **How:** ElevenLabs Python SDK
   - **Files involved:** `agents/tools.py` (`elevenlabs_speak` tool)
   - **Purpose:** Text-to-speech generation

5. **Integration with: Frontend ↔ Backend**
   - **How:** REST API (httpx client → FastAPI endpoints)
   - **Files involved:** `frontend/app.py` ↔ `backend/main.py`
   - **Endpoints:** POST /run, POST /replay, GET /runs, GET /runs/{id}

---

## Section 9: Task-Specific Research

### Current Project Status (From PROGRESS.md)

**Phase 1: COMPLETE ✅**
- All 19 tests passing
- Backend: instrumented agent, costs, DB, replay all implemented
- Frontend: 4-page Streamlit dashboard fully functional
- Infrastructure: Docker compose with backend + frontend
- W&B optional logging working

**Phase 2: PENDING**
- Memory snapshot capture (table exists but never written)
- Memory viewer in UI
- Evaluation metric logging (table exists but never written)
- Evaluation dashboard
- HuggingFace leaderboard export
- ElevenLabs TTS cost tracking
- W&B artifact upload
- Multi-agent run grouping

**Phase 3: PENDING**
- API key auth middleware
- GitHub Actions CI
- Cost spike alerts
- Fine-tune loop automation
- Kubernetes manifests

### Known Issues / Tech Debt
1. **Security fix applied:** `eval()` replaced with `json.loads()` in frontend ✅
2. **Feature gaps:**
   - `memory_snapshots` table never written (Phase 2)
   - `evaluations` table never written (Phase 2)
3. **Scalability concerns:**
   - Single DuckDB file, no WAL conflict protection
   - Acceptable for demo, use Postgres for production
4. **Reliability:**
   - No retry logic on Mistral API failures
   - Need exponential backoff for rate limits

### Architecture Strengths
1. **Clean separation of concerns:**
   - Agent logic separate from instrumentation
   - Tools modular and composable
   - Backend API well-structured
2. **Observability-first design:**
   - Every interaction logged
   - Full replay capability
   - Cost tracking built-in
3. **Optional integrations:**
   - W&B logging gracefully degrades if no API key
   - Tools can be disabled without breaking core workflow
4. **Docker-ready:**
   - Multi-service compose
   - Volume persistence for data
   - Proper PYTHONPATH configuration

### Potential Improvements
1. Add retry logic with exponential backoff for Mistral API
2. Implement Phase 2 features (memory, evaluations)
3. Add GitHub Actions CI/CD pipeline
4. Switch to PostgreSQL for production deployments
5. Add API authentication middleware
6. Implement cost spike alerting
7. Create comprehensive frontend tests

---

## Research Session Log

### Session 1: February 28, 2026
**Focus:** Complete codebase architecture and component discovery  
**Files Examined:**
- README.md (architecture overview)
- PROGRESS.md (project status)
- agents/main_agent.py (uninstrumented agent loop)
- agents/tools.py (W&B, HF, ElevenLabs tools)
- agents/config.yaml (model and tool settings)
- backend/instrumented_agent.py (observability wrapper)
- backend/costs.py (cost estimation)
- backend/db.py (DuckDB schema)
- backend/replay.py (replay with diffs)
- backend/main.py (FastAPI endpoints)
- frontend/app.py (Streamlit UI)
- tests/test_costs.py (cost test patterns)
- infra/docker-compose.yml (service setup)

**Key Discoveries:**
- Phase 1 MVP is **complete** with 19 passing tests
- Full observability stack implemented (agent → instrumentation → DB → UI)
- 4-endpoint REST API + 4-page Streamlit dashboard
- DuckDB with 6 tables (2 unused: memory_snapshots, evaluations)
- Docker-ready with backend (8000) + frontend (8501)
- W&B logging is optional and best-effort
- 7 Mistral models supported with accurate cost tracking
- Replay engine with unified diff generation
- Clean separation: main_agent.py (uninstrumented) vs instrumented_agent.py
- Tools: wandb_log, hf_list_models, elevenlabs_speak

**Next Steps:**
- Review and update task_plan.md with potential Phase 2 tasks
- Update progress.md with this research session
- Determine specific task to work on

---

## Quick Reference

### Critical Files for Current Task
- ✅ `agents/main_agent.py` - Base agent loop (uninstrumented)
- ✅ `agents/tools.py` - Tool definitions (W&B, HF, ElevenLabs)
- ✅ `agents/config.yaml` - Model and tool settings
- ✅ `backend/instrumented_agent.py` - Observability wrapper (Phase 1 complete)
- ✅ `backend/costs.py` - Cost estimation (7 Mistral models)
- ✅ `backend/db.py` - DuckDB schema (6 tables)
- ✅ `backend/replay.py` - Replay with diffs
- ✅ `backend/main.py` - FastAPI endpoints (4 routes)
- ✅ `frontend/app.py` - Streamlit dashboard (4 pages)
- ⏳ `tests/*` - 19 tests passing (gaps: instrumented_agent, frontend, agent integration)

### Key Functions/Classes
- `run_agent(prompt)` - [agents/main_agent.py] - Uninstrumented agent loop
- `run_instrumented(prompt, user_id, agent_id)` - [backend/instrumented_agent.py] - Observability-wrapped execution
- `estimate_cost(tokens, model)` - [backend/costs.py] - USD cost calculation
- `get_conn()` - [backend/db.py] - Get fresh DuckDB connection
- `init_schema()` - [backend/db.py] - Idempotent table creation
- `replay_run(run_id, override_params)` - [backend/replay.py] - Re-run with overrides + diff
- `_dispatch_tool(name, arguments)` - [agents/main_agent.py, backend/instrumented_agent.py] - Tool invocation
- `get_tools()` - [agents/tools.py] - Return Mistral function calling schemas
- `wandb_log()`, `hf_list_models()`, `elevenlabs_speak()` - [agents/tools.py] - External API tools

### Configuration Variables
- `MISTRAL_API_KEY` - **Required** - Mistral API authentication
- `MISTRAL_MODEL` - Optional - Model override (default: mistral-large-latest)
- `MAX_TOKENS` - Optional - Token limit (default: 2048)
- `TEMPERATURE` - Optional - Sampling temperature (default: 0.6)
- `WANDB_API_KEY` - Optional - Enables W&B logging
- `WANDB_PROJECT` - Optional - W&B project name (default: agentops-studio)
- `HF_TOKEN` - Optional - HuggingFace API token
- `ELEVENLABS_API_KEY` - Optional - ElevenLabs TTS API key
- `AGENTOPS_DB_PATH` - Optional - DuckDB file location (default: data/agentops.duckdb)
- `AGENTOPS_API_URL` - Optional - Frontend→Backend URL (default: http://localhost:8000)

### Database Schema Quick Reference
1. **runs** - Run metadata (id, agent_id, user_id, times, status, tokens, cost, config, prompt, response)
2. **messages** - Message log (id, run_id, role, content, timestamp, tokens, finish_reason)
3. **tool_calls** - Tool invocations (id, run_id, message_id, tool_name, args, return_value, latency_ms)
4. **memory_snapshots** - Memory state [UNUSED - Phase 2]
5. **diffs** - Replay comparisons (id, run_id, parameter_changed, before/after, diff_json)
6. **evaluations** - Metrics [UNUSED - Phase 2]

---

## Component Interaction Matrix

| Component | Depends On | Used By | Data Flow | Status |
|-----------|------------|---------|-----------|--------|
| main_agent | tools, Mistral API | instrumented_agent | Executes tasks, calls tools | ✅ Complete |
| instrumented_agent | main_agent, costs, db, Mistral API | FastAPI backend | Wraps execution with observability | ✅ Complete |
| costs | MISTRAL_PRICING dict | instrumented_agent | Calculates USD cost from tokens | ✅ Complete |
| db | DuckDB | instrumented_agent, replay, FastAPI | Persists all observability data | ✅ Complete |
| replay | db, instrumented_agent | FastAPI backend | Reproduces runs with overrides | ✅ Complete |
| tools (W&B, HF, ElevenLabs) | External APIs | main_agent, instrumented_agent | Provides agent capabilities | ✅ Complete |
| FastAPI backend | instrumented_agent, replay, db | frontend | REST API for UI | ✅ Complete |
| frontend | FastAPI backend (httpx) | user | Displays metrics, runs agent, shows diffs | ✅ Complete |

### Data Flow Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                      (Streamlit Frontend)                       │
│   Dashboard | Run Agent | Run Detail | Replay & Diff           │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP (httpx)
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│  POST /run | POST /replay | GET /runs | GET /runs/{id}         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
      ┌───────────┴──────────┬──────────────────┐
      ↓                      ↓                   ↓
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│ instrumented │  │     replay       │  │      db      │
│    _agent    │  │   (with diff)    │  │   (queries)  │
└──────┬───────┘  └────────┬─────────┘  └──────────────┘
       │                   │
       │ uses              │ uses
       ↓                   ↓
┌──────────────────────────────────────┐
│         main_agent + tools           │
│  (Mistral API + W&B/HF/ElevenLabs)   │
└──────────────────────────────────────┘
       │
       │ logs to
       ↓
┌──────────────────────────────────────┐
│        DuckDB (6 tables)             │
│  runs | messages | tool_calls |      │
│  memory | diffs | evaluations        │
└──────────────────────────────────────┘
```

---

## Unanswered Questions
1. ~~What database is used?~~ → **Answered:** DuckDB (file-based, 6 tables)
2. ~~How is cost tracking implemented?~~ → **Answered:** Per-model pricing dict, average input/output rates
3. ~~Is W&B logging required?~~ → **Answered:** No, optional, best-effort with silent failure
4. ~~What tools are available to the agent?~~ → **Answered:** wandb_log, hf_list_models, elevenlabs_speak
5. **Why are memory_snapshots and evaluations tables unused?** → Phase 2 feature, not yet implemented
6. **What's the plan for production scalability?** → PROGRESS.md mentions switching to PostgreSQL
7. **How to add a new tool?** → Add function to tools.py + JSON schema to get_tools()
8. **Can multiple agents run concurrently?** → Yes, but DuckDB may have WAL conflicts under heavy load

---

## Current State Summary

### ✅ What's Working (Phase 1 Complete)
- Full agent execution with Mistral API
- Complete observability logging (runs, messages, tool calls)
- Cost tracking for 7 Mistral models
- Replay with parameter overrides and unified diffs
- 4-page Streamlit dashboard (metrics, run agent, detail, replay)
- 4-endpoint REST API (run, replay, list, detail)
- Optional W&B logging
- 19 passing tests (costs, db, replay)
- Docker compose setup (backend + frontend)

### ⏳ What's Not Yet Implemented (Phase 2+)
- Memory snapshot capture and viewer
- Evaluation metric logging and dashboard
- HuggingFace leaderboard export
- ElevenLabs cost tracking
- W&B artifact upload
- Multi-agent session grouping
- API authentication
- GitHub Actions CI
- Cost spike alerts
- Retry logic for API failures

### 🎯 Recommended Next Tasks
1. **Add retry logic** to Mistral API calls (exponential backoff)
2. **Implement memory snapshot capture** in instrumented_agent.py
3. **Add evaluation logging** endpoint and dashboard
4. **Set up GitHub Actions CI** for automated testing
5. **Add API authentication** middleware
