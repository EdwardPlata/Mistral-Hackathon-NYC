# AgentOp-Studio — Progress Tracker

_Last updated: 2026-02-28_

---

## Phase 1 — Core Observability MVP

**Status: COMPLETE**

### Backend

| Component | File | Details | Status |
|-----------|------|---------|--------|
| DuckDB schema | `backend/db.py` | 6 tables: runs, messages, tool_calls, memory_snapshots, diffs, evaluations | ✅ Done |
| DB connection helper | `backend/db.py` | `get_conn()` / `init_schema()` — fresh connection per call, idempotent schema init | ✅ Done |
| Cost estimator | `backend/costs.py` | `estimate_cost(tokens, model)` — 7 Mistral models with input/output rates | ✅ Done |
| Instrumented agent | `backend/instrumented_agent.py` | Full observability loop: user msg, assistant msgs, tool calls (name/args/result/latency), tokens, cost | ✅ Done |
| W&B optional logging | `backend/instrumented_agent.py` | `_log_to_wandb()` — logs tokens, cost, status per run; silently skipped if no `WANDB_API_KEY` | ✅ Done |
| Replay & diff | `backend/replay.py` | `replay_run()` — env-var overrides, re-run, unified diff, stored in `diffs` table | ✅ Done |
| FastAPI: POST /run | `backend/main.py` | Runs instrumented agent, returns `{run_id, response, total_tokens, total_cost}` | ✅ Done |
| FastAPI: POST /replay | `backend/main.py` | Replays with override params, returns diff + both responses | ✅ Done |
| FastAPI: GET /runs | `backend/main.py` | Lists all runs, ordered by start_time DESC | ✅ Done |
| FastAPI: GET /runs/{id} | `backend/main.py` | Full run detail: summary + messages thread + tool calls | ✅ Done |

### Agents

| Component | File | Details | Status |
|-----------|------|---------|--------|
| Tool definitions | `agents/tools.py` | `wandb_log`, `hf_list_models`, `elevenlabs_speak` + Mistral JSON schemas | ✅ Done |
| Base agent loop | `agents/main_agent.py` | `run_agent()` — uninstrumented agentic loop, max 5 tool rounds | ✅ Done |
| Agent config | `agents/config.yaml` | Model/tool settings with env-var interpolation | ✅ Done |

### Frontend

| Component | File | Details | Status |
|-----------|------|---------|--------|
| Dashboard page | `frontend/app.py` | KPI metrics (runs, tokens, cost), bar charts, full runs table | ✅ Done |
| Run Agent page | `frontend/app.py` | Prompt input, user ID, submit → shows response + metrics | ✅ Done |
| Run Detail page | `frontend/app.py` | Message thread with role icons, tool calls table | ✅ Done |
| Replay & Diff page | `frontend/app.py` | Run selector, temperature/model overrides, side-by-side diff view | ✅ Done |

### Infrastructure

| Component | File | Details | Status |
|-----------|------|---------|--------|
| Dockerfile | `infra/Dockerfile` | Python 3.11-slim + uv, installs `agentops` extras | ✅ Done |
| docker-compose.yml | `infra/docker-compose.yml` | backend (port 8000) + frontend (port 8501), PYTHONPATH set correctly | ✅ Done |

### Tests

| Test File | Tests | Coverage | Status |
|-----------|-------|----------|--------|
| `tests/test_costs.py` | 6 | Cost estimation: known models, zero tokens, unknown model fallback, scaling | ✅ Done |
| `tests/test_replay.py` | 6 | Diff logic: identical strings, changed strings, unified format, multiline, add/remove | ✅ Done |
| `tests/test_db.py` | 7 | Schema creation, runs/messages/tool_calls/diffs/evaluations CRUD with tmp_path fixtures | ✅ Done |
| **Total** | **19** | | ✅ All passing |

---

## Phase 2 — Integrations & Evaluation

**Status: PENDING**

| Feature | Priority | Notes | Status |
|---------|----------|-------|--------|
| Memory snapshot capture | High | Populate `memory_snapshots` table during agent runs | Pending |
| Memory viewer (frontend) | High | New page or expander in Run Detail | Pending |
| Evaluation metric logging | High | `POST /eval` endpoint + evaluations table write | Pending |
| Evaluation dashboard | Medium | Charts: success rate, custom metrics over time | Pending |
| HuggingFace leaderboard export | Medium | Upload evaluation results as HF dataset | Pending |
| ElevenLabs TTS cost tracking | Low | Extend costs.py with ElevenLabs per-char pricing | Pending |
| W&B artifact upload | Medium | Upload DuckDB snapshot or run JSONs as W&B artifacts | Pending |
| Multi-agent run grouping | Low | Group runs by session/experiment in DB + UI | Pending |

---

## Phase 3 — Polish, Auth & CI/CD

**Status: PENDING**

| Feature | Priority | Notes | Status |
|---------|----------|-------|--------|
| API key auth middleware | Medium | FastAPI dependency injection, check `X-API-Key` header | Pending |
| GitHub Actions CI | Medium | Lint (ruff) + test on every push | Pending |
| Cost spike alerts | Low | Configurable threshold, email/webhook notification | Pending |
| Fine-tune loop | Low | Auto-collect failed runs → Mistral fine-tuning API | Pending |
| Kubernetes manifests | Low | `infra/k8s/` deployment configs | Pending |

---

## Known Issues / Tech Debt

| Issue | Impact | Fix |
|-------|--------|-----|
| ~~`eval()` used in frontend for config JSON parsing~~ | ~~Security risk~~ | Fixed: replaced with `json.loads()` in `frontend/app.py` | ✅ Fixed |
| `memory_snapshots` table never written | Feature gap | Wire into instrumented_agent.py in Phase 2 |
| `evaluations` table never written | Feature gap | Add POST /eval endpoint in Phase 2 |
| Single DuckDB file has no WAL conflict protection | Race condition risk under concurrent load | Acceptable for demo; use Postgres for production |
| No retry logic on Mistral API failures | Flaky under rate limits | Add exponential backoff |

---

## Architecture Summary

```
User
 └── Streamlit (port 8501)
      └── httpx → FastAPI (port 8000)
                   ├── instrumented_agent.py → Mistral API
                   │                         → agents/tools.py (W&B / HF / ElevenLabs)
                   │                         → DuckDB (data/agentops.duckdb)
                   │                         → W&B (optional)
                   ├── replay.py → instrumented_agent (re-run) → difflib
                   └── db.py (DuckDB reads for /runs endpoints)
```

---

## Quick Start

```bash
# From repo root
uv sync --extra agentops

# Export API key
export MISTRAL_API_KEY=your_key_here

# Backend
PYTHONPATH=AgentOp-Studio uvicorn backend.main:app --reload --port 8000

# Frontend (new terminal)
PYTHONPATH=AgentOp-Studio streamlit run AgentOp-Studio/frontend/app.py

# Tests
uv run pytest AgentOp-Studio/tests/ -v
```
