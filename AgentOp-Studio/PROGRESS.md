# AgentOp-Studio — Progress Tracker

_Last updated: 2026-03-01_

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
| `tests/test_costs.py` | 13 | Mistral cost estimation (6) + ElevenLabs cost estimation (7) | ✅ Done |
| `tests/test_replay.py` | 6 | Diff logic: identical strings, changed strings, unified format, multiline, add/remove | ✅ Done |
| `tests/test_db.py` | 6 | Schema creation, runs/messages/tool_calls/diffs/evaluations CRUD with tmp_path fixtures | ✅ Done |
| `tests/test_retry.py` | 12 | Retry logic: retryable/non-retryable errors, decorator behavior, config | ✅ Done |
| `tests/test_eval.py` | 11 | Memory snapshot DB (2), POST /eval (3), GET /evals (2), GET /evals/{run_id} (2), GET /runs/{id}/memory (2) | ✅ Done |
| **Total** | **48** | | ✅ All passing |

---

## Phase 2 — Integrations & Evaluation

**Status: MOSTLY COMPLETE**

| Feature | Priority | Notes | Status |
|---------|----------|-------|--------|
| Memory snapshot capture | High | Captures messages state after each tool-call round in `memory_snapshots` table | ✅ Done |
| Memory viewer (frontend) | High | Expandable per-snapshot JSON viewer in Run Detail page | ✅ Done |
| Evaluation metric logging | High | `POST /eval` + `GET /evals` + `GET /evals/{run_id}` endpoints, writes to evaluations table | ✅ Done |
| Evaluation dashboard | Medium | New "Evaluations" page: KPI metrics, bar chart, time-series scatter | ✅ Done |
| Log eval from UI | High | "Log Evaluation" form in Run Detail page | ✅ Done |
| ElevenLabs TTS cost tracking | Low | `estimate_elevenlabs_cost(chars, tier)` in costs.py; 5 tiers with real pricing | ✅ Done |
| W&B artifact upload | Medium | `_log_to_wandb()` now uploads full run JSON as `agent-run` artifact when W&B is configured | ✅ Done |
| HuggingFace leaderboard export | Medium | Upload evaluation results as HF dataset | Pending |
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
| ~~`memory_snapshots` table never written~~ | ~~Feature gap~~ | ✅ Fixed: instrumented_agent.py now captures a snapshot after each tool-call round |
| ~~`evaluations` table never written~~ | ~~Feature gap~~ | ✅ Fixed: POST /eval endpoint writes metrics; frontend has Log Evaluation form |
| ~~No retry logic on Mistral API failures~~ | ~~Flaky under rate limits~~ | ✅ Fixed: backend/retry.py with tenacity exponential backoff |
| ~~`@app.on_event` deprecation warning~~ | ~~Test noise~~ | ✅ Fixed: migrated to lifespan context manager |
| Single DuckDB file has no WAL conflict protection | Race condition risk under concurrent load | Acceptable for demo; use Postgres for production |

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
