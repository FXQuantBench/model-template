# model-template

A self-contained benchmark harness that runs an LLM in an autonomous agentic loop to research and improve a quantitative trading strategy on GBPUSD tick data. The loop issues commands, edits `strategy.py`, runs EDA and backtests, and tracks results on the shared leaderboard.

Supports **any OpenAI-compatible REST API** (OpenAI, Mistral, Together.ai, OpenRouter, Ollama, vLLM, and more), plus Anthropic and Google Gemini natively.

---

## Table of Contents

1. [Repository structure](#1-repository-structure)
2. [Prerequisites](#2-prerequisites)
3. [Step-by-step onboarding](#3-step-by-step-onboarding)
   - [3.1 Fork the template](#31-fork-the-template)
   - [3.2 Create a bot account and PAT](#32-create-a-bot-account-and-pat)
   - [3.3 Configure secrets](#33-configure-secrets)
   - [3.4 Configure variables](#34-configure-variables)
     - [3.4.1 Choose a provider and model](#341-choose-a-provider-and-model)
     - [3.4.2 Set the in-sample date window](#342-set-the-in-sample-date-window)
   - [3.5 Verify the setup](#35-verify-the-setup)
4. [How the agentic loop works](#4-how-the-agentic-loop-works)
5. [File ownership and protected files](#5-file-ownership-and-protected-files)
6. [strategy.py contract](#6-strategypy-contract)
7. [Running a backtest manually](#7-running-a-backtest-manually)
8. [Submitting a PR to main](#8-submitting-a-pr-to-main)
9. [Local development](#9-local-development)

---

## 1. Repository structure

```
model-template/
├── strategy.py              # Your model's strategy — agent-writable, you own this
├── prompt_context.md        # LLM system prompt context — editable by contributors
├── test_runner.py           # Execution engine — READ-ONLY (CODEOWNERS-protected)
├── releases.md              # Changelog — one [vN] entry required per PR to main
├── research_summary.md      # EDA findings table — agent-maintained
├── audit_logs/
│   └── thoughts.md          # Agent reasoning log — required before each command
├── research/                # EDA scripts and output logs — agent-generated
├── harness/
│   ├── providers.py         # LLM adapters: OpenAI-compatible (any base URL), Anthropic, Google
│   └── call_model.py        # CLI wrapper invoked by the agentic loop workflow
├── tests/                   # Unit tests
└── .github/
    ├── loop_state.json       # Agentic loop counters (last_run, daily_count)
    └── workflows/
        ├── agentic_loop.yml  # Core loop — triggers after EDA or backtest completes
        ├── run_eda.yml       # Runs a research/<file_id>.py EDA script in Docker
        ├── run_backtest.yml  # Runs a full vectorbt backtest in Docker
        ├── pr_guard.yml      # Enforces quality gates on PRs to main
        └── daily_eval.yml    # Out-of-sample eval, runs Tue–Sat at 08:30 UTC
```

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| GitHub account | Must have permission to fork the `fxquantbench/model-template` repo |
| `BENCHMARK_BOT_TOKEN` | **Provided automatically** — org-level secret on `fxquantbench`; no action required |
| `HF_TOKEN_RO` | **Provided automatically** — org-level secret on `fxquantbench`; no action required |
| LLM API key | Any OpenAI-compatible endpoint, Anthropic, or Google — one is enough |

No local tooling is required to run the benchmark — everything executes in GitHub Actions. A local Python ≥ 3.11 environment (or `uv`) is only needed to run tests and iterate on `strategy.py` before letting the agent take over.

---

## 3. Step-by-step onboarding

### 3.1 Fork the template

1. Click **Use this template → Create a new repository** (not a plain fork, so CI state is fresh).
2. Name your repo anything you like; keep it **private** until you are ready to submit.
3. Clone it locally:

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

### 3.2 Org secrets (no action required)

`BENCHMARK_BOT_TOKEN` and `HF_TOKEN_RO` are **organisation-level secrets** managed by `fxquantbench`. They are automatically inherited by every repository created from this template inside the org. You do not need to create a bot account, generate a PAT, or obtain a HuggingFace token.

If you created your repo outside the `fxquantbench` org, you need to transfer it into the org so the secrets are inherited automatically:

1. Open an issue in [fxquantbench/model-template](https://github.com/fxquantbench/model-template) letting the benchmark admin know you want to transfer your repo in.
2. Once the admin confirms they are ready to accept, go to your repo **Settings → General → Danger Zone → Transfer ownership**, enter `fxquantbench` as the destination, and confirm.
3. The benchmark admin will accept the transfer — your repo moves into the org and the secrets become available immediately. You retain admin access to your repo throughout.

### 3.3 Configure secrets

Go to **Settings → Secrets and variables → Actions → Secrets** in your repository and add:

| Secret name | Value |
|---|---|
| `MODEL_API_KEY` | Your LLM provider API key |

`BENCHMARK_BOT_TOKEN` and `HF_TOKEN_RO` are inherited from the org — do not add them manually.

### 3.4 Configure variables

Go to **Settings → Secrets and variables → Actions → Variables** and add:

| Variable name | Required | Description | Example |
|---|---|---|---|
| `MODEL_PROVIDER` | Yes | LLM provider — `openai`, `anthropic`, or `google` | `openai` |
| `MODEL_ID` | Yes | Model identifier string | `gpt-4o` |
| `IN_SAMPLE_START` | Yes | Inclusive start of the training window (`YYYY-MM-DD`) | `2022-01-03` |
| `IN_SAMPLE_END` | Yes | Exclusive end of the training window (`YYYY-MM-DD`) | `2024-01-01` |
| `MODEL_BASE_URL` | No | Override the API base URL for `openai` provider (any OpenAI-compatible endpoint) | `https://api.mistral.ai/v1` |
| `MAX_DAILY_ITERATIONS` | No | Max agentic loop runs per day (default: `6`) | `6` |
| `EDA_ARCHIVE_THRESHOLD` | No | Archive oldest EDA files when count exceeds this (default: `30`) | `30` |

#### 3.4.1 Choose a provider and model

Set `MODEL_PROVIDER` and `MODEL_ID` to select your model. The `openai` provider supports **any OpenAI-compatible REST API** — set `MODEL_BASE_URL` to point at a different endpoint.

| `MODEL_PROVIDER` | `MODEL_BASE_URL` | Example `MODEL_ID` values | Notes |
|---|---|---|---|
| `openai` | *(unset — default)* | `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini` | Official OpenAI API |
| `openai` | `https://openrouter.ai/api/v1` | `openai/gpt-5.5`, `anthropic/claude-opus-4-7`, `google/gemini-2.5-pro` | OpenRouter — access any provider via one key |
| `anthropic` | *(n/a)* | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` | Forced tool-use for structured output |
| `google` | *(n/a)* | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` | Uses `google.genai`; `response_mime_type="application/json"` |

> **Note:** Some OpenAI-compatible endpoints do not support the `json_schema` response format. If you hit errors with structured output, the model or endpoint may require a plain `json_object` mode — open an issue to request support for that endpoint.

The model receives the full `prompt_context.md` as the system prompt plus dynamic context (leaderboard summaries, current `strategy.py`, recent thoughts) as the user message.

#### 3.4.2 Set the in-sample date window

`IN_SAMPLE_START` and `IN_SAMPLE_END` define the GBPUSD tick data window the model is allowed to train on. The runner enforces a strict `[start, end)` window — no data outside this range is accessible during EDA or backtest.

Choose dates that leave at least 6 months of unseen data for out-of-sample evaluation. The daily eval job tests `strategy.py` on yesterday's ticks (always outside the in-sample window).

### 3.5 Verify the setup

Trigger the agentic loop manually to confirm everything is wired up:

1. Go to **Actions → Agentic Loop → Run workflow**.
2. Watch the run — the first iteration reads `loop_state.json`, calls the model, applies any `file_changes`, and updates the loop state.
3. Check that a commit appears on the `dev` branch and `audit_logs/thoughts.md` was updated.

If the run fails, the most common causes are:
- `MODEL_PROVIDER` or `MODEL_ID` variable is missing
- `MODEL_API_KEY` secret is not set or is set under a different name (must be exactly `MODEL_API_KEY`)
- `MODEL_BASE_URL` points to an endpoint that does not support `json_schema` response format
- `BENCHMARK_BOT_TOKEN` or `HF_TOKEN_RO` were not inherited from the org — backtest and eval run via reusable workflows that use org secrets from `fxquantbench/model-template` directly

---

## 4. How the agentic loop works

```
Agentic Loop
 │
 ├─ Trigger: workflow_dispatch  OR  after "Run EDA" / "Run Backtest" completes
 │
 ├─ Guard: < 30 min since last run?  → skip
 ├─ Guard: daily_count ≥ MAX_DAILY_ITERATIONS?  → skip
 │
 ├─ Fetch leaderboard summaries (via gh api)
 ├─ Build context file:  prompt_context.md + ---USER--- + dynamic context
 ├─ Call model:  python harness/call_model.py --context-file <path>
 │
 ├─ Model response (JSON):
 │     { "thoughts": "...",
 │       "file_changes": [{"path": "strategy.py", "content": "..."}],
 │       "commands": ["/run-eda <id>"]  |  ["/run-backtest"]  |  [] }
 │
 ├─ Append thoughts → audit_logs/thoughts.md
 ├─ Apply file_changes (skips test_runner.py and prompt_context.md)
 ├─ Dispatch first command → triggers run_eda.yml or run_backtest.yml
 ├─ Commit all changes to dev branch
 └─ Update .github/loop_state.json (last_run, daily_count)
```

The loop re-triggers itself after each EDA or backtest completes, so a single `workflow_dispatch` starts a self-sustaining research cycle up to the daily cap.

---

## 5. File ownership and protected files

| File | Who can modify | Notes |
|---|---|---|
| `strategy.py` | Agent and contributors | The only strategy file the runner executes |
| `audit_logs/thoughts.md` | Agent only | Updated every iteration; required before issuing commands |
| `releases.md` | Agent and contributors | Must contain a `## [vN]` entry before each PR to main |
| `research/` | Agent only | EDA scripts and output logs, auto-managed |
| `research_summary.md` | Agent only | EDA findings table, auto-maintained |
| `test_runner.py` | **CODEOWNERS only** | Protected — the agent cannot overwrite this file |
| `prompt_context.md` | Contributors | Defines the LLM's task and data contract — customise to guide your model |

The CODEOWNERS file enforces that `test_runner.py` requires approval from `@fxquantbench/benchmark-admin` before any PR touching it can be merged. `prompt_context.md` is no longer protected and can be freely edited by contributors.

---

## 6. strategy.py contract

The runner calls `run(conn, start_date, end_date)`. Your function must:

```python
def run(conn, start_date: str, end_date: str) -> pd.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `conn` | `duckdb.DuckDBPyConnection` | Connection with the `GBPUSD` view pre-loaded |
| `start_date` | `str` | Inclusive start date `"YYYY-MM-DD"` |
| `end_date` | `str` | Exclusive end date `"YYYY-MM-DD"` |

Return a `pd.DataFrame` with exactly these columns:

| Column | dtype | Description |
|---|---|---|
| `timestamp_utc` | `int64` | Unix timestamp in milliseconds (UTC) |
| `pair` | `str` | Always `"GBPUSD"` |
| `signal` | `float64` | Target position in `[-1.0, 1.0]` — `+1.0` = 100% long, `-1.0` = 100% short, `0.0` = flat |

Returning an empty DataFrame is valid and means "hold flat / no position". The runner clamps signals to `[-1, 1]` before simulation.

The GBPUSD view has these columns: `timestamp_utc` (int64 ms), `bid` (float), `ask` (float), `bid_volume` (float), `ask_volume` (float). Spread `(ask - bid)` is charged as a fee on every position change — no other commission applies.

---

## 7. Running a backtest manually

The `run_backtest.yml` workflow runs `strategy.py` inside a sandboxed Docker container, writes `result.json`, and commits it to the leaderboard.

**Requirements before triggering:**
- The latest commit on `dev` must include an update to `audit_logs/thoughts.md`.
- `strategy.py` must be syntactically valid Python with a top-level `run(conn, start_date, end_date)` function.

**To trigger:**
1. Go to **Actions → Run Backtest → Run workflow** (select the `dev` branch).
2. The workflow runs the container, validates the result against `ResultSchema`, and pushes `result.json` to the leaderboard under `<MODEL_ID>/results/backtest/<YYYY-MM-DD>-<short-sha>.json`.
3. A summary table with all 17 metrics is posted to the job summary.

**Result fields:**

| Field | Description |
|---|---|
| `sharpe` | Annualised Sharpe ratio |
| `max_drawdown` | Maximum drawdown (fraction, negative) |
| `win_rate` | Fraction of trades that were profitable |
| `calmar_ratio` | Annualised return / max drawdown |
| `annualized_return` | Annualised return (fraction) |
| `volatility` | Annualised return volatility |
| `total_trades` | Number of position changes |
| `avg_spread_cost_pips` | Average spread paid per trade in pips |
| `timed_out` | `true` if the 5-minute container timeout was hit |

---

## 8. Submitting a PR to main

When you are ready to submit, open a PR from `dev` → `main`. The `pr_guard.yml` workflow runs six automated checks:

| Check | What it verifies |
|---|---|
| 1 | `strategy.py` is valid Python (`ast.parse`) |
| 2 | `strategy.py` has a top-level `run(conn, start_date, end_date)` function |
| 3 | `audit_logs/thoughts.md` contains the PR branch name |
| 4 | `releases.md` has a `## [vN]` entry newer than the last merge to `main` |
| 5 | A backtest result JSON for the current HEAD short SHA exists in the leaderboard |
| 6 | The backtest result's `strategy_sha` matches the PR HEAD SHA, and `sharpe > -10.0` |

All six checks must pass for the PR to be mergeable.

**Pre-PR checklist:**
- [ ] `audit_logs/thoughts.md` contains the branch name
- [ ] `releases.md` has a new `## [vN] — <description>` entry
- [ ] A backtest has been run for the current HEAD commit (`/run-backtest` issued by the agent, or triggered manually)
- [ ] Sharpe > -10.0 in the latest backtest result

---

## 9. Local development

Set up a local environment to iterate on `strategy.py` and run tests:

```bash
# Create and activate venv (using uv, recommended)
uv venv --python 3.13
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install test dependencies
uv pip install pytest duckdb pandas numpy pydantic

# Run all unit tests
uv run pytest tests/ -v
```

The test suite runs entirely offline — no HuggingFace token or LLM API key required. `vectorbt` is stubbed in `tests/conftest.py` so the full install is not needed for testing.

To iterate on `strategy.py` locally, query DuckDB directly with your own tick data or a synthetic dataset:

```python
import duckdb
import pandas as pd
from strategy import run

conn = duckdb.connect()
# Create a minimal GBPUSD view for local testing
conn.execute("""
    CREATE VIEW GBPUSD AS
    SELECT * FROM read_parquet('path/to/local/ticks.parquet')
    WHERE pair = 'GBPUSD'
""")

signals = run(conn, "2022-01-03", "2023-01-01")
print(signals.head())
```
