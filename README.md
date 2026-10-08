# AI Literature Review Assistant

[![CI](https://github.com/JShi12/AI-literature-review-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/JShi12/AI-literature-review-assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

An AI tool to summarize research-paper PDFs into structured, citation-grounded literature review drafts.

**Live demo:** [ai-literature-review-assistant-2jld.onrender.com](https://ai-literature-review-assistant-2jld.onrender.com/)
(hosted on Render's free tier — the first request after a period of inactivity may take a little while to
wake up). Password: `AI Literature Review Assistant`. The demo is read-only
(`READ_ONLY_DEMO`, see [Deployment](#deployment)): you can browse the
pre-loaded example papers, claims, syntheses, and generated review draft, but actions that call the
OpenAI API or change stored data are disabled, so the password can be shared openly here without any
cost or data risk.

![Upload Papers tab of the Streamlit app](docs/screenshot.png)

Example output — a review draft generated end to end from three real papers on autonomous driving
(the same example data the live demo above is seeded with):

![Generated review draft with citation-grounded body text and a bullet-free reference list](docs/demo-review-draft.png)

## Why this is interesting

- **Sentence-level citation traceability**: every sentence in a generated review draft is linked back to
  the specific claims (and papers) that support it, not just a generic reference list.
- **Structured, validated LLM outputs**: claims, syntheses, and review drafts are all extracted via OpenAI
  structured outputs into Pydantic schemas, rather than parsed out of free-form text.
- **A real, if small, extraction pipeline**: PDFs are split into pages, academic sections are detected
  heuristically, and text is chunked section-aware before ever reaching the LLM.
- **Retrieval-augmented selection**: claims and syntheses are embedded (pgvector) as they're created, so
  entering a topic pulls in the claims/syntheses most semantically relevant to it instead of just the
  most recently created ones — falling back to recency automatically if no matching embeddings exist yet.
- **Agentic orchestration**: a PydanticAI agent (the **Ask the Assistant** tab) can chain the retrieval,
  synthesis, and review-drafting steps above from a single natural-language request, calling the same
  underlying functions as their respective tabs and showing exactly which tools it called.
- **Measured, not eyeballed**: an offline [evaluation harness](#evaluation) scores every stage (section
  detection, claims, retrieval, syntheses, review drafts, and the agent's tool use) with deterministic
  metrics, gold labels, and calibrated LLM judges. CI replays it on every pull request and blocks
  regressions. The harness found and drove fixes for several real problems (see the table below).

## Architecture

```mermaid
flowchart LR
    User -->|uploads PDFs| UI["Streamlit UI (app.py)"]
    UI --> Services["services.py / pipeline/*"]
    Services --> DB[("PostgreSQL + pgvector")]
    Services --> LLM["llm/* (claims, synthesis, review, agent)"]
    UI -->|"Ask the Assistant tab"| LLM
    LLM -->|calls, requesting structured output| OpenAI[("OpenAI API")]
    LLM --> DB
    UI --> DB
```

Data flows through the pipeline in stages, each persisted for traceability:

```text
PDFs -> pages -> sections -> chunks -> claims -> syntheses -> review draft
```

From claims onward, these stages can be driven either by hand (the Claims/Syntheses/Review Drafts
tabs) or by the **Ask the Assistant** agent from a single natural-language request — either way, the
same underlying functions run and the same data is persisted.

## Features

- Upload and ingest academic PDFs.
- Extract page text, paper metadata, academic sections, and section-aware chunks.
- Use OpenAI structured outputs to extract source-grounded claims from paper chunks.
- Generate claim-backed syntheses across themes, contradictions, gaps, method comparisons, and insights.
- Select the claims/syntheses most relevant to a given topic via pgvector embedding similarity, falling
  back to recency when no matching embeddings exist yet.
- Produce Markdown literature review drafts with numeric citations and references.
- Store papers, chunks, claims, syntheses, review drafts, and traceability links in PostgreSQL.
- Ask a PydanticAI agent (**Ask the Assistant** tab) to chain the retrieval/synthesis/review-drafting
  steps above from a single natural-language request, calling the same underlying functions as their
  respective tabs and showing which tools it called along the way.
- Run locally with Docker Compose or a Python environment.

## Tech Stack

- Python 3.12
- Streamlit
- PostgreSQL 16 + pgvector
- SQLAlchemy + Alembic
- PyMuPDF
- Pydantic
- PydanticAI (the **Ask the Assistant** tab's agent and tool calls)
- OpenAI Python SDK 
  
  In the demo I used `gpt-4.1-mini` for claim/synthesis/review-draft generation, and `text-embedding-3-small` for embeddings, both configurable via `OPENAI_CHAT_MODEL` / `OPENAI_EMBEDDING_MODEL`. Other models haven't been compared yet; the evaluation harness supports a side-by-side run
  (`python -m evals.run --model gpt-4.1-mini --model gpt-4.1`).
- Pytest, Ruff, Mypy, pre-commit, GitHub Actions
- Deployed on Render (Docker) with a managed Postgres from Neon/Supabase

## Project Structure

```text
.
+-- alembic/                      Database migrations
+-- src/lit_review_assistant/
|   +-- app.py                    Streamlit UI
|   +-- db/                       SQLAlchemy models and sessions
|   +-- llm/                      OpenAI claim, synthesis, and review workflows
|   +-- pipeline/                 PDF extraction, sectioning, chunking, traceability
|   +-- schemas.py                Structured-output schemas
|   +-- services.py               Application services
|   +-- logging_config.py         Logging setup
+-- evals/                        Offline evaluation harness (see evals/README.md)
|   +-- datasets/                 Gold labels, review scenarios, agent cases, judge calibration set
|   +-- recordings/               Recorded LLM responses and embeddings, for free offline replay
|   +-- baselines/                Committed metric baseline the CI gate compares against
+-- scripts/                      Demo data seeding and metadata fixes
+-- tests/                        Test suite
+-- .github/workflows/ci.yml      Lint, types, tests, and the eval replay + regression gate
+-- docker-compose.yml            App + PostgreSQL/pgvector services
+-- Dockerfile
+-- pyproject.toml
```

## Quick Start

Create a local environment file:

macOS/Linux:

```bash
cp .env.example .env
```

Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env` if you want to run claim extraction, synthesis generation, or review drafting.

Run with Docker:

```bash
docker compose up -d db
docker compose build app
docker compose run --rm app alembic upgrade head
docker compose up -d app
```

Open:

```text
http://localhost:8501
```

## Local Development

This runs the Python app directly on your machine, but it still needs a real PostgreSQL + pgvector
server to talk to -- the Python packages installed below are just client libraries (`psycopg`,
`sqlalchemy`, `pgvector`), not a database server. Easiest way to get one: reuse the `db` service
from `docker-compose.yml` (same as [Quick Start](#quick-start)), without running the app in Docker too:

```bash
docker compose up -d db
```

That starts Postgres 16 with pgvector on `localhost:5432` with the default credentials the app
expects (see below). Alternatively, install PostgreSQL natively and add the `vector` extension
yourself (e.g. via [pgvector's own install instructions](https://github.com/pgvector/pgvector#installation))
-- more setup, but no Docker dependency.

Then set up the Python app itself:

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
alembic upgrade head
streamlit run src/lit_review_assistant/app.py
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
alembic upgrade head
streamlit run src/lit_review_assistant/app.py
```

The default local database URL is:

```text
postgresql+psycopg://litreview:litreview@localhost:5432/litreview
```

PostgreSQL must have the `vector` extension available.

## Deployment

Deployed on [Render](https://render.com) (Docker) with an external Postgres from
[Neon](https://neon.tech)/[Supabase](https://supabase.com) -- Render's own free Postgres plan expires
after a fixed trial, so the database lives elsewhere. The live demo above runs in **read-only mode**
(`READ_ONLY_DEMO`): pre-seeded with example data, with every action that calls the OpenAI API or
writes to the database disabled.

Full deployment steps, environment variables, and the demo-seeding scripts are documented in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Testing

```bash
pytest
```

The tests cover PDF extraction, section detection, chunking offsets, metadata extraction, structured LLM
prompt construction, schema validation, support counting, review traceability helpers, embedding
generation/retrieval, quote-based claim location, database URL handling, and the evaluation harness
itself (metrics, record/replay, judges, the regression gate, agent trajectory checks). All tests are
self-contained unit tests (fakes and `monkeypatch`, no live database required). `OPENAI_API_KEY` is
blanked for every test, so no test can reach the live API. `pytest --cov` (configured by default)
reports coverage.

## Evaluation

`evals/` is an offline harness that scores the whole pipeline on a frozen snapshot of the three demo papers:

- **Section detection, claims, retrieval, syntheses, and review drafts.** These are scored with deterministic
  metrics: whether claim locations point at the claim text, recall and precision against gold claims,
  retrieval precision@k against labelled relevant chunks, invented citation IDs, and citation coverage.
- **LLM judges.** `gpt-4.1` judges claim grounding, synthesis faithfulness, and whether each review
  sentence is supported by its own citations. The judges are calibrated against a labelled set that
  includes deliberately broken items.
- **The Ask the Assistant agent.** Checks cover which tools it calls and in what order, and whether it
  passes made-up IDs to tools.

Every LLM response is recorded, so runs replay offline for free. In CI, the eval replays on every push and
pull request and fails if any metric drops below the committed baseline.

```bash
python -m evals.snapshot                 # once: build the input snapshot (downloads the demo papers)
python -m evals.run                      # score all stages, recording new LLM responses
python -m evals.run --mode replay        # re-score from recordings: free, no API key
python -m evals.gate                     # compare with the committed baseline
python -m evals.calibrate                # how far the LLM judges can be trusted
```

Problems the harness found, and the effect of the fixes (gpt-4.1-mini):

| Problem | Before | After |
|---|---|---|
| Claim locations pointing at the claim text (the model now quotes; offsets are computed in code) | 0.8% | 81% |
| Section headings detected (precision 100%) | 18% | 99% |
| Chunks given the correct section type | 27% | 86% |
| Review sentences fully supported by their citations (LLM judge) | 42% | 77–82% |
| Syntheses drawing on evidence from 2+ papers | 32% | 77–91% |
| Agent review requests saved as traceable drafts | 2/8 | 7/8 |

Also fixed:
- **Rolled-back extraction runs:** one bad claim offset rolled back an entire claim-extraction run.
- **Synthesis IDs cited as claim IDs** in review drafts.
- **Phantom support:** sentences were stored as supported with no claims behind them.

Synthesis faithfulness is the one thing the harness can't yet judge reliably: with 11–19 syntheses per
run, it moves by about 25 points between runs. See [evals/README.md](evals/README.md) for every metric,
the calibration results, and how to change a prompt.

## Code Quality / CI

Ruff (lint + format), Mypy, and the full test suite run in GitHub Actions on every push and pull request.
A second job, `eval-replay`, does the following:
- builds the eval snapshot;
- replays the evaluation and the judge calibration from committed recordings (no API key, no cost);
- fails if any metric is worse than `evals/baselines/baseline.json`, or if a prompt changed without
  re-recording.

Locally, `pre-commit install` wires the lint, format, and type checks into `git commit`.

## What's not done yet

- Scanned or image-only PDFs aren't supported -- there's no OCR step, so the pipeline only works with
  PDFs that already have a selectable text layer.
- Authentication is a single shared password (`APP_PASSWORD`), not per-user accounts — the app has no
  user/tenant concept, so everyone with the password sees the same shared workspace and data. See
  [Deployment](#deployment) for how it's configured.
- The evaluation's gold labels and judge-calibration labels were drafted by Claude and haven't been
  reviewed by a person yet, so treat the eval numbers as indicative.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
