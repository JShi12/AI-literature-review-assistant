# AI Literature Review Assistant

[![CI](https://github.com/JShi12/AI-literature-review-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/JShi12/AI-literature-review-assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

An AI tool to summarize research-paper PDFs into structured, citation-grounded literature review drafts.

**Live demo:** [ai-literature-review-assistant-2jld.onrender.com](https://ai-literature-review-assistant-2jld.onrender.com/)
(hosted on Render's free tier — the first request after a period of inactivity may take a little while to
wake up). The demo is password-protected; reach out if you'd like access.

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

## Architecture

```mermaid
flowchart LR
    User -->|uploads PDFs| UI["Streamlit UI (app.py)"]
    UI --> Services["services.py / pipeline/*"]
    Services --> DB[("PostgreSQL + pgvector")]
    Services --> LLM["llm/* (claims, synthesis, review)"]
    LLM -->|structured outputs| OpenAI[("OpenAI API")]
    LLM --> DB
    UI --> DB
```

Data flows through the pipeline in stages, each persisted for traceability:

```text
PDFs -> pages -> sections -> chunks -> claims -> syntheses -> review draft
```

## Features

- Upload and ingest academic PDFs.
- Extract page text, paper metadata, academic sections, and section-aware chunks.
- Use OpenAI structured outputs to extract source-grounded claims from paper chunks.
- Generate claim-backed syntheses across themes, contradictions, gaps, method comparisons, and insights.
- Select the claims/syntheses most relevant to a given topic via pgvector embedding similarity, falling
  back to recency when no matching embeddings exist yet.
- Produce Markdown literature review drafts with numeric citations and references.
- Store papers, chunks, claims, syntheses, review drafts, and traceability links in PostgreSQL.
- Run locally with Docker Compose or a Python environment.

## Tech Stack

- Python 3.12
- Streamlit
- PostgreSQL 16 + pgvector
- SQLAlchemy + Alembic
- PyMuPDF
- Pydantic
- OpenAI Python SDK
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
+-- tests/                        Test suite
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

## Deployment (Render free tier + external Postgres)

The repo includes a `Dockerfile` that auto-runs `alembic upgrade head` on startup and binds to the
`PORT` environment variable most PaaS hosts inject (falling back to 8501 locally), plus a `render.yaml`
Blueprint for a one-click deploy. Render's Blueprint YAML fields change over time, so if the Blueprint
doesn't work as-is, use the manual steps below instead -- they don't depend on `render.yaml` at all.

The database is hosted externally (e.g. [Neon](https://neon.tech) or [Supabase](https://supabase.com)),
not on Render's own managed Postgres -- see "Why an external database" below.

**Manual setup:**

1. Push this repo to GitHub (it already is, if you're reading this from there).
2. Create a Postgres database on Neon (or Supabase), with the `pgvector` extension available. Copy its
   connection string.
3. On [Render](https://render.com), create a **Web Service** from this GitHub repo, runtime **Docker**
   (it will pick up the `Dockerfile` automatically).
4. Set these environment variables on the web service:
   - `DATABASE_URL` -- the connection string from step 2 (any `postgres://` or `postgresql://` scheme is
     fine; the app normalizes it to the driver it needs).
   - `OPENAI_API_KEY` -- your key.
   - `OPENAI_CHAT_MODEL` / `OPENAI_EMBEDDING_MODEL` -- optional, default to `gpt-4.1-mini` /
     `text-embedding-3-small`.
   - `APP_PASSWORD` -- optional; set this to put a login screen in front of the whole app, since a
     public URL otherwise has no access control. Leave unset for no login screen.
   - `READ_ONLY_DEMO` -- optional; set to `true` to disable every action that calls the OpenAI API
     or writes to the database (uploading, extracting claims, generating syntheses/review drafts,
     backfilling metadata). Recommended alongside `APP_PASSWORD` for a public demo, since a shared
     password can end up more widely known than intended -- visitors can browse real, pre-seeded
     output without being able to trigger new (billed) API calls. See "Seeding example data" below.
5. Set the health check path to `/_stcore/health`.
6. Deploy. The `vector` extension is created automatically by the first migration.

**Seeding example data for a read-only demo:** `scripts/seed_demo_data.py` downloads three real,
open-access papers on autonomous driving from arXiv, ingests them, and runs the full pipeline
(claims -> syntheses -> a review draft) so a `READ_ONLY_DEMO` deployment has real content to show.
Run it once, locally, with `DATABASE_URL` pointed at your production database (not local Postgres):

```bash
DATABASE_URL=<your Neon/Supabase connection string> OPENAI_API_KEY=<your key> \
  python scripts/seed_demo_data.py
```

It's not designed to be re-run repeatedly against the same database -- paper ingestion is
deduplicated by file hash, but claim/synthesis/review-draft generation is not.

If you already ran an older version of this script and the References section of the generated
draft looks garbled (author names mixed into titles, "Anonymous, Submission" as an author, etc.),
that's a known extraction quirk for these three papers' specific byline format, fixed as of this
version. Rather than re-run the whole (billed) pipeline, `scripts/fix_demo_paper_metadata.py`
corrects the three papers' stored title/author/year and regenerates only the review draft:

```bash
DATABASE_URL=<your Neon/Supabase connection string> OPENAI_API_KEY=<your key> \
  python scripts/fix_demo_paper_metadata.py
```

**Why an external database:** Render's free PostgreSQL plan is a fixed-length trial -- the database is
deleted after it expires, not just paused, regardless of activity. Neon and Supabase both have a
pgvector-capable free tier that pauses/scales to zero on inactivity instead of being deleted, which
suits an infrequently-visited portfolio deployment much better. See their current docs for exact free
tier terms, since these change over time.

**Free tier limitations worth knowing regardless:** Render's free web services spin down after ~15
minutes of inactivity, so the first request after idling takes a while to cold-start; an external
database that's paused will add its own wake-up delay on top of that for the very first request.

**One-click alternative:** click "New Blueprint Instance" on Render and point it at this repo --
`render.yaml` provisions the web service. You'll be prompted to fill in `DATABASE_URL`, `OPENAI_API_KEY`,
and (optionally) `APP_PASSWORD` manually, since none of those are stored in the Blueprint.

## Testing

```bash
pytest
```

The tests cover PDF extraction, section detection, chunking offsets, metadata extraction, structured LLM
prompt construction, schema validation, support counting, review traceability helpers, embedding
generation/retrieval, and database URL handling. All tests are self-contained unit tests (fakes and
`monkeypatch`, no live database or `OPENAI_API_KEY` required). `pytest --cov` (configured by default)
reports coverage.

## Code Quality / CI

Ruff (lint + format), Mypy, and the full test suite run in GitHub Actions on every push and pull request
(see the badge above). Locally, `pre-commit install` wires the same checks into `git commit`.

## What's not done yet

- There's no backfill for embeddings: claims/syntheses created before this feature (or created while
  embedding generation failed) won't be selectable by topic similarity until they're regenerated.
- Scanned/image-only PDFs are not OCR'd.
- Authentication is a single shared password (`APP_PASSWORD`), not per-user accounts — the app has no
  user/tenant concept, so everyone with the password sees the same shared workspace and data. See
  [Deployment](#deployment-render-free-tier--external-postgres) for how it's configured.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
