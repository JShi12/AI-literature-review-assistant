# AI Literature Review Assistant

Local MVP for turning research-paper PDFs into claim-backed literature review drafts.

```text
PDFs -> pages -> sections -> chunks -> extracted claims -> syntheses -> cited review draft
```

The Streamlit app supports PDF ingestion, paper-structure inspection, LLM claim extraction, synthesis generation, review-draft generation, and database count inspection. Generated review drafts are built from intermediate syntheses and store sentence-level support metadata linking evidence-backed sentences to their source claims.

## Features

- Upload and ingest one or more PDF papers.
- Extract page text with PyMuPDF.
- Detect common academic sections such as Abstract, Introduction, Methods, Results, Discussion, Limitations, and Conclusion.
- Create section-aware text chunks with stable page-level character offsets.
- Avoid duplicate ingestion by hashing uploaded PDFs with SHA-256.
- Extract structured, source-grounded claims from chunks using OpenAI structured outputs.
- Generate claim-backed syntheses across themes, contradictions, gaps, method comparisons, and higher-level insights.
- Generate Markdown literature review drafts with numeric academic-style citations and a References section.
- Store papers, pages, sections, chunks, claims, syntheses, review drafts, sentence support, LLM runs, and embeddings metadata in PostgreSQL.
- Run locally with Docker Compose or directly with Python.

## Current Status

This project is a local MVP, not a production deployment. The core workflow is implemented through Streamlit tabs:

- `Upload Papers`: save PDFs, extract text, detect sections, and create chunks.
- `Paper Structure`: inspect recent papers and their extracted structure.
- `Claims`: run LLM claim extraction over selected paper chunks.
- `Syntheses`: generate claim-supported synthesis items.
- `Review Drafts`: generate and display the latest literature review draft.
- `Database`: inspect table counts.

The database includes an `embeddings` table using `pgvector`, and the environment configuration includes an embedding model setting. Embedding generation and retrieval are not currently wired into the UI workflow.

## Stack

- Python 3.12
- Streamlit
- PostgreSQL 16 with `pgvector`
- SQLAlchemy + Alembic
- PyMuPDF
- Pydantic
- OpenAI Python SDK
- Docker Compose
- Pytest

## Repository Layout

```text
.
+-- alembic/                         Database migration environment
+-- src/lit_review_assistant/
|   +-- app.py                       Streamlit application entry point
|   +-- db/                          SQLAlchemy models and session handling
|   +-- llm/                         Structured OpenAI workflows
|   +-- pipeline/                    PDF, sectioning, chunking, and traceability helpers
|   +-- schemas.py                   Pydantic structured-output schemas
|   +-- services.py                  Application service functions
+-- tests/                           Pytest test suite
+-- Dockerfile                       App container image
+-- docker-compose.yml               App + PostgreSQL/pgvector stack
+-- pyproject.toml                   Package metadata and dependencies
+-- .env.example                     Environment variable template
```

## Environment

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Important variables:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `DATABASE_URL` | Yes | `postgresql+psycopg://litreview:litreview@localhost:5432/litreview` | SQLAlchemy database connection string. Docker Compose overrides this to use the `db` service. |
| `OPENAI_API_KEY` | For LLM workflows | Empty | Required for claim extraction, synthesis generation, and review drafting. |
| `OPENAI_CHAT_MODEL` | No | `gpt-4.1-mini` | Chat model used for structured LLM outputs. |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Reserved for embedding workflows. |

PDF ingestion does not require an OpenAI API key. The `Claims`, `Syntheses`, and `Review Drafts` tabs do require one.

Do not commit `.env`, real API keys, uploaded PDFs, or local data. The repository `.gitignore` excludes these by default.

## Run With Docker

From the repository root:

```powershell
docker compose up -d db
docker compose build app
docker compose run --rm app alembic upgrade head
docker compose up -d app
```

Open:

```text
http://localhost:8501
```

Useful Docker commands:

```powershell
docker compose logs -f app
docker compose logs -f db
docker compose down
```

Use `docker compose down -v` only when you intentionally want to delete the local PostgreSQL volume.

## Run Locally Without Docker

Prerequisites:

- Python 3.12
- PostgreSQL 16
- `pgvector` extension available in PostgreSQL

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Set the database URL in the same terminal:

```powershell
$env:DATABASE_URL="postgresql+psycopg://litreview:litreview@localhost:5432/litreview"
```

Set an OpenAI key if you want to run LLM workflows:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
$env:OPENAI_CHAT_MODEL="gpt-4.1-mini"
```

Apply migrations:

```powershell
alembic upgrade head
```

Start Streamlit:

```powershell
streamlit run src/lit_review_assistant/app.py
```

Open:

```text
http://localhost:8501
```

## Database Setup Notes

For a local PostgreSQL database, create the expected user and database:

```sql
CREATE USER litreview WITH PASSWORD 'litreview';
CREATE DATABASE litreview OWNER litreview;
```

Connect to the `litreview` database and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The initial Alembic migration creates these core tables:

- `llm_runs`
- `papers`
- `pages`
- `sections`
- `chunks`
- `claims`
- `syntheses`
- `synthesis_claims`
- `review_drafts`
- `review_sentences`
- `review_sentence_claims`
- `embeddings`

## Using The App

1. Open `http://localhost:8501`.
2. Go to `Upload Papers`.
3. Choose chunk size and chunk overlap.
4. Upload one or more PDFs.
5. Click `Ingest PDFs`.
6. Review extracted structure in `Paper Structure`.
7. Go to `Claims` and extract claims from selected chunks.
8. Go to `Syntheses` and generate one or more synthesis types.
9. Go to `Review Drafts`, enter a topic, and generate a draft.

Chunking defaults:

- `1500` characters with `150` overlap for smaller, more precise chunks.
- `3000` characters with `250` overlap for fewer calls while keeping chunks manageable.
- `4000` characters with `300` overlap for larger, lower-count chunks.

Chunk settings apply only to newly ingested PDFs. If a file has already been ingested, the existing paper record is reused based on its SHA-256 hash.

## LLM Workflow

Claim extraction:

- Processes chunks in paper order, then page order, then character-offset order.
- Produces structured claims with claim type, source IDs, page number, character offsets, and confidence.
- Validates claim offsets against the source chunk before persisting.

Synthesis generation:

- Uses selected recent claims.
- Supports `theme`, `contradiction`, `gap`, `method_comparison`, and `insight`.
- Stores supporting claim links and support counts.

Review drafting:

- Uses selected recent syntheses.
- Generates Markdown.
- Replaces internal claim identifiers with numeric citations.
- Rebuilds the References section from supporting papers.
- Stores sentence-level support links to claims.

## Tests

Run the test suite from the repository root:

```powershell
pytest
```

The tests cover PDF extraction, section detection, chunking offsets, ingestion settings, prompt construction, schema validation, support counting, and review traceability helpers.

## Migrations

Apply migrations:

```powershell
alembic upgrade head
```

Create a new migration after model changes:

```powershell
alembic revision --autogenerate -m "describe_change"
```

Review autogenerated migrations before applying them, especially if they contain destructive schema changes.

## Data And Secrets

Local uploaded PDFs are stored under:

```text
data/uploads
```

The following should remain local and uncommitted:

- `.env`
- API key files
- `data/`
- `.venv/`
- Python caches
- local database dumps

If an API key is exposed, revoke it from the OpenAI dashboard and create a new one.

## Limitations

- Scanned or image-only PDFs are not OCR'd.
- Embedding storage is modeled, but embedding generation and retrieval are not yet part of the UI workflow.
- There is no authentication layer.
- The default Docker database credentials are for local development only.
- Uploaded PDFs may contain sensitive or licensed content, so retention and access controls should be decided before wider deployment.

## Production Notes

Before exposing this outside a trusted local environment:

- Replace default database credentials.
- Store secrets in a secret manager.
- Add authentication.
- Serve Streamlit behind a reverse proxy.
- Use HTTPS.
- Configure persistent storage and backups for PostgreSQL and uploaded files.
- Add resource limits for the app and database containers.
- Run migrations as a release step.
- Keep PostgreSQL private to the application network.

## Useful Commands

```powershell
# Start services
docker compose up -d

# Run migrations in Docker
docker compose run --rm app alembic upgrade head

# Start local Streamlit app
streamlit run src/lit_review_assistant/app.py

# Run tests
pytest

# Check Git status
git status --short
```
