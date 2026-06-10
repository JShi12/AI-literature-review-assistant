# AI Literature Review Assistant

Local MVP for an AI literature review workflow:

```text
PDF -> pages -> sections -> chunks -> claims -> syntheses -> review with sentence-level traceability
```

The current Streamlit UI supports PDF upload/ingestion, paper structure inspection, and database count views. The repository also contains structured OpenAI modules for claim extraction, synthesis generation, review drafting, and traceability, but those LLM actions are not yet wired into the Streamlit UI.

## Stack

- Python 3.12
- Streamlit
- PostgreSQL 16 with `pgvector`
- SQLAlchemy + Alembic
- PyMuPDF for PDF text extraction
- OpenAI Python SDK for structured LLM calls
- Docker Compose for local/container deployment

## Repository Layout

```text
.
+-- alembic/                         Database migration environment
+-- src/lit_review_assistant/
|   +-- app.py                       Streamlit application entry point
|   +-- db/                          SQLAlchemy models and sessions
|   +-- llm/                         Structured OpenAI workflows
|   +-- pipeline/                    PDF extraction, sectioning, chunking, traceability
|   +-- schemas.py                   Pydantic structured-output schemas
|   +-- services.py                  Application service functions
+-- tests/                           Pytest test suite
+-- Dockerfile                       App image
+-- docker-compose.yml               App + PostgreSQL/pgvector stack
+-- pyproject.toml                   Python package metadata
+-- .env.example                     Environment variable template
```

## Runtime Components

The deployed app has two services:

- `db`: PostgreSQL 16 using the `pgvector/pgvector:pg16` image.
- `app`: Streamlit app served on port `8501`.

The Compose deployment stores PostgreSQL data in the named volume `postgres_data` and uploaded PDFs under the host directory `./data`, mounted into the app container at `/app/data`.

## Prerequisites

For Docker deployment:

- Docker Desktop or Docker Engine with Docker Compose v2.
- An OpenAI API key if you plan to run LLM workflows outside the current UI.

For local development without Docker:

- Python 3.12.
- PostgreSQL 16 with the `vector` extension available.
- A database reachable by `DATABASE_URL`.

## Environment Configuration

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

For Docker Compose, `.env` is read by Docker Compose for variable substitution. The database URL is supplied by `docker-compose.yml`, so the important values are:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

For a direct host-based Python run, set `DATABASE_URL` to a database reachable from your machine:

```env
DATABASE_URL=postgresql+psycopg://litreview:litreview@localhost:5432/litreview
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Environment variables:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `DATABASE_URL` | Yes | `postgresql+psycopg://litreview:litreview@localhost:5432/litreview` | SQLAlchemy database connection string. Compose overrides this to use host `db`. |
| `OPENAI_API_KEY` | Only for LLM workflows | Empty | API key used by `OpenAIStructuredLLM`. PDF ingestion does not currently require it. |
| `OPENAI_CHAT_MODEL` | No | `gpt-4.1-mini` | Structured-output chat model for claim/synthesis/review generation. |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model name reserved for embedding workflows. |

Do not commit `.env` or real API keys.

## OpenAI API Key Setup

The PDF upload and ingestion workflow does not require an OpenAI API key. The LLM workflows do require one:

- claim extraction
- synthesis generation
- review draft generation

To create an API key:

1. Sign in to the OpenAI Platform:

```text
https://platform.openai.com/
```

2. Create or select a project.
3. Open the API keys page for that project.
4. Create a new secret key.
5. Copy it immediately and store it somewhere private. You will not be able to view the full key again later.

OpenAI's API docs recommend loading API keys from an environment variable or secret manager instead of hardcoding them in client-side code or committing them to a repository.

For a local PowerShell session, set the key before starting Streamlit:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
$env:OPENAI_CHAT_MODEL="gpt-4.1-mini"
```

You must set this in the same terminal where you run:

```powershell
streamlit run src/lit_review_assistant/app.py
```

To check whether PowerShell has the key for the current session:

```powershell
$env:OPENAI_API_KEY
```

Do not print or paste the value into logs, GitHub, screenshots, or chat messages. If a key is exposed, delete it from the OpenAI dashboard and create a new one.

## Docker Deployment

From the repository root, build and start the database first:

```powershell
docker compose up -d db
```

Build the application image:

```powershell
docker compose build app
```

Run database migrations:

```powershell
docker compose run --rm app alembic upgrade head
```

Start the Streamlit app:

```powershell
docker compose up -d app
```

Open:

```text
http://localhost:8501
```

To view logs:

```powershell
docker compose logs -f app
docker compose logs -f db
```

To stop the stack without deleting data:

```powershell
docker compose down
```

To stop the stack and delete the PostgreSQL volume:

```powershell
docker compose down -v
```

Only use `down -v` when you intentionally want to remove the local database.

## One-Command Local MVP Startup

For quick local testing, this also works:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Then run migrations in another terminal:

```powershell
docker compose exec app alembic upgrade head
```

The more explicit deployment flow above is preferred because the Streamlit app expects migrated tables when it renders database-backed tabs.

## Local Web App Testing Without Docker

This project is a Streamlit web app. There is no separate local FastAPI service to start. For a host-based test run, Streamlit connects directly to PostgreSQL through `DATABASE_URL`.

The expected local database connection is:

```text
postgresql+psycopg://litreview:litreview@localhost:5432/litreview
```

That URL means:

- database type and driver: `postgresql+psycopg`
- database user: `litreview`
- database password: `litreview`
- host: `localhost`
- port: `5432`
- database name: `litreview`

### 1. Install PostgreSQL 16

Install PostgreSQL 16 for Windows from:

```text
https://www.postgresql.org/download/windows/
```

During installation:

- Keep the default PostgreSQL port: `5432`.
- Remember the password you set for the default `postgres` admin user.
- Keep the command-line tools selected.
- The default install directory is usually `C:\Program Files\PostgreSQL\16`.

After installation, open a new PowerShell terminal and make sure `psql` is available:

```powershell
$env:Path += ";C:\Program Files\PostgreSQL\16\bin"
psql --version
```

If `psql` is not found, add this folder to your Windows `Path` environment variable permanently:

```text
C:\Program Files\PostgreSQL\16\bin
```

Confirm the PostgreSQL service is running:

```powershell
Get-Service *postgres*
```

If the service is stopped, start it:

```powershell
Start-Service postgresql-x64-16
```

If your service has a different name, use the name shown by `Get-Service`.

### 2. Create the Local Database and User

Connect as the default PostgreSQL admin user:

```powershell
psql -U postgres
```

Enter the password you set during PostgreSQL installation. At the `postgres=#` prompt, run:

```sql
CREATE USER litreview WITH PASSWORD 'litreview';
CREATE DATABASE litreview OWNER litreview;
```

Connect to the new database:

```sql
\c litreview
```

You should now be connected to the `litreview` database.

### 3. Install and Enable pgvector

The app schema includes an `embeddings` table with a vector column, so PostgreSQL needs the `vector` extension.

First try enabling it:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Verify it:

```sql
SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';
```

If this returns one row for `vector`, pgvector is ready and you can skip to the Python setup.

If PostgreSQL reports that the `vector` extension is not available, install pgvector into PostgreSQL 16:

1. Install Git for Windows if `git` is not already installed.
2. Install Visual Studio Build Tools with the `Desktop development with C++` workload.
3. Open an `x64 Native Tools Command Prompt` or Visual Studio Developer Command Prompt that has `nmake` available.
4. Run:

```cmd
set "PGROOT=C:\Program Files\PostgreSQL\16"
cd %TEMP%
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

Then return to PowerShell and enable the extension:

```powershell
psql -U postgres -d litreview
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';
\q
```

### 4. Create and Activate the Python Environment

From the repository root:

```powershell
cd "c:\Users\jings\_ML projects\AI literature assistent"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If `.venv` already exists, just activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Set Environment Variables

Set the database URL in the same PowerShell terminal where you will run migrations and Streamlit:

```powershell
$env:DATABASE_URL="postgresql+psycopg://litreview:litreview@localhost:5432/litreview"
```

The app can ingest PDFs without an OpenAI key. Set `OPENAI_API_KEY` if you are testing LLM workflows from the `Claims`, `Syntheses`, or `Review Drafts` tabs:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
$env:OPENAI_CHAT_MODEL="gpt-4.1-mini"
```

Note: `.env.example` documents the expected variables, but the current host-based Streamlit run does not automatically load `.env`. Set variables in PowerShell before running `alembic` and `streamlit`.

### 6. Run Database Migrations

Apply the project schema to the `litreview` database:

```powershell
alembic upgrade head
```

Successful output should include:

```text
Context impl PostgresqlImpl.
Running upgrade  -> 0001_simplified_v1_schema
```

This creates the tables used by the app, including `papers`, `pages`, `sections`, `chunks`, `claims`, `syntheses`, `review_drafts`, and `embeddings`.

### 7. Start the Web App

Run Streamlit:

```powershell
streamlit run src/lit_review_assistant/app.py
```

Open:

```text
http://localhost:8501
```

### 8. Test the App

1. Open `http://localhost:8501`.
2. Go to the `Upload Papers` tab.
3. Choose `Chunk size` and `Chunk overlap`.
4. Upload one or more PDF files.
5. Click `Ingest PDFs`.
6. Go to `Paper Structure` and confirm the uploaded paper appears.
7. Go to `Database` and confirm the table counts increased.

Chunking options:

- `1500` characters with `150` overlap is the default and gives more precise, smaller chunks.
- `3000` characters with `250` overlap usually reduces API calls while keeping chunks manageable.
- `4000` characters with `300` overlap gives fewer, larger chunks and may be useful for cost-sensitive tests.

Chunk settings affect only newly ingested PDFs. If a PDF was already ingested, the app reuses the existing paper record based on file hash.

To test the LLM workflows:

1. Start Streamlit from a terminal where `OPENAI_API_KEY` is set.
2. Ingest at least one PDF.
3. Go to `Claims`.
4. Choose a paper or keep `All papers`.
5. Set a small `Max chunks to process` value such as `1` or `3` for the first test. For a full review, increase it up to the total selected chunk count.
6. Click `Extract Claims`.
7. Go to `Syntheses`.
8. Keep all synthesis types selected for a comprehensive review, or choose a smaller subset for a focused test.
9. Click `Generate Syntheses`.
10. Go to `Review Drafts`.
11. Enter a review topic.
12. Click `Generate Review Draft`.

The available synthesis types are:

- `theme`: recurring patterns across claims.
- `contradiction`: disagreements or conflicting evidence.
- `gap`: missing evidence, limitations, or unresolved questions.
- `method_comparison`: comparisons between methods, datasets, or experimental setups.
- `insight`: higher-level interpretations supported by the claims.

For a comprehensive literature-review draft, generate all synthesis types first, then create the review draft from the combined recent syntheses.

When claim extraction is run with `All papers`, chunks are processed in paper order, then page order, then character-offset order. If you set `Max chunks to process` lower than the total chunk count, only the first chunks in that order are processed.

Review drafts are prompted to use academic-style numbered citations, such as `[1]` or `[1, 2]`, and to include a `References` section at the end. Citation numbers are built from the papers that support the claims used in the selected syntheses.

The LLM workflow writes results to these tables:

- `llm_runs`
- `claims`
- `syntheses`
- `synthesis_claims`
- `review_drafts`
- `review_sentences`
- `review_sentence_claims`

Uploaded PDFs are stored under:

```text
data/uploads
```

### 9. Stop the Local App

In the terminal running Streamlit, press:

```text
Ctrl+C
```

PostgreSQL can keep running in the background for future local tests.

## Database Migrations

Alembic reads `DATABASE_URL` from the environment. If it is not set, it falls back to the URL in `alembic.ini`.

Apply all migrations:

```powershell
alembic upgrade head
```

Apply migrations inside Docker:

```powershell
docker compose run --rm app alembic upgrade head
```

Create a new migration after model changes:

```powershell
alembic revision --autogenerate -m "describe_change"
```

Review generated migrations before applying them, especially for destructive schema changes.

## Using the App

1. Open `http://localhost:8501`.
2. Go to the `Upload Papers` tab.
3. Upload one or more PDF files.
4. Click `Ingest PDFs`.
5. Review detected pages, sections, and chunks in `Paper Structure`.
6. Use `Database` to verify table counts.

Ingestion behavior:

- PDFs are saved under `data/uploads`.
- The app extracts page text with PyMuPDF.
- Sections are detected from common academic headings such as Abstract, Introduction, Methods, Results, Discussion, Limitations, and Conclusion.
- Text is chunked with stable page-level character offsets.
- Duplicate uploads are detected by SHA-256 file hash and reuse the existing paper record.

## Tests

Run the test suite from the repository root:

```powershell
python -m pip install -e ".[dev]"
pytest
```

The tests cover PDF extraction, section detection, chunking offsets, database-adjacent service behavior, prompt construction, and traceability helpers.

## Production Deployment Notes

This repository is currently shaped as a local MVP. Before exposing it beyond a trusted local network, address these items:

- Replace the default database username, password, and database name in `docker-compose.yml`.
- Put secrets in your deployment platform's secret manager instead of committing them to files.
- Add authentication in front of Streamlit if the app will be reachable by other users.
- Serve Streamlit behind a reverse proxy such as Nginx, Caddy, Traefik, or a platform load balancer.
- Use HTTPS at the reverse proxy or hosting platform.
- Configure persistent storage for both PostgreSQL and `data/uploads`.
- Add database backups and restore testing.
- Decide whether uploaded PDFs contain sensitive or licensed material and set retention rules accordingly.
- Add resource limits for the app and database containers.
- Run migrations as a release step before routing users to the new app container.

Example production-style Compose overrides to consider:

```yaml
services:
  app:
    restart: unless-stopped
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
  db:
    restart: unless-stopped
```

For public deployments, do not publish PostgreSQL directly to the internet. Keep port `5432` private to the application network and expose only the Streamlit/reverse-proxy endpoint.

## Backup and Restore

Create a database backup from the Compose database:

```powershell
docker compose exec db pg_dump -U litreview -d litreview > litreview_backup.sql
```

Restore into a running database:

```powershell
Get-Content .\litreview_backup.sql | docker compose exec -T db psql -U litreview -d litreview
```

Back up uploaded PDFs separately:

```powershell
Compress-Archive -Path .\data -DestinationPath .\data_backup.zip
```

## Troubleshooting

If the app shows missing-table errors, run migrations:

```powershell
docker compose run --rm app alembic upgrade head
```

If the app cannot connect to the database in Docker, confirm the app uses the Compose service hostname:

```text
postgresql+psycopg://litreview:litreview@db:5432/litreview
```

If a local host-based run cannot connect, use `localhost` instead of `db`:

```text
postgresql+psycopg://litreview:litreview@localhost:5432/litreview
```

If migration fails with a missing `vector` extension, make sure you are using the `pgvector/pgvector:pg16` image or install pgvector in your PostgreSQL server.

If OpenAI calls fail, verify:

- `OPENAI_API_KEY` is set in the environment where the Python process runs.
- `OPENAI_CHAT_MODEL` names a model available to your OpenAI account.
- Network egress to the OpenAI API is allowed.

If uploaded PDFs ingest with little or no text, the PDF may be scanned/image-only. The current pipeline uses text extraction and does not perform OCR.

## Useful Commands

```powershell
# Build app image
docker compose build app

# Start services
docker compose up -d

# Stop services
docker compose down

# Run migrations
docker compose run --rm app alembic upgrade head

# Open an app shell
docker compose run --rm app sh

# Run tests locally
pytest
```
