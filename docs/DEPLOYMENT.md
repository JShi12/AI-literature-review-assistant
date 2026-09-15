# Deployment (Render free tier + external Postgres)

The repo includes a `Dockerfile` that auto-runs `alembic upgrade head` on startup and binds to the
`PORT` environment variable most PaaS hosts inject (falling back to 8501 locally), plus a `render.yaml`
Blueprint for a one-click deploy. Render's Blueprint YAML fields change over time, so if the Blueprint
doesn't work as-is, use the manual steps below instead -- they don't depend on `render.yaml` at all.

The database is hosted externally (e.g. [Neon](https://neon.tech) or [Supabase](https://supabase.com)),
not on Render's own managed Postgres -- see "Why an external database" below.

## Manual setup

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

## Seeding example data for a read-only demo

`scripts/seed_demo_data.py` downloads three real, open-access papers on autonomous driving from
arXiv, ingests them, and runs the full pipeline (claims -> syntheses -> a review draft) so a
`READ_ONLY_DEMO` deployment has real content to show. Run it once, locally, with `DATABASE_URL`
pointed at your production database (not local Postgres):

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

## Why an external database

Render's free PostgreSQL plan is a fixed-length trial -- the database is deleted after it expires,
not just paused, regardless of activity. Neon and Supabase both have a pgvector-capable free tier
that pauses/scales to zero on inactivity instead of being deleted, which suits an
infrequently-visited portfolio deployment much better. See their current docs for exact free tier
terms, since these change over time.

## Free tier limitations worth knowing regardless

Render's free web services spin down after ~15 minutes of inactivity, so the first request after
idling takes a while to cold-start; an external database that's paused will add its own wake-up
delay on top of that for the very first request.

## One-click alternative

Click "New Blueprint Instance" on Render and point it at this repo -- `render.yaml` provisions the
web service. You'll be prompted to fill in `DATABASE_URL`, `OPENAI_API_KEY`, and (optionally)
`APP_PASSWORD` manually, since none of those are stored in the Blueprint.
