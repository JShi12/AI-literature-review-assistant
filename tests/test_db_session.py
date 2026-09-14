from __future__ import annotations

from lit_review_assistant.db.session import get_database_url


def test_get_database_url_normalizes_bare_postgres_scheme(monkeypatch) -> None:
    # Hosting providers (Render, Heroku-style platforms) commonly hand out "postgres://" URLs;
    # SQLAlchemy needs an explicit driver, and this project uses psycopg (v3), not psycopg2.
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host:5432/dbname")

    assert get_database_url() == "postgresql+psycopg://user:pass@host:5432/dbname"


def test_get_database_url_normalizes_bare_postgresql_scheme(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/dbname")

    assert get_database_url() == "postgresql+psycopg://user:pass@host:5432/dbname"


def test_get_database_url_leaves_explicit_driver_untouched(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@host:5432/dbname")

    assert get_database_url() == "postgresql+psycopg://user:pass@host:5432/dbname"


def test_get_database_url_falls_back_to_local_default(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert get_database_url() == "postgresql+psycopg://litreview:litreview@localhost:5432/litreview"
