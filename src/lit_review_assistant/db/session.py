"""Database engine/session setup and the session_scope transactional context manager."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Read DATABASE_URL, normalized to the psycopg (v3) driver.

    Hosting providers (Render, Heroku-style platforms, ...) typically hand out a bare
    "postgres://" or "postgresql://" URL. SQLAlchemy needs an explicit driver in the scheme,
    and this project depends on psycopg (v3), not psycopg2, so rewrite the scheme rather than
    asking every deployment target to know that detail.
    """
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://litreview:litreview@localhost:5432/litreview",
    )
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


engine = create_engine(get_database_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Yield a session that commits on success and rolls back and re-raises on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        logger.exception("Database session error; rolling back")
        session.rollback()
        raise
    finally:
        session.close()
