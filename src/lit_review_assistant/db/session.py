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
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://litreview:litreview@localhost:5432/litreview",
    )


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
