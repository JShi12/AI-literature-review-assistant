from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

from lit_review_assistant.db.models import Base
from scripts import seed_demo_data as seed


def test_reset_order_covers_every_table_and_deletes_children_first() -> None:
    order = [model.__table__ for model in seed.RESET_ORDER]

    assert set(order) == set(Base.metadata.tables.values())
    for table in order:
        for foreign_key in table.foreign_keys:
            referenced = foreign_key.column.table
            assert order.index(table) < order.index(referenced), (
                f"{table.name} must be deleted before {referenced.name}"
            )


def test_reset_deletes_in_reset_order() -> None:
    class RecordingSession:
        def __init__(self) -> None:
            self.tables: list[str] = []

        def execute(self, statement) -> None:
            self.tables.append(statement.table.name)

    session = RecordingSession()

    seed.reset_database(session)  # type: ignore[arg-type]

    assert session.tables == [model.__tablename__ for model in seed.RESET_ORDER]


def test_reset_aborts_without_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    class CountingSession:
        def scalar(self, _statement) -> int:
            return 3

    @contextmanager
    def fake_scope():
        yield CountingSession()

    def fail(*_args, **_kwargs):
        raise AssertionError("must not run after an aborted reset")

    monkeypatch.setattr(seed, "session_scope", fake_scope)
    monkeypatch.setattr(seed, "reset_database", fail)
    monkeypatch.setattr(seed, "ingest_papers", fail)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    monkeypatch.setattr(sys, "argv", ["seed_demo_data.py", "--reset"])

    seed.main()
